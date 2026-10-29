# Clover-35B-A3B-Preview 参考项目分析：moex 与 OLED-MoE

分析日期：2026 年 10 月 28 日

参考项目：

- `neerajdad123-byte/moe`，检查到的 HEAD：`8148499a4672892d1b2510c48bd46ff12e5c4452`
- `flashserve/OLED-MoE`，检查到的 HEAD：`deec1388e043d6c0887743c2000354f03f3a519d`

上游源码：

- [moex](https://github.com/neerajdad123-byte/moe/tree/8148499a4672892d1b2510c48bd46ff12e5c4452)
- [OLED-MoE](https://github.com/flashserve/OLED-MoE/tree/deec1388e043d6c0887743c2000354f03f3a519d)

## 结论

这两个项目都值得参考，但不能直接把其中任何一个当作 Clover 的现成后端。

最合适的组合是：

```text
moex：三层存储、SSD 预取模拟、低显存量化专家、CPU fallback 的工程思路
OLED-MoE：扩散模型跨 denoising iteration 的路由局部性、pinned host bank、GPU slot cache、异步 H2D、LRU/优先级淘汰
Clover：保留自己的 GDN/full attention、KV cache、coreset、top-k 和 block sampler
```

Clover 应该做的是替换 routed expert 的执行层，不要替换整个 Clover 的 attention、GDN、KV cache 或生成流程。

## moex 可以借鉴的部分

moex 的 README 明确采用了三层思路：热门专家预加载到 VRAM，全部专家保留在 RAM，SSD 只作为冷数据来源。它还把专家缓存容量、预取和 SSD 读延迟放进了单独的模拟器。

最有价值的文件是：

- [README.md](https://github.com/neerajdad123-byte/moe/blob/8148499a4672892d1b2510c48bd46ff12e5c4452/README.md)
- [prefetch_sim.py](https://github.com/neerajdad123-byte/moe/blob/8148499a4672892d1b2510c48bd46ff12e5c4452/tools/prefetch_sim.py)
- [ssd_bench.py](https://github.com/neerajdad123-byte/moe/blob/8148499a4672892d1b2510c48bd46ff12e5c4452/tools/ssd_bench.py)
- [moex.cu](https://github.com/neerajdad123-byte/moe/blob/8148499a4672892d1b2510c48bd46ff12e5c4452/src/moex.cu)
- [cpu_expert.h](https://github.com/neerajdad123-byte/moe/blob/8148499a4672892d1b2510c48bd46ff12e5c4452/src/cpu_expert.h)

它的 `prefetch_sim.py` 将 VRAM、RAM、SSD 分开建模，并模拟：

- VRAM 热专家缓存；
- RAM 专家缓存；
- SSD demand read；
- layer-ahead 预取；
- 预取并发度；
- SSD 读请求排队；
- GPU cache 容量和每层 admission 带宽预算；
- RAM 和 VRAM 之间的 victim/回收关系。

这正好对应 Clover 当前需要的离线模拟器。不过 moex 的专家是量化格式，README 中的单专家大小约为 1.77 MB；Clover 当前 BF16 routed expert 约为 6 MiB。因此它的“6GB GPU 可放约 1496 个专家”不能直接套到 Clover，只能借用缓存算法和测试框架。

moex 还做了两个和 Clover 目标一致的选择：

1. 生成期间尽量避免等待磁盘，把专家预先放到 RAM；
2. GPU cache miss 时允许 CPU 直接执行专家，而不是强制每个 miss 都先搬进 GPU。

第二点很重要。Clover 的动态策略应该比较：

```text
命中 GPU：GPU 执行
GPU miss + CPU 中有专家：CPU 执行，避免 H2D 等待
GPU miss + GPU 计算足够空闲：异步搬运后 GPU 执行
CPU/RAM miss：SSD 预取或 demand read
```

不能默认“所有 miss 都搬进 GPU”一定更快。

## OLED-MoE 可以借鉴的部分

OLED-MoE 的设计与 Clover 更接近。它的架构文档把推理过程拆成：

```text
模型壳层 / KV cache / diffusion decoder
                    |
                MoE runtime
        placement / scheduler / memory / metrics
```

核心文件：

- [architecture.md](https://github.com/flashserve/OLED-MoE/blob/deec1388e043d6c0887743c2000354f03f3a519d/docs/architecture.md)
- [moe_runtime.py](https://github.com/flashserve/OLED-MoE/blob/deec1388e043d6c0887743c2000354f03f3a519d/src/oled_moe/runtime/moe_runtime.py)
- [device_cache.py](https://github.com/flashserve/OLED-MoE/blob/deec1388e043d6c0887743c2000354f03f3a519d/src/oled_moe/memory/device_cache.py)
- [expert_store.py](https://github.com/flashserve/OLED-MoE/blob/deec1388e043d6c0887743c2000354f03f3a519d/src/oled_moe/memory/expert_store.py)
- [priority.py](https://github.com/flashserve/OLED-MoE/blob/deec1388e043d6c0887743c2000354f03f3a519d/src/oled_moe/placement/priority.py)
- [moe_layer.py](https://github.com/flashserve/OLED-MoE/blob/deec1388e043d6c0887743c2000354f03f3a519d/src/oled_moe/models/llada2/moe_layer.py)

它的实现包含几个适合直接移植的机制：

### 1. pinned host bank

`LayerExpertStore` 为每层创建按专家连续排列的 CPU bank。每个专家的 gate/up/down 权重被打包成一个连续切片，这样一次 H2D 可以搬运一个完整专家。

Clover 当前 `CloverExperts` 的权重是：

```python
 gate_up_proj: [num_experts, 2 * intermediate_size, hidden_size]
 down_proj:    [num_experts, hidden_size, intermediate_size]
```

这与 OLED-MoE 的 expert-major 存储方式兼容。Clover 可以把每个 `(layer_id, expert_id)` 转换成一个连续的 `[w13 | w2]` host slice。

### 2. 预分配 GPU slot

`LayerDeviceCache` 不在每次 miss 时创建新的 CUDA tensor，而是预先分配固定数量的 GPU slot，再将专家拷贝到 slot。这样可以避免反复分配、释放和显存碎片。

缓存键必须是：

```text
(layer_id, expert_id)
```

不同层的同编号专家不能共享 slot 的逻辑身份。

### 3. 异步 H2D

OLED-MoE 使用独立 CUDA copy stream，把 pinned host bank 中的专家异步复制到 GPU slot，再用 CUDA event 让计算 stream 等待复制完成。

Clover 可以在当前层的 routed expert 计算期间搬运当前层剩余 miss，具体流程是：

```text
得到当前层最终 expert_ids
→ 划分 resident / miss
→ copy stream 搬运 miss
→ 计算已 resident 的专家
→ event 等待搬运完成
→ 计算 transferred 专家
→ 合并结果
```

### 4. persistent cache 与 transient buffer 分离

OLED-MoE 把专家分成两类：

- 长期保留在 GPU cache 中的专家；
- 这一次 forward 临时搬进 GPU、使用后可以丢弃的专家。

这个设计很适合 Clover，因为 32-token block 的跨轮路由集合较大，若每次都把所有 miss 永久塞进 LRU cache，容易发生 thrashing。Clover 应该保留一部分 persistent hot cache，另设 transient buffer 承接本轮 miss。

### 5. CPU fallback

OLED-MoE 的 scheduler 会把 active experts 分成 GPU hit、GPU transfer 和 CPU execution。moex 也采用了类似思路。

Clover 的 CPU fallback 必须根据以下条件决定：

```text
专家权重是否已经在 RAM
该专家的 token 数量
CPU 执行成本
H2D 搬运成本
当前 GPU copy/compute 是否繁忙
```

对于只有一个或很少 token 使用的冷专家，CPU 执行可能比等待 H2D 更快。对于同一个专家被 block 中多个位置反复使用的情况，搬到 GPU 才更划算。

## 与 Clover 的关键适配点

### Clover 已有的路由逻辑必须保留

Clover 的 `CloverMoeBlock.forward()` 先计算 router logits，再在 preview 阶段执行 coreset，然后选 top-k，并归一化 gate weights。OLED-MoE 的 adapter 可以接入 Clover 的最终 `ids` 和 `weights`，但不能在外部重新做一次普通 top-k，否则会破坏 Clover 的 coreset 行为。

适配点应位于：

```text
CloverMoeBlock.forward
    gate(x)
    coreset_torch(...)       # 保留
    top-k + normalize        # 保留
    CloverExperts.execute(...)  # 替换为 cache runtime
```

### Clover 的 block diffusion 需要跨轮缓存策略

OLED-MoE 重点利用相邻 diffusion iteration 的 expert locality。Clover 也应该把以下信息交给 cache policy：

- request_id；
- block_id；
- denoising_round；
- layer_id；
- 当前最终 active expert IDs；
- gate weights；
- 当前 token 的 confidence；
- 上一轮相同 block 的 route 集合。

当前的 route 统计已经看到：128-token prompt、block_size=32 时，每层单次 preview 的 unique experts 平均约 41 个；跨完整 block 多轮 preview 的专家并集平均约 85 个。这说明“跨轮保留热点专家”比单纯按 token 做 cache 更合理。

### KV cache 不应交给专家缓存管理器

OLED-MoE 把 KV cache 留在 dInfer 的模型壳层；Clover 也应保持这个边界。专家 runtime 只管理 routed expert projections，不应该把 KV cache、GDN state 或 full-attention state 混进 expert slot。

满 128K 时，Clover 的持久化 full-attention KV cache约为 2.5 GiB；当前 attention 的 KV head 复制还会带来临时工作集。因此专家 runtime 的 GPU 预算必须由上层显存规划器传入，而不能直接使用“剩余全部显存”。

建议接口类似：

```python
MoERuntime(
    gpu_expert_budget_bytes=budget_after_kv_and_workspace,
    host_expert_store=pinned_store,
    cpu_executor=..., 
    ssd_store=optional_cold_store,
)
```

## 推荐的 Clover 实现顺序

### 阶段 1：只做 trace 和离线模拟

复用 moex 的 `prefetch_sim.py` 思路，把当前 Clover profile 输出变成：

```text
request_id, block_id, denoising_round, layer_id, expert_id, token_count, confidence
```

模拟每层 2、4、8、16、32、64 个专家容量，并输出：

- expert hit rate；
- token-weighted hit rate；
- H2D bytes；
- CPU fallback bytes/route；
- SSD reads；
- prefetch coverage；
- 每个 block 的等待时间估计。

先做这一阶段，才能知道 dynamic LRU、LFU、confidence priority 哪个适合 Clover。

### 阶段 2：pinned RAM bank + GPU slot

按 OLED-MoE 的结构实现：

- 每层一个 expert-major pinned host bank；
- 固定数量 GPU slot；
- layer-local mapping；
- copy stream + CUDA event；
- persistent cache 与 transient buffer；
- metrics。

先不接 SSD，不做 CPU fallback，验证输出与原始 Clover 逐 token 对齐。

### 阶段 3：加入 CPU fallback

复用 moex/OLED-MoE 的策略，根据 token 数和传输时间选择 GPU 或 CPU。此时要特别注意 Clover 的 BF16 权重和 CPU BF16 计算性能；如果 CPU 端没有高效的量化 kernel，CPU fallback 可能只适合极少量冷路由。

### 阶段 4：SSD 冷数据与预测预取

只有在系统内存确实不足，或者希望减少启动加载时间时才加入 SSD。正常生成路径应优先：

```text
SSD → pinned RAM → GPU transient/persistent slot
```

SSD demand read 只作为最后保底。优先使用 layer-ahead 的历史路由分布预取，而不要假设能提前准确知道下一层的 expert ID；下一层 ID 依赖下一层 hidden state 和 router。

## 最终判断

这两个项目让原方案更可信，但也明确了边界：

- 你的“低频大权重放外部存储，热权重留 GPU”方向是对的；
- SSD 不应成为正常生成路径的同步随机读取层；
- 系统内存应保存完整 routed expert bank，最好使用 pinned memory；
- GPU 应使用预分配 slot，而不是每次 `.to(cuda)`；
- GPU miss 应允许 CPU fallback 和 transient GPU transfer 两种路径竞争；
- OLED-MoE 的跨 denoising iteration locality 机制比普通 token LRU 更适合 Clover；
- KV cache、GDN state、full attention 临时空间必须由上层单独预留；
- 下一步最值得实现的是 Clover 专用 trace replay simulator，然后再写真正的 `CloverExperts` runtime。

