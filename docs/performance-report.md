# Clover-35B-A3B-Preview 输出速度与显存测试报告

测试日期：2026 年 10 月 28 日  
测试机器：Windows；NVIDIA GeForce RTX 5060 Ti，显存 16,311 MiB；系统内存约 128 GiB；SSD 为 ZHITAI TiPlus7100。  
模型：[Diffuspace/Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview)  
推理环境：Python 3.11，PyTorch 2.11.0+cu128，CUDA 12.8，BF16。  
测试目标：预填充可以慢，但生成阶段尽量快；验证专家缓存是否值得做，并把 128K 上下文的 KV cache 和 full attention 临时工作集算进去。

## 一、模型权重和缓存规模

从本地 safetensors 索引统计出的 BF16 权重大小：

| 部分 | 大小 |
|---|---:|
| routed experts | 60.000 GiB |
| GDN / linear attention | 1.886 GiB |
| input embedding | 0.947 GiB |
| lm_head | 0.947 GiB |
| full attention | 0.508 GiB |
| shared experts + gate | 0.235 GiB |
| router | 0.039 GiB |
| 其他 | 约 0.0003 GiB |
| **总计** | **64.562 GiB** |

配置中共有 40 层、256 个 routed experts、每个 token top-8 路由；每 4 层有一个 full attention 层，因此共有 10 个 full attention 层。

单 batch、128K 上下文时，full attention 的持久化 BF16 KV cache 为：

```text
10 层 × 2(K/V) × 2 KV heads × 131072 token × 256 head_dim × 2 bytes
= 2.5 GiB
```

当前实现会把 2 个 KV head 复制成 16 个 attention head。组件实测中，128K cache 加上一个 full attention 层的复制后 K/V 工作集，显存峰值约为 9.41 GiB，其中包含测试脚本保留的非 routed 权重和状态；单看 KV 结构本身，持久化 cache 是 2.5 GiB，单层复制后的 K/V 是约 2.0 GiB。

因此，满 128K 时不能按“固定权重 4.6 GiB，剩下 7～8 GiB 都给专家缓存”规划。需要先为 KV cache、attention 临时张量、激活和 CUDA allocator 余量留空间。

## 二、真实 Clover 路由采样

采样脚本对每层 `CloverExperts.forward()` 收集最终的 `ids.unique()`，因此统计的是经过 coreset 和 top-k 之后真正访问的专家集合。测试使用当前 HF Clover 实现，未修改模型权重。

### 128 token prompt，4 token 输出，block_size=4

- GPU 层数：6 层
- preview calls：40
- causal replay calls：40
- 总耗时：5.91 秒
- CUDA peak allocated：11.342 GiB
- CUDA peak reserved：11.355 GiB
- 每层每次 preview 的 unique experts：约 12～21 个

### 128 token prompt，4 token 输出，block_size=32

- GPU 层数：6 层
- preview calls：440
- causal replay calls：40
- 总耗时：17.52 秒
- CUDA peak allocated：11.396 GiB
- CUDA peak reserved：11.465 GiB
- 每层单次 preview 的 unique experts：8～70 个，平均约 41.05 个
- 每层跨完整 block 多轮 preview 的专家并集：51～117 个，平均约 85.35 个
- 40 层专家实例的跨轮并集总量：3,414 个

这说明实际路由存在明显局部性，低于独立均匀路由的理论估计，但一个完整 32-token block 仍然会访问很多专家。专家缓存有价值，但不能只缓存“单次 top-8”，也不能假设一个 block 只需要少量固定专家。

### 4096 token prompt，4 token 输出，block_size=32

- GPU 层数：6 层
- preview calls：80
- causal replay calls：40
- 总耗时：49.34 秒
- CUDA peak allocated：11.849 GiB
- CUDA peak reserved：11.922 GiB

这个样本使用重复构造的 prompt，主要用于观察长 prefix 的代价，不能当作真实聊天质量或生产吞吐 benchmark。相较 128 token prompt，预填充和每次长 prefix attention 的开销明显增加。

## 三、128K full attention 组件实测

测试使用真实 Clover full attention 层权重，加上合成的 BF16 KV cache；这是 attention 组件测试，不是完整 128K 生成。

| 总上下文 | 持久化 KV | 原实现 median | 原实现峰值 | native GQA median | native GQA 峰值 |
|---:|---:|---:|---:|---:|---:|
| 8K | 0.156 GiB | 2.10 ms | 4.98 GiB | 3.56 ms | 5.28 GiB |
| 32K | 0.624 GiB | 6.20 ms | 5.83 GiB | 15.70 ms | 7.02 GiB |
| 64K | 1.249 GiB | 13.53 ms | 7.02 GiB | 30.07 ms | 9.40 GiB |
| 128K | 2.499 GiB | 26.63 ms | 9.41 GiB | 149.48 ms | 14.17 GiB |

本次 native GQA 修改使用 `scaled_dot_product_attention(..., enable_gqa=True)`，输出与当前显式复制实现在测试条件下接近，最大绝对差约 0.004，但在当前 PyTorch/CUDA/attention mask 组合下速度反而更慢、峰值显存更高。因此目前不能直接采用这个改动；需要换 attention kernel 或进一步处理 mask 和布局后再测。

## 四、CPU 到 GPU 搬运测试

测试使用 pinned 和非 pinned CPU 内存向 GPU 拷贝，结果是裸拷贝带宽，不包含专家 dispatch、同步和计算开销：

| 搬运大小 | 非 pinned | pinned |
|---:|---:|---:|
| 6 MiB | 约 17.9 GB/s | 约 23.2 GB/s |
| 48 MiB | 约 16.4 GB/s | 约 28.0 GB/s |
| 192 MiB | 约 14.3 GB/s | 约 28.6 GB/s |

单个 BF16 routed expert 约 6 MiB。理论上搬运一个专家需要约 0.21 ms，但真实专家访问还包括路由、缓存查找、分配、同步和矩阵计算。实际缓存设计必须使用 pinned system memory，并尽量把搬运和当前层计算重叠。

## 五、阶段性结论

1. 冷 routed experts 放系统内存是可行的。当前机器的系统内存足以容纳约 64.6 GiB 的完整 BF16 权重。
2. 生成时直接从 SSD 按专家随机读取不适合追求高输出速度。SSD 应作为启动加载和极冷数据后备，正常路径应是 SSD → 系统内存 → GPU cache。
3. 满 128K 上下文时，KV cache 和 full attention 临时工作集会显著压缩专家缓存空间。专家缓存预算应从 3～5 GiB 起测，不应默认有 7～8 GiB。
4. 真实 block32 路由显示每层跨轮专家并集平均约 85 个，说明缓存有价值，但路由按“预览轮次→层”执行后，固定每层配额比全局 LRU 更适合当前访问模式。完整 block 仍会触达很多专家。
5. 16GB 显存下，优先保证 embedding、lm_head、GDN、full attention、shared expert、router、KV cache 和运行时工作区。专家只使用明确测出来的剩余空间。
6. 当前 native GQA 试验没有带来收益，不能因为理论上减少重复 K/V 就直接合入。
7. 下一步应把缓存运行时做成 pinned RAM + GPU slot，并优先测试“按层配额 + decayed frequency”策略；SSD 先不进入生成同步路径。

## 六、原始数据和脚本

- [clover_profile.py](../clover_profile.py)：真实 Clover 路由采样脚本
- [clover_profile.json](../benchmarks/results/clover_profile.json)：block_size=4 原始结果
- [clover_profile_block32.json](../benchmarks/results/clover_profile_block32.json)：128 token prompt、block_size=32 原始结果
- [clover_profile_4k_block32.json](../benchmarks/results/clover_profile_4k_block32.json)：4096 token prompt、block_size=32 原始结果
- [clover_kv_probe.py](../clover_kv_probe.py)：128K KV 结构容量测试脚本
- [attention_probe.py](../benchmarks/attention_probe.py)：8K～128K attention 组件和 H2D 测试脚本
- [attention-results.json](../benchmarks/results/attention-results.json)：attention 组件原始结果

## 七、修正后的路由缓存模拟

本轮修正了两个影响结果的问题：事件顺序改为真实执行顺序“preview round → layer”；`static_hot` 改为按 `(layer, expert)` 统计，并在按层模式下独立选取每层热门专家。模拟仍只使用每次 preview 的 `unique_ids`，没有 token 数、gate mass、confidence、矩阵计算时间和 CPU fallback 时间，因此命中率与搬运量是方向性数据。

测试轨迹为 `clover_profile_block32.json` 和 `clover_profile_4k_block32.json`，每个 profile 前 50% 作为预热，后 50% 计入指标。每个 BF16 routed expert 按 6 MiB 计算，H2D 下界按已测 pinned 带宽 28.6 GiB/s 估算。

| GPU 专家缓存 | 策略 | 组织方式 | 命中率 | H2D 数据量 | 串行搬运下界 |
|---:|---|---|---:|---:|---:|
| 3 GiB | decayed | 按层配额 | 20.9% | 59.16 GiB | 2.07 s |
| 4 GiB | decayed | 按层配额 | 29.3% | 52.88 GiB | 1.85 s |
| 5 GiB | decayed | 按层配额 | 35.9% | 47.92 GiB | 1.68 s |
| 5 GiB | lfu | 按层配额 | 34.9% | 48.66 GiB | 1.70 s |
| 5 GiB | static_hot | 全局池 | 33.7% | 49.55 GiB | 1.73 s |
| 5 GiB | lru | 全局池 | 0.0% | 74.78 GiB | 2.61 s |

全局 LRU 在这条轨迹上几乎不能命中：40 个层轮流访问，5 GiB 只能容纳约 21 个专家实例/层的平均额度，最近访问的层在下一轮回来前已经被其他层挤出。因此需要为层保留工作集，或者使用明确的按层配额；简单的全局 LRU 不能作为 Clover 的缓存策略。

5 GiB、按层配额、decayed frequency 是当前离线轨迹中最好的组合，但命中率只有约 36%。它可以减少约 36% 的专家 H2D 请求，却不能单独把专家搬运开销消除。按 28.6 GiB/s 的串行下界计算，两条短轨迹仍对应约 1.68 秒的专家数据搬运；真实运行还要加上 dispatch、同步、矩阵计算和未重叠的等待时间。GPU copy stream、预分配 slot、layer-ahead 预取和 CPU fallback 仍然需要实测。

这也修正了报告前面“全局 LRU 约 72% 命中”的旧模拟结果：旧结果按层拼接了事件，违反真实时间顺序，不能继续作为依据。修正后的原始数据保存在 [route-cache-sim.json](../benchmarks/results/route-cache-sim.json)，模拟器在 [route_cache_sim.py](../benchmarks/route_cache_sim.py)。

## 八、Clover 专家冷读 SSD 实测

按 Clover 的真实 safetensors 文件布局建立了 10,240 个 `(layer, expert)` 索引，每个专家由 `gate_up_proj` 和 `down_proj` 两段组成，合计 6 MiB BF16。使用 Windows `FILE_FLAG_NO_BUFFERING` 绕过文件缓存，随机读取 96 个专家。

| 读取方式 | 结果 |
|---|---:|
| 单专家串行 median | 3.72 ms/专家 |
| 单专家串行 p90 | 5.91 ms/专家 |
| 单专家串行 mean | 38.91 ms/专家 |
| 2 路并发 | 2.98 GiB/s |
| 4 路并发 | 3.54 GiB/s |
| 8 路并发 | 3.65 GiB/s |
| 16 路并发 | 5.70 GiB/s |

单专家从 pinned RAM 搬到 GPU 的实测下界约 0.21 ms，而 SSD 冷读单专家的中位延迟约 3.72 ms，约为 H2D 的 18 倍；串行均值受少量高延迟样本影响更大。即使 SSD 有并发吞吐，生成时一旦发生同步冷读，也会把单个专家 miss 变成毫秒级等待。因此推荐完整 routed expert bank 常驻系统内存，SSD 只用于启动加载、内存不足时的冷数据后备和离线预取。

原始 SSD 测试结果保存在 [clover-ssd-results.json](../benchmarks/results/clover-ssd-results.json)，测试脚本在 [clover_ssd_bench.py](../benchmarks/clover_ssd_bench.py)。

## 九、两个参考项目对 Clover 的适用性

`moe` 项目适合参考三层存储和离线预取模型：VRAM 保存热点专家，系统内存保存完整专家池，SSD 保存冷数据。它的专家经过量化，单专家体积和 Clover 的 BF16 6 MiB 专家不同，所以其中的吞吐数字不能直接套用到 Clover；可以复用它的 layer-ahead、频率衰减和 SSD 队列模拟思路。参见 [moex 上游仓库](https://github.com/neerajdad123-byte/moe/tree/8148499a4672892d1b2510c48bd46ff12e5c4452)。

`OLED-MoE` 更接近 Clover 的运行时需求：按 `(layer_id, expert_id)` 管理专家，使用 pinned host bank、预分配 GPU slot、独立 CUDA copy stream、CUDA event、持久 GPU cache 和 CPU fallback。它的扩散迭代局部性也对应 Clover 的 block preview 多轮访问。Clover 仍需要自己处理 GDN 状态、full attention 和 KV cache，不能让 MoE cache manager 接管这些状态。参见 [OLED-MoE 上游仓库](https://github.com/flashserve/OLED-MoE/tree/deec1388e043d6c0887743c2000354f03f3a519d)。

综合当前实测，最稳妥的 Clover 原型顺序是：先把 64.6 GiB 权重中的 routed experts 作为 pinned RAM bank，GPU 只保留 3～5 GiB、按层配额的专家 slot；用 decayed frequency 做保留优先级，用 copy stream 和 event 做异步搬运；确认输出与原实现一致后，再加入 layer-ahead 预取。SSD 只作为 RAM bank 的后备，不应在 token 生成的同步路径中读取。

## 十、第一版 Clover 专家缓存运行时实测

已实现独立的 [clover_expert_runtime.py](../clover_expert_runtime.py)，没有修改原始模型文件和原始服务器。它把每层完整 routed expert bank 保存在 CPU，使用 pinned host memory；GPU 侧只分配固定 slot，冷专家通过 copy stream 和 CUDA event 搬运，slot 满时使用 decayed frequency 淘汰，并支持 CPU fallback。原始服务器仍可独立运行；缓存版本入口是 [clover_cached_server.py](../clover_cached_server.py)。

实现先经过测试驱动开发：

- 专家计算与原 Clover FP32 down projection 累加结果一致；
- LRU slot 命中、淘汰和计数正确；
- pinned host bank、异步 H2D、临时 slot 反复复用通过 CUDA churn 测试；
- 没有临时 slot 时，连续冷专家也能正确回退到单个 scratch slot；
- invalid capacity 和预算不足会直接报错。

测试文件为 [test_clover_expert_runtime.py](../test_clover_expert_runtime.py)。

### 真实模型短上下文结果

在同一台 RTX 5060 Ti 上，使用相同 prompt、相同 greedy 解码，缓存后端与原后端生成的 token ID 全部一致。代表性结果如下：

| 配置 | 原后端生成部分 | 5 GiB 缓存生成部分 | 原后端速度 | 5 GiB 缓存速度 |
|---|---:|---:|---:|---:|
| block4，4 token | 0.59～0.75 s | 0.28～0.36 s | 5.3～6.8 tok/s | 11.2～14.4 tok/s |
| block4，16 token | 5.06～5.15 s | 3.38～3.41 s | 3.1～3.2 tok/s | 4.7 tok/s |
| block32，32 token | 12.03～12.33 s | 6.11～6.22 s | 2.60～2.66 tok/s | 5.15～5.24 tok/s |

这些速度是单进程、包含模型内部 block preview/replay 的实测值；预填充单独计时后，缓存后端也更快，但本轮优化的主要收益来自把 40 层的非专家壳体放到 GPU，并减少了原服务器的逐层 CPU/GPU activation hopping。它仍然要搬运大量冷专家，不能说明已经达到理论上限。

5 GiB 运行时的 GPU 分配约 9.63 GiB；3 GiB 运行时约 7.52 GiB。5 GiB 128 token / block32 测例中累计 H2D 约 89～91 GiB，3 GiB 测例约 105 GiB，命中率随缓存增加而提高。

完整 token ID、阶段时间和缓存计数保存在 [runtime-validation.json](../benchmarks/results/runtime-validation.json)。

### 128K 状态压力测试

使用真实模型的全部 40 层、真实 KV/GDN 状态形状和初始化的 131,040 token prefix，测试 preview 32 token 和 causal replay 到 131,072 token。这个测试验证显存与状态接口，prefix 内容没有经过完整真实 prefill，因此不能作为 128K 质量或端到端速度结果。

| 专家缓存 | preview 峰值 allocated | causal replay 峰值 allocated | reserved 是否仍在可用预算内 |
|---:|---:|---:|---|
| 3 GiB | 14.46 GiB | 14.52 GiB | 否，reserved 约 15.22 GiB，接近物理上限 |
| 2 GiB | 13.59 GiB | 13.65 GiB | 是，reserved 约 14.36 GiB |

因此，若目标明确包含 128K 上下文，第一版默认应使用 **2 GiB 左右 GPU expert cache**，并把 3～5 GiB 作为短上下文或经过显存监控后的配置。2 GiB 压力测试中 logits 有限、cache 长度正确从 131,040 变为 131,072，preview 和 causal replay 均成功。

之前 5 GiB 测试出现约 16.63 GiB allocated 峰值，说明即使某次运行没有立即报 CUDA OOM，也不能把它视为安全配置。报告现在按 2 GiB 作为满上下文保守基线。

原始压力测试结果在 [runtime-full-context-validation.json](../benchmarks/results/runtime-full-context-validation.json)。一次直接把 131,072 token prompt 和 1 个新 token 交给 `generate` 的测试被模型自身的 block 边界检查拒绝；`block_size=32` 时 prompt 必须不超过 131,040。随后用 131,040 直接 prefill 测试，长 causal prefill 在显存接近上限时未完成，因此当前 128K 端到端生成仍需专门的分块 prefill 实现。该次边界和中断记录保存在 [cached-full-128k-2g.json](../benchmarks/results/cached-full-128k-2g.json)。

### 当前建议配置

- 短上下文优先输出速度：5 GiB cache，2 个 transient slot，pinned RAM bank，`transfer` backend；
- 128K 长上下文优先稳定性：2 GiB cache，2 个 transient slot，保留更多 KV 和 attention workspace；
- SSD 不进入同步生成路径；
- 下一版优化重点是按 preview round 预取下一批专家、把 token route 聚合后再搬运，并在长 prompt 中按块 prefill；
- 每次切换 cache budget 后都要重新做完整 token ID 对照。

