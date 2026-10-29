import torch
import torch.nn.functional as F

import importlib.util


def reference(gate_up, down, x, ids, weights):
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in ids.unique().tolist():
        token, slot = torch.where(ids == expert)
        gate, up = F.linear(x[token], gate_up[expert]).chunk(2, -1)
        hidden = (F.silu(gate.float()) * up.float()).to(x.dtype)
        y = F.linear(hidden.float(), down[expert].float())
        routed = (y * weights[token, slot, None].float()).to(x.dtype)
        out.index_add_(0, token, routed.float())
    return out.to(x.dtype)


def test_runtime_matches_original_expert_math():
    assert importlib.util.find_spec('clover_expert_runtime') is not None, 'expert runtime is not implemented'
    from clover_expert_runtime import CloverExpertRuntime
    torch.manual_seed(1)
    gate_up = torch.randn(4, 8, 3, dtype=torch.bfloat16)
    down = torch.randn(4, 3, 4, dtype=torch.bfloat16)
    x = torch.randn(7, 3, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [2, 0], [3, 1], [2, 3], [0, 2], [1, 3], [0, 1]])
    weights = torch.softmax(torch.randn(7, 2), dim=-1).to(torch.bfloat16)

    runtime = CloverExpertRuntime(gate_up, down, capacity=2, device="cpu", policy="decayed")
    actual = runtime(x, ids, weights)
    expected = reference(gate_up, down, x, ids, weights)

    assert torch.allclose(actual, expected, atol=0.01, rtol=0.01)


def test_runtime_reuses_and_evicts_slots_with_lru():
    from clover_expert_runtime import CloverExpertRuntime
    gate_up = torch.randn(4, 8, 3, dtype=torch.bfloat16)
    down = torch.randn(4, 3, 4, dtype=torch.bfloat16)
    x = torch.randn(2, 3, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [0, 1]])
    weights = torch.full((2, 2), 0.5, dtype=torch.bfloat16)

    runtime = CloverExpertRuntime(gate_up, down, capacity=2, device="cpu", policy="lru")
    runtime(x, ids, weights)
    runtime(x, ids, weights)
    runtime(x, torch.tensor([[2, 3], [2, 3]]), weights)
    stats = runtime.stats()

    assert stats["hits"] == 2
    assert stats["misses"] == 4
    assert stats["evictions"] == 2
    assert stats["resident_experts"] == 2


def test_runtime_rejects_invalid_capacity():
    from clover_expert_runtime import CloverExpertRuntime
    gate_up = torch.randn(4, 8, 3, dtype=torch.bfloat16)
    down = torch.randn(4, 3, 4, dtype=torch.bfloat16)

    try:
        CloverExpertRuntime(gate_up, down, capacity=0, device="cpu")
    except ValueError as exc:
        assert "capacity" in str(exc)
    else:
        raise AssertionError("capacity=0 must be rejected")

def test_cache_slots_reserves_capacity_per_layer():
    from clover_expert_runtime import cache_slots_for_budget

    assert cache_slots_for_budget(5.0, 40) == 21
    assert cache_slots_for_budget(3.0, 40) == 12


def test_cuda_churn_and_fallback():
    from clover_expert_runtime import CloverExpertRuntime, linear_fp32_output
    assert torch.cuda.is_available(), "GPU tests require CUDA"
    torch.manual_seed(42)
    gu = torch.randn(8, 64, 64, dtype=torch.bfloat16) * 0.05
    dn = torch.randn(8, 64, 32, dtype=torch.bfloat16) * 0.05
    x = torch.randn(32, 64, dtype=torch.bfloat16, device='cuda:0')
    weights = torch.softmax(torch.randn(32, 2, device='cuda:0'), -1)
    gpu_gu, gpu_dn = gu.cuda(), dn.cuda()
    for fallback in ('transfer', 'cpu'):
        rt = CloverExpertRuntime(gu, dn, capacity=2, device='cuda:0',
                                 pin_memory=True, miss_backend=fallback, transient_slots=2)
        refs, actuals = [], []
        for iteration in range(12):
            ids = (torch.arange(64, device='cuda:0').reshape(32, 2) + iteration) % 8
            ref = torch.zeros_like(x, dtype=torch.float32)
            for expert in ids.unique().tolist():
                token, route = torch.where(ids == expert)
                gate, up = F.linear(x[token], gpu_gu[expert]).chunk(2, -1)
                h = (F.silu(gate.float()) * up.float()).to(x.dtype)
                y = linear_fp32_output(h, gpu_dn[expert])
                ref.index_add_(0, token, (y * weights[token,route,None]).to(x.dtype).float())
            refs.append(ref.to(x.dtype))
            actuals.append(rt(x, ids, weights))
        torch.cuda.synchronize()
        for actual, ref in zip(actuals, refs):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
        assert rt.bank_gate.is_pinned() and rt.bank_down.is_pinned()
        assert rt.stats()['resident_experts'] <= 2
        if fallback == 'cpu':
            assert rt.stats()['cpu_experts'] > 0
        else:
            assert rt.stats()['transient_copies'] > 0
        assert rt.stats()['device_bytes'] == 4 * (gu[0].nbytes + dn[0].nbytes)


def test_tiny_budget_is_rejected():
    from clover_expert_runtime import cache_slots_for_budget
    try:
        cache_slots_for_budget(0.001, 40)
    except ValueError:
        return
    raise AssertionError('must not silently exceed budget')

def test_install_replaces_each_layer_and_reports_budget():
    from clover_expert_runtime import install_clover_expert_cache, CloverExpertRuntime
    class Mlp:
        pass
    class Layer:
        def __init__(self):
            self.mlp = Mlp()
            self.mlp.experts = Mlp()
            self.mlp.experts.gate_up_proj = torch.randn(4, 8, 3, dtype=torch.bfloat16)
            self.mlp.experts.down_proj = torch.randn(4, 3, 4, dtype=torch.bfloat16)
    class Model:
        pass
    model = Model()
    model.model = Model()
    model.model.layers = [Layer(), Layer()]
    report = install_clover_expert_cache(model, cache_gib=(2*72/2**30), device='cpu', pin_memory=False, transient_slots=0)
    assert report['layers'] == 2
    assert report['slots_per_layer'] == 1
    assert all(isinstance(layer.mlp.experts, CloverExpertRuntime) for layer in model.model.layers)


def test_no_transient_many_cold_experts():
    from clover_expert_runtime import CloverExpertRuntime
    torch.manual_seed(87)
    gu=torch.randn(8,8,3,dtype=torch.bfloat16)
    dn=torch.randn(8,3,4,dtype=torch.bfloat16)
    x=torch.randn(4,3,dtype=torch.bfloat16)
    ids=torch.arange(8).reshape(4,2)
    weights=torch.full((4,2),.5)
    rt=CloverExpertRuntime(gu,dn,capacity=1,transient_slots=0)
    torch.testing.assert_close(rt(x,ids,weights),reference(gu,dn,x,ids,weights),rtol=0,atol=0)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            with torch.inference_mode():
                fn()
            print('PASS', name, flush=True)

