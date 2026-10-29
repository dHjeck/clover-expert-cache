"""Clover routed experts: CPU weight bank, bounded GPU slots and event-safe H2D.

Inference only, one caller at a time. Attach after moving the non-expert shell.
Host banks and slots intentionally are not nn.Parameters/buffers: do not call
model.to()/state_dict() to move/save a model after installing this backend.
"""
import math
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn


def linear_fp32_output(x, weight):
    # Matches Clover's down-projection rounding (FP32 until route weighting).
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        try:
            return torch.mm(x, weight.T, out_dtype=torch.float32)
        except TypeError:
            pass
    with torch.autocast(device_type=x.device.type, enabled=False):
        return F.linear(x.float(), weight.float())


def cache_slots_for_budget(cache_gib, layer_count, expert_bytes=6 * 2**20):
    """Total slots per layer, INCLUDING transient buffers; never exceed budget."""
    if cache_gib <= 0 or layer_count <= 0 or expert_bytes <= 0:
        raise ValueError('budget, layer_count and expert_bytes must be positive')
    result = int(cache_gib * 2**30 // expert_bytes) // layer_count
    if result < 1:
        raise ValueError('budget cannot reserve one slot per layer')
    return result


class CloverExpertRuntime(nn.Module):
    def __init__(self, gate_up, down, capacity, device='cpu', policy='decayed', tau=10.0,
                 pin_memory=False, async_copy=True, miss_backend='transfer', transient_slots=0):
        super().__init__()
        if gate_up.ndim != 3 or down.ndim != 3:
            raise ValueError('expert weights must have three dimensions')
        e, twice_i, h = gate_up.shape
        if tuple(down.shape) != (e, h, twice_i // 2) or twice_i % 2 or gate_up.dtype != down.dtype:
            raise ValueError('incompatible expert weight shapes or dtypes')
        if not 1 <= capacity <= e:
            raise ValueError('capacity must be between 1 and the expert count')
        if policy not in ('lru', 'decayed') or tau <= 0:
            raise ValueError('invalid policy or tau')
        if miss_backend not in ('transfer', 'cpu') or transient_slots < 0:
            raise ValueError('invalid miss backend or transient_slots')
        device = torch.device(device)
        self.device = torch.device('cuda', torch.cuda.current_device()) if device == torch.device('cuda') else device
        self.capacity, self.policy, self.tau = int(capacity), policy, float(tau)
        self.transient_slots = int(transient_slots)
        self.miss_backend = miss_backend
        self.clock = 0
        self.scores, self.last, self.resident = {}, {}, {}
        self.counters = Counter()
        # Pin one layer at a time during installation; failure is explicit, not hidden.
        self.bank_gate = gate_up.detach().cpu().contiguous()
        self.bank_down = down.detach().cpu().contiguous()
        if pin_memory and self.device.type == 'cuda':
            if not self.bank_gate.is_pinned():
                self.bank_gate = self.bank_gate.pin_memory()
            if not self.bank_down.is_pinned():
                self.bank_down = self.bank_down.pin_memory()
        total = self.capacity + self.transient_slots
        self.slot_gate = torch.empty((total, *gate_up.shape[1:]), device=self.device, dtype=gate_up.dtype)
        self.slot_down = torch.empty((total, *down.shape[1:]), device=self.device, dtype=down.dtype)
        self._done = [None] * total
        self._ready = [None] * total
        self._transient_next = 0
        self.copy_stream = torch.cuda.Stream(device=self.device) if async_copy and self.device.type == 'cuda' else None
        if self.copy_stream is not None:
            self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            self.slot_gate.record_stream(self.copy_stream)
            self.slot_down.record_stream(self.copy_stream)

    def _priority(self, e):
        if self.policy == 'lru':
            return (self.last.get(e, -1), -e)
        return (self.scores.get(e, 0) * math.exp(-(self.clock-self.last.get(e,self.clock))/self.tau), -e)

    def _copy(self, e, slot):
        if self.copy_stream is None:
            if self.device.type == 'cuda' and self._done[slot] is not None:
                torch.cuda.current_stream(self.device).wait_event(self._done[slot])
            self.slot_gate[slot].copy_(self.bank_gate[e])
            self.slot_down[slot].copy_(self.bank_down[e])
        else:
            with torch.cuda.stream(self.copy_stream):
                # Only wait for the computation using THIS slot, not the whole stream.
                if self._done[slot] is not None:
                    self.copy_stream.wait_event(self._done[slot])
                self.slot_gate[slot].copy_(self.bank_gate[e], non_blocking=True)
                self.slot_down[slot].copy_(self.bank_down[e], non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(self.copy_stream)
                self._ready[slot] = ready
            self.counters['async_copies'] += 1
        self.counters['h2d_bytes' if self.device.type == 'cuda' else 'cpu_copy_bytes'] += self.expert_bytes

    @property
    def expert_bytes(self):
        return self.bank_gate[0].nbytes + self.bank_down[0].nbytes

    @torch.inference_mode()
    def forward(self, x, ids, weights):
        if ids.ndim != 2 or weights.shape != ids.shape or x.ndim != 2 or x.shape[0] != ids.shape[0]:
            raise ValueError('expected x[tokens, hidden], ids/weights[tokens, routes]')
        if x.device != self.device or ids.device != x.device or weights.device != x.device:
            raise ValueError('x, ids, weights must be on runtime device')
        if x.dtype != self.bank_gate.dtype or x.shape[1] != self.bank_gate.shape[2]:
            raise ValueError('input shape/dtype disagrees with bank')
        active = ids.unique().tolist()
        if active and (min(active) < 0 or max(active) >= len(self.bank_gate)):
            raise ValueError('expert id out of bounds')
        self.clock += 1
        self.counters['calls'] += 1
        before = dict(self.resident)
        self.counters['hits'] += sum(e in before for e in active)
        self.counters['misses'] += sum(e not in before for e in active)
        for e in active:
            old = self.scores.get(e,0) * math.exp(-(self.clock-self.last.get(e,self.clock))/self.tau)
            self.scores[e], self.last[e] = old + 1, self.clock
        selected = set(sorted(set(before) | set(active), key=self._priority, reverse=True)[:self.capacity])
        for e in list(self.resident):
            if e not in selected:
                self.resident.pop(e)
                self.counters['evictions'] += 1
        free = iter(i for i in range(self.capacity) if i not in self.resident.values())
        for e in active:
            if e in selected and e not in self.resident:
                self.resident[e] = next(free)

        scratch_slot = None

        def prepare(e):
            nonlocal scratch_slot
            if e in self.resident:
                slot = self.resident[e]
                if e not in before:
                    self._copy(e,slot)
                return slot
            if self.miss_backend == 'cpu':
                return None
            if self.transient_slots:
                slot = self.capacity + self._transient_next % self.transient_slots
                self._transient_next += 1
                self.counters['transient_copies'] += 1
            else:
                # Capacity-only mode: sacrifice one retained slot for cold work.
                if scratch_slot is None:
                    victim = min(self.resident,key=self._priority)
                    scratch_slot = self.resident.pop(victim)
                    self.counters['evictions'] += 1
                slot = scratch_slot
            self._copy(e,slot)
            return slot

        out = torch.zeros_like(x,dtype=torch.float32)
        prepared = prepare(active[0]) if active else None
        for n,e in enumerate(active):
            slot = prepared
            if slot is not None and self._ready[slot] is not None:
                torch.cuda.current_stream(self.device).wait_event(self._ready[slot])
            token, route = torch.where(ids == e)
            # Two transient buffers let the next copy overlap this expert's kernels.
            lookahead = self.transient_slots >= 2
            if lookahead and n+1 < len(active):
                prepared = prepare(active[n+1])
            if slot is None:
                xx = x[token].cpu()
                ww = weights[token,route,None].float().cpu()
                gu, dn = self.bank_gate[e], self.bank_down[e]
                self.counters['cpu_experts'] += 1
            else:
                xx, ww = x[token], weights[token,route,None].float()
                gu, dn = self.slot_gate[slot], self.slot_down[slot]
            gate, up = F.linear(xx,gu).chunk(2,-1)
            hidden = (F.silu(gate.float()) * up.float()).to(x.dtype)
            yy = linear_fp32_output(hidden,dn)
            routed = (yy * ww).to(x.dtype).to(x.device)
            out.index_add_(0,token,routed.float())
            if slot is not None and self.device.type == 'cuda':
                done = torch.cuda.Event()
                done.record(torch.cuda.current_stream(self.device))
                self._done[slot] = done
            if not lookahead and n+1 < len(active):
                prepared = prepare(active[n+1])
        return out.to(x.dtype)

    def stats(self):
        names = ('calls','hits','misses','evictions','async_copies','cpu_experts','transient_copies','h2d_bytes')
        return {**{k:self.counters[k] for k in names},
                'resident_experts':len(self.resident), 'capacity':self.capacity,
                'transient_slots':self.transient_slots, 'policy':self.policy,
                'pinned':self.bank_gate.is_pinned() and self.bank_down.is_pinned(),
                'device_bytes':self.slot_gate.nbytes+self.slot_down.nbytes}

def install_clover_expert_cache(model, cache_gib=5.0, device='cuda:0', policy='decayed',
                               pin_memory=True, transient_slots=2, miss_backend='transfer',
                               async_copy=True, progress=None):
    """Replace routed expert modules only. Caller then places the shell on GPU.

    Budget includes transient buffers. Host tensors remain ordinary attributes so
    the shell's .to(device) does not accidentally upload the 60 GiB expert bank.
    CPU bank pinning failure aborts explicitly. Installation is intended for a
    newly loaded private model, not a live concurrent server.
    """
    layers = model.model.layers
    first = layers[0].mlp.experts
    expert_bytes = first.gate_up_proj[0].nbytes + first.down_proj[0].nbytes
    slots = cache_slots_for_budget(cache_gib, len(layers), expert_bytes)
    slots = min(slots, first.gate_up_proj.shape[0] + transient_slots)
    capacity = slots - transient_slots
    if capacity < 1:
        raise ValueError('budget must include at least one persistent and all transient slots per layer')
    for i, layer in enumerate(layers):
        old = layer.mlp.experts
        if isinstance(old, CloverExpertRuntime):
            raise ValueError('cache is already installed')
        if old.gate_up_proj[0].nbytes + old.down_proj[0].nbytes != expert_bytes:
            raise ValueError('heterogeneous expert sizes are unsupported')
        layer.mlp.experts = CloverExpertRuntime(
            old.gate_up_proj, old.down_proj, capacity, device=device, policy=policy,
            pin_memory=pin_memory, transient_slots=transient_slots,
            miss_backend=miss_backend, async_copy=async_copy)
        del old
        if progress:
            progress(i+1, len(layers))
    return {'layers':len(layers), 'slots_per_layer':slots,
            'persistent_slots_per_layer':capacity, 'transient_slots_per_layer':transient_slots,
            'expert_bytes':expert_bytes, 'device_bytes':slots*len(layers)*expert_bytes,
            'cache_budget_gib':cache_gib, 'miss_backend':miss_backend, 'policy':policy}
