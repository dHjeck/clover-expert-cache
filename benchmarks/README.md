# Benchmarks

The scripts in this directory accompany
[the performance report](../docs/performance-report.md).

- `attention_probe.py` measures full-attention execution with real Clover
  weights and synthetic KV tensors.
- `route_cache_sim.py` replays captured routed-expert events for cache-policy
  simulation.
- `clover_ssd_bench.py` measures direct random reads of routed experts on
  Windows using unbuffered I/O.

All scripts accept or read the local model directory through
`--model`/arguments or `CLOVER_MODEL`. Raw results used by the report are in
`results/`.
