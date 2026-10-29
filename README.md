# Clover Expert Cache

Experimental CPU-to-GPU routed-expert caching for
[Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview),
targeting a 16 GB consumer GPU. The runtime keeps routed expert weights in a
pinned host-memory bank and uses a bounded set of preallocated GPU slots,
asynchronous copies, CUDA events, decayed-frequency eviction, and optional CPU
fallback.

> This is an independent research prototype. It is not an official Clover
> implementation and is not yet a production serving stack. Model weights are
> not included in this repository.

## What is included

- `clover_expert_runtime.py`: routed-expert cache runtime.
- `clover_server.py`: standalone OpenAI-compatible server using decoder-layer
  CPU/GPU offload.
- `clover_cached_server.py`: the same API with the expert cache backend.
- Route profiling, cache simulation, attention, storage, and validation tools.
- Unit and CUDA churn tests.
- Raw benchmark results and two Chinese research reports under `docs/`.

## Validated result

The measurements below come from one Windows machine with an RTX 5060 Ti
(16 GB), approximately 128 GiB system RAM, Python 3.11, PyTorch 2.11.0+cu128,
and Transformers 5.19.0. They are not a general performance promise.

- Greedy token IDs matched the original backend in the reported validation runs.
- A 5 GiB expert cache improved the measured short-context generation paths,
  while retaining many cold-expert transfers.
- About 2 GiB is the conservative starting point when 128K KV/GDN state must
  coexist with the runtime. A 3 GiB configuration came too close to the
  physical VRAM limit in the synthetic full-context test.
- SSD demand reads are unsuitable for the synchronous generation path. The
  tested median cold read was about 3.72 ms per 6 MiB routed expert.

See [docs/performance-report.md](docs/performance-report.md) for the complete
methodology, caveats, and result tables.

## Requirements

- Python 3.11 or newer.
- A CUDA-capable GPU for the cached backend.
- Enough host RAM for the complete model and its CPU expert bank.
- The local Clover model directory, downloaded from Hugging Face.

The CUDA backend has been tested with PyTorch 2.11.0+cu128. Install the PyTorch
build appropriate for your driver before installing the remaining packages:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Quick start

Set the model directory:

```bash
export CLOVER_MODEL=/path/to/Clover-35B-A3B-Preview
```

PowerShell equivalent:

```powershell
$env:CLOVER_MODEL = 'D:\models\Clover-35B-A3B-Preview'
```

Start the cached server:

```bash
python clover_cached_server.py --cache-gib 2 --port 8322
```

For a shorter context and faster iteration, 3-5 GiB can be tested:

```bash
python clover_cached_server.py --cache-gib 5 --port 8322
```

The original layer-offload backend remains available for comparison:

```bash
python clover_cached_server.py --backend original --port 8321
```

Test the HTTP endpoint:

```bash
python clover_local_client_test.py
```

The client reads `CLOVER_ENDPOINT` and `CLOVER_RESULTS_DIR` if those variables
are set.

## Reproduce measurements

Route profile:

```bash
python clover_profile.py --prompt-tokens 128 --max-new 4 --block-size 32 \
  --output route-profile.json
```

Cached generation:

```bash
python clover_cached_benchmark.py --cache-gib 5 --block-size 32 \
  --output cached-generation.json
```

Cache-policy simulation from existing route profiles:

```bash
python benchmarks/route_cache_sim.py \
  benchmarks/results/clover_profile_block32.json \
  benchmarks/results/clover_profile_4k_block32.json
```

Full parity and near-128K state validation:

```bash
python clover_runtime_validation.py --output runtime-validation.json
python clover_full_context_validation.py --output full-context-validation.json
```

The long-context validation initializes state shapes; it does not establish
end-to-end 128K prefill quality or speed.

## Tests

Run CPU tests:

```bash
pytest -q test_clover_expert_runtime.py -k "not cuda"
```

Run all tests on a CUDA machine:

```bash
pytest -q test_clover_expert_runtime.py
```

## Safety and limitations

The runtime assumes one inference caller and a newly loaded model. Do not call
`model.to(...)` or serialize `state_dict()` after installing the cache backend:
the host bank and slots are intentionally ordinary tensor attributes so they
are not uploaded with the model shell. Do not share a cached model instance
across concurrent requests without adding locking around the cache state.

The expert cache preserves routed-expert math under the tested configurations.
Text parity across all GPUs, PyTorch versions, kernels, sampling settings, and
long contexts is not guaranteed. Choose the cache budget from observed free
VRAM, not from total device memory alone.

## Attribution and license

The project code is distributed under Apache-2.0. Clover is derived from
Qwen3.6-35B-A3B and is distributed by Diffuspace under Apache-2.0. Upstream
attributions are recorded in [NOTICE](NOTICE). External projects discussed in
the analysis are linked rather than vendored.

No model weights or third-party source snapshots are included.
