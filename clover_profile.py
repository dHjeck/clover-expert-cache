import argparse, json, os, time, types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("CLOVER_MODEL"),
                    help="Local Clover model directory, or set CLOVER_MODEL.")
    ap.add_argument("--vram-gb", type=float, default=13.0)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--output", default="clover-route-profile.json")
    args = ap.parse_args()
    if not args.model:
        ap.error("--model is required (or set CLOVER_MODEL)")

    device = torch.device("cuda:0")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    load_s = time.perf_counter() - t0

    model.model.embed_tokens.to(device)
    model.lm_head.to(device)
    model.model.norm.to(device)
    layers = model.model.layers
    per_layer = sum(p.numel() * p.element_size() for p in layers[0].parameters()) / 1e9
    base = torch.cuda.memory_allocated() / 1e9
    n_gpu = 0
    for layer in layers:
        if base + per_layer * (n_gpu + 1) > args.vram_gb:
            break
        layer.to(device)
        n_gpu += 1

    recorder = {
        "calls": 0,
        "preview_calls": 0,
        "causal_calls": 0,
        "by_layer": {},
        "events": [],
        "phase": None,
    }

    for li, layer in enumerate(layers):
        orig_layer = layer.forward
        def layer_forward(self, x, cache, valid, prefix_valid, positions, preview, _orig=orig_layer):
            dev = self.input_layernorm.weight.device
            src = x.device
            if dev != src:
                x = x.to(dev)
                valid = valid.to(dev)
                prefix_valid = prefix_valid.to(dev)
                positions = positions.to(dev)
            out, new_cache = _orig(x, cache, valid, prefix_valid, positions, preview)
            if dev != src:
                out = out.to(src)
            return out, new_cache
        layer.forward = types.MethodType(layer_forward, layer)
        block = layer.mlp
        experts = block.experts
        orig_block = block.forward
        orig_experts = experts.forward

        def block_forward(self, x, valid, preview, _orig=orig_block):
            recorder["phase"] = "preview" if preview else "causal"
            if preview:
                recorder["preview_calls"] += 1
            else:
                recorder["causal_calls"] += 1
            return _orig(x, valid, preview)

        def experts_forward(self, x, ids, weights, _orig=orig_experts, _li=li):
            unique = torch.unique(ids).detach().cpu().tolist()
            key = str(_li)
            rec = recorder["by_layer"].setdefault(key, {"preview": [], "causal": []})
            rec[recorder["phase"]].append({
                "unique_count": len(unique),
                "unique_ids": unique,
                "token_count": int(ids.shape[0]),
            })
            recorder["calls"] += 1
            return _orig(x, ids, weights)

        block.forward = types.MethodType(block_forward, block)
        experts.forward = types.MethodType(experts_forward, experts)

    # Build a deterministic prompt with approximately the requested token count.
    seed = "Explain the following code carefully and propose a small safe improvement. "
    seed_ids = tok(seed, add_special_tokens=False, return_tensors="pt").input_ids[0]
    repeats = (args.prompt_tokens + seed_ids.numel() - 1) // seed_ids.numel()
    prompt = seed_ids.repeat(repeats)[: args.prompt_tokens].unsqueeze(0).to(device)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            input_ids=prompt,
            max_new_tokens=args.max_new,
            block_size=args.block_size,
            threshold=args.threshold,
            do_sample=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    by_layer = {}
    for li, records in sorted(recorder["by_layer"].items(), key=lambda x: int(x[0])):
        preview = records["preview"]
        causal = records["causal"]
        all_preview = [x for r in preview for x in r["unique_ids"]]
        per_forward = [r["unique_count"] for r in preview]
        by_layer[li] = {
            "preview_forward_count": len(preview),
            "causal_forward_count": len(causal),
            "preview_unique_per_forward": per_forward,
            "preview_unique_union": len(set(all_preview)),
            "preview_unique_union_ids": sorted(set(all_preview)),
            "preview_events": preview,
        }

    result = {
        "model": args.model,
        "load_seconds": load_s,
        "prompt_tokens": int(prompt.shape[1]),
        "requested_max_new_tokens": args.max_new,
        "block_size": args.block_size,
        "threshold": args.threshold,
        "gpu_layers": n_gpu,
        "elapsed_seconds": elapsed,
        "output_tokens": int(out.shape[1] - prompt.shape[1]),
        "cuda_allocated_gib_after": torch.cuda.memory_allocated(device) / 2**30,
        "cuda_reserved_gib_after": torch.cuda.memory_reserved(device) / 2**30,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "preview_calls": recorder["preview_calls"],
        "causal_calls": recorder["causal_calls"],
        "by_layer": by_layer,
        "decoded": tok.decode(out[0, prompt.shape[1]:], skip_special_tokens=True),
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "by_layer"}, ensure_ascii=False, indent=2))
    print("output:", args.output)

if __name__ == "__main__":
    main()
