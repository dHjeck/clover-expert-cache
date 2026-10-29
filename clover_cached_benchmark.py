import argparse
import json
import os
import time
from pathlib import Path

import torch
from clover_cached_server import build_cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.environ.get('CLOVER_MODEL'),
                    help='Local Clover model directory, or set CLOVER_MODEL.')
    ap.add_argument('--cache-gib', type=float, default=3.0)
    ap.add_argument('--prompt-tokens', type=int, default=128)
    ap.add_argument('--max-new', type=int, default=4)
    ap.add_argument('--block-size', type=int, default=32)
    ap.add_argument('--threshold', type=float, default=0.95)
    ap.add_argument('--output', default='cached-generation.json')
    args = ap.parse_args()
    if not args.model:
        ap.error('--model is required (or set CLOVER_MODEL)')
    model, tok, report = build_cached(args.model, args.cache_gib)
    device = torch.device('cuda:0')
    seed = 'Explain the following code carefully and propose a small safe improvement. '
    seed_ids = tok(seed, add_special_tokens=False, return_tensors='pt').input_ids[0]
    repeats = (args.prompt_tokens + seed_ids.numel() - 1) // seed_ids.numel()
    prompt = seed_ids.repeat(repeats)[:args.prompt_tokens].unsqueeze(0).to(device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(input_ids=prompt, max_new_tokens=args.max_new,
                             block_size=args.block_size, threshold=args.threshold,
                             do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    layer_stats = []
    for li, layer in enumerate(model.model.layers):
        row = layer.mlp.experts.stats()
        row['layer'] = li
        layer_stats.append(row)
    result = {
        'model': args.model, 'cache_gib': args.cache_gib,
        'prompt_tokens': int(prompt.shape[1]), 'max_new_tokens': args.max_new,
        'block_size': args.block_size, 'threshold': args.threshold,
        'elapsed_seconds': elapsed, 'output_tokens': int(out.shape[1]-prompt.shape[1]),
        'tok_per_s': (out.shape[1]-prompt.shape[1])/elapsed,
        'decoded': tok.decode(out[0, prompt.shape[1]:], skip_special_tokens=True),
        'cuda_allocated_gib': torch.cuda.memory_allocated(device)/2**30,
        'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated(device)/2**30,
        'cuda_peak_reserved_gib': torch.cuda.max_memory_reserved(device)/2**30,
        'install': report,
        'layers': layer_stats,
        'aggregate': {
            'hits': sum(x['hits'] for x in layer_stats),
            'misses': sum(x['misses'] for x in layer_stats),
            'evictions': sum(x['evictions'] for x in layer_stats),
            'transient_copies': sum(x['transient_copies'] for x in layer_stats),
            'h2d_bytes': sum(x['h2d_bytes'] for x in layer_stats),
            'cpu_experts': sum(x['cpu_experts'] for x in layer_stats),
        },
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('elapsed_seconds','output_tokens','tok_per_s','decoded','cuda_allocated_gib','cuda_peak_allocated_gib','cuda_peak_reserved_gib','aggregate')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

