"""OpenAI-compatible Clover server with an optional expert cache backend.

Examples:
    python clover_cached_server.py --model /path/to/Clover --cache-gib 5
    python clover_cached_server.py --model /path/to/Clover --backend original

The checkpoint is loaded read-only. ``--backend original`` uses the included
layer-offload server, while ``cached`` replaces routed expert execution with the
CPU-bank and GPU-slot runtime.
"""
import argparse
import os
import time

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

import torch
import clover_server
from clover_expert_runtime import install_clover_expert_cache


def build_cached(model_path, cache_gib=5.0, miss_backend='transfer', async_copy=True):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,
              local_files_only=True,dtype=torch.bfloat16,low_cpu_mem_usage=True).eval()
    print('weights loaded; pinning host banks and allocating slots',flush=True)
    report = install_clover_expert_cache(model,cache_gib=cache_gib,miss_backend=miss_backend,
             async_copy=async_copy,progress=lambda i,n: print(f'expert banks {i}/{n}',flush=True) if i%5==0 else None)
    model.to('cuda:0')  # bank/slot tensors deliberately are not registered buffers
    torch.cuda.synchronize()
    report.update(load_seconds=time.perf_counter()-started,
                  allocated_gib=torch.cuda.memory_allocated()/2**30)
    return model,tokenizer,report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',default=os.environ.get('CLOVER_MODEL'),
                    help='Local Clover model directory, or set CLOVER_MODEL.')
    ap.add_argument('--backend',choices=['cached','original'],default='cached')
    ap.add_argument('--cache-gib',type=float,default=5.0)
    ap.add_argument('--miss-backend',choices=['transfer','cpu'],default='transfer')
    ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--host',default='127.0.0.1')
    ap.add_argument('--port',type=int,default=8322)
    ap.add_argument('--load-only',action='store_true')
    args=ap.parse_args()
    if not args.model:
        ap.error('--model is required (or set CLOVER_MODEL)')
    torch.set_num_threads(args.threads)
    os.environ['CLOVER_MODEL']=args.model
    server=clover_server
    server.MODEL_PATH=args.model
    if args.backend=='original':
        server.build()
    else:
        model,tok,report=build_cached(args.model,args.cache_gib,args.miss_backend)
        server.state.update(model=model,tok=tok,gpu_layers=len(model.model.layers),load_s=report['load_seconds'])
        print(report,flush=True)
    if args.load_only:
        return
    http=server.ThreadingHTTPServer((args.host,args.port),server.Handler)
    http.daemon_threads=True
    print(f'listening on {args.host}:{args.port}, backend={args.backend}',flush=True)
    http.serve_forever()


if __name__=='__main__':
    main()
