"""Chunked causal prefill stress: no block preview, one next-token logits step."""
import argparse,gc,json,os,time
from pathlib import Path
import torch
from clover_cached_server import build_cached

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--model',default=os.environ.get('CLOVER_MODEL'),help='Local Clover model directory, or set CLOVER_MODEL.')
 ap.add_argument('--cache-gib',type=float,default=2)
 ap.add_argument('--tokens',type=int,default=131040)
 ap.add_argument('--chunk',type=int,default=4096)
 ap.add_argument('--out',default='chunked-prefill.json')
 a=ap.parse_args()
 if not a.model: ap.error('--model is required (or set CLOVER_MODEL)')
 model,tok,install=build_cached(a.model,a.cache_gib)
 seed=tok('Explain the following code carefully and propose a small safe improvement. ',add_special_tokens=False,return_tensors='pt').input_ids[0]
 ids=seed.repeat((a.tokens+len(seed)-1)//len(seed))[:a.tokens][None].cuda()
 cache=None; rows=[]; t0=time.perf_counter()
 for start in range(0,a.tokens,a.chunk):
  part=ids[:,start:min(a.tokens,start+a.chunk)]
  torch.cuda.synchronize();s=time.perf_counter()
  out=model(input_ids=part,past_key_values=cache,use_cache=True)
  torch.cuda.synchronize();dt=time.perf_counter()-s;cache=out.past_key_values
  rows.append({'start':start,'tokens':part.shape[1],'seconds':dt,'prefix_after':cache.attention_mask.shape[-1],'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
  print(json.dumps(rows[-1]),flush=True)
 # one causal next token; this is the first output distribution after full prefill
 nxt=torch.full((1,1),0,dtype=torch.long,device='cuda')
 torch.cuda.reset_peak_memory_stats();s=time.perf_counter();out=model(input_ids=nxt,past_key_values=cache,use_cache=True);torch.cuda.synchronize();next_id=int(out.logits[0,-1].argmax());dt=time.perf_counter()-s
 result={'kind':'real_chunked_causal_prefill_then_one_next_token','cache_gib':a.cache_gib,'tokens':a.tokens,'chunk':a.chunk,'prefill_seconds':time.perf_counter()-t0-dt,'next_token_seconds':dt,'next_token_id':next_id,'decoded':tok.decode([next_id],skip_special_tokens=True),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'chunks':rows,'install':install}
 Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
