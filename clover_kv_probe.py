import json, time, torch

def gib(x): return x / 2**30

def main():
    device='cuda:0'; seq=131072; layers=10; kv=2; dim=256; heads=16; dtype=torch.bfloat16
    torch.empty(1, device=device); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t=time.perf_counter()
    cache=[]
    for _ in range(layers):
        k=torch.empty((1,kv,seq,dim),device=device,dtype=dtype)
        v=torch.empty((1,kv,seq,dim),device=device,dtype=dtype)
        cache.append((k,v))
    torch.cuda.synchronize(); cache_s=time.perf_counter()-t
    alloc_cache=torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats()
    t=time.perf_counter()
    k,v=cache[0]
    kr=k.repeat_interleave(heads//kv,dim=1)
    vr=v.repeat_interleave(heads//kv,dim=1)
    torch.cuda.synchronize(); repeat_s=time.perf_counter()-t
    result={
      'seq':seq,'layers':layers,'kv_heads':kv,'attention_heads':heads,'head_dim':dim,
      'dtype':str(dtype),'theoretical_persistent_kv_gib':gib(layers*2*kv*seq*dim*2),
      'measured_cache_allocated_gib':gib(alloc_cache),
      'one_layer_repeated_kv_gib':gib(2*heads*seq*dim*2),
      'measured_peak_with_one_layer_repeat_gib':gib(torch.cuda.max_memory_allocated(device)),
      'cache_allocation_seconds':cache_s,'repeat_seconds':repeat_s,
    }
    print(json.dumps(result,indent=2))
    del kr,vr,cache,k,v
    torch.cuda.empty_cache()

if __name__=='__main__': main()

