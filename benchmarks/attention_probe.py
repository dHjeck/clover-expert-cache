"""Isolated real Clover attention weights + synthetic initialized KV. Not full-model throughput."""
import argparse,os,json,time,statistics,gc,types,inspect,textwrap,hashlib
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from safetensors import safe_open
ROOT=None
OUT=Path(__file__).parent/'results'/'attention-results.json'
def sync(): torch.cuda.synchronize()
def gib(n): return n/2**30

def timed(fn,reps=5):
    times=[]
    for _ in range(2):
        x=fn(); sync(); del x
    gc.collect(); torch.cuda.empty_cache(); sync()
    base=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    for _ in range(reps):
        sync(); start=time.perf_counter(); x=fn(); sync(); times.append((time.perf_counter()-start)*1000); del x
    return dict(ms=times,median_ms=statistics.median(times),base_gib=gib(base),peak_allocated_gib=gib(torch.cuda.max_memory_allocated()),peak_reserved_gib=gib(torch.cuda.max_memory_reserved()))

@torch.inference_mode()
def main():
    global ROOT, OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',default=os.environ.get('CLOVER_MODEL'),help='Local Clover model directory, or set CLOVER_MODEL.')
    ap.add_argument('--output',default=str(OUT))
    args=ap.parse_args()
    if not args.model: ap.error('--model is required (or set CLOVER_MODEL)')
    ROOT=Path(args.model)
    OUT=Path(args.output)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20261028); torch.cuda.init(); torch.set_num_threads(16)
    config=AutoConfig.from_pretrained(str(ROOT),trust_remote_code=True)
    cls=get_class_from_dynamic_module('modeling_clover_moe.CloverAttention',str(ROOT),local_files_only=True)
    attn=cls(config).to(dtype=torch.bfloat16).eval()
    idx=json.loads((ROOT/'model.safetensors.index.json').read_text())['weight_map']
    prefix='model.layers.3.self_attn.'
    weights={}
    for shard in sorted({v for k,v in idx.items() if k.startswith(prefix)}):
        with safe_open(str(ROOT/shard),framework='pt',device='cpu') as f:
            for key in idx:
                if key.startswith(prefix) and idx[key]==shard:weights[key[len(prefix):]]=f.get_tensor(key)
    attn.load_state_dict(weights,strict=True); del weights; attn.cuda()
    source=textwrap.dedent(inspect.getsource(cls.forward))
    modified=source.replace('    k = k.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1)\n','').replace('    v = v.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1)\n','')
    modified=modified.replace('attn_mask=allow, dropout_p=0.0','attn_mask=allow, dropout_p=0.0, enable_gqa=True')
    assert modified!=source and 'repeat_interleave' not in modified
    ns={};exec(modified,cls.forward.__globals__,ns)
    gqa=types.MethodType(ns['forward'],attn)
    result={'kind':'isolated_real_attention_weights_synthetic_KV_not_full_generation','torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'rows':[],'h2d':[]}
    def save():OUT.write_text(json.dumps(result,indent=2),encoding='utf8')
    # Reserve/touch memory matching all non-routed weights, less this real attention module.
    fixed_bytes=4898192128 # replaced below by exact safetensors header accounting
    import struct
    fixed_bytes=0
    for p in ROOT.glob('*.safetensors'):
        with p.open('rb') as f: h=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
        for k,v in h.items():
            if k!='__metadata__' and '.mlp.experts.' not in k:fixed_bytes+=v['data_offsets'][1]-v['data_offsets'][0]
    attn_bytes=sum(p.numel()*p.element_size() for p in attn.parameters())
    reservation=torch.zeros(fixed_bytes-attn_bytes,dtype=torch.uint8,device='cuda')
    gdn=[torch.zeros((1,32,128,128),dtype=torch.float32,device='cuda') for _ in range(30)]
    conv=[torch.zeros((1,8192,3),dtype=torch.bfloat16,device='cuda') for _ in range(30)]
    result['non_routed_weight_reservation_gib']=gib(fixed_bytes)
    result['gdn_state_gib']=gib(sum(t.numel()*t.element_size() for t in gdn+conv))
    for total in (8192,32768,65536,131072):
        prefix_len=total-32
        caches=[{'key':torch.randn((1,2,prefix_len,256),device='cuda',dtype=torch.bfloat16),'value':torch.randn((1,2,prefix_len,256),device='cuda',dtype=torch.bfloat16)} for _ in range(10)]
        x=torch.randn((1,32,2048),device='cuda',dtype=torch.bfloat16)
        valid=torch.ones((1,32),device='cuda',dtype=torch.bool)
        pv=torch.ones((1,prefix_len),device='cuda',dtype=torch.bool)
        positions=torch.arange(prefix_len,total,device='cuda')[None]
        args=(x,caches[0],valid,pv,positions,True)
        a,_=attn(*args); b,_=gqa(*args); sync()
        diff=(a.float()-b.float()).abs()
        row={'total_positions':total,'prefix_tokens':prefix_len,'block_size':32,'persistent_kv_gib':gib(10*2*2*prefix_len*256*2),'allclose_rtol_0_01_atol_0_001':torch.allclose(a,b,rtol=.01,atol=.001),'max_output_abs_diff':diff.max().item(),'mean_output_abs_diff':diff.mean().item()}
        del a,b,diff,_
        for name,func in [('original_repeat',attn),('native_gqa',gqa)]:
            try: row[name]=timed(lambda:func(*args))
            except Exception as e:row[name]={'error':str(e)}
        result['rows'].append(row);save();print(json.dumps(row),flush=True)
        del caches,x,valid,pv,positions,args
        gc.collect();torch.cuda.empty_cache()
    del reservation,gdn,conv,attn,gqa
    gc.collect();torch.cuda.empty_cache()
    for mib in (6,48,192):
        for pin in (False,True):
            cpu=torch.zeros(mib*2**20,dtype=torch.uint8,pin_memory=pin)
            gpu=torch.empty_like(cpu,device='cuda')
            r=timed(lambda:gpu.copy_(cpu,non_blocking=pin),reps=9)
            r.update(mib=mib,pinned=pin,effective_GBps=mib*2**20/(r['median_ms']/1000)/1e9)
            result['h2d'].append(r);save();print('H2D',json.dumps(r),flush=True)
            del cpu,gpu
    save()
if __name__=='__main__':main()
