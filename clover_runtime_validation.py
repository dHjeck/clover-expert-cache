"""End-to-end original/cache comparison + synthetic 128K state stress, one model load."""
import argparse, gc, json, os, sys, time, types, hashlib
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import torch
import clover_server
from clover_expert_runtime import install_clover_expert_cache
OUT=Path('runtime-validation.json')
result={'kind':'real_model_generation_and_synthetic_long_context_stress','runs':[]}
def save(): OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

def stats(model):
    rows=[l.mlp.experts.stats() for l in model.model.layers if hasattr(l.mlp.experts,'stats')]
    return {k:sum(r[k] for r in rows) for k in ('hits','misses','h2d_bytes','cpu_experts')}

@torch.inference_mode()
def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',default=os.environ.get('CLOVER_MODEL'),
                    help='Local Clover model directory, or set CLOVER_MODEL.')
    ap.add_argument('--output',default='runtime-validation.json')
    args=ap.parse_args()
    if not args.model:
        ap.error('--model is required (or set CLOVER_MODEL)')
    OUT=Path(args.output)
    torch.set_num_threads(16)
    torch.manual_seed(42)
    clover_server.MODEL_PATH=args.model
    os.environ['CLOVER_MODEL']=args.model
    server=clover_server
    result.update(torch=torch.__version__,gpu=torch.cuda.get_device_name(),threads=torch.get_num_threads(),
                  runtime_sha256=hashlib.sha256(Path('clover_expert_runtime.py').read_bytes()).hexdigest())
    server.build()
    model,tok=server.state['model'],server.state['tok']
    seed=tok('Explain the following code carefully and propose a small safe improvement. ',add_special_tokens=False,return_tensors='pt').input_ids[0]
    prompt=seed.repeat(20)[:128][None].cuda()
    result['prompt_ids']=prompt[0].tolist()
    phases=[]
    original_forward=model.model.forward
    def timed_forward(*args,**kwargs):
        phase='preview' if kwargs.get('preview',False) else ('replay' if kwargs.get('past_key_values') is not None else 'prefill')
        torch.cuda.synchronize(); start=time.perf_counter()
        output=original_forward(*args,**kwargs)
        torch.cuda.synchronize()
        phases.append({'phase':phase,'seconds':time.perf_counter()-start})
        return output
    model.model.forward=timed_forward
    # Preserve a real routed activation, not a synthetic matrix, for direct parity.
    captured={}
    def capture(module,args,out):
        if not captured:
            captured.update(x=args[0].detach().clone(),ids=args[1].detach().clone(),
                            weights=args[2].detach().clone(),expected=out.detach().clone())
    hook=model.model.layers[0].mlp.experts.register_forward_hook(capture)
    cases=[(4,4),(32,4),(4,16),(32,32)]
    for backend in ('original','cached_5g'):
        if backend=='cached_5g':
            hook.remove()
            print('moving original GPU layers to CPU before installing all-GPU shell',flush=True)
            model.cpu(); gc.collect(); torch.cuda.empty_cache()
            report=install_clover_expert_cache(model,cache_gib=5,progress=lambda i,n:print(f'pin {i}/{n}',flush=True) if i%10==0 else None)
            model.cuda(); torch.cuda.synchronize()
            result['install']=report
            actual=model.model.layers[0].mlp.experts(captured['x'],captured['ids'],captured['weights'])
            diff=(actual.float()-captured['expected'].float()).abs()
            result['real_expert_parity']={'equal':torch.equal(actual,captured['expected']),
                                         'max_abs_diff':diff.max().item(),'tokens':captured['x'].shape[0]}
            del actual,diff
            captured.clear()
            save()
        for block,new in cases:
            for rep in range(2):
                phases.clear(); before=stats(model)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                output=model.generate(input_ids=prompt,max_new_tokens=new,block_size=block,threshold=.95,do_sample=False)
                torch.cuda.synchronize(); elapsed=time.perf_counter()-start
                tokens=output[0,128:].tolist()
                prefill=sum(p['seconds'] for p in phases if p['phase']=='prefill')
                after=stats(model)
                row={'backend':backend,'block':block,'requested_tokens':new,'repeat':rep,
                     'token_ids':tokens,'decoded':tok.decode(tokens,skip_special_tokens=True),
                     'elapsed_seconds':elapsed,'prefill_seconds':prefill,
                     'generation_seconds_excluding_prefill':elapsed-prefill,
                     'generation_tok_s_excluding_prefill':len(tokens)/(elapsed-prefill),
                     'phase_seconds':{k:sum(p['seconds'] for p in phases if p['phase']==k) for k in ('prefill','preview','replay')},
                     'preview_forwards':sum(p['phase']=='preview' for p in phases),
                     'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                     'cache_delta':{k:after[k]-before[k] for k in after}}
                if backend!='original':
                    base=next(r for r in result['runs'] if r['backend']=='original' and r['block']==block and r['requested_tokens']==new and r['repeat']==rep)
                    row['token_ids_equal_original']=tokens==base['token_ids']
                result['runs'].append(row); save()
                print(json.dumps(row,ensure_ascii=True),flush=True)
    # Full model shell + all expert slots + initialized synthetic states.
    # This tests capacity/forward execution, not real 128K prefill or long-context quality.
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    module=sys.modules[type(model.model).__module__]
    length=131072-32
    states=[]
    for layer in model.model.layers:
        if hasattr(layer,'self_attn'):
            states.append({'key':torch.zeros((1,2,length,256),device='cuda',dtype=torch.bfloat16),
                           'value':torch.zeros((1,2,length,256),device='cuda',dtype=torch.bfloat16)})
        else:
            a=layer.linear_attn
            states.append({'conv':torch.zeros((1,a.channels,a.config.linear_conv_kernel_dim-1),device='cuda',dtype=torch.bfloat16),
                           'ssm':torch.zeros((1,a.vh,a.vd,a.kd),device='cuda',dtype=torch.float32)})
    cache=module.CloverCache(states,torch.ones((1,length),dtype=torch.bool,device='cuda'))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    long={'kind':'synthetic_128K_initialized_KV_GDN_all_40_layers_not_real_prefill','prefix_tokens':length,'preview_tokens':32,
          'base_allocated_gib':torch.cuda.memory_allocated()/2**30}
    try:
        start=time.perf_counter()
        output=model(input_ids=prompt[:,:32],past_key_values=cache,use_cache=True,preview=True)
        torch.cuda.synchronize()
        long.update(success=True,seconds=time.perf_counter()-start,finite_logits=bool(torch.isfinite(output.logits).all()),
                    same_cache_object=output.past_key_values is cache)
        del output
        # Also exercise causal commit, when new KV coexists with the old cache.
        start=time.perf_counter()
        replay=model.model(prompt[:,:32],past_key_values=cache,use_cache=True)
        torch.cuda.synchronize()
        long.update(replay_success=True,replay_seconds=time.perf_counter()-start,
                    replay_length=replay.past_key_values.attention_mask.shape[-1])
        del replay
    except Exception as exc:
        long.update(success=False,error=repr(exc))
    long.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    result['long_context']=long; save(); print(json.dumps(long),flush=True)
    result['status']='complete';save()

if __name__=='__main__': main()
