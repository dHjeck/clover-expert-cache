"""Chat parity and full-state memory validation at conservative cache budgets."""
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import torch
import clover_server
from clover_expert_runtime import CloverExpertRuntime, install_clover_expert_cache, cache_slots_for_budget
OUT=Path('runtime-full-context-validation.json')
result={'runtime_sha256':hashlib.sha256(Path('clover_expert_runtime.py').read_bytes()).hexdigest(),'chat_runs':[],'long_context':[]}
def save(): OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

def resize(model,gib):
    torch.cuda.synchronize()
    n=cache_slots_for_budget(gib,len(model.model.layers))
    for layer in model.model.layers:
        old=layer.mlp.experts
        layer.mlp.experts=CloverExpertRuntime(old.bank_gate,old.bank_down,n-2,
             device='cuda:0',pin_memory=True,transient_slots=2)
        del old
    gc.collect(); torch.cuda.empty_cache()

@torch.inference_mode()
def long_check(model,gib):
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    physical=torch.cuda.get_device_properties(0).total_memory
    available=torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()
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
    tokens=torch.full((1,32),279,dtype=torch.long,device='cuda')
    row={'cache_gib':gib,'kind':'synthetic_initialized_states_not_real_128K_prefill_or_quality',
         'physical_gib':physical/2**30,'estimated_process_budget_gib':available/2**30,
         'kv_bytes':sum(t.nbytes for layer in states for k,t in layer.items() if k in ('key','value'))}
    for preview in (True,False):
        phase='preview' if preview else 'causal_replay'
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        try:
            output=model(input_ids=tokens,past_key_values=cache,use_cache=True,preview=preview)
            torch.cuda.synchronize()
            row[phase]={'success':True,'seconds':time.perf_counter()-start,
                        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                        'allocated_within_available_budget':torch.cuda.max_memory_allocated() <= available,
                        'reserved_within_available_budget':torch.cuda.max_memory_reserved() <= available,
                        'finite_logits':bool(torch.isfinite(output.logits).all()),
                        'cache_length':output.past_key_values.attention_mask.shape[-1],
                        'same_cache_object':output.past_key_values is cache}
            del output
        except Exception as exc:
            row[phase]={'success':False,'error':repr(exc)}
        print(json.dumps({phase:row[phase],'cache_gib':gib}),flush=True)
    del cache,states,tokens
    gc.collect();torch.cuda.empty_cache()
    return row

@torch.inference_mode()
def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',default=os.environ.get('CLOVER_MODEL'),
                    help='Local Clover model directory, or set CLOVER_MODEL.')
    ap.add_argument('--output',default='runtime-full-context-validation.json')
    args=ap.parse_args()
    if not args.model:
        ap.error('--model is required (or set CLOVER_MODEL)')
    OUT=Path(args.output)
    torch.set_num_threads(16);torch.manual_seed(42)
    clover_server.MODEL_PATH=args.model
    os.environ['CLOVER_MODEL']=args.model
    server=clover_server
    server.build()
    model,tok=server.state['model'],server.state['tok']
    prompts=[('zh','用一句话解释为什么天空是蓝色的。'),('code','What does this Python function return? def f(x): return x * x')]
    captures={}
    def capture(module,args,out):
        if not captures:
            captures.update(x=args[0].clone(),ids=args[1].clone(),weights=args[2].clone(),expected=out.clone())
    hook=model.model.layers[0].mlp.experts.register_forward_hook(capture)
    for backend in ('original','cached_3g','cached_2g'):
        if backend=='cached_3g':
            hook.remove();model.cpu();gc.collect();torch.cuda.empty_cache()
            result['install_3g']=install_clover_expert_cache(model,cache_gib=3,
                  progress=lambda i,n: print(f'pin {i}/{n}',flush=True) if i%10==0 else None)
            model.cuda();torch.cuda.synchronize()
            rt=model.model.layers[0].mlp.experts
            actual=rt(captures['x'],captures['ids'],captures['weights'])
            result['real_weight_transfer_parity']={'equal':torch.equal(actual,captures['expected']),
                             'max_abs_diff':(actual.float()-captures['expected'].float()).abs().max().item()}
            rt.miss_backend='cpu'
            actual=rt(captures['x'],captures['ids'],captures['weights'])
            result['real_weight_cpu_fallback_parity']={'equal':torch.equal(actual,captures['expected']),
                'allclose_1pct_1e3':torch.allclose(actual,captures['expected'],rtol=.01,atol=.001),
                'max_abs_diff':(actual.float()-captures['expected'].float()).abs().max().item(),
                'cpu_experts':rt.stats()['cpu_experts']}
            rt.miss_backend='transfer';captures.clear();del actual;save()
        elif backend=='cached_2g':
            resize(model,2)
        for name,text in prompts:
            chat=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prompt=tok(chat,add_special_tokens=False,return_tensors='pt').input_ids.cuda()
            start=time.perf_counter();out=model.generate(input_ids=prompt,max_new_tokens=8,block_size=4,threshold=.95,do_sample=False)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            tokens=out[0,prompt.shape[1]:].tolist()
            row={'backend':backend,'case':name,'prompt_ids':prompt[0].tolist(),'token_ids':tokens,
                 'decoded':tok.decode(tokens,skip_special_tokens=True),'elapsed_seconds':seconds}
            if backend!='original':
                ref=next(r for r in result['chat_runs'] if r['backend']=='original' and r['case']==name)
                row['equal_original']=tokens==ref['token_ids']
            result['chat_runs'].append(row);save();print(json.dumps(row,ensure_ascii=True),flush=True)
        if backend!='original':
            result['long_context'].append(long_check(model,3 if backend=='cached_3g' else 2));save()
    result['status']='complete';save()

if __name__=='__main__':main()
