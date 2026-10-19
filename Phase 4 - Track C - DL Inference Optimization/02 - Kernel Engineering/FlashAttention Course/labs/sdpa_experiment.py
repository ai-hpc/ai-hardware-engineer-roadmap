"""Controlled SDPA forward comparison; CPU math executed, CUDA path optional."""
import argparse
import json
import math
import statistics
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from attention_contract import causal_mask, count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--backend',choices=['math','flash','efficient','cudnn'],default='math')
    p.add_argument('--dtype',choices=['float64','float32','float16','bfloat16'],default='float64')
    p.add_argument('--q-len',type=int,default=7);p.add_argument('--kv-len',type=int,default=11)
    p.add_argument('--dim',type=int,default=8);p.add_argument('--heads',type=int,default=2)
    p.add_argument('--alignment',choices=['none','upper_left','lower_right'],default='lower_right')
    p.add_argument('--repeats',type=int,default=20)
    args=p.parse_args()
    for key in ['q_len','kv_len','dim','heads','repeats']:count(getattr(args,key),key)
    if args.q_len*args.kv_len*args.heads>16_000_000: p.error('dense CPU oracle limited to 16 million scores')
    if args.repeats>1000:p.error('repeats limited to 1000')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable; no GPU result produced')
    if args.device=='cpu' and args.backend!='math':p.error('this CPU experiment explicitly selects the math backend')
    torch.set_num_threads(1);g=torch.Generator().manual_seed(17)
    dtype=getattr(torch,args.dtype)
    q=torch.randn(1,args.heads,args.q_len,args.dim,generator=g,dtype=torch.float64).to(dtype=dtype,device=args.device)
    k=torch.randn(1,args.heads,args.kv_len,args.dim,generator=g,dtype=torch.float64).to(dtype=dtype,device=args.device)
    v=torch.randn(1,args.heads,args.kv_len,args.dim,generator=g,dtype=torch.float64).to(dtype=dtype,device=args.device)
    mask=None if args.alignment=='none' else torch.tensor(causal_mask(args.q_len,args.kv_len,args.alignment),device=args.device)
    selected={'math':SDPBackend.MATH,'flash':SDPBackend.FLASH_ATTENTION,'efficient':SDPBackend.EFFICIENT_ATTENTION,'cudnn':SDPBackend.CUDNN_ATTENTION}[args.backend]
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        reference=F.scaled_dot_product_attention(q.cpu().double(),k.cpu().double(),v.cpu().double(),attn_mask=None if mask is None else mask.cpu(),dropout_p=0)
    def execute():return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0)
    samples=[]
    with torch.inference_mode(),sdpa_kernel(selected):
        actual=execute()
        for _ in range(5):execute()
        if args.device=='cuda':torch.cuda.synchronize()
        for _ in range(args.repeats):
            if args.device=='cuda':
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();execute();end.record();end.synchronize();samples.append(start.elapsed_time(end))
            else:
                start=time.perf_counter_ns();execute();samples.append((time.perf_counter_ns()-start)/1e6)
    actual_cpu=actual.cpu().double()
    tolerance={'float64':(1e-12,1e-12),'float32':(2e-5,2e-5),'float16':(.003,.003),'bfloat16':(.01,.01)}[args.dtype]
    passed=torch.allclose(actual_cpu,reference,atol=tolerance[0],rtol=tolerance[1])
    report={'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'device':args.device,
            'device_name':torch.cuda.get_device_name() if args.device=='cuda' else 'CPU',
            'backend_requested':args.backend,'dtype':args.dtype,'shape':[1,args.heads,args.q_len,args.kv_len,args.dim],
            'alignment':args.alignment,'dropout_p':0,'seed':17,'threads':torch.get_num_threads(),
            'oracle':'CPU math float64 on the same quantized inputs','max_abs_error':float((actual_cpu-reference).abs().max()),
            'atol':tolerance[0],'rtol':tolerance[1],'passes_fixture_tolerance':passed,
            'warmup':5,'forward_ms_samples':samples,'forward_ms_median':statistics.median(samples),
            'timing_boundary':'forward SDPA including explicit mask; excludes allocation, transfer, reference and compilation warmup',
            'scope':'No FlashAttention distribution installed by this script; CPU numbers do not predict GPU speed or serving TPS.'}
    print(json.dumps(report,indent=2,allow_nan=False))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
