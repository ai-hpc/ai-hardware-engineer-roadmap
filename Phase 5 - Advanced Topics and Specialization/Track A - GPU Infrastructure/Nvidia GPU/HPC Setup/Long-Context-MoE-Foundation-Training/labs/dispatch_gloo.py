"""Real two-process CPU AllToAll dispatch/combine, checked against serial experts.

No training, CUDA, NCCL, bandwidth or large-model claim is made by this fixture.
"""
import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import torch
import torch.distributed as dist


def expert(x, rank):
    return (rank + 2) * x + rank


def main():
    parser=argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--skew',action='store_true')
    modes.add_argument('--empty-expert',action='store_true')
    modes.add_argument('--empty-source',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('dispatch.json'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group('gloo',timeout=timedelta(seconds=20))
    try:
        rank,world=dist.get_rank(),dist.get_world_size()
        if world != 2: raise ValueError('fixture requires two ranks')
        x=torch.arange(1.,rank+3,dtype=torch.float64).reshape(-1,1)+rank*10
        if args.empty_source and rank == 0:
            x = torch.empty(0, 1, dtype=torch.float64)
        routes=[]
        for token in range(len(x)):
            if args.empty_expert:
                routes.append([(0, 1.)])
            elif args.skew:
                routes.append([(0 if rank==0 else token%2,1.)])
            else:
                routes.append([(0,.25),(1,.75)])
        # Pack in destination order. Local lookup is retained for the inverse permutation.
        packed=[];lookup=[];send_splits=[]
        for destination in range(world):
            start=len(packed)
            for token,row in enumerate(routes):
                for chosen,weight in row:
                    if chosen==destination:
                        packed.append(x[token]);lookup.append((token,weight))
            send_splits.append(len(packed)-start)
        send_counts=torch.tensor(send_splits,dtype=torch.int64)
        recv_counts=torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts,send_counts)
        recv_splits=recv_counts.tolist()
        send=torch.stack(packed) if packed else x.new_empty((0, x.shape[1]))
        received=torch.empty(sum(recv_splits),1,dtype=torch.float64)
        dist.all_to_all_single(received,send,output_split_sizes=recv_splits,input_split_sizes=send_splits)
        computed=expert(received,rank)
        returned=torch.empty_like(send)
        # The reverse exchange reverses the split vectors, preserving sender-local order.
        dist.all_to_all_single(returned,computed,output_split_sizes=send_splits,input_split_sizes=recv_splits)
        combined=torch.zeros_like(x)
        for value,(token,weight) in zip(returned,lookup): combined[token]+=weight*value
        reference=torch.zeros_like(x)
        for token,row in enumerate(routes):
            for chosen,weight in row: reference[token]+=weight*expert(x[token],chosen)
        torch.testing.assert_close(combined,reference,atol=0,rtol=0)
        row={'rank':rank,'input':x.flatten().tolist(),'send_splits':send_splits,'receive_splits':recv_splits,
             'output':combined.flatten().tolist(),'serial':reference.flatten().tolist(),
             'max_abs_error':(combined-reference).abs().max().item() if x.numel() else 0.,
             'source_tokens':len(x),'received_assignments':len(received)}
        rows=[None]*world
        dist.all_gather_object(rows,row)  # trusted local fixture peers only
        if rank==0:
            report={'backend':'gloo','device':'cpu','torch':torch.__version__,'skew':args.skew,'empty_expert':args.empty_expert,'empty_source':args.empty_source,'ranks':rows,
                    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    finally: dist.destroy_process_group()


if __name__=='__main__': main()
