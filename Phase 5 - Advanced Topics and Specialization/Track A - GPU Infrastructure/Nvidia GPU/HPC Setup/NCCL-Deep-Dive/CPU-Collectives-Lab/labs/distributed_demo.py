"""Run with torchrun --standalone --nproc-per-node=2 distributed_demo.py.

Original deterministic synthetic training examples; no downloaded dataset.
This is correctness evidence for CPU/Gloo, never a GPU bandwidth benchmark.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from accounting import validate_signatures


def shard(rank, equal):
    # Unequal shards model two robots contributing different numbers of samples.
    values = ([1., 2.], [3., 4.] if equal else [3., 4., 5.])[rank]
    x = torch.tensor(values, dtype=torch.float64).reshape(-1, 1)
    return x, 2 * x + 1


def model():
    result = torch.nn.Linear(1, 1, dtype=torch.float64)
    with torch.no_grad():
        result.weight.fill_(.25)
        result.bias.fill_(-.5)
    return result


def gradients(module):
    return torch.cat([p.grad.flatten() for p in module.parameters()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--equal-batches", action="store_true")
    parser.add_argument("--mismatch", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("collectives-report.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 2:
            raise ValueError("this worked example requires exactly two ranks")
        value = torch.full((3 if args.mismatch and rank == 1 else 2,), rank + 1., dtype=torch.float64)
        signature = (list(value.shape), str(value.dtype), "sum")
        signatures = [None] * world
        # Trusted local metadata only; object collectives use pickle.
        dist.all_gather_object(signatures, signature)
        try:
            validate_signatures(signatures)
        except ValueError as error:
            if not args.mismatch: raise
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"rejected_before_data_collective": str(error)}, indent=2) + "\n")
            return
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(value, torch.tensor([3., 3.], dtype=torch.float64))
        gathered = [torch.empty(1, dtype=torch.float64) for _ in range(world)]
        dist.all_gather(gathered, torch.tensor([rank + 1.], dtype=torch.float64))
        torch.testing.assert_close(torch.cat(gathered), torch.tensor([1., 2.], dtype=torch.float64))

        x, y = shard(rank, args.equal_batches)
        count = torch.tensor(float(len(x)), dtype=torch.float64)
        dist.all_reduce(count)
        all_x = torch.cat([shard(r, args.equal_batches)[0] for r in range(world)])
        all_y = 2 * all_x + 1
        reference = model()
        ((reference(all_x) - all_y) ** 2).mean().backward()
        ref_gradient = gradients(reference)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        ref_optimizer.step()

        results = {}
        for mode in ("local_mean", "global_sample_mean"):
            distributed = DDP(model())
            errors = (distributed(x) - y) ** 2
            # DDP averages across ranks. Correct it to average across samples.
            loss = errors.mean() if mode == "local_mean" else errors.sum() * world / count
            loss.backward()
            grad = gradients(distributed.module).clone()
            optimizer = torch.optim.SGD(distributed.parameters(), lr=.01)
            optimizer.step()
            parameter_error = max((a - b).abs().max().item() for a, b in
                                  zip(distributed.module.parameters(), reference.parameters()))
            results[mode] = {"gradient": grad.tolist(),
                             "max_gradient_error": (grad - ref_gradient).abs().max().item(),
                             "max_parameter_error": parameter_error}
            if mode == "global_sample_mean" or args.equal_batches:
                torch.testing.assert_close(grad, ref_gradient, rtol=1e-12, atol=1e-12)
                assert parameter_error < 1e-12
            else:
                assert results[mode]["max_gradient_error"] > 1
        if rank == 0:
            report = {"backend": "gloo", "device": "cpu", "ranks": world,
                      "python": platform.python_version(), "torch": torch.__version__,
                      "per_rank_samples": [len(shard(r, args.equal_batches)[0]) for r in range(world)],
                      "all_reduce_sum": value.tolist(), "all_gather": torch.cat(gathered).tolist(),
                      "serial_gradient": ref_gradient.tolist(), "comparisons": results,
                      "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__": main()
