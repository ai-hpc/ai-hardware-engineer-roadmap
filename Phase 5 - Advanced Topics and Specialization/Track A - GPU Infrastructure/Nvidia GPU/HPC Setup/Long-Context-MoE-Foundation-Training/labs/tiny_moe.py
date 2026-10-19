"""CPU top-k MoE with explicit capacity semantics and real optimization.

This is a teaching network, not a transformer or a distributed training engine.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ledger import capacity, positive_int


class TopKMoE(nn.Module):
    def __init__(self, hidden=4, width=16, experts=4, top_k=2, capacity_factor=None):
        super().__init__()
        for name, value in (("hidden", hidden), ("width", width),
                            ("experts", experts), ("top_k", top_k)):
            positive_int(name, value)
        if top_k > experts:
            raise ValueError('top_k cannot exceed experts')
        if capacity_factor is not None:
            capacity(1, top_k, experts, capacity_factor)
        self.num_experts, self.top_k, self.capacity_factor = experts, top_k, capacity_factor
        self.router = nn.Linear(hidden, experts, bias=False)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(hidden, width), nn.GELU(), nn.Linear(width, hidden)) for _ in range(experts))

    def forward(self, x, aux_weight=.01, z_weight=.001, dense_reference=False):
        for name, value in (("aux_weight", aux_weight), ("z_weight", z_weight)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point()
                or x.ndim != 2 or x.shape[0] == 0
                or x.shape[1] != self.router.in_features or not torch.isfinite(x).all()):
            raise ValueError('x must be finite nonempty [tokens,hidden]')
        logits = self.router(x)
        if not torch.isfinite(logits).all():
            raise ValueError('router logits must remain finite; inspect inputs and weights')
        probabilities = logits.softmax(-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        limit = x.shape[0] if self.capacity_factor is None else capacity(x.shape[0], self.top_k, self.num_experts, self.capacity_factor)
        keep = torch.zeros_like(indices, dtype=torch.bool)
        counts = torch.bincount(indices.flatten(), minlength=self.num_experts)
        for expert in range(self.num_experts):
            token, slot = torch.where(indices == expert)
            order = weights[token, slot].argsort(descending=True, stable=True)[:limit]
            keep[token[order], slot[order]] = True
        out = torch.zeros_like(x)
        if dense_reference:
            # Deliberately expensive all-expert oracle with the same admitted assignments.
            dense_weights = torch.zeros_like(probabilities).scatter_add(1, indices, weights * keep)
            for expert, network in enumerate(self.experts):
                out = out + network(x) * dense_weights[:, expert:expert+1]
        else:
            for expert, network in enumerate(self.experts):
                token, slot = torch.where((indices == expert) & keep)
                if token.numel():
                    out = out.index_add(0, token, network(x[token]) * weights[token, slot, None])
        f = counts.to(x.dtype) / indices.numel()
        aux = aux_weight * self.num_experts * (f * probabilities.mean(0)).sum()
        z = z_weight * logits.logsumexp(-1).square().mean() if z_weight else logits.new_zeros(())
        if not torch.isfinite(out).all() or not torch.isfinite(aux + z):
            raise ValueError("expert outputs and regularizer must remain finite")
        stats = {'assignment_shares': f.detach().tolist(),
                 'dropped_assignment_fraction': (~keep).float().mean().item(),
                 'tokens_with_any_drop_fraction': (~keep.all(-1)).float().mean().item(),
                 'tokens_with_no_expert_fraction': (~keep.any(-1)).float().mean().item(),
                 'capacity': limit}
        return out, aux + z, stats


def dataset(seed, count):
    positive_int("count", count)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(count, 4, generator=generator)
    y = torch.stack((x[:,0].sin(), x[:,1].square()*.3, x[:,2]*x[:,3], (x[:,0]+x[:,3]).tanh()), dim=1)
    return x, y


def train(aux_weight, steps=120, capacity_factor=None):
    positive_int("steps", steps)
    torch.manual_seed(7)
    model = TopKMoE(capacity_factor=capacity_factor)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    x, y = dataset(41, 512)
    valid_x, valid_y = dataset(42, 256)
    history = []
    with torch.no_grad(): initial = F.mse_loss(model(valid_x, aux_weight)[0], valid_y).item()
    for step in range(steps):
        optimizer.zero_grad()
        predicted, regularizer, stats = model(x, aux_weight)
        loss = F.mse_loss(predicted, y)
        (loss + regularizer).backward()
        optimizer.step()
        if step % 20 == 0 or step == steps-1:
            history.append({'step':step, 'training_mse':loss.item(), 'regularizer':regularizer.item(), **stats})
    with torch.no_grad():
        prediction, _, stats = model(valid_x, aux_weight)
        final = F.mse_loss(prediction, valid_y).item()
    return model, {'aux_weight':aux_weight, 'steps':steps, 'capacity_factor':capacity_factor,
                   'initial_validation_mse':initial, 'final_validation_mse':final,
                   'final_validation_routing':stats, 'history':history}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('moe-results'))
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--capacity-factor',type=float,default=None)
    args=parser.parse_args()
    if args.steps < 1: parser.error('--steps must be positive')
    if args.capacity_factor is not None: capacity(1,2,4,args.capacity_factor)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for weight in (0., .01, .5):
        model,row=train(weight,args.steps,args.capacity_factor)
        path=args.output/f'model-aux-{weight}.pt'
        torch.save(model.state_dict(),path)
        reloaded=TopKMoE(capacity_factor=args.capacity_factor)
        reloaded.load_state_dict(torch.load(path,weights_only=True))
        probe,_=dataset(43,17)
        with torch.no_grad(): row['reload_max_abs_error']=(model(probe)[0]-reloaded(probe)[0]).abs().max().item()
        results.append(row)
    report={'torch':torch.__version__,'device':'cpu','dtype':'float32','train_seed':41,'validation_seed':42,
            'task':'four-dimensional synthetic nonlinear regression; no language/robot training',
            'ablations':results}
    (args.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
