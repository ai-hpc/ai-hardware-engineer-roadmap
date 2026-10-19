"""CPU teaching layout: low nibble first, padded rows, E4M3 scales, FP32 global.
Not a vendor checkpoint ABI and not native FP4 execution.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class PackedFP4:
    codes: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor
    shape: tuple
    padded_cols: int

    @property
    def payload_bytes(self):
        return self.codes.numel() + self.scales.numel() + self.global_scale.numel()*4


def pack_fp4(weight, clip_ratio=1.):
    if weight.device.type != 'cpu' or weight.ndim != 2 or weight.numel()==0 or not 0 < clip_ratio <= 1 or not torch.isfinite(weight).all():
        raise ValueError('finite nonempty CPU matrix and clipping ratio in (0,1] required')
    w=weight.detach().float()
    if not torch.isfinite(w).all():
        raise ValueError('values must be representable in float32 reference arithmetic')
    rows,cols=w.shape
    padded_cols=((cols+15)//16)*16
    blocks=F.pad(w,(0,padded_cols-cols)).reshape(rows,-1,16)
    amax=blocks.abs().amax(-1)*clip_ratio
    peak=float(amax.max())
    global_scale=torch.tensor(max(peak/(6*448),torch.finfo(torch.float32).tiny) if peak else 1.,dtype=torch.float32)
    # E4M3FN's smallest positive subnormal is 2^-9. Positive scales also handle
    # blocks that would underflow; all-zero blocks receive an arbitrary scale 1.
    raw=(amax/(6*global_scale)).clamp(min=2**-9,max=448)
    raw=torch.where(amax==0,torch.ones_like(raw),raw)
    scales=raw.to(torch.float8_e4m3fn)
    normalized=blocks/(scales.float()*global_scale).unsqueeze(-1)
    magnitudes=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.])
    distance=(normalized.abs().unsqueeze(-1)-magnitudes).abs()
    nearest=distance.min(-1,keepdim=True).values
    indices=torch.arange(8)
    # On exact halfway ties, choose the code with an even least-significant bit.
    rank=indices%2*8+indices
    selected=torch.where(distance==nearest,rank,torch.full_like(rank,100)).argmin(-1).to(torch.uint8)
    selected=selected | (torch.signbit(normalized).to(torch.uint8)<<3)
    flat=selected.reshape(-1)
    packed=flat[::2] | (flat[1::2]<<4)
    return PackedFP4(packed,scales,global_scale,(rows,cols),padded_cols)


def unpack_fp4(artifact):
    rows,cols=artifact.shape
    if artifact.codes.dtype != torch.uint8 or artifact.scales.dtype != torch.float8_e4m3fn or artifact.global_scale.dtype != torch.float32 or artifact.global_scale.numel()!=1:
        raise ValueError('invalid artifact dtypes')
    if rows<=0 or cols<=0 or artifact.padded_cols!=((cols+15)//16)*16 or artifact.codes.numel()!=rows*artifact.padded_cols//2 or artifact.scales.shape!=(rows,artifact.padded_cols//16):
        raise ValueError('invalid artifact shape')
    if not torch.isfinite(artifact.scales.float()).all() or not (artifact.scales.float()>0).all() or not torch.isfinite(artifact.global_scale).all() or not (artifact.global_scale>0).all():
        raise ValueError('finite positive scales required')
    codes=torch.stack((artifact.codes&15,artifact.codes>>4),-1).reshape(rows,-1,16)
    grid=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.])
    sign=torch.where((codes&8)!=0,-1.,1.)
    values=grid[(codes&7).long()]*sign*artifact.scales.float().unsqueeze(-1)*artifact.global_scale
    return values.reshape(rows,artifact.padded_cols)[:,:cols]


def select_clipping(weight, inputs=None, ratios=(.5,.65,.8,.9,1.)):
    if not ratios or 1. not in ratios:
        raise ValueError('candidate grid must include absmax ratio 1')
    if inputs is not None and (inputs.ndim!=2 or inputs.shape[1]!=weight.shape[1] or inputs.numel()==0 or not torch.isfinite(inputs).all()):
        raise ValueError('finite aligned nonempty calibration activations required')
    candidates=[]
    for ratio in ratios:
        artifact=pack_fp4(weight,ratio); error=unpack_fp4(artifact)-weight
        loss=error.square().mean() if inputs is None else (inputs@error.T).square().mean()
        candidates.append((float(loss),float(ratio),artifact))
    best=min(candidates,key=lambda row:(row[0],-row[1]))
    return best[2], {'selected_ratio':best[1],'calibration_loss':best[0],
                    'candidates':[{'ratio':r,'loss':loss} for loss,r,_ in candidates]}
