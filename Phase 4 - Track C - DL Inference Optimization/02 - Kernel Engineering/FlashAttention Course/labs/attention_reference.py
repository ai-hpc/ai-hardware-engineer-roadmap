"""Small actual NumPy attention references; masks True=allowed, empty rows=zero."""
from __future__ import annotations
import argparse
import json
import math
import numpy as np
from attention_contract import causal_mask, count


def prepare(q,k,v,mask,dtype=np.float64):
    arrays=[np.asarray(x,dtype=dtype) for x in (q,k,v)]
    if any(x.ndim!=2 or not all(x.shape) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Q, K, V must be finite, nonempty rank-2 arrays')
    q,k,v=arrays
    if q.shape[1]!=k.shape[1] or k.shape[0]!=v.shape[0]:raise ValueError('incompatible dimensions')
    allowed=np.ones((len(q),len(k)),dtype=bool) if mask is None else np.asarray(mask)
    if allowed.dtype!=np.bool_ or allowed.shape!=(len(q),len(k)):raise ValueError('mask must be boolean [L,S]')
    return q,k,v,allowed


def dense(q,k,v,mask=None,*,scale=None,dtype=np.float64):
    q,k,v,allowed=prepare(q,k,v,mask,dtype)
    factor=1/math.sqrt(q.shape[1]) if scale is None else scale
    if not math.isfinite(factor):raise ValueError('scale must be finite')
    scores=(q@k.T)*factor
    if not np.isfinite(scores).all():raise ValueError('nonfinite dot product')
    scores=np.where(allowed,scores,-np.inf)
    active=allowed.any(axis=1)
    maximum=np.max(scores,axis=1)
    safe=np.where(active,maximum,0)
    probabilities=np.exp(scores-safe[:,None])
    denominator=probabilities.sum(axis=1)
    probabilities/=np.where(active,denominator,1)[:,None]
    lse=np.full(len(q),-np.inf,dtype=dtype)
    lse[active]=maximum[active]+np.log(denominator[active])
    return probabilities@v,lse,probabilities


def online(q,k,v,mask=None,*,row_tile=3,column_tile=4,scale=None,dtype=np.float64):
    count(row_tile,'row_tile');count(column_tile,'column_tile')
    q,k,v,allowed=prepare(q,k,v,mask,dtype)
    factor=1/math.sqrt(q.shape[1]) if scale is None else scale
    if not math.isfinite(factor):raise ValueError('scale must be finite')
    output=np.zeros((len(q),v.shape[1]),dtype=dtype)
    lse=np.full(len(q),-np.inf,dtype=dtype)
    peak_score_elements=0
    for i in range(0,len(q),row_tile):
        stop=min(i+row_tile,len(q));rows=stop-i
        maximum=np.full(rows,-np.inf,dtype=dtype)
        denominator=np.zeros(rows,dtype=dtype)
        numerator=np.zeros((rows,v.shape[1]),dtype=dtype)
        for j in range(0,len(k),column_tile):
            end=min(j+column_tile,len(k))
            scores=(q[i:stop]@k[j:end].T)*factor
            if not np.isfinite(scores).all():raise ValueError('nonfinite dot product')
            peak_score_elements=max(peak_score_elements,scores.size)
            scores=np.where(allowed[i:stop,j:end],scores,-np.inf)
            next_max=np.maximum(maximum,np.max(scores,axis=1))
            safe=np.where(np.isfinite(next_max),next_max,0)
            alpha=np.zeros(rows,dtype=dtype)
            had_values=denominator>0
            alpha[had_values]=np.exp(maximum[had_values]-safe[had_values])
            weights=np.exp(scores-safe[:,None])
            numerator=alpha[:,None]*numerator+weights@v[j:end]
            denominator=alpha*denominator+weights.sum(axis=1)
            maximum=next_max
        active=denominator>0
        output[i:stop]=numerator/np.where(active,denominator,1)[:,None]
        local_lse=np.full(rows,-np.inf,dtype=dtype)
        local_lse[active]=maximum[active]+np.log(denominator[active])
        lse[i:stop]=local_lse
    return output,lse,{'largest_score_tile_elements':peak_score_elements,'dense_score_elements':len(q)*len(k)}


def backward(q,k,v,dout,mask=None,*,scale=None):
    q,k,v,allowed=prepare(q,k,v,mask)
    factor=1/math.sqrt(q.shape[1]) if scale is None else scale
    out,lse,p=dense(q,k,v,allowed,scale=factor)
    dout=np.asarray(dout,dtype=np.float64)
    if dout.shape!=out.shape or not np.isfinite(dout).all():raise ValueError('invalid output gradient')
    dv=p.T@dout
    dp=dout@v.T
    ds=p*(dp-(dp*p).sum(axis=1,keepdims=True))
    return ds@k*factor,ds.T@q*factor,dv


def rope_interleaved(x,positions,*,base=10000.0):
    """Toy standard RoPE only: all dimensions, adjacent pairs, absolute positions."""
    x=np.asarray(x,dtype=np.float64);positions=np.asarray(positions,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]%2 or positions.shape!=(len(x),):raise ValueError('invalid RoPE shape')
    if not np.isfinite(x).all() or not np.isfinite(positions).all() or base<=0:raise ValueError('invalid RoPE values')
    frequencies=base**(-np.arange(0,x.shape[1],2)/x.shape[1])
    angles=positions[:,None]*frequencies[None,:]
    c,s=np.cos(angles),np.sin(angles);out=np.empty_like(x)
    out[:,0::2]=x[:,0::2]*c-x[:,1::2]*s
    out[:,1::2]=x[:,0::2]*s+x[:,1::2]*c
    return out


def report():
    rng=np.random.default_rng(17);rows=[]
    for length,keys in [(7,11),(1,9),(9,2)]:
        q=rng.normal(size=(length,8));k=rng.normal(size=(keys,8));v=rng.normal(size=(keys,5))
        mask=np.array(causal_mask(length,keys,'lower_right'))
        for dtype in (np.float64,np.float32):
            reference=dense(q,k,v,mask)[0]
            output,lse,scratch=online(q,k,v,mask,dtype=dtype)
            rows.append({'shape':[length,keys,8,5],'accumulator':np.dtype(dtype).name,
                         'max_abs_error_vs_float64':float(np.max(abs(output-reference))),
                         'fully_masked_rows':int((~mask.any(axis=1)).sum()),**scratch})
    return {'numpy_version':np.__version__,'scope':'NumPy CPU math; no FlashAttention package or GPU run','rows':rows}


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(report(),indent=2,allow_nan=False))
