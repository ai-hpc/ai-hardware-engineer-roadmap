from pathlib import Path
import sys
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attention_reference import dense,online,backward
from attention_contract import causal_mask

torch.set_num_threads(1)


def math_sdpa(q,k,v,**kw):
    with sdpa_kernel(SDPBackend.MATH):return F.scaled_dot_product_attention(q,k,v,**kw)


class TorchAttention(unittest.TestCase):
    def inputs(self,l=7,s=11,dtype=torch.float64):
        g=torch.Generator().manual_seed(17)
        return (torch.randn(1,1,l,8,generator=g,dtype=dtype),
                torch.randn(1,1,s,8,generator=g,dtype=dtype),
                torch.randn(1,1,s,5,generator=g,dtype=dtype))

    def test_actual_math_forward_and_gradients_match_numpy(self):
        q,k,v=[x.requires_grad_() for x in self.inputs()]
        mask=torch.tensor(causal_mask(7,11,'lower_right'))
        out=math_sdpa(q,k,v,attn_mask=mask,dropout_p=0)
        grad=torch.randn(out.shape,generator=torch.Generator().manual_seed(5),dtype=torch.float64)
        out.backward(grad)
        arrays=[x.detach().numpy()[0,0] for x in (q,k,v)]
        reference=dense(*arrays,mask.numpy())[0]
        np.testing.assert_allclose(out.detach().numpy()[0,0],reference,atol=1e-12,rtol=1e-12)
        for actual,expected in zip([x.grad.numpy()[0,0] for x in (q,k,v)],backward(*arrays,grad.numpy()[0,0],mask.numpy())):
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)

    def test_sdpa_is_causal_is_upper_left_for_rectangular_inputs(self):
        q,k,v=self.inputs(1,9)
        upper=math_sdpa(q,k,v,is_causal=True)
        lower=math_sdpa(q,k,v,attn_mask=torch.tensor(causal_mask(1,9,'lower_right')))
        torch.testing.assert_close(upper,v[:,:,:1])
        self.assertFalse(torch.allclose(upper,lower))
        torch.testing.assert_close(lower,math_sdpa(q,k,v))

    def test_fully_masked_rows_are_zero_and_gradients_finite(self):
        q,k,v=[x.requires_grad_() for x in self.inputs(9,2)]
        out=math_sdpa(q,k,v,attn_mask=torch.tensor(causal_mask(9,2,'lower_right')))
        self.assertTrue(torch.equal(out[:,:,:7],torch.zeros_like(out[:,:,:7])))
        out.sum().backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in (q,k,v)))
        self.assertTrue(torch.equal(q.grad[:,:,:7],torch.zeros_like(q.grad[:,:,:7])))

    def test_gqa_matches_explicit_head_repetition(self):
        g=torch.Generator().manual_seed(8)
        q=torch.randn(1,4,3,8,generator=g,dtype=torch.float64)
        k=torch.randn(1,2,5,8,generator=g,dtype=torch.float64)
        v=torch.randn(1,2,5,6,generator=g,dtype=torch.float64)
        actual=math_sdpa(q,k,v,enable_gqa=True)
        expected=math_sdpa(q,k.repeat_interleave(2,-3),v.repeat_interleave(2,-3))
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)

    def test_noncontiguous_last_dimension_and_zero_scale(self):
        q,k,v=self.inputs()
        q= torch.stack([q,torch.zeros_like(q)],dim=-1).flatten(-2)[...,::2]
        self.assertEqual(q.stride(-1),2)
        out=math_sdpa(q,k,v,scale=0)
        torch.testing.assert_close(out,v.mean(dim=-2,keepdim=True).expand_as(out),atol=1e-12,rtol=1e-12)

    def test_bf16_error_separates_input_quantization_and_computation(self):
        q,k,v=self.inputs(dtype=torch.float32)
        qb,kb,vb=[x.to(torch.bfloat16) for x in (q,k,v)]
        actual=math_sdpa(qb,kb,vb).float().numpy()[0,0]
        quantized_reference=dense(*[x.float().numpy()[0,0] for x in (qb,kb,vb)])[0]
        original_reference=dense(*[x.numpy()[0,0] for x in (q,k,v)])[0]
        self.assertGreater(float(abs(original_reference-quantized_reference).max()),0)
        np.testing.assert_allclose(actual,quantized_reference,atol=.005,rtol=.005)

    def test_dropout_is_not_disabled_by_inference_mode(self):
        q,k,v=self.inputs()
        with torch.inference_mode():
            torch.manual_seed(2);drop=math_sdpa(q,k,v,dropout_p=.5)
            no_drop=math_sdpa(q,k,v,dropout_p=0)
        self.assertFalse(torch.equal(drop,no_drop))


if __name__=='__main__':unittest.main()
