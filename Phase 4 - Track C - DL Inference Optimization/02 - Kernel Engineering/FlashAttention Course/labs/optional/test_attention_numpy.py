from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attention_reference import dense,online,backward,rope_interleaved
from attention_contract import causal_mask


class NumpyAttention(unittest.TestCase):
    def inputs(self,l=7,s=11):
        r=np.random.default_rng(17)
        return r.normal(size=(l,8)),r.normal(size=(s,8)),r.normal(size=(s,5))

    def test_tails_and_masked_tiles_match_dense(self):
        for l,s in [(7,11),(1,9),(9,2)]:
            q,k,v=self.inputs(l,s);mask=np.array(causal_mask(l,s,'lower_right'))
            ref,lse,_=dense(q,k,v,mask)
            for br,bc in [(1,1),(3,4),(20,20)]:
                out,stats,meta=online(q,k,v,mask,row_tile=br,column_tile=bc)
                np.testing.assert_allclose(out,ref,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(stats,lse,atol=1e-12,rtol=1e-12)
                self.assertLessEqual(meta['largest_score_tile_elements'],min(br,l)*min(bc,s))
                self.assertTrue(np.isfinite(out).all())

    def test_large_max_change_requires_rescaling(self):
        q=np.array([[1.]]);k=np.array([[-1000.],[0.],[1000.]])
        v=np.array([[1.],[2.],[3.]])
        out,_,_=online(q,k,v,column_tile=1)
        np.testing.assert_allclose(out,[[3.]],atol=1e-12)
        self.assertNotAlmostEqual(float(out[0,0]),float(v.mean()))

    def test_empty_rows_and_all_masked_blocks_have_declared_result(self):
        q,k,v=self.inputs();mask=np.zeros((7,11),dtype=bool)
        out,lse,_=online(q,k,v,mask)
        self.assertTrue((out==0).all());self.assertTrue(np.isneginf(lse).all())
        dq,dk,dv=backward(q,k,v,np.ones_like(out),mask)
        self.assertTrue(all((g==0).all() for g in (dq,dk,dv)))

    def test_backward_finite_difference(self):
        q,k,v=self.inputs(3,4);mask=np.array(causal_mask(3,4,'lower_right'))
        grad=np.random.default_rng(2).normal(size=(3,5));analytic=backward(q,k,v,grad,mask)
        arrays=[q,k,v];epsilon=1e-6
        for which,position in [(0,(1,2)),(1,(2,3)),(2,(0,1))]:
            original=arrays[which][position]
            arrays[which][position]=original+epsilon;plus=(dense(*arrays,mask)[0]*grad).sum()
            arrays[which][position]=original-epsilon;minus=(dense(*arrays,mask)[0]*grad).sum()
            arrays[which][position]=original
            self.assertAlmostEqual((plus-minus)/(2*epsilon),analytic[which][position],places=8)

    def test_zero_scale_is_supported_and_not_replaced_by_default(self):
        q,k,v=self.inputs();out,_,_=online(q,k,v,scale=0)
        np.testing.assert_allclose(out,np.broadcast_to(v.mean(axis=0),out.shape),atol=1e-12)

    def test_incremental_rope_keeps_absolute_position(self):
        rng=np.random.default_rng(6);q=rng.normal(size=(6,8));k=rng.normal(size=(6,8));v=rng.normal(size=(6,3))
        rotated_q=rope_interleaved(q,np.arange(6));rotated_k=rope_interleaved(k,np.arange(6))
        full=dense(rotated_q,rotated_k,v,np.array(causal_mask(6,6,'upper_left')))[0][-1]
        cached=np.concatenate([rope_interleaved(k[:5],np.arange(5)),rope_interleaved(k[5:],[5])])
        decoded=dense(rope_interleaved(q[5:],[5]),cached,v)[0][0]
        np.testing.assert_allclose(decoded,full,atol=1e-12)
        wrong=dense(rope_interleaved(q[5:],[0]),cached,v)[0][0]
        self.assertGreater(float(abs(wrong-full).max()),.01)

    def test_invalid_mask_and_nonfinite_input_rejected(self):
        q,k,v=self.inputs()
        for mask in (np.ones((7,11)),np.ones((7,10),dtype=bool)):
            with self.assertRaises(ValueError):online(q,k,v,mask)
        q[0,0]=float('nan')
        with self.assertRaises(ValueError):dense(q,k,v)


if __name__=='__main__':unittest.main()
