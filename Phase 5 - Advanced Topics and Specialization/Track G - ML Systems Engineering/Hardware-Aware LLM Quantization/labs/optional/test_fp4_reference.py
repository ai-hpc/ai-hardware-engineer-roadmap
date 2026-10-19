import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fp4_reference import pack_fp4, unpack_fp4, select_clipping
from quant_contracts import fp4_payload_bytes


class FP4Reference(unittest.TestCase):
    def test_exact_grid_and_signed_nibbles(self):
        grid=torch.tensor([[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]])
        a=pack_fp4(grid);restored=unpack_fp4(a)
        torch.testing.assert_close(restored,grid,atol=1e-6,rtol=0)
        self.assertEqual(a.codes.tolist(),[16,50,84,118,152,186,220,254])

    def test_halfway_ties_even(self):
        # Largest magnitude 6 makes effective block scale exactly 1.
        w=torch.tensor([[.25,.75,1.25,1.75,2.5,3.5,5.,6.]+[0.]*8])
        out=unpack_fp4(pack_fp4(w))
        torch.testing.assert_close(out[0,:7],torch.tensor([0.,1.,1.,2.,2.,4.,4.]),atol=1e-6,rtol=0)

    def test_zero_tail_and_payload(self):
        for shape in [(2,3),(1,17),(3,32)]:
            a=pack_fp4(torch.zeros(shape));self.assertEqual(a.payload_bytes,fp4_payload_bytes(*shape))
            torch.testing.assert_close(unpack_fp4(a),torch.zeros(shape))
        a=pack_fp4(torch.arange(17.).reshape(1,17));self.assertEqual(unpack_fp4(a).shape,(1,17))
        self.assertAlmostEqual(float(unpack_fp4(a)[0,-1]),16.,places=5)

    def test_actual_e4m3_rounding(self):
        values=torch.tensor([3.5,3.75,3.95,4.25]).to(torch.float8_e4m3fn).float()
        self.assertEqual(values.tolist(),[3.5,3.75,4.,4.])

    def test_calibration_selects_no_worse_searched_loss(self):
        torch.manual_seed(3);w=torch.randn(4,32);x=torch.randn(80,32);x[:,0]*=20
        for inputs in (None,x):
            _,report=select_clipping(w,inputs)
            absmax=next(c['loss'] for c in report['candidates'] if c['ratio']==1.)
            self.assertLessEqual(report['calibration_loss'],absmax+1e-12)

    def test_clipping_changes_reconstruction(self):
        w=torch.ones(1,16);w[0,0]=23.7
        a=unpack_fp4(pack_fp4(w));b=unpack_fp4(pack_fp4(w,.5))
        self.assertGreater(float((a-b).abs().max()),10)

    def test_reject_invalid_input_and_corrupt_scale(self):
        for w in (torch.tensor([[float('nan')]]),torch.empty(0,1),torch.ones(2),torch.tensor([[1e100]],dtype=torch.float64)):
            with self.assertRaises(ValueError):pack_fp4(w)
        a=pack_fp4(torch.ones(2,3));a.scales=torch.zeros_like(a.scales)
        with self.assertRaises(ValueError):unpack_fp4(a)
        with self.assertRaises(ValueError):select_clipping(torch.ones(1,3),torch.ones(2,4))

if __name__=='__main__':unittest.main()
