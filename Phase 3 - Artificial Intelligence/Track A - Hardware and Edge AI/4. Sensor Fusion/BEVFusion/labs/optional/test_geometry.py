import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from bev_geometry import lift_pixels, bev_sum


class TestGeometry(unittest.TestCase):
    def test_depth_and_pixel_augmentation(self):
        identity = torch.eye(3,dtype=torch.float64)[None,None]
        k = torch.tensor([[100.,0,50.],[0,100.,40.],[0,0,1.]],dtype=torch.float64)[None,None]
        # Original pixel (60,60), augmented by 2x scale then crop (10,20) => (110,100).
        a = torch.tensor([[2.,0,-10.],[0,2.,-20.],[0,0,1.]],dtype=torch.float64)[None,None]
        t = torch.tensor([[[1.,0.,0.]]],dtype=torch.float64)
        pixels = torch.tensor([[[[110.,100.],[110.,100.]]]],dtype=torch.float64)
        result = lift_pixels(pixels,torch.tensor([[[2.,4.]]]),k,identity,t,a)
        torch.testing.assert_close(result,torch.tensor([[[[1.2,.4,2.],[1.4,.8,4.]]]],dtype=torch.float64))

    def test_batch_isolation_floor_boundaries_sum_and_gradient(self):
        points = torch.tensor([[[[.1,.1,.5],[.2,.2,.5],[-.01,.1,.5],[2.,.1,.5]]],
                               [[[.1,.1,.5],[.2,.2,.5],[-.01,.1,.5],[2.,.1,.5]]]])
        features=torch.tensor([[[[1.],[2.],[100.],[100.]]],[[[10.],[20.],[100.],[100.]]]],requires_grad=True)
        result=bev_sum(points,features,(0,2,1),(0,2,1),(0,1))
        self.assertEqual(tuple(result.shape),(2,1,2,2))
        self.assertEqual(result[0,0,0,0].item(),3)
        self.assertEqual(result[1,0,0,0].item(),30)
        result.sum().backward()
        torch.testing.assert_close(features.grad,torch.tensor([[[[1.],[1.],[0.],[0.]]],[[[1.],[1.],[0.],[0.]]]]))

    def test_invalid_grid_rejected(self):
        with self.assertRaises(ValueError):
            bev_sum(torch.zeros(1,1,1,3),torch.ones(1,1,1,1),(0,2,0),(0,2,1),(0,1))


if __name__=='__main__':unittest.main()
