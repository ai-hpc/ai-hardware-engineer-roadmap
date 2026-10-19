import unittest
import math
try:
    import torch
    from tiny_moe import TopKMoE, dataset, train
except ImportError:
    torch=None


@unittest.skipIf(torch is None, 'optional: install requirements-torch.txt for real CPU MoE checks')
class MoETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_sparse_dispatch_forward_and_backward_match_dense_oracle(self):
        for factor in (None,.5,1.):
            model=TopKMoE(capacity_factor=factor).double()
            x=torch.randn(7,4,dtype=torch.float64,requires_grad=True)
            weights=torch.randn_like(x)
            a,reg,stats=model(x)
            params=(x,*model.parameters())
            ga=torch.autograd.grad((a*weights).sum()+reg,params,allow_unused=True)
            b,reg2,_=model(x,dense_reference=True)
            gb=torch.autograd.grad((b*weights).sum()+reg2,params,allow_unused=True)
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
            for pa,pb,p in zip(ga,gb,params):
                torch.testing.assert_close(torch.zeros_like(p) if pa is None else pa,torch.zeros_like(p) if pb is None else pb,atol=1e-11,rtol=1e-11)

    def test_router_gradient_finite_difference_away_from_ties(self):
        model=TopKMoE().double()
        x=torch.randn(5,4,dtype=torch.float64)
        def objective():
            y,reg,_=model(x)
            return y.square().sum()+reg
        objective().backward()
        analytical=model.router.weight.grad[0,0].item()
        original=model.router.weight[0,0].item();delta=1e-6
        with torch.no_grad():
            model.router.weight[0,0]=original+delta;plus=objective().item()
            model.router.weight[0,0]=original-delta;minus=objective().item()
            model.router.weight[0,0]=original
        self.assertAlmostEqual(analytical,(plus-minus)/(2*delta),places=7)

    def test_top_one_renormalization_removes_task_router_gradient(self):
        model=TopKMoE(top_k=1).double()
        x=torch.randn(5,4,dtype=torch.float64)
        y,_,_=model(x,aux_weight=0,z_weight=0)
        y.square().sum().backward()
        self.assertLess(model.router.weight.grad.abs().max().item(),1e-12)

    def test_capacity_zero_coverage_is_reported(self):
        model=TopKMoE(capacity_factor=.5)
        with torch.no_grad():
            model.router.weight.zero_();model.router.weight[:,0]=torch.tensor([4.,3.,2.,1.])
        y,_,stats=model(torch.ones(4,4))
        self.assertEqual(stats['capacity'],1)
        self.assertEqual(stats['dropped_assignment_fraction'],.75)
        self.assertEqual(stats['tokens_with_no_expert_fraction'],.75)
        self.assertTrue(torch.equal(y[1:],torch.zeros_like(y[1:])))

    def test_capacity_admits_highest_weights_without_rescaling_survivors(self):
        model = TopKMoE(hidden=1, width=1, experts=2, top_k=2,
                        capacity_factor=.5).double()
        with torch.no_grad():
            model.router.weight.copy_(torch.tensor([[1.], [-1.]], dtype=torch.float64))
            for i, network in enumerate(model.experts):
                for parameter in network.parameters(): parameter.zero_()
                network[2].bias.fill_(10. * (i + 1))
        y, _, stats = model(torch.tensor([[-2.], [0.], [2.]], dtype=torch.float64))
        high = 1. / (1. + math.exp(-4.))
        expected = torch.tensor([[20. * high], [15.], [10. * high]], dtype=torch.float64)
        torch.testing.assert_close(y, expected, rtol=1e-14, atol=1e-14)
        self.assertEqual(stats['capacity'], 2)
        self.assertAlmostEqual(stats['dropped_assignment_fraction'], 1/3, places=7)
        self.assertAlmostEqual(stats['tokens_with_any_drop_fraction'], 2/3, places=7)
        self.assertEqual(stats['tokens_with_no_expert_fraction'], 0.)

    def test_invalid_dimensions_coefficients_and_nonfinite_logits(self):
        for kwargs in ({'hidden':True}, {'width':2.5}, {'experts':0}, {'top_k':5}):
            with self.assertRaises(ValueError): TopKMoE(**kwargs)
        model = TopKMoE()
        for kwargs in ({'aux_weight':-1}, {'aux_weight':float('nan')},
                       {'z_weight':float('inf')}, {'z_weight':True}):
            with self.assertRaises(ValueError): model(torch.ones(2,4), **kwargs)
        for x in ([1.,2.], torch.ones(2,4,dtype=torch.int64)):
            with self.assertRaises(ValueError): model(x)
        with torch.no_grad(): model.router.weight.fill_(torch.finfo(torch.float32).max)
        with self.assertRaises(ValueError): model(torch.full((2,4),2.))
        scalar = TopKMoE(hidden=1, width=1, experts=1, top_k=1)
        with torch.no_grad(): scalar.router.weight.fill_(1e30)
        with self.assertRaisesRegex(ValueError, "regularizer"):
            scalar(torch.ones(1,1))
        # Disabling z-loss avoids evaluating a squared logit that would overflow.
        _, regularizer, _ = scalar(torch.ones(1,1), z_weight=0)
        self.assertTrue(torch.isfinite(regularizer))
        with torch.no_grad(): scalar.experts[0][2].bias.fill_(float('inf'))
        with self.assertRaisesRegex(ValueError, "expert outputs"):
            scalar(torch.ones(1,1), z_weight=0)
        with self.assertRaises(ValueError): train(.01, steps=0)
        with self.assertRaises(ValueError): dataset(41, 0)

    def test_real_training_reduces_held_out_loss(self):
        _,row=train(.01,steps=80)
        self.assertLess(row['final_validation_mse'],.2*row['initial_validation_mse'])

    def test_split_and_invalid_input(self):
        a,_=dataset(41,8);b,_=dataset(42,8)
        self.assertFalse(torch.equal(a,b))
        for x in (torch.empty(0,4),torch.ones(2,3),torch.full((2,4),float('nan'))):
            with self.assertRaises(ValueError): TopKMoE()(x)


if __name__=='__main__': unittest.main()
