"""Independent numerical checks for exact lesson snippets; no simulator or model download."""
import ast
import itertools
import math
from pathlib import Path
import re
import types
import unittest

try:
    import torch
    from torch import nn
    from torch.distributions import Normal
except ImportError:
    torch = None
try:
    import numpy as np
    from scipy.stats import binomtest, norm, trim_mean
except ImportError:
    np = None

COURSE = Path(__file__).resolve().parents[2]


def snippets(name, names, namespace):
    """Compile only named definitions from our lesson; exclude incomplete trainer top levels."""
    found = []
    for source in re.findall(r'```python\n(.*?)```', (COURSE / name).read_text(), re.S):
        module = ast.parse(source)
        found.extend(n for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
    if {n.name for n in found} != set(names):
        raise AssertionError('missing or renamed lesson definitions')
    exec(compile(ast.Module(body=found, type_ignores=[]), str(COURSE / name), 'exec'), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


@unittest.skipIf(np is None, 'optional: requires NumPy')
class CapstoneWeightNumerics(unittest.TestCase):
    def setUp(self):
        self.f = snippets('Lecture-14.md', ['group_advantages', 'regression_weights'], {'np': np})

    def test_group_filter_and_success_filter_are_different_objectives(self):
        outcomes = np.array([[0, 1], [1, 1], [0, 0]])
        advantages, fraction = self.f.group_advantages(outcomes)
        np.testing.assert_array_equal(advantages, [[-.5, .5], [0, 0], [0, 0]])
        self.assertAlmostEqual(fraction, 2 / 3)
        filtered = self.f.regression_weights(advantages, mode='filtered')
        np.testing.assert_array_equal(filtered, [[0, 1], [0, 0], [0, 0]])
        self.assertEqual(filtered.sum(), 1)
        self.assertEqual(outcomes.sum(), 3)  # All-success group still supplies two BC targets.
        np.testing.assert_array_equal(self.f.regression_weights(advantages)[1:], np.ones((2, 2)))

    def test_small_temperature_caps_weights_without_exp_overflow(self):
        with np.errstate(over='raise', invalid='raise'):
            weights = self.f.regression_weights(np.array([-.5, 0, .5]), beta=.001)
        np.testing.assert_array_equal(weights, [0, 1, 20])
        self.assertTrue(np.isfinite(weights).all())

    def test_invalid_groups_and_temperatures_do_not_become_training_weights(self):
        for outcomes in ([], [[1]], [[.2, 1]], [[float('nan'), 1]]):
            with self.subTest(outcomes=outcomes), self.assertRaises(ValueError):
                self.f.group_advantages(outcomes)
        for kwargs in ({'beta': 0}, {'beta': -1}, {'beta': float('nan')}, {'mode': 'typo'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.f.regression_weights(np.array([-.5, .5]), **kwargs)


@unittest.skipIf(torch is None, 'optional: requires PyTorch CPU')
class FlowNumerics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.f = snippets('Lecture-11.md', ['_sigma_vector', 'sample_chain', 'chain_logprob', 'group_advantages', 'grpo_loss'], {'torch': torch, 'math': math})
        self.obs = torch.zeros(1, 2, dtype=torch.float64)
        self.chain = torch.tensor([0., 1., 1.5], dtype=torch.float64).reshape(1, 3, 1, 1)
        self.sigma = torch.tensor([.5, 1.], dtype=torch.float64)
        self.theta = torch.tensor(2., dtype=torch.float64, requires_grad=True)
        self.policy = lambda x, t, o: self.theta.expand_as(x)

    def test_log_density_and_gradient_match_scalar_derivation(self):
        lp, mu = self.f.chain_logprob(self.policy, self.obs, self.chain, self.sigma)
        expected = torch.tensor([[-math.log(.5)-.5*math.log(2*math.pi), -.125-.5*math.log(2*math.pi)]], dtype=torch.float64)
        torch.testing.assert_close(lp, expected, atol=1e-13, rtol=0)
        torch.testing.assert_close(mu.flatten(), torch.tensor([1., 2.], dtype=torch.float64))
        lp.sum().backward()
        self.assertAlmostEqual(self.theta.grad.item(), -.25, places=13)

    def test_vectorization_matches_independent_normal_distribution(self):
        torch.manual_seed(31)
        obs = torch.randn(3, 2, dtype=torch.float64)
        _, chain = self.f.sample_chain(self.policy, obs, 2, 4, 3, [.2,.3,.4])
        lp, mu = self.f.chain_logprob(self.policy, obs, chain, [.2,.3,.4])
        sd = torch.tensor([.2,.3,.4], dtype=torch.float64)[None,:,None,None]
        oracle = Normal(mu, sd).log_prob(chain[:,1:]).sum((-1,-2))
        torch.testing.assert_close(lp, oracle, atol=1e-12, rtol=1e-12)
        self.assertFalse(chain.requires_grad)

    def test_fixed_rollout_inputs_are_detached(self):
        chain = self.chain.clone().requires_grad_()
        obs = self.obs.clone().requires_grad_()
        lp, _ = self.f.chain_logprob(self.policy, obs, chain, self.sigma)
        lp.sum().backward()
        self.assertIsNone(chain.grad)
        self.assertIsNone(obs.grad)
        self.assertAlmostEqual(self.theta.grad.item(), -.25, places=13)

    def test_gaussian_kl_and_ppo_gradient_are_separate(self):
        lp, mu = self.f.chain_logprob(self.policy, self.obs, self.chain, self.sigma)
        old = lp.detach().clone().requires_grad_()
        ref = mu.detach().clone().requires_grad_()
        loss, stats = self.f.grpo_loss(self.policy, self.obs, self.chain, old, ref, torch.ones(1,dtype=torch.float64), self.sigma, beta=0)
        loss.backward()
        self.assertAlmostEqual(self.theta.grad.item(), .125, places=13)
        self.assertIsNone(old.grad)
        self.assertIsNone(ref.grad)
        self.assertEqual(stats['kl'], 0.)
        _, shifted = self.f.grpo_loss(self.policy,self.obs,self.chain,old,ref-.2,torch.zeros(1,dtype=torch.float64),self.sigma)
        self.assertAlmostEqual(shifted['kl'], (.2**2/(2*.5**2)+.2**2/(2*1**2))/2, places=13)

    def test_sampling_zero_noise_is_allowed_but_density_is_not(self):
        action, _ = self.f.sample_chain(self.policy,self.obs,1,1,2,[0.,0.])
        self.assertTrue(torch.isfinite(action).all())
        for sigma in ([0.,1.],[-1.,1.],[float('nan'),1.],[1.]):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                self.f.chain_logprob(self.policy,self.obs,self.chain,sigma)
        with self.assertRaises(ValueError):
            self.f.sample_chain(self.policy,self.obs,1,1,0,[])
        with self.assertRaises(ValueError):
            self.f.chain_logprob(self.policy,self.obs,self.chain[:,0],self.sigma)

    def test_group_mean_bias_by_exact_bernoulli_enumeration(self):
        p = .3
        estimate = {'group': 0., 'loo': 0.}
        for a,b in itertools.product((0,1), repeat=2):
            probability = (p if a else 1-p)*(p if b else 1-p)
            rewards = torch.tensor([a,b],dtype=torch.float64)
            scores = torch.tensor([a-p,b-p],dtype=torch.float64)  # derivative of Bernoulli log p wrt logit
            for name,loo in [('group',False),('loo',True)]:
                adv,_ = self.f.group_advantages(rewards,2,normalize_std=False,leave_one_out=loo)
                estimate[name] += probability * (adv*scores).mean().item()
        self.assertAlmostEqual(estimate['loo'], p*(1-p), places=13)
        self.assertAlmostEqual(estimate['group'], .5*p*(1-p), places=13)

    def test_groups_reject_partial_singleton_and_nonfinite_rewards(self):
        rewards=torch.tensor([0.,1.,1.,1.],dtype=torch.float64)
        adv,keep=self.f.group_advantages(rewards,2)
        torch.testing.assert_close(adv,torch.tensor([-1/(1+2e-6),1/(1+2e-6),0.,0.],dtype=torch.float64))
        self.assertEqual(keep.tolist(),[True,True,False,False])
        for bad,g in [(rewards,1),(rewards[:3],2),(rewards[:0],2),(torch.tensor([float('nan'),1.]),2)]:
            with self.assertRaises(ValueError):self.f.group_advantages(bad,g)


@unittest.skipIf(torch is None, 'optional: requires PyTorch CPU')
class PPONumerics(unittest.TestCase):
    def test_singleton_last_minibatch_does_not_make_std_nan(self):
        f=snippets('Lecture-06.md',['gaussian_kl','ppo_update'],{'torch':torch,'nn':nn,'Normal':Normal})
        actor=nn.Linear(2,1).double();critic=nn.Linear(2,1).double()
        std=nn.Parameter(torch.zeros(1,dtype=torch.float64))
        opt=torch.optim.SGD([*actor.parameters(),*critic.parameters(),std],lr=1e-5)
        obs=torch.arange(10,dtype=torch.float64).reshape(5,2)/10
        with torch.no_grad():
            action=actor(obs)+.1
            old=Normal(actor(obs),std.exp()).log_prob(action).sum(-1)
        batch=dict(obs=obs,act=action,logp_old=old,adv=torch.arange(5,dtype=torch.float64),ret=torch.ones(5,dtype=torch.float64))
        args=types.SimpleNamespace(minibatches=2,epochs=1,clip=.2,target_kl=1.,vf_coef=.5,ent_coef=0.,kl_ref=0.,max_grad_norm=1.)
        result=f.ppo_update(actor,critic,std,opt,batch,args)
        self.assertEqual(result['updates'],3)
        self.assertTrue(all(torch.isfinite(p).all() for p in [*actor.parameters(),*critic.parameters(),std]))
        args.minibatches=6
        with self.assertRaises(ValueError):f.ppo_update(actor,critic,std,opt,batch,args)


@unittest.skipIf(np is None, 'optional: requires NumPy and SciPy')
class StatisticalNumerics(unittest.TestCase):
    def setUp(self):
        self.f=snippets('Lecture-13.md',['wilson','episodes_needed','episodes_to_detect','iqm','stratified_bootstrap','mcnemar_paired'],{'np':np,'sqrt':math.sqrt,'norm':norm,'trim_mean':trim_mean,'binomtest':binomtest})

    def test_wilson_endpoint_and_invalid_counts(self):
        lo,hi=self.f.wilson(50,50)
        z=norm.ppf(.975)
        self.assertAlmostEqual(lo,50/(50+z*z),places=13)
        self.assertAlmostEqual(hi,1.,places=13)
        for k,n in [(0,0),(51,50),(-1,50)]:
            with self.assertRaises(ValueError):self.f.wilson(k,n)

    def test_exact_discordant_pairs_are_not_marginal_ci_overlap(self):
        a=np.ones(6,dtype=bool);b=np.zeros(6,dtype=bool)
        self.assertAlmostEqual(self.f.mcnemar_paired(a,b),2*(.5**6),places=13)
        self.assertEqual(self.f.mcnemar_paired(a,a),1.)
        for bad in (b[:2],np.zeros(6,dtype=int)):
            with self.assertRaises(ValueError):self.f.mcnemar_paired(a,bad)

    def test_shared_run_bootstrap_preserves_cross_task_dependence(self):
        scores=np.array([[0.,1.],[1.,0.]])
        _,shared=self.f.stratified_bootstrap(scores,stat=np.mean,reps=2000,shared_runs=True)
        _,independent=self.f.stratified_bootstrap(scores,stat=np.mean,reps=2000,shared_runs=False)
        self.assertEqual(shared,(.5,.5))
        self.assertEqual(independent,(0.,1.))

    def test_sample_size_inputs_and_known_half_width_count(self):
        self.assertEqual(self.f.episodes_needed(.5,.05),385)
        for p,h in [(0.,.05),(1.,.05),(.5,0.)]:
            with self.assertRaises(ValueError):self.f.episodes_needed(p,h)
        with self.assertRaises(ValueError):self.f.episodes_to_detect(.5,.5)


if __name__=='__main__':unittest.main()
