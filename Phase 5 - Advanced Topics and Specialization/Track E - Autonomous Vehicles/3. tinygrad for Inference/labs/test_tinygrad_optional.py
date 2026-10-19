"""Actual CPU framework checks; explicit skip when pinned optional runtime is absent."""
import importlib.metadata
from pathlib import Path
import shutil
import sys
import unittest


class TinygradExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: installed=importlib.metadata.version('tinygrad')
        except importlib.metadata.PackageNotFoundError:
            raise unittest.SkipTest('optional tinygrad 0.14.0 environment is not installed')
        if installed!='0.14.0': raise unittest.SkipTest('requires the pinned tinygrad 0.14.0 API')
        if shutil.which('clang') is None: raise unittest.SkipTest('optional CPU compiler Clang is unavailable')
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'projects'))
        import course
        cls.course=course

    def test_scalar_and_nonsquare_known_answers(self):
        self.assertEqual(self.course.intro()['output'],[0.,1.,5.])
        self.assertEqual(self.course.operations()['manual_matmul'],[[2.,11.],[-3.,6.]])

    def test_gradient_and_analytic_optimizer_update(self):
        result=self.course.autograd()
        self.assertLess(result['finite_difference_max_error'],2e-4)
        self.assertAlmostEqual(result['sgd_weight'][0],2*(1-.8**20),places=6)

    def test_synthetic_training_and_checkpoint_roundtrip(self):
        r=self.course.classifier(steps=20)
        self.assertLess(r['final_training_loss'],r['first_training_loss'])
        self.assertEqual(r['test_accuracy'],1.)
        self.assertEqual(r['save_reload_max_error'],0.)

    def test_changing_inputs_reach_jit_output(self):
        r=self.course.compiler()
        self.assertEqual(r['changing_input_jit_outputs'][-1],[7.,1.,0.])

    def test_masked_attention_and_correct_swiglu(self):
        r=self.course.custom_ops()
        self.assertEqual(r['causal_attention'][0],[2.,3.])
        self.assertEqual(r['swiglu'][1],0.)

    def test_model_output_contract(self):
        self.assertTrue(self.course.model_contract()['wrong_units_rejected'])


if __name__=='__main__':unittest.main()
