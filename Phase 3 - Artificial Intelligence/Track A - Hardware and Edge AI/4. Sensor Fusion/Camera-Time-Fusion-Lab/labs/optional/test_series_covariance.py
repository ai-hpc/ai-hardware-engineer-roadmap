"""Check the earlier scalar-arithmetic class against an independent matrix filter."""
import ast
from pathlib import Path
import unittest
import numpy as np


class TestSeriesCovariance(unittest.TestCase):
    def test_full_covariance_matches_matrix_reference(self):
        source = Path(__file__).resolve().parents[3]/'kalman-filter'/'kalman_1d.py'
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'KalmanFilter1D')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        kf = namespace['KalmanFilter1D'](P_x0=1., P_v0=2., P_xv0=.5, Q_x=.01, Q_v=.02, R=.3)
        x, p = np.array([0.,1.]), np.array([[1.,.5],[.5,2.]])
        for step in range(12):
            if step:
                dt = .2 if step % 2 else .4
                f = np.array([[1.,dt],[0.,1.]])
                x, p = f@x, f@p@f.T + np.diag([.01,.02])
                kf.predict(dt)
            z = .3 * step + .1 * (-1)**step
            gain = p[:,0] / (p[0,0]+.3)
            x = x + gain * (z-x[0])
            a = np.eye(2)-np.outer(gain,[1.,0.])
            p = a@p@a.T + np.outer(gain,gain)*.3
            kf.update(z)
            np.testing.assert_allclose([kf.x,kf.v],x,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose([[kf.P_xx,kf.P_xv],[kf.P_xv,kf.P_vv]],p,rtol=1e-12,atol=1e-12)




class TestCheckedSeriesExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        cls.modules = {}
        root = Path(__file__).resolve().parents[3] / 'kalman-filter'
        for name in ('kalman_1d', 'kalman_2d', 'kalman_6d'):
            spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.modules[name] = module

    def make(self, **overrides):
        values = dict(F=[[1.,1.],[0.,1.]], H=[[1.,0.]],
                      Q=np.zeros((2,2)), R=[[25.]], x0=[[0.],[25.]],
                      P0=np.diag([100.,25.]))
        values.update(overrides)
        return self.modules['kalman_6d'].KalmanFilter(**values)

    def test_train_example_full_state_and_covariance(self):
        kf = self.make()
        kf.predict()
        np.testing.assert_allclose(kf.P, [[125.,25.],[25.,25.]])
        kf.update([23.])
        np.testing.assert_allclose(kf.x[:,0], [70/3,74/3])
        np.testing.assert_allclose(kf.P, [[125/6,25/6],[25/6,125/6]])
        kf.predict()
        self.assertAlmostEqual(kf.x[0,0],48.)

    def test_flat_and_column_observations_are_equivalent(self):
        for name, create in [('general', lambda: self.make(H=np.eye(2),R=np.eye(2))),
                             ('2d', self.modules['kalman_2d'].KalmanFilter2DTracking)]:
            with self.subTest(name=name):
                flat, column = create(), create()
                flat.predict(); column.predict()
                flat.update([3.,4.]); column.update([[3.],[4.]])
                np.testing.assert_array_equal(flat.x,column.x)
                np.testing.assert_array_equal(flat.P,column.P)
                self.assertEqual(flat.x.shape,(flat.P.shape[0],1))
                for bad in ([[3.,4.]], [3.], [3.,float('nan')]):
                    old=flat.x.copy()
                    with self.assertRaises(ValueError): flat.update(bad)
                    np.testing.assert_array_equal(flat.x,old)

    def test_constructor_rejects_invalid_covariance_and_shapes(self):
        for kwargs in (dict(F=np.zeros((2,3))),dict(H=[[1.]]),dict(x0=[0.,25.]),
                       dict(P0=[[1.,2.],[2.,1.]]),dict(P0=[[1.,.2],[0.,1.]]),
                       dict(Q=[[-1.,0.],[0.,1.]]),dict(R=[[0.]]),dict(R=[[-1.]]),
                       dict(x0=[[0.],[float('inf')]])):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): self.make(**kwargs)
        # Semidefinite P and Q are valid; R is positive definite in this example.
        self.make(P0=np.zeros((2,2)))

    def test_constructor_copies_caller_arrays(self):
        state=np.array([[0.],[25.]])
        covariance=np.eye(2)
        kf=self.make(x0=state,P0=covariance)
        state[:]=99;covariance[:]=99
        np.testing.assert_array_equal(kf.x,[[0.],[25.]])
        np.testing.assert_array_equal(kf.P,np.eye(2))

    def test_2d_matches_general_filter_over_long_sequence(self):
        kf=self.modules['kalman_2d'].KalmanFilter2DTracking(.1)
        reference=self.modules['kalman_6d'].KalmanFilter(kf.F,kf.H,kf.Q,kf.R,kf.x,kf.P)
        rng=np.random.default_rng(2026)
        for i in range(500):
            z=np.array([.1*i,.2*i])+rng.normal(0,1,2)
            kf.predict();reference.predict()
            kf.update(z);reference.update(z)
            np.testing.assert_allclose(kf.x,reference.x,atol=1e-11)
            np.testing.assert_allclose(kf.P,reference.P,atol=1e-11)
            np.testing.assert_allclose(kf.P,kf.P.T,atol=1e-14)
            self.assertGreaterEqual(np.linalg.eigvalsh(kf.P).min(),0.)

    def test_scalar_rejects_invalid_values_without_changing_state(self):
        constructor=self.modules['kalman_1d'].KalmanFilter1D
        for kwargs in ({'R':0},{'Q_x':-1},{'P_xv0':2},{'v0':float('nan')}):
            with self.assertRaises(ValueError): constructor(**kwargs)
        kf=constructor()
        for dt in (0,-1,float('inf')):
            with self.assertRaises(ValueError): kf.predict(dt)
        for z in ([3.],float('nan')):
            with self.assertRaises(ValueError): kf.update(z)
        self.assertEqual((kf.x,kf.v),(0.,1.))

    def test_gain_is_not_always_a_percentage_and_prediction_can_contract(self):
        kf=self.make(F=[[.5]],H=[[-.5]],Q=[[0.]],R=[[1.]],x0=[[0.]],P0=[[16.]])
        kf.predict()
        np.testing.assert_array_equal(kf.P,[[4.]])
        kf.update([1.])  # S=2, K=4*(-.5)/2=-1.
        np.testing.assert_allclose(kf.x,[[-1.]])
        np.testing.assert_allclose(kf.P,[[2.]])


if __name__ == '__main__':
    unittest.main()
