"""The simulator boundary's configuration/units checks need no physics package."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mujoco_bridge as bridge
import policy_lab


class BridgeContractTests(unittest.TestCase):
    def test_invalid_dynamics_and_fractional_periods(self):
        for field, value in [("mass_kg", 0), ("assumed_mass_kg", True), ("force_limit_n", float("inf")),
                             ("damping_n_s_m", -1), ("physics_dt_s", .03),
                             ("duration_s", .15), ("physics_dt_s", .00000001),
                             ("physics_dt_s", 1e-320)]:
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                replace(bridge.Dynamics(), **{field:value})

    def test_acceleration_and_force_are_separately_bounded(self):
        cfg=bridge.Dynamics(mass_kg=3, assumed_mass_kg=3, force_limit_n=.5)
        self.assertEqual(bridge.control_force(2,cfg),(.5, True, True))
        self.assertEqual(bridge.control_force(-.1,cfg),(-.30000000000000004, False, False))
        with self.assertRaises(ValueError):bridge.control_force(float("nan"),cfg)

    def test_saved_preprocessing_identity_and_rejection(self):
        report={"script_sha256":hashlib.sha256(Path(policy_lab.__file__).read_bytes()).hexdigest(),
                "weights_normalized_inputs":[2.,-1.,.5],"scaler_training_mean":[1.,2.],
                "scaler_training_std":[2.,4.]}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"policy.json";path.write_text(json.dumps(report))
            self.assertEqual(bridge.load_policy(path)(3.,6.),1.5)
            for key,value in [("script_sha256","changed"),("weights_normalized_inputs",[2.,1.]),
                              ("scaler_training_std",[0.,4.]),("scaler_training_mean",[True,2.])]:
                bad=dict(report);bad[key]=value;path.write_text(json.dumps(bad))
                with self.subTest(key=key),self.assertRaises(ValueError):bridge.load_policy(path)
            for invalid in ([], None, 1, "not an object"):
                path.write_text(json.dumps(invalid))
                with self.subTest(invalid=invalid),self.assertRaises(ValueError):bridge.load_policy(path)

    def test_independent_semiimplicit_step(self):
        trajectory=bridge.analytical_trace(lambda e,v:.5,(1.,-.2),dt=.1,steps=1)
        self.assertAlmostEqual(trajectory[1][0],.985)
        self.assertAlmostEqual(trajectory[1][1],-.15)

if __name__ == "__main__":unittest.main()
