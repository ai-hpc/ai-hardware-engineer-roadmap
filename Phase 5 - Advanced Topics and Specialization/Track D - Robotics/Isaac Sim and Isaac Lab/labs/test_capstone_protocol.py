import copy
import unittest
from capstone_protocol import (CommandGate, Rejected, canonical, command_for,
                               observation, policy_observation, strict_loads)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.gate = CommandGate([-1]*7, [1]*7)
        self.gate.reset(0, [0]*7)

    def issue(self, now=100):
        return self.gate.issue([0]*9, [0]*9, [.5, 0, .025], [.6, 0, .025], now, 20)

    def test_applied_once_and_last_action_owned_by_simulator(self):
        obs = self.issue()
        cmd = command_for(obs, [.5]*7)
        result = self.gate.accept(cmd, 115)
        self.assertEqual(result.target, (.025,)*7)
        self.assertEqual(result.age_ns, 15)
        with self.assertRaises(Rejected): self.gate.accept(cmd, 116)
        self.assertEqual(self.issue(120)["last_action"], [.5]*7)

    def test_expiry_hold_does_not_integrate_or_advance_policy_state(self):
        obs = self.issue()
        with self.assertRaises(Rejected): self.gate.accept(command_for(obs, [1]*7), 121)
        self.gate.expire()
        self.assertEqual(self.gate.target, (0,)*7)
        self.assertEqual(self.gate.last_action, (0,)*7)

    def test_future_timestamp_and_boundary(self):
        cmd = command_for(self.issue(), [0]*7)
        with self.assertRaises(Rejected): self.gate.accept(cmd, 99)
        self.gate.accept(cmd, 120)

    def test_old_episode_cannot_act_after_reset(self):
        cmd = command_for(self.issue(), [1]*7)
        self.gate.reset(1, [.2]*7)
        with self.assertRaises(Rejected): self.gate.accept(cmd, 110)
        self.assertEqual(self.gate.target, (.2,)*7)
        self.assertEqual(self.gate.last_action, (0,)*7)

    def test_new_observation_supersedes_old_sequence(self):
        old = command_for(self.issue(), [1]*7)
        self.issue(110)
        with self.assertRaises(Rejected): self.gate.accept(old, 115)

    def test_atomic_observation_hash_detects_mixed_sensor_sample(self):
        obs = self.issue()
        obs["cube"][0] += .1
        with self.assertRaises(Rejected): self.gate.accept(command_for(obs, [0]*7), 110)

    def test_invalid_numeric_payload_is_transactional(self):
        cmd = command_for(self.issue(), [0]*7)
        for bad in (float("nan"), float("inf"), True, 10**400, 1.001):
            variant = copy.deepcopy(cmd); variant["action"][3] = bad
            with self.assertRaises(Rejected): self.gate.accept(variant, 110)
            self.assertEqual(self.gate.target, (0,)*7)
        self.gate.accept(cmd, 110)

    def test_wrong_order_schema_and_bool_identifiers(self):
        cmd = command_for(self.issue(), [0]*7)
        variants = []
        x=copy.deepcopy(cmd); x["names"].reverse(); variants.append(x)
        x=copy.deepcopy(cmd); x["episode"]=False; variants.append(x)
        x=copy.deepcopy(cmd); x["names"]=None; variants.append(x)
        x=copy.deepcopy(cmd); x["extra"]=0; variants.append(x)
        for value in variants:
            with self.assertRaises(Rejected): self.gate.accept(value, 110)

    def test_target_limits_and_observation_layout(self):
        self.gate.reset(1, [.99]*7)
        obs=self.issue(); values=policy_observation(obs, [.1]*9)
        self.assertEqual(len(values), 31)
        self.assertEqual(values[:9], [-.1]*9)
        self.assertEqual(self.gate.accept(command_for(obs, [1]*7), 110).target, (1,)*7)

    def test_strict_json_and_roundtrip(self):
        obs=self.issue()
        self.assertEqual(observation(strict_loads(canonical(obs))), obs)
        for bad in ('{"x":1,"x":2}', '{"x":NaN}', '[1]', 'null', '{} trailing', '{"x":'+'9'*5000+'}'):
            with self.assertRaises(Rejected): strict_loads(bad)

    def test_invalid_issue_and_reset_do_not_mutate(self):
        self.issue()
        with self.assertRaises(Rejected): self.gate.issue([0]*8, [0]*9, [0]*3, [0]*3, 110, 20)
        self.assertEqual(self.gate.sequence, 0)
        with self.assertRaises(Rejected): self.gate.reset(0, [0]*7)
        with self.assertRaises(Rejected): self.gate.reset(1, [2]*7)
        self.assertEqual(self.gate.episode, 0)


if __name__ == "__main__": unittest.main()
