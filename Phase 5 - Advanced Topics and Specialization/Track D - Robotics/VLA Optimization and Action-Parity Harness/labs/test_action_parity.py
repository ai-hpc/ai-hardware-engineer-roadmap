from dataclasses import replace
import math
import unittest

import action_parity as lab


def tape(actions):
    return [{'episode':0,'frame':i,'observation_sha256':lab.digest({'frame':i}),
             'action':value} for i,value in enumerate(actions)]


class ReplayTests(unittest.TestCase):
    def test_hand_calculated_mse_and_sign_cancellation(self):
        row=lab.replay_metrics(tape([0.0,0.0]),tape([0.2,-0.2]))
        self.assertAlmostEqual(row['raw_mse_m2_s4'],0.04)
        self.assertEqual(row['signed_mean_error_m_s2'],0.0)
        self.assertEqual(row['raw_max_error_m_s2'],0.2)

    def test_clipping_can_hide_a_wrong_raw_policy(self):
        row=lab.replay_metrics(tape([2.0]),tape([3.0]))
        self.assertEqual(row['clipped_mse_m2_s4'],0.0)
        self.assertEqual(row['raw_mse_m2_s4'],1.0)

    def test_declared_limit_controls_the_comparison(self):
        wider=replace(lab.CONTRACT,limit=4.0)
        row=lab.replay_metrics(tape([2.0]),tape([3.0]),wider,wider)
        self.assertEqual(row['clipped_mse_m2_s4'],1.0)
        invalid=replace(lab.CONTRACT,limit=math.nan)
        with self.assertRaises(ValueError):
            lab.replay_metrics(tape([0]),tape([0]),invalid,invalid)

    def test_empty_missing_extra_and_duplicate_rejected(self):
        for left,right in [([],[]),(tape([0]),tape([])),(tape([0]),tape([0,1])),
                           (tape([0]),tape([0])+tape([0]))]:
            with self.subTest(left=left,right=right), self.assertRaises(ValueError):
                lab.replay_metrics(left,right)

    def test_identity_and_contract_are_checked_before_error(self):
        candidate=tape([0]);candidate[0]['observation_sha256']='a'*64
        with self.assertRaises(ValueError):
            lab.replay_metrics(tape([0]),candidate)
        for changed in (replace(lab.CONTRACT,action_kind='velocity'),
                        replace(lab.CONTRACT,action_unit='cm/s^2'),
                        replace(lab.CONTRACT,timestamp_kind='receive_ms'),
                        replace(lab.CONTRACT,frame='opposite_positive_axis')):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                lab.replay_metrics(tape([0]),tape([0]),candidate_contract=changed)

    def test_nonfinite_action_and_difference_overflow_rejected(self):
        for value in (math.nan,math.inf,-math.inf,True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                lab.replay_metrics(tape([0]),tape([value]))
        with self.assertRaises(ValueError):
            lab.replay_metrics(tape([-1e308]),tape([1e308]))

    def test_folded_normalization_known_answer(self):
        folded=lab.fold_normalization([2,-3,1],([10,20],[2,5]))
        self.assertEqual(folded,[1,-0.6,3])
        self.assertAlmostEqual(folded[0]*12+folded[1]*25+folded[2],0)
        with self.assertRaises(ValueError):
            lab.fold_normalization([2,-3,1],([10,20],[2,0]))

    def test_quantizer_preserves_scale_and_integer_bounds(self):
        restored,info=lab.quantized_weights([1,-0.5,0])
        self.assertEqual(info['integers'],[127,-64,0])
        self.assertEqual(restored[0],1)
        self.assertLessEqual(abs(restored[1]+0.5),info['scale']/2)
        self.assertEqual(lab.quantized_weights([0,0,0])[0],[0,0,0])


class StatisticalTests(unittest.TestCase):
    def test_exact_zero_harm_upper_bound(self):
        self.assertAlmostEqual(lab.binomial_upper(0,12),1-0.05**(1/12))
        self.assertAlmostEqual(lab.binomial_upper(0,50),0.058155079116972264)
        self.assertGreater(lab.binomial_upper(0,149),0.019)
        self.assertLess(lab.binomial_upper(0,149),0.02)

    def test_two_trial_one_harm_has_hand_solved_bound(self):
        # P(X<=1)=1-p^2=0.05, so p=sqrt(0.95).
        self.assertAlmostEqual(lab.binomial_upper(1,2),math.sqrt(0.95),places=13)
        self.assertEqual(lab.binomial_upper(2,2),1)

    def test_zero_losses_twelve_pairs_is_inconclusive(self):
        rows={i:True for i in range(12)}
        result=lab.paired_outcomes(rows,rows)
        self.assertEqual(result['observed_success_difference'],0)
        self.assertEqual(result['population_gate'],'INCONCLUSIVE')
        self.assertGreater(result['harm_probability_upper'],0.22)

    def test_many_independent_zero_losses_can_pass_declared_margin(self):
        rows={i:True for i in range(150)}
        result=lab.paired_outcomes(rows,rows)
        self.assertEqual(result['population_gate'],'PASS')
        self.assertLess(result['harm_probability_upper'],0.02)
        # This checks arithmetic; synthetic booleans are not 150 actual robot trials.

    def test_paired_counts_and_missing_episode_rejection(self):
        ref={0:True,1:False,2:True};cand={0:False,1:True,2:False}
        row=lab.paired_outcomes(ref,cand)
        self.assertEqual((row['harmed'],row['helped']),(2,1))
        self.assertEqual(row['observed_success_difference'],-1/3)
        for r,c in [({},{}),({0:True},{1:True}),({0:1},{0:True})]:
            with self.assertRaises(ValueError):lab.paired_outcomes(r,c)

    def test_invalid_counts_alpha_and_margin(self):
        for args in [(0,0),(2,1),(True,2),(0,2,0),(0,2,1),(0,2,math.nan)]:
            with self.subTest(args=args),self.assertRaises(ValueError):lab.binomial_upper(*args)
        with self.assertRaises(ValueError):lab.paired_outcomes({0:True},{0:True},margin=-0.1)


class TrainedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=lab.run()
        cls.rows={r['candidate']:r for r in cls.report['candidates']}

    def test_actual_training_and_reference_identity(self):
        self.assertAlmostEqual(self.report['held_out_teacher_rmse_m_s2'],0.021929528127230737)
        self.assertEqual(self.report['reference_tape_frames'],180)
        self.assertEqual(self.rows['identity']['fixed_fixture_gate'],'PASS')
        self.assertEqual(self.rows['identity']['max_position_divergence_m'],0)

    def test_folded_and_integer_arithmetic_preserve_declared_fixture(self):
        self.assertLess(self.rows['folded_normalization']['numerics']['raw_max_error_m_s2'],1e-12)
        self.assertEqual(self.rows['folded_normalization']['fixed_fixture_gate'],'PASS')
        self.assertEqual(self.rows['int8_weight_arithmetic']['fixed_fixture_gate'],'PASS')
        self.assertGreater(self.rows['int8_weight_arithmetic']['numerics']['raw_mse_m2_s4'],0)

    def test_same_numerics_can_fail_freshness_and_closed_loop(self):
        row=self.rows['same_weights_slow_fifo']
        self.assertEqual(row['numerics']['raw_mse_m2_s4'],0)
        self.assertEqual(row['successes'],0)
        self.assertIn('expired_observation_applied',row['failure_reasons'])
        self.assertGreater(row['max_applied_age_ms'],250)

    def test_success_count_can_hide_trajectory_difference(self):
        row=self.rows['same_weights_slow_latest']
        self.assertEqual(row['successes'],12)
        self.assertLessEqual(row['max_applied_age_ms'],250)
        self.assertIn('trajectory_divergence',row['failure_reasons'])

    def test_fallback_success_cannot_credit_an_unused_policy(self):
        row=self.rows['same_weights_over_budget']
        self.assertEqual(row['successes'],2)
        self.assertEqual(row['applied_policy_intervals'],0)
        self.assertIn('no_policy_action_applied',row['failure_reasons'])
        self.assertIsNone(row['max_applied_age_ms'])

    def test_wrong_normalization_is_rejected(self):
        row=self.rows['missing_normalization_fault']
        self.assertIn('numerical_action_error',row['failure_reasons'])
        self.assertEqual(row['successes'],5)


if __name__=='__main__':unittest.main()
