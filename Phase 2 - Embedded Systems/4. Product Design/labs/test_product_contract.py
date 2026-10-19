import unittest
from product_contract import Appliance, demonstration


class ProductTests(unittest.TestCase):
    def device(self):
        d = Appliance(session='test-session'); d.ready(0)
        return d

    def test_network_attachment_is_not_readiness(self):
        d = Appliance(); d.network_connected = True
        self.assertEqual(d.state, 'starting')
        self.assertFalse(d.frame(b'frame'))
        with self.assertRaises(ValueError): d.request('typed', 0)
        d.ready(0); self.assertEqual(d.state, 'ready')

    def test_mute_clears_queue_and_pending_voice_result(self):
        d = self.device(); token = d.request('voice', 1)
        self.assertTrue(d.frame(b'queued'))
        d.set_mute(True, 2)
        self.assertFalse(d.frame(b'new')); self.assertEqual(len(d.frames), 0)
        self.assertFalse(d.result(token, 20, 3))
        self.assertIsNone(d.timer_due_ms)
        d.set_mute(False, 4)
        self.assertFalse(d.result(token, 20, 5))

    def test_local_timer_is_independent_of_mute_and_network(self):
        d = self.device(); d.set_mute(True, 0)
        token = d.request('typed', 10)
        self.assertTrue(d.result(token, 1, 20))
        self.assertEqual(d.state, 'timer_active')
        self.assertFalse(d.capture_permitted)
        d.advance(1019); self.assertEqual(d.expiry_events, 0)
        d.advance(1020); self.assertEqual(d.expiry_events, 1)
        d.advance(5000); self.assertEqual(d.expiry_events, 1)

    def test_cancel_does_not_allow_delayed_result_to_resurrect_task(self):
        d = self.device(); token = d.request('typed', 10)
        d.cancel(20)
        self.assertFalse(d.result(token, 10, 30))
        self.assertEqual(d.state, 'ready')
        new = d.request('typed', 40)
        self.assertTrue(d.result(new, 1, 50))
        d.cancel(60); d.advance(1050)
        self.assertEqual(d.expiry_events, 0)

    def test_deadline_exact_boundary_and_duplicate_result(self):
        d = self.device(); token = d.request('typed', 10)
        self.assertFalse(d.result(token, 1, 510))
        token = d.request('typed', 520)
        self.assertTrue(d.result(token, 1, 1019))
        self.assertFalse(d.result(token, 1, 1020))
        self.assertEqual(d.timer_due_ms, 2019)

    def test_prior_boot_token_cannot_match_reused_sequence(self):
        old = self.device(); old_token = old.request('typed', 0)
        new = Appliance(session='different-boot'); new.ready(0)
        token = new.request('typed', 0)
        self.assertEqual(token.sequence, old_token.sequence)
        self.assertFalse(new.result(old_token, 10, 5))
        self.assertTrue(new.result(token, 1, 10))

    def test_frames_bounded_and_input_contract_strict(self):
        d = self.device()
        for _ in range(6): d.frame(b'frame')
        self.assertEqual(len(d.frames), 4); self.assertEqual(d.dropped_frames, 2)
        token = d.request('typed', 1)
        for bad in (True, -1, 0, 3601, 1.5):
            with self.assertRaises(ValueError): d.result(token, bad, 2)
        with self.assertRaises(ValueError): d.advance(1)
        with self.assertRaises(ValueError): d.set_mute(1, 2)
        with self.assertRaises(ValueError): d.frame(b'x'*1025)

    def test_demo_matches_product_claims(self):
        report = demonstration()
        self.assertFalse(report['late_voice_result_accepted'])
        self.assertEqual(report['network_only']['state'], 'starting')
        self.assertEqual(report['muted_local_timer']['state'], 'timer_active')
        self.assertEqual(report['final']['expiry_events'], 1)


if __name__ == '__main__': unittest.main()
