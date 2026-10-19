import socket
import struct
import time
import unittest
from localhost_lab import CommandGate, encode_frame, percentile, receive_frame, tcp_peer, tcp_experiment, udp_experiment


class FramingTests(unittest.TestCase):
    def test_two_coalesced_frames_with_unicode_and_short_reads(self):
        messages = [{'text': 'café 🤖'}, {'sequence': 8, 'text': 'next'}]
        with tcp_peer(lambda s: s.sendall(b''.join(map(encode_frame, messages)))) as client:
            self.assertEqual([receive_frame(client, time.monotonic() + 2) for _ in messages], messages)

    def test_reject_truncation_and_unbounded_length(self):
        with tcp_peer(lambda s: s.sendall(struct.pack('!I', 10) + b'{}')) as client:
            with self.assertRaises(EOFError): receive_frame(client, time.monotonic() + 2)
        for n in (0, 4097, 2**32-1):
            with tcp_peer(lambda s: s.sendall(struct.pack('!I', n))) as client:
                with self.assertRaises(ValueError): receive_frame(client, time.monotonic() + 2)

    def test_whole_frame_deadline_not_reset_after_each_fragment(self):
        def delayed(s):
            s.sendall(b'\x00')
            time.sleep(.07)
        with tcp_peer(delayed) as client:
            with self.assertRaises(TimeoutError): receive_frame(client, time.monotonic() + .02)

    def test_nonfinite_encoding_and_percentile_definition(self):
        with self.assertRaises(ValueError): encode_frame({'x': float('nan')})
        self.assertEqual(percentile(list(range(1, 21)), 95), 19)


class GateTests(unittest.TestCase):
    def message(self, sequence=0, delta=2, deadline=1001):
        return {'session': 'fixture-session', 'sequence': sequence, 'delta': delta, 'deadline_ns': deadline}

    def test_ack_replay_does_not_reapply_and_conflicting_retry_fails(self):
        gate=CommandGate()
        self.assertEqual(gate.accept(self.message(), 1000)['status'], 'applied')
        self.assertTrue(gate.accept(self.message(), 2000)['duplicate'])
        self.assertEqual(gate.value, 2)
        self.assertEqual(gate.accept(self.message(delta=3), 1000)['status'], 'id_conflict')

    def test_expiry_order_epoch_limit_and_invalid_values(self):
        gate=CommandGate(limit=1)
        self.assertEqual(gate.accept(self.message(deadline=1000), 1000)['status'], 'expired')
        self.assertEqual(gate.accept(self.message(sequence=4), 1000)['status'], 'applied')
        self.assertEqual(gate.accept(self.message(sequence=3), 1000)['status'], 'out_of_order')
        self.assertEqual(gate.accept(self.message(sequence=5), 1000)['status'], 'session_full')
        msg=self.message();msg['session']='old-session'
        self.assertEqual(gate.accept(msg, 1000)['status'], 'wrong_session')
        self.assertEqual(gate.accept(self.message(delta=True), 1000)['status'], 'invalid')
        self.assertEqual(gate.accept(self.message(sequence=7, deadline=6_000_000_000),1000)['status'],'invalid_deadline')


class RealLoopbackTests(unittest.TestCase):
    def test_tcp_roundtrips_have_correct_sequences_and_actual_timings(self):
        report=tcp_experiment(5, 2)
        self.assertEqual(report['injected_server_work_ms'],[0,0,0,0,2])
        self.assertEqual(len(report['rtt_ms']),5)
        self.assertTrue(all(v>0 for v in report['rtt_ms']))
        self.assertGreaterEqual(report['max_ms'],report['p50_ms'])

    def test_udp_loss_duplicate_reordering_and_expiry(self):
        report=udp_experiment()
        self.assertEqual(report['final_counter'],10)
        self.assertEqual(report['unique_applied_commands'],4)
        rows=report['requests']
        self.assertEqual([r['attempts'] for r in rows],[2,2,1,1,1,1])
        self.assertEqual([r['reply']['status'] for r in rows],
                         ['applied','applied','applied','out_of_order','expired','applied'])
        self.assertTrue(rows[1]['reply']['duplicate'])
        self.assertFalse(rows[0]['reply']['duplicate'])


if __name__=='__main__': unittest.main()
