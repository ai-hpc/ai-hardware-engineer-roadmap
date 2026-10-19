import shutil
import socket
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from sensor_protocol import (Sample, pack, unpack, SequenceTracker, filter_matches,
                             FRAME, CAN_EFF_FLAG, CAN_RTR_FLAG, CAN_ERR_FLAG, demo)


class ProtocolTests(unittest.TestCase):
    def test_independent_golden_payload_and_signed_units(self):
        frame = pack(Sample(127, -250, 1200, 15))
        self.assertEqual(frame[8:], bytes.fromhex('017f06ffb0040f00'))
        self.assertEqual(unpack(frame), Sample(127, -250, 1200, 15))

    def test_boundary_values(self):
        for sample in (Sample(0, -32768, 0, 0), Sample(255, 32767, 65535, 65535)):
            self.assertEqual(unpack(pack(sample)), sample)

    def test_invalid_units_types_and_overflow(self):
        for args in ((True, 0, 1, 1), (256, 0, 1, 1), (0, -32769, 1, 1),
                     (0, 32768, 1, 1), (0, 0, -1, 1), (0, 0, 1, 65536),
                     (0, 0.5, 1, 1), (0, 0, 1, float('nan'))):
            with self.subTest(args=args), self.assertRaises(ValueError): Sample(*args)

    def test_all_truncated_and_concatenated_frames(self):
        frame = pack(Sample(1, 2, 3, 4))
        for length in range(16):
            with self.subTest(length=length), self.assertRaises(ValueError): unpack(frame[:length])
        with self.assertRaises(ValueError): unpack(frame + frame)

    def test_remote_error_extended_and_wrong_id_rejected(self):
        frame = pack(Sample(1, 2, 3, 4))
        for can_id in (0x320, 0x321 | CAN_EFF_FLAG, 0x321 | CAN_RTR_FLAG,
                       0x321 | CAN_ERR_FLAG, 0x1321):
            with self.subTest(can_id=can_id), self.assertRaises(ValueError):
                unpack(struct.pack('=I', can_id) + frame[4:])

    def test_payload_length_and_version_rejected(self):
        frame = bytearray(pack(Sample(1, 2, 3, 4)))
        for length in (0, 7, 9, 64):
            frame[4] = length
            with self.assertRaises(ValueError): unpack(bytes(frame))
        frame[4], frame[8] = 8, 2
        with self.assertRaises(ValueError): unpack(bytes(frame))

    def test_filter_distinguishes_standard_remote_and_extended(self):
        self.assertTrue(filter_matches(0x321))
        for can_id in (0x320, 0x321 | CAN_EFF_FLAG, 0x321 | CAN_RTR_FLAG, 0x321 | CAN_ERR_FLAG):
            self.assertFalse(filter_matches(can_id))

    def test_wrap_duplicates_loss_and_reordering(self):
        tracker = SequenceTracker()
        self.assertEqual([tracker.observe(s) for s in (254, 255, 0, 0, 3, 2, 4)],
                         [('first', 0), ('next', 0), ('next', 0), ('duplicate', 0),
                          ('gap', 2), ('old_or_ambiguous', 0), ('next', 0)])
        self.assertEqual(tracker.last, 4)

    def test_half_range_ambiguity_does_not_advance_tracker(self):
        tracker = SequenceTracker()
        tracker.observe(0)
        self.assertEqual(tracker.observe(128), ('old_or_ambiguous', 0))
        self.assertEqual(tracker.observe(1), ('next', 0))
        # Delta 127 is the last unambiguous forward advance: 126 missed frames.
        boundary = SequenceTracker()
        boundary.observe(0)
        self.assertEqual(boundary.observe(127), ('gap', 126))
        self.assertEqual(boundary.observe(255), ('old_or_ambiguous', 0))
        self.assertEqual(boundary.last, 127)

    def test_actual_datagram_preserves_message_boundaries(self):
        first, second = Sample(1, -1, 10, 0), Sample(2, 1, 20, 1)
        tx, rx = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        with tx, rx:
            rx.settimeout(1)
            tx.send(pack(first)); tx.send(pack(second))
            self.assertEqual(unpack(rx.recv(64)), first)
            self.assertEqual(unpack(rx.recv(64)), second)
        self.assertEqual(demo()['payload_hex'], '017f06ffb0040f00')


@unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('cc'),
                     'Linux and cc required for native ABI fixture')
class NativeABITests(unittest.TestCase):
    def test_real_linux_header_layout_and_cross_language_exchange(self):
        with tempfile.TemporaryDirectory(prefix='course-can-abi-') as directory:
            binary = str(Path(directory) / 'can_abi')
            subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
                            str(Path(__file__).with_name('can_abi.c')), '-o', binary], check=True)
            fixture = subprocess.check_output([binary, 'emit'])
            sample = Sample(127, -250, 1200, 15)
            self.assertEqual(unpack(fixture), sample)
            subprocess.run([binary, 'check'], input=pack(sample), check=True, capture_output=True)
            self.assertEqual(subprocess.run([binary, 'check'], input=fixture[:-1],
                                           capture_output=True).returncode, 2)
            malformed = bytearray(fixture); malformed[8] = 2
            self.assertEqual(subprocess.run([binary, 'check'], input=malformed,
                                           capture_output=True).returncode, 3)


if __name__ == '__main__':
    unittest.main()
