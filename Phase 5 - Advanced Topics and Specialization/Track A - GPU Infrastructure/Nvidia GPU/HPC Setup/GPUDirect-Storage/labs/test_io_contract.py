import io
import unittest

from io_contract import BufferState, aligned_cover, chunks, experiment, read_exact


class ShortReader(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 3))


class IOContractTests(unittest.TestCase):
    def test_irregular_tail_covers_every_byte_once(self):
        for total in (0, 1, 4095, 4096, 4097, 12345):
            plan = chunks(total, 4096)
            seen = [position for offset, length in plan for position in range(offset, offset + length)]
            self.assertEqual(seen, list(range(total)))
            self.assertTrue(all(0 < length <= 4096 for _, length in plan))

    def test_bad_configuration_is_rejected(self):
        for total, size in ((-1, 4096), (10, 0), (1.5, 2), (True, 2)):
            with self.assertRaises(ValueError):
                chunks(total, size)

    def test_aligned_cover_keeps_logical_slice(self):
        for offset, length, alignment in ((100, 5000, 4096), (4096, 4096, 4096), (5, 4, 3)):
            cover = aligned_cover(offset, length, alignment)
            self.assertEqual(cover["start"] % alignment, 0)
            self.assertEqual(cover["length"] % alignment, 0)
            self.assertEqual(cover["start"] + cover["prefix"], offset)
            self.assertGreaterEqual(cover["length"], cover["prefix"] + length)

    def test_aligned_cover_does_not_prove_readable_range(self):
        cover = aligned_cover(100, 5000, 4096)
        with self.assertRaises(EOFError):
            read_exact(io.BytesIO(b"x" * 5100), cover["start"], cover["length"])

    def test_short_reads_are_accumulated(self):
        self.assertEqual(read_exact(ShortReader(b"0123456789"), 2, 7), b"2345678")

    def test_unexpected_eof_and_bad_offsets(self):
        with self.assertRaises(EOFError):
            read_exact(io.BytesIO(b"abc"), 1, 3)
        with self.assertRaises(ValueError):
            read_exact(io.BytesIO(b"abc"), -1, 1)
        self.assertEqual(read_exact(io.BytesIO(b"abc"), 3, 0), b"")

    def test_ownership_requires_both_completions(self):
        state = BufferState()
        for action in (state.finish_read, state.submit_compute, state.finish_compute):
            with self.assertRaises(RuntimeError):
                action()
        state.submit_read()
        with self.assertRaises(RuntimeError):
            state.submit_compute()
        state.finish_read()
        state.submit_compute()
        with self.assertRaises(RuntimeError):
            state.submit_read()
        state.finish_compute()
        state.submit_read()  # Reuse is valid after compute completion.
        self.assertEqual(state.state, "reading")

    def test_temporary_file_roundtrip_and_faults(self):
        result = experiment()
        self.assertEqual(result["logical_bytes"], 12345)
        self.assertEqual(result["chunks"][-1], (12288, 57))
        self.assertTrue(result["truncation_rejected"])
        self.assertTrue(result["early_reuse_rejected"])
        self.assertEqual(result["final_buffer_state"], "free")


if __name__ == "__main__":
    unittest.main()
