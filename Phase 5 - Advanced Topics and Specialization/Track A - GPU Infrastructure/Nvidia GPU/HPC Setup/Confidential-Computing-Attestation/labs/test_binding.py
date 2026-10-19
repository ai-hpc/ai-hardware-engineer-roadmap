import unittest
from binding import bindings, canonical, strict_load


class TranscriptTests(unittest.TestCase):
    def test_length_ambiguity_removed(self):
        self.assertEqual("ab" + "c", "a" + "bc")
        self.assertNotEqual(canonical(["ab", "c"]), canonical(["a", "bc"]))

    def test_order_and_full_field_binding(self):
        a = {"nonce": "123", "key": "abc"}
        self.assertEqual(bindings(a), bindings(dict(reversed(list(a.items())))))
        changed = bindings({**a, "key": "abd"})
        for role in ("cpu", "gpu"):
            self.assertNotEqual(bindings(a)[role], changed[role])
        self.assertEqual(len(bytes.fromhex(bindings(a)["cpu"])), 64)
        self.assertEqual(len(bytes.fromhex(bindings(a)["gpu"])), 32)

    def test_duplicate_and_noncanonical_values(self):
        for raw in ['{"nonce":"a","nonce":"b"}', '{"x":NaN}', '{"x":1.5}']:
            with self.assertRaises(ValueError):
                strict_load(raw)
        with self.assertRaises(ValueError):
            canonical({"x": "caf\u00e9"})

    def test_resource_bound(self):
        with self.assertRaises(ValueError):
            strict_load(' ' * 65537)


if __name__ == "__main__":
    unittest.main()
