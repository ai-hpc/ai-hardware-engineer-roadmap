from copy import deepcopy
import json
from pathlib import Path
import unittest

from cache_plan import Allocation, cache_bytes, lesson_allocations, report, summarize


class CacheTests(unittest.TestCase):
    def test_worked_cache_values(self):
        self.assertEqual(cache_bytes(lesson_allocations(), 8192), 76 * 2**20)
        self.assertEqual(cache_bytes(lesson_allocations(), 16384), 140 * 2**20)
        self.assertEqual(cache_bytes(lesson_allocations(), 256), 8 * 2**20)

    def test_local_saturates_global_grows(self):
        local, global_ = Allocation("local", 2, 32, 64), Allocation("global", 2, 64)
        self.assertEqual(cache_bytes([local], 100), cache_bytes([local], 200))
        self.assertEqual(cache_bytes([global_], 200), 2 * cache_bytes([global_], 100))

    def test_share_physical_allocations_not_consumers(self):
        a = Allocation("shared", 4, 128)
        self.assertEqual(cache_bytes([a, a], 8192), cache_bytes([a], 8192))
        self.assertEqual(cache_bytes([a, Allocation("separate",4,128)],8192), 2*cache_bytes([a],8192))

    def test_conflicting_alias_fails(self):
        with self.assertRaises(ValueError):
            cache_bytes([Allocation("same",4,128),Allocation("same",4,256)],8192)

    def test_batch_and_precision(self):
        a = lesson_allocations()
        self.assertEqual(cache_bytes(a,8192,batch=2),2*cache_bytes(a,8192))
        self.assertEqual(cache_bytes(a,8192,element_bytes=1)*2,cache_bytes(a,8192))

    def test_target_and_assistant_are_different(self):
        result=report()
        self.assertEqual(result["target"]["layer_types"], {"sliding_attention":28,"full_attention":7})
        self.assertEqual(result["assistant"]["layer_types"], {"sliding_attention":3,"full_attention":1})
        self.assertEqual(result["target"]["text"]["hidden_size"],1536)
        self.assertEqual(result["assistant"]["text"]["hidden_size"],256)
        self.assertEqual(result["assistant_backbone_hidden_size"],1536)
        self.assertFalse(result["target"]["text"]["attention_k_eq_v"])

    def test_malformed_configuration_fails(self):
        config=json.loads(Path(__file__).with_name("target-config.json").read_text())
        bad=deepcopy(config);bad["text_config"]["num_hidden_layers"]+=1
        with self.assertRaises(ValueError):summarize(bad)
        for context in (0,-1,True,1.5):
            with self.assertRaises(ValueError):cache_bytes(lesson_allocations(),context)


if __name__ == "__main__":
    unittest.main()
