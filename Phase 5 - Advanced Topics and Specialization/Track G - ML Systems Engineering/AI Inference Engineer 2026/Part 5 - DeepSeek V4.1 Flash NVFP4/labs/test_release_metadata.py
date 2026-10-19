import copy
import unittest

from release_metadata import FILES, compare, fetch_release, shard_manifest


class ReleaseMetadataTests(unittest.TestCase):
    def test_equal_metadata_can_hide_changed_weights(self):
        left = {"revision": "a" * 40, "metadata_receipts": {name: {"sha256": "same"} for name in FILES},
                "shards": {"one.safetensors": {"declared_sha256": "1" * 64}}}
        right = copy.deepcopy(left)
        right["revision"] = "b" * 40
        right["shards"]["one.safetensors"]["declared_sha256"] = "2" * 64
        result = compare(left, right)
        self.assertEqual(len(result["identical_metadata_files"]), 3)
        self.assertEqual(result["changed_declared_shard_hashes"], 1)
        self.assertEqual(result["runtime_compatibility"], "not established by metadata")

    def test_added_and_removed_shards_are_reported(self):
        left = {"revision": "a", "metadata_receipts": {name: {"sha256": "1"} for name in FILES}, "shards": {"old": {"declared_sha256": "1"}}}
        right = copy.deepcopy(left)
        right["shards"] = {"new": {"declared_sha256": "2"}}
        self.assertEqual(compare(left, right)["only_left"], ["old"])
        self.assertEqual(compare(left, right)["only_right"], ["new"])

    def test_missing_or_invalid_weight_hash_is_rejected(self):
        for digest in (None, "a" * 40, "z" * 64):
            item = {"rfilename": "one.safetensors", "size": 10}
            if digest is not None:
                item["lfs"] = {"sha256": digest}
            with self.assertRaises(ValueError):
                shard_manifest({"siblings": [item]})

    def test_empty_and_duplicate_manifests_are_rejected(self):
        with self.assertRaises(ValueError):
            shard_manifest({"siblings": []})
        item = {"rfilename": "one.safetensors", "size": 10, "lfs": {"sha256": "a" * 64}}
        with self.assertRaises(ValueError):
            shard_manifest({"siblings": [item, item]})

    def test_mutable_revision_rejected_before_network(self):
        for revision in ("main", "v1", "a" * 7):
            with self.assertRaises(ValueError):
                fetch_release("example/model", revision)


if __name__ == "__main__":
    unittest.main()
