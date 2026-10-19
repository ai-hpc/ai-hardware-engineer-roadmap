import json
from pathlib import Path
import shutil
import tempfile
import unittest

from signed_update import canonical, make_keys, manifest_for, parse_manifest, policy, sign, verify


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.artifact = b"model-v3"
        self.manifest = manifest_for(self.artifact)

    def test_exact_canonical_manifest_roundtrip(self):
        self.assertEqual(parse_manifest(canonical(self.manifest)), self.manifest)

    def test_duplicate_unknown_and_boolean_fields_rejected(self):
        duplicate = b'{"schema":1,"schema":1}'
        with self.assertRaises(ValueError): parse_manifest(duplicate)
        for change in ({"unknown":1}, {"version":True}, {"schema":True}, {"size_bytes":-1},
                       {"sha256":"g"*64}, {"target":"../device"}):
            with self.assertRaises(ValueError): parse_manifest(canonical(self.manifest | change))

    def test_noncanonical_bytes_rejected(self):
        with self.assertRaises(ValueError): parse_manifest(json.dumps(self.manifest).encode())

    def test_size_and_hash_both_checked(self):
        for bad in (b"short", b"model-v4"):
            with self.assertRaises(ValueError): policy(self.manifest, bad, "course-simulator-v1", 2)

    def test_target_and_rollback_policy(self):
        for target, floor in (("wrong-board",2), ("course-simulator-v1",4)):
            with self.assertRaises(ValueError): policy(self.manifest,self.artifact,target,floor)
        self.assertTrue(policy(self.manifest,self.artifact,"course-simulator-v1",3)["accepted"])


@unittest.skipUnless(shutil.which("openssl"), "native signature tests require OpenSSL")
class SignatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="course-signature-tests-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.private, cls.public = make_keys(Path(cls.directory.name))
        cls.artifact = b"model-v3"
        cls.raw = canonical(manifest_for(cls.artifact))
        cls.signature = sign(cls.raw, cls.private)

    def check(self, raw=None, signature=None, artifact=None, floor=2, target="course-simulator-v1"):
        return verify(self.raw if raw is None else raw,
                      self.signature if signature is None else signature,
                      self.artifact if artifact is None else artifact, self.public, target, floor)

    def test_actual_signature_accepts_valid_artifact(self):
        self.assertTrue(self.check()["accepted"])

    def test_changed_metadata_invalidates_signature(self):
        changed = canonical(manifest_for(self.artifact,version=4))
        with self.assertRaisesRegex(ValueError,"signature"): self.check(raw=changed)

    def test_changed_payload_fails_despite_valid_manifest_signature(self):
        with self.assertRaisesRegex(ValueError,"digest"): self.check(artifact=b"model-v4")

    def test_truncated_or_empty_signature_rejected(self):
        for signature in (self.signature[:-1], b""):
            with self.assertRaises(ValueError): self.check(signature=signature)

    def test_valid_signature_still_needs_target_and_version_policy(self):
        with self.assertRaisesRegex(ValueError,"rollback"): self.check(floor=4)
        with self.assertRaisesRegex(ValueError,"target"): self.check(target="other-simulator")

    def test_different_trust_anchor_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, other_public = make_keys(Path(directory))
            with self.assertRaisesRegex(ValueError,"signature"):
                verify(self.raw,self.signature,self.artifact,other_public,"course-simulator-v1",2)


if __name__ == "__main__":
    unittest.main()
