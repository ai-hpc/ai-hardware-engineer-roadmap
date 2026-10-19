"""Local signed-artifact exercise. Does not provision keys, flash, or install updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

MAX_ARTIFACT_BYTES = 1024 * 1024
FIELDS = {"schema", "target", "version", "size_bytes", "sha256"}


def canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def manifest_for(artifact: bytes, version: int = 3, target: str = "course-simulator-v1") -> dict:
    return {"schema": 1, "target": target, "version": version,
            "size_bytes": len(artifact), "sha256": hashlib.sha256(artifact).hexdigest()}


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def parse_manifest(raw: bytes) -> dict:
    if len(raw) > 4096: raise ValueError("manifest too large")
    try:
        manifest = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("invalid manifest JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != FIELDS:
        raise ValueError("manifest has missing or unknown fields")
    if type(manifest["schema"]) is not int or manifest["schema"] != 1:
        raise ValueError("unsupported schema")
    if type(manifest["version"]) is not int or not 0 <= manifest["version"] < 2**31:
        raise ValueError("invalid version")
    if type(manifest["size_bytes"]) is not int or not 0 <= manifest["size_bytes"] <= MAX_ARTIFACT_BYTES:
        raise ValueError("invalid size")
    if not isinstance(manifest["target"], str) or not re.fullmatch(r"[a-z0-9-]{1,64}", manifest["target"]):
        raise ValueError("invalid target")
    if not isinstance(manifest["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", manifest["sha256"]):
        raise ValueError("invalid digest")
    if raw != canonical(manifest):
        raise ValueError("manifest must use this lab's exact canonical serialization")
    return manifest


def policy(manifest: dict, artifact: bytes, expected_target: str, minimum_version: int) -> dict:
    if type(minimum_version) is not int or minimum_version < 0:
        raise ValueError("trusted minimum version must be a nonnegative integer")
    if manifest["target"] != expected_target: raise ValueError("target mismatch")
    if manifest["version"] < minimum_version: raise ValueError("rollback rejected")
    if len(artifact) != manifest["size_bytes"]: raise ValueError("artifact size mismatch")
    if hashlib.sha256(artifact).hexdigest() != manifest["sha256"]:
        raise ValueError("artifact digest mismatch")
    return {"accepted": True, "version": manifest["version"], "target": manifest["target"],
            "sha256": manifest["sha256"]}


def openssl(*arguments: str) -> subprocess.CompletedProcess:
    if not shutil.which("openssl"): raise RuntimeError("OpenSSL executable unavailable")
    return subprocess.run(["openssl", *map(str, arguments)], capture_output=True,
                          text=True, timeout=30)


def make_keys(directory: Path) -> tuple[Path, Path]:
    private, public = directory / "ephemeral-private.pem", directory / "public.pem"
    result = openssl("genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", private)
    if result.returncode: raise RuntimeError("test key generation failed")
    private.chmod(0o600)
    result = openssl("pkey", "-in", private, "-pubout", "-out", public)
    if result.returncode: raise RuntimeError("public key export failed")
    return private, public


def sign(raw: bytes, private: Path) -> bytes:
    with tempfile.TemporaryDirectory(prefix="course-sign-") as name:
        directory = Path(name)
        source, signature = directory / "manifest", directory / "signature"
        source.write_bytes(raw)
        result = openssl("dgst", "-sha256", "-sign", private, "-out", signature, source)
        if result.returncode: raise RuntimeError("test signing failed")
        return signature.read_bytes()


def verify(raw: bytes, signature: bytes, artifact: bytes, public: Path,
           expected_target: str, minimum_version: int) -> dict:
    manifest = parse_manifest(raw)
    if not 1 <= len(signature) <= 1024: raise ValueError("invalid signature size")
    if len(artifact) > MAX_ARTIFACT_BYTES: raise ValueError("artifact too large")
    # Verify the same captured bytes subsequently parsed and checked. The public
    # key and policy are trusted inputs, not keys/version floors supplied by the bundle.
    with tempfile.TemporaryDirectory(prefix="course-verify-") as name:
        directory = Path(name)
        source, sig = directory / "manifest", directory / "signature"
        source.write_bytes(raw)
        sig.write_bytes(signature)
        result = openssl("dgst", "-sha256", "-verify", public, "-signature", sig, source)
        if result.returncode: raise ValueError("signature verification failed")
    return policy(manifest, artifact, expected_target, minimum_version)


def demo(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    artifact = b"course synthetic robot model revision 3\n"
    raw = canonical(manifest_for(artifact))
    with tempfile.TemporaryDirectory(prefix="course-test-key-") as name:
        private, public = make_keys(Path(name))
        signature = sign(raw, private)
        result = verify(raw, signature, artifact, public, "course-simulator-v1", 2)
        (output / "public.pem").write_bytes(public.read_bytes())
        # The demonstration private key is deleted with its temporary directory.
    for filename, data in (("artifact.bin", artifact), ("manifest.json", raw), ("manifest.sig", signature)):
        (output / filename).write_bytes(data)
    result.update({"openssl": openssl("version").stdout.strip(),
                   "private_key_retained": False,
                   "boundary": "local signature/hash/policy exercise; no device installation or protected counter"})
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(demo(arguments.output_dir), indent=2))
