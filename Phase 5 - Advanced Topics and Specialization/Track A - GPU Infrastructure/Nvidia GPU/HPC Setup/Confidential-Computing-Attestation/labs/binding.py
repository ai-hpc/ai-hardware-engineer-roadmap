"""Strict teaching transcript encoding, not vendor evidence or RFC 8785."""
import hashlib
import json

DOMAIN = b"course/binding/v1\0"


def canonical(value):
    def check(v):
        if type(v) is str:
            if not v.isascii():
                raise ValueError("ASCII strings required by the teaching protocol")
        elif type(v) in (int, bool) or v is None:
            pass
        elif type(v) is list:
            for item in v:
                check(item)
        elif type(v) is dict:
            for key, item in v.items():
                if type(key) is not str:
                    raise ValueError("string keys required")
                check(key)
                check(item)
        else:
            raise ValueError("unsupported transcript type")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_load(raw):
    if len(raw) > 65536:
        raise ValueError("teaching bundle size limit")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    canonical(value)
    return value


def bindings(transcript):
    data = DOMAIN + canonical(transcript)
    return {"cpu": hashlib.sha512(data).hexdigest(),
            "gpu": hashlib.sha256(data).hexdigest()}
