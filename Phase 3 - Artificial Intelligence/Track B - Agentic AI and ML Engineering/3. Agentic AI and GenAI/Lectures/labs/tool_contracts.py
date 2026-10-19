"""Bounded, side-effect-free arithmetic tool used by provider protocol tests."""
import json

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"},
          "b": {"type": "integer"}}, "required": ["a", "b"], "additionalProperties": False}


def parse_arguments(raw):
    if not isinstance(raw, str) or len(raw) > 4096:
        raise ValueError("argument-size")
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate-key")
            result[k] = v
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    if type(value) is not dict:
        raise ValueError("argument-object")
    return value


def dispatch(name, arguments):
    if name != "add_integers":
        return {"ok": False, "error": "unknown-tool"}
    if type(arguments) is not dict or set(arguments) != {"a", "b"}:
        return {"ok": False, "error": "argument-fields"}
    if any(type(v) is not int or abs(v) > 1000000 for v in arguments.values()):
        return {"ok": False, "error": "argument-range-or-type"}
    return {"ok": True, "sum": arguments["a"] + arguments["b"]}


class CallBudget:
    def __init__(self, limit=8):
        if type(limit) is not int or limit < 1:
            raise ValueError("positive call limit required")
        self.limit = limit
        self.seen = set()

    def admit(self, ids):
        if any(type(i) is not str or not i or len(i) > 128 for i in ids):
            raise ValueError("invalid-call-id")
        if len(set(ids)) != len(ids) or self.seen.intersection(ids):
            raise ValueError("duplicate-call-id")
        if len(self.seen) + len(ids) > self.limit:
            raise ValueError("tool-call-budget")
        self.seen.update(ids)
