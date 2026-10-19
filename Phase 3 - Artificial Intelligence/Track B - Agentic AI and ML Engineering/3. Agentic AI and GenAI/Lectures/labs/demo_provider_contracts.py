"""Run actual SDK serialization against local fixtures; never contacts a provider."""
import json
from importlib.metadata import version
from optional.test_provider_sdk import FixtureTransport, claude, ccall, responses, ocall, otext
from provider_loops import run_claude, run_responses


def main():
    rows = []
    cases = [
        ("claude", run_claude, [claude([ccall()], "tool_use"), claude([{"type": "text", "text": "5"}])]),
        ("openai", run_responses, [responses([ocall()]), responses([otext()])]),
    ]
    for provider, run, replies in cases:
        fixture = FixtureTransport(replies)
        with fixture.client(provider) as client:
            answer = run(client, "fixture-model", "Add 2 and 3.")
        assert answer == "5" and len(fixture.requests) == 2
        # Bodies contain fictional data only; never record authorization headers.
        rows.append({"provider": provider, "answer": answer,
                     "request_bodies": fixture.requests})
    print(json.dumps({"evidence": "real SDKs with local HTTP fixtures; no model execution",
                      "packages": {name: version(name) for name in
                                   ["anthropic", "openai", "httpx", "httpx2", "pydantic"]},
                      "runs": rows}, indent=2))

if __name__ == "__main__":
    main()
