import json
from pathlib import Path
import sys
import unittest
import httpx
import httpx2
import anthropic
import openai
from pydantic import BaseModel, ConfigDict, Field, ValidationError
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provider_loops import run_claude, run_responses


class FixtureTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        if not self.responses: raise AssertionError("unexpected API request")
        value = self.responses.pop(0)
        return self.http.Response(200, json=value, request=request)

    def client(self, provider):
        self.http = httpx2 if provider == "claude" else httpx
        transport = self.http.Client(transport=self.http.MockTransport(self))
        cls = anthropic.Anthropic if provider == "claude" else openai.OpenAI
        return cls(api_key="fixture-not-a-real-key", base_url="https://fixture.invalid/v1",
                   http_client=transport, max_retries=0, timeout=2.0)


def claude(content, stop="end_turn"):
    return {"id": "msg_fixture", "type": "message", "role": "assistant",
            "model": "fixture-model", "content": content, "stop_reason": stop,
            "stop_sequence": None, "usage": {"input_tokens": 4, "output_tokens": 2}}


def ccall(identifier="tool-1", name="add_integers", arguments=None):
    return {"type": "tool_use", "id": identifier, "name": name,
            "input": arguments if arguments is not None else {"a": 2, "b": 3}}


def responses(output, status="completed"):
    return {"id": "resp_fixture", "object": "response", "created_at": 1,
            "model": "fixture-model", "status": status, "output": output,
            "parallel_tool_calls": True, "tools": [], "tool_choice": "auto",
            "error": None, "incomplete_details": None}


def ocall(identifier="call-1", arguments='{"a":2,"b":3}'):
    return {"type": "function_call", "id": "fc_" + identifier,
            "call_id": identifier, "name": "add_integers", "arguments": arguments,
            "status": "completed"}


def otext(text="5"):
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


class SDKTests(unittest.TestCase):
    def test_current_anthropic_requires_httpx2_transport(self):
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as wrong:
            with self.assertRaisesRegex(TypeError, "httpx2"):
                anthropic.Anthropic(api_key="fixture-not-real", http_client=wrong)

    def test_claude_two_calls_correlated_and_text_joined(self):
        fixture = FixtureTransport([claude([ccall(), ccall("tool-2", arguments={"a": 7, "b": -2})], "tool_use"),
                                    claude([{"type": "text", "text": "5"}, {"type": "text", "text": " and 5"}])])
        with fixture.client("claude") as client:
            self.assertEqual(run_claude(client, "fixture-model", "add"), "5 and 5")
        second = fixture.requests[1]["messages"]
        self.assertEqual(second[-1]["role"], "user")
        self.assertEqual([b["tool_use_id"] for b in second[-1]["content"]], ["tool-1", "tool-2"])
        self.assertTrue(all(json.loads(b["content"])["sum"] == 5 for b in second[-1]["content"]))
        self.assertNotIn("temperature", fixture.requests[0])

    def test_claude_bad_args_return_error(self):
        fixture = FixtureTransport([claude([ccall(arguments={"a": True, "b": 3})], "tool_use"), claude([{"type": "text", "text": "invalid"}])])
        with fixture.client("claude") as client: run_claude(client, "fixture-model", "add")
        self.assertTrue(fixture.requests[1]["messages"][-1]["content"][0]["is_error"])

    def test_claude_unknown_tool(self):
        fixture = FixtureTransport([claude([ccall(name="shell")], "tool_use"), claude([{"type": "text", "text": "no tool"}])])
        with fixture.client("claude") as client: run_claude(client, "fixture-model", "add")
        result = fixture.requests[1]["messages"][-1]["content"][0]
        self.assertEqual(json.loads(result["content"])["error"], "unknown-tool")

    def test_claude_incomplete_stops(self):
        for reason in ["max_tokens", "refusal", "pause_turn"]:
            fixture = FixtureTransport([claude([ccall()], reason)])
            with fixture.client("claude") as client:
                with self.assertRaisesRegex(RuntimeError, "incomplete-claude"):
                    run_claude(client, "fixture-model", "add")
            self.assertEqual(len(fixture.requests), 1)

    def test_claude_duplicate_ids_rejected(self):
        fixture = FixtureTransport([claude([ccall(), ccall()], "tool_use")])
        with fixture.client("claude") as client:
            with self.assertRaisesRegex(ValueError, "duplicate-call-id"):
                run_claude(client, "fixture-model", "add")

    def test_turn_and_call_limits(self):
        for kwargs, message in [({"max_turns": 1}, "turn-budget"), ({"max_calls": 1}, "tool-call-budget")]:
            fixture = FixtureTransport([claude([ccall(), ccall("tool-2")], "tool_use")])
            with fixture.client("claude") as client:
                with self.assertRaisesRegex((RuntimeError, ValueError), message):
                    run_claude(client, "fixture-model", "add", **kwargs)

    def test_responses_preserves_all_output_and_matches_call(self):
        reasoning = {"id": "rs_1", "type": "reasoning", "summary": []}
        fixture = FixtureTransport([responses([reasoning, ocall()]), responses([otext()])])
        with fixture.client("openai") as client:
            self.assertEqual(run_responses(client, "fixture-model", "add"), "5")
        history = fixture.requests[1]["input"]
        self.assertEqual([item.get("type") for item in history[1:]], ["reasoning", "function_call", "function_call_output"])
        self.assertEqual(history[-1]["call_id"], "call-1")
        self.assertEqual(json.loads(history[-1]["output"]), {"ok": True, "sum": 5})
        self.assertFalse(fixture.requests[0]["store"])

    def test_responses_rejects_duplicate_json(self):
        fixture = FixtureTransport([responses([ocall(arguments='{"a":2,"a":9,"b":3}')]), responses([otext("invalid")])])
        with fixture.client("openai") as client: run_responses(client, "fixture-model", "add")
        self.assertEqual(json.loads(fixture.requests[1]["input"][-1]["output"])["error"], "invalid-json-arguments")

    def test_responses_incomplete(self):
        fixture = FixtureTransport([responses([ocall()], "incomplete")])
        with fixture.client("openai") as client:
            with self.assertRaisesRegex(RuntimeError, "incomplete-responses"):
                run_responses(client, "fixture-model", "add")

    def test_responses_refusal(self):
        message = otext(); message["content"] = [{"type": "refusal", "refusal": "fixture refusal"}]
        fixture = FixtureTransport([responses([message])])
        with fixture.client("openai") as client:
            with self.assertRaisesRegex(RuntimeError, "responses-refusal"):
                run_responses(client, "fixture-model", "add")

    def test_current_anthropic_removes_temperature_parameter(self):
        fixture = FixtureTransport([])
        with fixture.client("claude") as client:
            with self.assertRaisesRegex(TypeError, "temperature"):
                client.messages.create(model="fixture-model", max_tokens=100,
                    messages=[{"role": "user", "content": "hi"}], temperature=0)
        self.assertEqual(fixture.requests, [])

    def test_responses_explicit_continuation_is_not_final(self):
        first = responses([otext("Working")]); first["end_turn"] = False
        last = responses([otext("Finished")]); last["end_turn"] = True
        fixture = FixtureTransport([first, last])
        with fixture.client("openai") as client:
            self.assertEqual(run_responses(client, "fixture-model", "task"), "Finished")
        self.assertEqual(len(fixture.requests), 2)
        self.assertEqual(fixture.requests[1]["input"][-1]["content"][0]["text"], "Working")

    def test_responses_malformed_continuation_fails(self):
        first = responses([otext()]); first["end_turn"] = "false"
        fixture = FixtureTransport([first])
        with fixture.client("openai") as client:
            with self.assertRaisesRegex(RuntimeError, "invalid-end-turn"):
                run_responses(client, "fixture-model", "task")

    def test_responses_text_continuation_obeys_budget(self):
        first = responses([otext("Working")]); first["end_turn"] = False
        fixture = FixtureTransport([first])
        with fixture.client("openai") as client:
            with self.assertRaisesRegex(RuntimeError, "turn-budget"):
                run_responses(client, "fixture-model", "task", max_turns=1)

    def test_refusal_beside_call_stops_before_result_submission(self):
        msg = otext(); msg["content"] = [{"type": "refusal", "refusal": "no"}]
        fixture = FixtureTransport([responses([msg, ocall()])])
        with fixture.client("openai") as client:
            with self.assertRaisesRegex(RuntimeError, "responses-refusal"):
                run_responses(client, "fixture-model", "task")
        self.assertEqual(len(fixture.requests), 1)

    def test_unsupported_server_tools_are_not_final_success(self):
        fixture = FixtureTransport([responses([{"type": "web_search_call", "id": "ws1", "status": "completed"}])])
        with fixture.client("openai") as client:
            with self.assertRaisesRegex(RuntimeError, "unsupported-responses-item"):
                run_responses(client, "fixture-model", "task")
        fixture = FixtureTransport([claude([{"type": "server_tool_use", "id": "srv1", "name": "web_search", "input": {}}])])
        with fixture.client("claude") as client:
            with self.assertRaisesRegex(RuntimeError, "unsupported-claude-content"):
                run_claude(client, "fixture-model", "task")

    def test_duplicate_response_call_id_across_turns(self):
        fixture = FixtureTransport([responses([ocall()]), responses([ocall()])])
        with fixture.client("openai") as client:
            with self.assertRaisesRegex(ValueError, "duplicate-call-id"):
                run_responses(client, "fixture-model", "task")
        self.assertEqual(len(fixture.requests), 2)

    def test_structured_parse_sdk_generates_output_config(self):
        class Spec(BaseModel):
            model_config = ConfigDict(extra="forbid", strict=True)
            device: str
            bandwidth_gb_s: float = Field(gt=0)
        fixture = FixtureTransport([claude([{"type": "text", "text": '{"device":"Demo-A","bandwidth_gb_s":40.0}'}])])
        with fixture.client("claude") as client:
            result = client.messages.parse(model="fixture-model", max_tokens=100,
                    messages=[{"role": "user", "content": "extract"}], output_format=Spec)
        self.assertEqual(result.parsed_output.bandwidth_gb_s, 40.0)
        self.assertEqual(fixture.requests[0]["output_config"]["format"]["type"], "json_schema")

    def test_structured_parse_enforces_local_range(self):
        class Spec(BaseModel):
            bandwidth_gb_s: float = Field(gt=0)
        fixture = FixtureTransport([claude([{"type": "text", "text": '{"bandwidth_gb_s":-1}'}])])
        with fixture.client("claude") as client:
            with self.assertRaises(ValidationError):
                client.messages.parse(model="fixture-model", max_tokens=100,
                    messages=[{"role": "user", "content": "extract"}], output_format=Spec)


if __name__ == "__main__": unittest.main()
