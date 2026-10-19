"""Real SDK adapters, exercised with local HTTP fixtures. No live API calls in tests."""
import json
from tool_contracts import SCHEMA, CallBudget, dispatch, parse_arguments

CLAUDE_TOOLS = [{"name": "add_integers", "description": "Add two bounded integers.",
                 "input_schema": SCHEMA, "strict": True}]
RESPONSES_TOOLS = [{"type": "function", "name": "add_integers",
                    "description": "Add two bounded integers.", "parameters": SCHEMA,
                    "strict": True}]


def run_claude(client, model, prompt, max_turns=4, max_calls=8):
    messages = [{"role": "user", "content": prompt}]
    budget = CallBudget(max_calls)
    for _ in range(max_turns):
        response = client.messages.create(model=model, max_tokens=256,
                                          tools=CLAUDE_TOOLS, messages=messages)
        if response.stop_reason not in {"end_turn", "tool_use"}:
            raise RuntimeError(f"incomplete-claude:{response.stop_reason}")
        if any(b.type not in {"text", "thinking", "redacted_thinking", "tool_use"}
               for b in response.content):
            raise RuntimeError("unsupported-claude-content")
        messages.append({"role": "assistant", "content": [
            b.model_dump(mode="json", exclude_none=True) for b in response.content]})
        calls = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason == "end_turn":
            if calls: raise RuntimeError("inconsistent-claude-stop")
            return "".join(b.text for b in response.content if b.type == "text")
        if not calls: raise RuntimeError("missing-client-tool-call")
        budget.admit([b.id for b in calls])
        results = []
        for b in calls:
            if getattr(b, "toolset_name", None):
                raise RuntimeError("unsupported-toolset")
            result = dispatch(b.name, b.input)
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": json.dumps(result), "is_error": not result["ok"]})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("turn-budget")


def run_responses(client, model, prompt, max_turns=4, max_calls=8):
    items = [{"role": "user", "content": prompt}]
    budget = CallBudget(max_calls)
    for _ in range(max_turns):
        response = client.responses.create(model=model, max_output_tokens=256,
                                           tools=RESPONSES_TOOLS, input=items, store=False)
        if response.status != "completed":
            raise RuntimeError(f"incomplete-responses:{response.status}")
        if any(b.type not in {"message", "reasoning", "function_call"} for b in response.output):
            raise RuntimeError("unsupported-responses-item")
        if any(c.type == "refusal" for b in response.output if b.type == "message" for c in b.content):
            raise RuntimeError("responses-refusal")
        end_turn = getattr(response, "end_turn", None)
        if end_turn is not None and type(end_turn) is not bool:
            raise RuntimeError("invalid-end-turn")
        items.extend(b.model_dump(mode="json", exclude_none=True) for b in response.output)
        calls = [b for b in response.output if b.type == "function_call"]
        if not calls:
            if end_turn is False:
                continue  # A completed response is not necessarily a completed turn.
            return response.output_text
        budget.admit([b.call_id for b in calls])
        for b in calls:
            try:
                result = dispatch(b.name, parse_arguments(b.arguments))
            except (ValueError, TypeError):
                result = {"ok": False, "error": "invalid-json-arguments"}
            items.append({"type": "function_call_output", "call_id": b.call_id,
                          "output": json.dumps(result)})
    raise RuntimeError("turn-budget")
