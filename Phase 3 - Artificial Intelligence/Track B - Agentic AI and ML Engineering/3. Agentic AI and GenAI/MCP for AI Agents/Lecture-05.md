# Lecture 05 — Build and test a current Python MCP server

**Collection:** [MCP for AI Agents](README.md) | [Previous](Lecture-04.md) | [Next](Lecture-06.md)

**Executed 18 October 2026:** official Python SDK `mcp==2.3.0`, Python 3.12.3. This lesson uses the current `MCPServer` API and an actual in-memory client plus a real stdio subprocess. Read the [2026-07-28 protocol lesson](Protocol-2026-07-28.md) for current wire behavior; older lectures' explicitly scoped compatibility examples do not define this SDK's Python names.

**Time:** 45–60 minutes. **Outcome:** a discoverable, validated read-only tool, resource and prompt, with four passing SDK tests. No model account or robot hardware is required.

## 1. Separate package version from protocol version

| Implementation | Checked state | Consequence |
|---|---|---|
| Official Python `mcp` |2.3.0|High-level server is `from mcp.server import MCPServer`; SDK model fields use snake_case |
| Standalone `fastmcp` |4.1.1 metadata checked|Separate project at PrefectHQ; its migration is not a simple import swap |
| MCP wire protocol |2026-07-28 current lesson|Wire JSON names and negotiation are distinct from Python attribute names |

SDK v2 is no longer a future beta. `mcp.types` moved to the separate types package; direct result fields such as `structured_content` and `is_error` use Python naming. Transport configuration, context access and error behavior also changed. Use the [official v1→v2 migration guide](https://py.sdk.modelcontextprotocol.io/migration/) instead of combining fragments from different versions.

Standalone FastMCP 4 builds on SDK v2 and removes APIs deprecated through its v3 transition. Follow its [2→3](https://gofastmcp.com/getting-started/upgrading/from-fastmcp-2) and [3→4](https://gofastmcp.com/getting-started/upgrading/from-fastmcp-3) guides in order. This course executed the official SDK below; it does not claim a standalone FastMCP 4 runtime test or a market-share percentage.

## 2. Install the tested dependency set

From the repository root:

```bash
python3.12 -m venv /tmp/course-mcp-v2
/tmp/course-mcp-v2/bin/python -m pip install --require-hashes -r \
  'Phase 3 - Artificial Intelligence/Track B - Agentic AI and ML Engineering/3. Agentic AI and GenAI/MCP for AI Agents/labs/requirements-linux-cpu-py312.txt'
```

The complete wheel/hash lock targets Linux x86_64/Python 3.12. It includes the official SDK and its transitive dependencies. Do not install floating latest packages into an unrelated working environment and assume the same API contract.

## 3. Inspect the actual server

[server_v2.py](labs/server_v2.py) exposes a bounded millimeters-to-meters conversion, a static lesson contract and a prompt template. Its key tool is:

```python
from typing import Annotated
from mcp.server import MCPServer
from pydantic import BaseModel, Field

mcp = MCPServer('Robot observation lesson', version='1.0.0')

class Distance(BaseModel):
    value_m: float
    unit: str = 'm'
    frame: str = 'synthetic_sensor'

@mcp.tool()
def millimeters_to_meters(
    value_mm: Annotated[int, Field(strict=True, ge=0, le=1000000)]
) -> Distance:
    return Distance(value_m=value_mm / 1000)
```

The type/field constraints describe a nonnegative bounded integer. Strict validation rejects `true` and the string `"1250"` rather than silently converting them. The returned model gives the client structured data with units and an explicit synthetic frame. `1250mm` becomes `1.25m`; it is not a physical sensor measurement.

A generated input schema is useful, but runtime validation remains necessary. Tool metadata or a read-only annotation is a hint to a client, not an access-control mechanism. This particular implementation performs only arithmetic and does not expose actuator or filesystem operations.

## 4. Resources and prompts have different jobs

The resource `lesson://contract` returns the declared fixture contract. A resource URI identifies data; it is not permission to read arbitrary paths. If extending this to files or records, map identifiers to authorized resources instead of interpolating untrusted strings into unrestricted access.

The prompt `inspect_reading` returns instructions for inspecting a supplied reading. The input reading is explicitly treated as untrusted text. Creating a prompt does not execute its contents or authorize an action. Prompt/resource content can still steer a downstream model, so the receiving application must maintain that trust distinction.

Do not add new dependencies on deprecated roots, sampling or logging merely because a compatibility API exposes them. Use [the current protocol feature status](Protocol-2026-07-28.md) and the SDK's corresponding documentation. Progress, cancellation and elicitation must also follow the chosen protocol and request context.

## 5. Test through a real client

An in-memory client tests actual SDK serialization, discovery and handler behavior without opening a port:

```python
import asyncio
from mcp import Client
from server_v2 import mcp

async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print([tool.name for tool in tools.tools])
        result = await client.call_tool('millimeters_to_meters', {'value_mm': 1250})
        print(result.structured_content)

asyncio.run(main())
```

Run this from the `labs` directory in the environment above, or run the complete tests directly:

```bash
/tmp/course-mcp-v2/bin/python -m unittest discover -s \
  'Phase 3 - Artificial Intelligence/Track B - Agentic AI and ML Engineering/3. Agentic AI and GenAI/MCP for AI Agents/labs/optional' -v
```

The four tests cover discovery/structured output, invalid types/bounds and unknown tool, resource/prompt data, and a **real stdio subprocess** launched with the tested Python executable. They do not connect to a desktop host, call a model, expose HTTP or validate OAuth. [Official client testing guidance](https://py.sdk.modelcontextprotocol.io/get-started/).

## 6. Understand stdio's boundary

The client launches the server and connects its stdin/stdout. Keep protocol output on stdout; diagnostic text belongs on stderr. The subprocess path and environment are part of the trust decision. A local server can use any credentials and resources available to its OS process unless an external boundary restricts them.

The HTTP OAuth profile is not used for this local pipe, but that does not make all requested operations authorized or eliminate downstream authentication. Avoid inheriting unrelated credentials into a tool process. The lab's server needs no credentials at all. Continue with [transport and authorization](Lecture-06.md) for a remote deployment.

## 7. Completion and variations

1. Call the tool with −1, a value above its bound, a boolean and a string. Explain each error.
2. Add an output timestamp only if you can define its clock and acquisition meaning. A call-completion timestamp is not sensor acquisition time.
3. Add a second read-only synthetic conversion and test its schema/result. Keep physical actuation out of this first server.
4. Explain why a valid tool call does not establish the caller's permission or the data's freshness.

**Submit:** the discovery output, one valid structured result, one rejected call and the subprocess-test result. Record SDK/protocol scope and any unexecuted host/network integration separately.

## Reproduction record

[Four executed SDK tests and installed versions](labs/validation-2026-10-18.json) · [primary-source snapshot and SDK revision](labs/source-evidence-2026-10-18.json). No remote HTTP/OAuth server or model call was run.
