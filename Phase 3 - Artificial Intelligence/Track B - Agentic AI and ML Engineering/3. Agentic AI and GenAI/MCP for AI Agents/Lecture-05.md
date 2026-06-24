# Lecture 05 - Building an MCP Server (FastMCP)

**Collection:** [MCP for AI Agents](README.md) | **Previous:** [← Lecture 04](Lecture-04.md) | **Next:** [Lecture 06](Lecture-06.md)

---

The previous four lectures were spent reading the protocol: the host/client/server split, the lifecycle, and the six primitives — Tools, Resources, Prompts going server→client, and Sampling, Roots, Elicitation running backwards. Now you stop reading the spec and start emitting it. By the end of this lecture you will have a server that a real host — Claude Desktop, Claude Code, or the MCP Inspector — can connect to, enumerate, and call.

The good news for your wrists: you are not going to hand-write JSON-RPC envelopes or JSON Schema. The **FastMCP** API — the high-level layer that ships *inside* the official `mcp` Python SDK — turns a decorated Python function into a fully-formed tool, reading your type hints to build the `inputSchema` and your docstring to build the description. The protocol machinery from Lectures 02–04 is still down there; FastMCP just generates it for you so you can spend your attention on the actual capability you are exposing.

One standing warning before any code: **the SDK surface is a moving target.** The official package is heading toward a v2 (beta in 2026), and the standalone FastMCP 2.x evolves release-to-release. Decorator names, the `Context` method set, and the lifespan signature have all shifted across versions. Treat every snippet here as *current-as-of* and verify against the version you actually `pip install`ed — `mcp version`, then read the module's own docstrings if a call doesn't resolve.

---

## Learning objectives

By the end of this lecture you should be able to:

- Choose between the official `mcp` SDK (FastMCP), the standalone FastMCP 2.x, and the TypeScript SDK, and install the right one.
- Write a complete FastMCP server exposing a `@mcp.tool()`, and explain how Python type hints become the `inputSchema` the host sees.
- Expose templated **resources** with `@mcp.resource(...)` and reusable **prompts** with `@mcp.prompt()`.
- Use the injected **`Context`** object for logging, progress, resource reads, sampling, and elicitation, and hold shared state in a **lifespan**.
- Return a Pydantic model so the SDK emits an `outputSchema` and structured content.
- Test the server with the **MCP Inspector** and wire it into a host via a JSON config block.

---

## 1. The SDK landscape & install

Three SDKs matter in 2026, and it is worth being precise about which is which because their names collide.

- **The official `mcp` Python SDK.** Maintained alongside the spec. It ships a low-level `Server` class *and* a high-level ergonomic layer called **FastMCP** — imported as `from mcp.server.fastmcp import FastMCP`. This is the canonical, spec-tracking choice, and it is what most of this lecture uses.
- **The standalone FastMCP 2.x** (`jlowin/Prefect`, `pip install fastmcp`). The original FastMCP project, now a superset with extra batteries — auth helpers, server composition, testing utilities, deployment glue. It powers roughly **70% of community servers**. The core decorator API is nearly identical to the official one, so most of what you learn here transfers; FastMCP 2.x just adds more around the edges.
- **The official TypeScript SDK** (`@modelcontextprotocol/sdk`). The same protocol for the Node/browser world — covered briefly in §7.

How to choose, in one sentence: **start on the official `mcp` SDK's FastMCP for anything you want maximally spec-aligned and dependency-light, and reach for standalone FastMCP 2.x when you need its extra deployment/auth/composition machinery.** They are close enough that moving between them is rarely more than an import change plus a few signature tweaks.

Install the official SDK with the CLI extra (which gives you `mcp dev`, `mcp install`, and friends):

```bash
pip install "mcp[cli]"
```

Or, the idiomatic 2026 way, with [`uv`](https://docs.astral.sh/uv/) — fast, reproducible, and what the host config in §7 will invoke:

```bash
uv init my-mcp-server
cd my-mcp-server
uv add "mcp[cli]"
```

If you instead want the standalone project, it is `pip install fastmcp` (or `uv add fastmcp`) and `from fastmcp import FastMCP`. The rest changes very little.

---

## 2. A minimal server

Here is a complete, runnable server. It does one thing — exposes a single tool — but it is a *real* MCP server: a host can connect to it over stdio, list its tools, and call them.

```python
# server.py
from mcp.server.fastmcp import FastMCP

# The server's name is what shows up in the host's UI.
mcp = FastMCP("Demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    return a + b


if __name__ == "__main__":
    # Run over stdio — the host launches this process and talks to it
    # over stdin/stdout. This is the default, local transport.
    mcp.run()
```

That is the whole server. Run it directly with `python server.py` (it will sit waiting for a client on stdio), or — far more usefully during development — point the Inspector at it (§6).

The part worth slowing down on is **how `add` became a protocol-legal tool**. You wrote a plain Python function; FastMCP did the protocol work:

- **The function name `add`** becomes the tool's `name`.
- **The docstring** `"Add two integers and return their sum."` becomes the tool's `description` — the text the host's model reads to decide *whether and how* to call it. This is not a comment; it is load-bearing prompt material, so write it for the model.
- **The type hints** `a: int, b: int` are introspected and compiled into a JSON Schema `inputSchema`. The host receives, roughly:

```json
{
  "name": "add",
  "description": "Add two integers and return their sum.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "a": { "type": "integer" },
      "b": { "type": "integer" }
    },
    "required": ["a", "b"]
  }
}
```

This is the same `inputSchema` you would have hand-authored in Lecture 03 — FastMCP just derived it from the signature. Richer hints produce richer schemas: a `str` becomes `"type": "string"`, a parameter with a default (`unit: str = "celsius"`) becomes optional and drops out of `required`, a `Literal["celsius", "fahrenheit"]` becomes an `enum`, and a Pydantic model parameter expands into a nested object schema. The lesson: **the schema quality the model sees is exactly the type-annotation quality you write.** Vague hints (`a` with no type, or `dict` for a structured argument) produce a vague schema and worse tool selection.

---

## 3. Resources & prompts in FastMCP

Tools are model-controlled actions. The other two server→client primitives from Lecture 03 — **resources** (app-controlled data) and **prompts** (user-controlled templates) — get their own decorators.

A **resource** is addressable data, identified by a URI. FastMCP lets you template the URI: path parameters in the URI string map to function arguments, so one function serves a whole family of resources.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")

# Static, well-known config values keyed by name.
CONFIG = {"theme": "dark", "max_retries": "3", "region": "us-east-1"}


@mcp.resource("config://{key}")
def read_config(key: str) -> str:
    """Return the value of a single configuration key."""
    return CONFIG.get(key, f"(no such key: {key})")
```

A host reading `config://region` invokes `read_config("region")` and gets back `"us-east-1"`. The `{key}` segment in the URI scheme becomes the `key` parameter — the same hint-to-schema mechanism as tools, applied to a URI template. Use resources for data the application wants to *pull into context* (files, records, config) rather than actions the model *performs*; that controller distinction from Lecture 03 is what keeps a clean server clean.

A **prompt** is a reusable, parameterized message template the user can invoke — think of the slash-commands a host surfaces. A FastMCP prompt returns the messages that should be injected:

```python
@mcp.prompt()
def review_code(language: str, code: str) -> str:
    """A prompt that asks for a focused code review."""
    return (
        f"Please review the following {language} code for bugs, security "
        f"issues, and style. Be specific and cite line numbers.\n\n"
        f"```{language}\n{code}\n```"
    )
```

Returning a plain string produces a single user message. When you need multi-turn structure or non-user roles, return a list of message objects instead:

```python
from mcp.server.fastmcp.prompts import base


@mcp.prompt()
def debug_session(error: str) -> list[base.Message]:
    """Seed a debugging conversation with the error already in context."""
    return [
        base.UserMessage("I hit this error and need help:"),
        base.UserMessage(error),
        base.AssistantMessage("Let's work through it. What were you running?"),
    ]
```

The exact import path for the message helpers (`base.UserMessage`, etc.) is one of the things that drifts between SDK versions — if it doesn't resolve, returning a list of plain `{"role": ..., "content": ...}` dicts is the version-robust fallback.

---

## 4. The Context object & lifespan

Everything so far has been stateless and one-directional. Real servers need two more things: a way for a tool to *talk back* to the host while it runs, and a way to hold expensive shared state (a database pool, an HTTP client) across the server's life. FastMCP gives you `Context` for the first and **lifespan** for the second.

### The Context object

Declare a parameter type-hinted as `Context` and FastMCP injects it — it does **not** appear in the tool's `inputSchema`, so the model never sees it; it is purely your handle to the session. Through it, a tool can do the things you met as the *reverse primitives* in Lecture 04:

- **Log** to the host: `ctx.info(...)`, `ctx.debug(...)`, `ctx.warning(...)`, `ctx.error(...)`.
- **Report progress** for long operations: `await ctx.report_progress(current, total)`.
- **Read a resource** without the client round-tripping: `await ctx.read_resource(uri)`.
- **Sample** — borrow the host's LLM for a sub-completion: `await ctx.session.create_message(...)` (some SDK versions expose a convenience `ctx.sample(...)`).
- **Elicit** — pause and ask the user for structured input mid-task: `await ctx.elicit(...)`.

```python
from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("Demo")


@mcp.tool()
async def summarize_files(uris: list[str], ctx: Context) -> str:
    """Read several resources and return a combined, LLM-written summary."""
    await ctx.info(f"Summarizing {len(uris)} resource(s)")

    chunks: list[str] = []
    for i, uri in enumerate(uris):
        await ctx.report_progress(i, len(uris))      # progress bar in the host UI
        content, _mime = await ctx.read_resource(uri)  # server-side resource read
        chunks.append(content)

    corpus = "\n\n---\n\n".join(chunks)

    # Sampling: ask the HOST's model to do the summary. The server has no
    # API key of its own — it borrows the host's LLM (see Lecture 04).
    result = await ctx.session.create_message(
        messages=[
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"Summarize these documents in 5 bullet points:\n\n{corpus}",
                },
            }
        ],
        max_tokens=400,
    )
    return result.content.text
```

Two things to internalize. First, `ctx` is injected, not passed by the model — the host never supplies it and never sees it in the schema. Second, **sampling is what makes a server able to "use an LLM" without holding any credentials**: it does not call Anthropic (or anyone) directly; it requests a completion *from the host*, which runs its own model under its own keys and policy. That is the whole point of the reverse direction, and it is what keeps servers cheap, portable, and out of the credential-management business.

### Lifespan — shared startup/shutdown state

You do not want to open a database connection per tool call. The **lifespan** is an async context manager you hand to `FastMCP`; whatever it yields on startup is held for the server's life and torn down on shutdown, and tools reach it through the context.

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP


@dataclass
class AppState:
    """Shared resources, created once at startup."""
    db: "Database"  # e.g. an asyncpg pool, an httpx.AsyncClient, etc.


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    db = await Database.connect("postgres://localhost/app")  # open once
    try:
        yield AppState(db=db)        # everything yielded is the shared context
    finally:
        await db.close()             # closed once, on shutdown


mcp = FastMCP("Demo", lifespan=lifespan)


@mcp.tool()
async def get_user(user_id: int, ctx: Context) -> str:
    """Look up a user by id using the shared DB pool."""
    state: AppState = ctx.request_context.lifespan_context
    row = await state.db.fetch_one("SELECT name FROM users WHERE id = $1", user_id)
    return row["name"] if row else "(not found)"
```

The shape to remember: **open before `yield`, close in the `finally`, reach the yielded object via `ctx.request_context.lifespan_context`.** This is also where startup failures should surface loudly — a server that can't reach its database should fail in the lifespan, not silently degrade on every tool call. (The exact attribute path to the lifespan context is another version-sensitive spot; if `ctx.request_context.lifespan_context` doesn't resolve, check the `Context` docstring in your installed version.)

---

## 5. Structured output

By default a tool returns text. But hosts increasingly want **structured, typed** results they can parse reliably — and the model benefits from knowing the shape in advance. Return a Pydantic model (or a typed `dict`/`TypedDict`) and the SDK does two things: it emits an `outputSchema` alongside the tool definition, and it returns *structured content* the host can consume as data rather than re-parsing a string.

```python
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


class Weather(BaseModel):
    """A structured weather report."""
    location: str
    temperature_c: float = Field(description="Temperature in degrees Celsius")
    conditions: str
    humidity_pct: int


@mcp.tool()
def get_weather(city: str) -> Weather:
    """Get the current weather for a city as a structured report."""
    # (a real implementation would call a weather API here)
    return Weather(
        location=city,
        temperature_c=18.5,
        conditions="partly cloudy",
        humidity_pct=64,
    )
```

Because the return annotation is `Weather`, the tool definition now carries an `outputSchema` derived from the model — the mirror image of the `inputSchema` derivation in §2 — and the host receives the result as a validated object with named fields, not a blob it has to scrape. Field-level `Field(description=...)` annotations flow into the schema too, so you can document each field the way you documented each parameter. Prefer this for anything downstream code (or another tool) will consume; reserve plain-string returns for genuinely free-form text.

---

## 6. Testing with MCP Inspector

Before you wire a server into a host, test it in isolation. The **MCP Inspector** is the official interactive test/debug UI — a web app that launches your server, speaks the protocol to it, and lets you click through everything it exposes. No host, no model, no config file: just you and the server.

Run it against your server with `uv` (no global install needed):

```bash
npx @modelcontextprotocol/inspector uv run server.py
```

If you installed with plain `pip`, swap the command: `npx @modelcontextprotocol/inspector python server.py`. The official SDK also ships a shortcut — `mcp dev server.py` — which launches the same Inspector wired to your server.

Once it opens, work through this checklist — it maps one-to-one onto the lifecycle and primitives from Lectures 02–04:

- **Capabilities.** On connect, confirm the server reports the capabilities you expect (tools / resources / prompts). This is the `initialize` handshake from Lecture 02, made visible.
- **`tools/list`.** Open the Tools tab and verify each tool's name, description, and — critically — its generated `inputSchema`. This is where a missing or wrong type hint shows up as a bad schema.
- **Call a tool.** Fill in arguments and invoke it. Check the result *and* the structured output / `outputSchema` if you returned a model (§5). Watch the log pane for any `ctx.info(...)` messages and progress updates.
- **Read a resource.** Open the Resources tab, resolve a templated URI (e.g. `config://region`), and confirm the content and MIME type.
- **Run a prompt.** Invoke a prompt with arguments and inspect the messages it produces.

If it is green in the Inspector, it will behave in a host. If it is *not* green here, debugging it inside Claude Desktop — where failures are swallowed and surface only as "the tool didn't work" — is far more painful. **Inspector first, host second** is the workflow that saves you hours.

---

## 7. Wiring into a host & packaging

A host discovers your server through a small JSON config block that tells it *what command to run*. For a stdio server, that is the command and its arguments, plus any environment the server needs. Here is a Claude Desktop / Claude Code config (`claude_desktop_config.json` on Desktop; the equivalent MCP config for Claude Code):

```json
{
  "mcpServers": {
    "demo": {
      "command": "uv",
      "args": ["--directory", "/abs/path/to/my-mcp-server", "run", "server.py"],
      "env": {
        "WEATHER_API_KEY": "sk-..."
      }
    }
  }
}
```

The host launches `uv --directory ... run server.py`, speaks MCP over the process's stdio, and surfaces the server's tools, resources, and prompts in its UI. The `env` block is how secrets reach a *local* server — note this is the stdio/local story; **remote servers authenticate completely differently, over OAuth 2.1, and that is the entire subject of Lecture 06.** Do not try to ship a secret in an env var to a server running on someone else's machine.

The official SDK gives you a one-liner that writes this block for you during development:

```bash
mcp install server.py
```

For **distribution** to other people, the 2026 norms are: publish the server as a package and let users run it with `uvx your-server` (zero-install execution, the analog of `npx`) or `pip install your-server`; ship a `pyproject.toml` with a console-script entry point so the command is stable; and — for public discoverability — list it in the **MCP registry**. Packaging conventions and the registry are their own topic, covered in **Lecture 08**.

### The TypeScript equivalent

The same protocol, the same mental model, different language. The TS SDK's high-level server registers a tool with a [Zod](https://zod.dev/) schema standing in for Python's type hints:

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "Demo", version: "1.0.0" });

server.registerTool(
  "add",
  {
    description: "Add two integers and return their sum.",
    inputSchema: { a: z.number().int(), b: z.number().int() },
  },
  async ({ a, b }) => ({ content: [{ type: "text", text: String(a + b) }] }),
);

await server.connect(new StdioServerTransport());
```

Same three ingredients as the Python `add`: a name, a description for the model, and a schema (Zod here, inferred from type hints there). The host config is identical in shape — just point `command`/`args` at `node build/server.js` instead of `uv run server.py`.

---

You now have the full local build loop: write a FastMCP server with tools, resources, and prompts; reach back to the host through `Context` for logging, progress, sampling, and elicitation; hold shared state in a lifespan; return Pydantic models for structured output; test it green in the Inspector; and wire it into a host. Everything to this point has run on `localhost` over stdio. **Lecture 06** takes the same server to the internet — Streamable HTTP, sessions, and the OAuth 2.1 authorization that local env vars cannot provide.

---

## Current as of

June 2026, pinned to the **2025-11-25** MCP specification revision. The official `mcp` Python SDK is heading toward a **v2 (beta in 2026)**, and both it and the standalone FastMCP 2.x evolve release-to-release — decorator names, the `Context` method set (`ctx.sample` vs `ctx.session.create_message`), the lifespan signature, and prompt-message import paths have all shifted across versions. Treat every snippet as a snapshot and verify against your installed version (`mcp version`, then the module docstrings) before relying on an exact call.
