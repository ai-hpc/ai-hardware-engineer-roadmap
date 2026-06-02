# Lab 01 — Build a Research Agent with Tool Use

**Track B · Agentic AI & GenAI** | [← Index](README.md) | [Next → Lab 02](Lab-02-Multi-Agent-Pipeline.md)

---

## Overview

In this lab you build a fully working ReAct research agent from scratch using the raw Anthropic SDK. The agent can search the web, read URLs, and save notes to disk. It persists memory between runs in a JSON file and streams its progress to the terminal.

**Estimated time:** 60–90 minutes
**Difficulty:** Intermediate

**What you will build:**

```
research_agent/
├── agent.py          ← main agent loop
├── tools.py          ← tool implementations
├── memory.py         ← persistent memory (JSON)
├── main.py           ← entry point
├── notes/            ← saved research notes (created at runtime)
└── requirements.txt
```

---

## Step 1 — Project Setup

### 1.1 Create the project directory

```bash
mkdir research_agent && cd research_agent
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 1.2 Install dependencies

```bash
pip install anthropic tavily-python python-dotenv rich
```

Create `requirements.txt`:

```
anthropic>=0.25.0
tavily-python>=0.3.0
python-dotenv>=1.0.0
rich>=13.0.0
```

### 1.3 Environment variables

Create `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...   # get free key at tavily.com
```

If you do not have a Tavily key, the mocked version in Step 2 works offline.

---

## Step 2 — Define the Tools

Create `tools.py`:

```python
# tools.py
"""
Tool implementations for the research agent.
Each tool is a regular Python function.
Mock versions are included so the lab works without API keys.
"""

import json
import os
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Notes directory ────────────────────────────────────────────────────────────
NOTES_DIR = Path("notes")
NOTES_DIR.mkdir(exist_ok=True)


# ── Tool 1: Web Search ─────────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information. Returns a formatted string of results.
    Uses Tavily API if available, otherwise returns mock data.
    """
    tavily_key = os.environ.get("TAVILY_API_KEY", "")

    if tavily_key and not tavily_key.startswith("tvly-MOCK"):
        # Real Tavily search
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True,
            )
            results = []
            if response.get("answer"):
                results.append(f"Summary: {response['answer']}\n")
            for r in response.get("results", []):
                results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL: {r.get('url', 'N/A')}\n"
                    f"Content: {r.get('content', '')[:300]}\n"
                )
            return "\n---\n".join(results) if results else "No results found."
        except Exception as e:
            return f"Search error: {e}"
    else:
        # Mock search results for offline development
        mock_data = {
            "ml compiler": (
                "Top ML Compilers in 2025:\n\n"
                "1. XLA (Accelerated Linear Algebra) — Used by JAX and TensorFlow. "
                "Compiles computation graphs to optimized CPU/GPU/TPU code. "
                "Key feature: whole-program optimization via HLO IR.\n\n"
                "2. TVM (Apache TVM) — Open-source deep learning compiler stack. "
                "Supports NVIDIA, AMD, ARM, RISC-V. Key feature: auto-tuning with Ansor.\n\n"
                "3. MLIR (Multi-Level Intermediate Representation) — LLVM-based compiler "
                "infrastructure from Google. Key feature: dialect system for hardware targets."
            ),
            "default": (
                f"Mock search results for: '{query}'\n"
                "Result 1: Example result about the query topic.\n"
                "Result 2: Another relevant source with detailed information.\n"
                "Result 3: Technical documentation with specifications."
            ),
        }
        for key, value in mock_data.items():
            if key.lower() in query.lower():
                return value
        return mock_data["default"]


# ── Tool 2: Read URL ───────────────────────────────────────────────────────────
def read_url(url: str, max_chars: int = 3000) -> str:
    """
    Fetch and return the text content of a web page.
    Strips HTML tags and returns clean text.
    """
    # Mock mode: detect fake URLs
    if "example.com" in url or "mock" in url.lower():
        return (
            f"[Mock content for {url}]\n"
            "This is example content from the URL. "
            "In production, this would contain the actual page text. "
            "The page discusses technical details about the requested topic."
        )

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Research Agent 1.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")

        # Strip HTML tags
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    except Exception as e:
        return f"Failed to read URL: {e}"


# ── Tool 3: Save Note ──────────────────────────────────────────────────────────
def save_note(title: str, content: str, tags: list[str] | None = None) -> str:
    """
    Save a research note to disk as a markdown file.
    Returns a confirmation message with the file path.
    """
    if tags is None:
        tags = []

    # Sanitize filename
    safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{safe_title[:40]}.md"
    filepath = NOTES_DIR / filename

    note_content = f"""# {title}

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Tags:** {", ".join(tags) if tags else "none"}

---

{content}
"""
    filepath.write_text(note_content, encoding="utf-8")
    return f"Note saved successfully: {filepath} ({len(content)} characters)"


# ── Tool schema (Anthropic format) ────────────────────────────────────────────
TOOL_SCHEMAS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for current information about a topic. "
            "Use this to find facts, recent news, or technical details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific for better results.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_url",
        "description": "Fetch and read the text content of a specific URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to read."},
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 3000)",
                    "default": 3000,
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "save_note",
        "description": "Save important research findings as a markdown note file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The note title."},
                "content": {"type": "string", "description": "The note content in markdown format."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of tags for organization.",
                },
            },
            "required": ["title", "content"],
        },
    },
]

# ── Tool dispatcher ────────────────────────────────────────────────────────────
def run_tool(name: str, inputs: dict) -> str:
    """Execute a tool by name and return the result as a string."""
    if name == "web_search":
        return web_search(**inputs)
    elif name == "read_url":
        return read_url(**inputs)
    elif name == "save_note":
        return save_note(**inputs)
    else:
        return f"Unknown tool: {name}"
```

---

## Step 3 — Implement the ReAct Agent Loop

Create `agent.py`:

```python
# agent.py
"""
ReAct agent loop using the raw Anthropic SDK.
ReAct = Reasoning + Acting: the model reasons, picks a tool, observes
the result, then reasons again until it decides it is done.
"""

import json
import os
from typing import Generator

import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from tools import TOOL_SCHEMAS, run_tool

console = Console()

SYSTEM_PROMPT = """You are an expert AI research assistant. Your job is to thoroughly
research topics and produce well-structured, accurate reports.

You have access to three tools:
- web_search: search for current information
- read_url: read a specific webpage
- save_note: save your findings as a markdown file

Research methodology:
1. Start with a broad search to understand the landscape.
2. Do 2-3 targeted follow-up searches for specific details.
3. Optionally read key URLs for more depth.
4. Compile findings into a well-structured note using save_note.
5. Provide a final summary to the user.

Be thorough but efficient. Cite sources when possible."""


class ResearchAgent:
    def __init__(self, max_iterations: int = 10):
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "sk-ant-fake")
        )
        self.max_iterations = max_iterations
        self.messages: list[dict] = []

    def _print_step(self, step_type: str, content: str, style: str = "white"):
        """Pretty-print an agent step to the terminal."""
        styles = {
            "thinking":   ("blue",   "Thinking"),
            "tool_call":  ("yellow", "Tool Call"),
            "tool_result":("green",  "Tool Result"),
            "answer":     ("cyan",   "Final Answer"),
            "error":      ("red",    "Error"),
        }
        color, label = styles.get(step_type, ("white", step_type))
        console.print(Panel(
            Text(content[:800] + ("..." if len(content) > 800 else ""), style=color),
            title=f"[bold {color}]{label}[/bold {color}]",
            border_style=color,
        ))

    def run(self, task: str) -> str:
        """
        Run the agent on a task. Returns the final text answer.
        Uses the ReAct loop: reason → act → observe → repeat.
        """
        console.rule(f"[bold]Research Task[/bold]")
        console.print(f"[bold cyan]{task}[/bold cyan]\n")

        # Add the user's task to the conversation
        self.messages.append({"role": "user", "content": task})

        for iteration in range(self.max_iterations):
            console.rule(f"[dim]Iteration {iteration + 1}/{self.max_iterations}[/dim]")

            # Call the model
            response = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=self.messages,
            )

            # Collect text and tool use blocks
            text_blocks = []
            tool_use_blocks = []
            for block in response.content:
                if block.type == "text":
                    text_blocks.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            # Print thinking text
            if text_blocks:
                self._print_step("thinking", "\n".join(text_blocks))

            # Add assistant response to message history
            self.messages.append({"role": "assistant", "content": response.content})

            # Check if the model is done (no tool calls)
            if response.stop_reason == "end_turn" or not tool_use_blocks:
                final_text = "\n".join(text_blocks)
                self._print_step("answer", final_text)
                return final_text

            # Execute all tool calls
            tool_results = []
            for tool_use in tool_use_blocks:
                self._print_step(
                    "tool_call",
                    f"Tool: {tool_use.name}\nInputs: {json.dumps(tool_use.input, indent=2)}"
                )

                result = run_tool(tool_use.name, tool_use.input)
                self._print_step("tool_result", result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                })

            # Add tool results to the conversation
            self.messages.append({"role": "user", "content": tool_results})

        return "Max iterations reached. Research incomplete."
```

---

## Step 4 — Add Persistent Memory

Create `memory.py`:

```python
# memory.py
"""
Persistent memory for the research agent.
Stores past research sessions in a JSON file.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

MEMORY_FILE = Path("agent_memory.json")


class AgentMemory:
    """
    Persists research summaries and key facts between agent runs.
    Stored as a JSON file with a list of research sessions.
    """

    def __init__(self, memory_file: Path = MEMORY_FILE):
        self.memory_file = memory_file
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.memory_file.exists():
            try:
                return json.loads(self.memory_file.read_text())
            except (json.JSONDecodeError, IOError):
                return {"sessions": [], "facts": {}}
        return {"sessions": [], "facts": {}}

    def _save(self):
        self.memory_file.write_text(json.dumps(self._data, indent=2))

    def save_session(self, task: str, summary: str, notes_saved: list[str] = None):
        """Record a completed research session."""
        session = {
            "id": len(self._data["sessions"]) + 1,
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "summary": summary[:500],
            "notes_saved": notes_saved or [],
        }
        self._data["sessions"].append(session)
        self._save()
        print(f"\n[Memory] Session {session['id']} saved.")

    def save_fact(self, key: str, value: str):
        """Store a key fact for future sessions."""
        self._data["facts"][key] = {
            "value": value,
            "saved_at": datetime.now().isoformat(),
        }
        self._save()

    def get_context(self, max_sessions: int = 3) -> str:
        """
        Build a context string from recent sessions.
        This is injected into the agent system prompt for continuity.
        """
        sessions = self._data["sessions"][-max_sessions:]
        facts = self._data["facts"]

        if not sessions and not facts:
            return ""

        lines = ["## Previous Research Sessions\n"]
        for s in sessions:
            lines.append(f"**Session {s['id']}** ({s['timestamp'][:10]}): {s['task']}")
            lines.append(f"Summary: {s['summary'][:200]}\n")

        if facts:
            lines.append("## Stored Facts\n")
            for key, info in facts.items():
                lines.append(f"- **{key}**: {info['value']}")

        return "\n".join(lines)

    def list_sessions(self):
        """Print all past sessions."""
        if not self._data["sessions"]:
            print("No past sessions found.")
            return
        for s in self._data["sessions"]:
            print(f"  [{s['id']}] {s['timestamp'][:16]} — {s['task'][:60]}")
```

---

## Step 5 — Test with a Real Research Task

Create `main.py`:

```python
# main.py
"""
Entry point for the research agent.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

from agent import ResearchAgent
from memory import AgentMemory

console = Console()


def run_research(task: str, show_history: bool = False):
    memory = AgentMemory()

    if show_history:
        console.print("\n[bold]Past Research Sessions:[/bold]")
        memory.list_sessions()
        print()

    # Include memory context for continuity
    context = memory.get_context()
    full_task = task
    if context:
        full_task = f"{task}\n\n[Context from previous sessions]\n{context}"

    # Run the agent
    agent = ResearchAgent(max_iterations=8)
    result = agent.run(full_task)

    # Save the session to memory
    memory.save_session(task=task, summary=result[:500])

    console.print("\n[bold green]Research complete![/bold green]")
    return result


if __name__ == "__main__":
    task = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (
            "What are the top 3 ML compilers in 2025 and their key features? "
            "Include XLA, TVM, and MLIR. Save a structured note with your findings."
        )
    )
    run_research(task, show_history=True)
```

Run the agent:

```bash
python main.py
# or with a custom task:
python main.py "What are the latest advances in transformer inference optimization?"
```

---

## Step 6 — Add Streaming Output

Streaming shows tokens as they are generated, rather than waiting for the full response. Update `agent.py` with a streaming variant:

```python
# Add to agent.py

def run_streaming(self, task: str) -> str:
    """
    Run the agent with streaming output.
    Tokens are printed to the terminal as they arrive.
    """
    console.rule("[bold]Research Task (Streaming)[/bold]")
    console.print(f"[bold cyan]{task}[/bold cyan]\n")

    self.messages.append({"role": "user", "content": task})
    full_text_parts = []

    for iteration in range(self.max_iterations):
        console.rule(f"[dim]Iteration {iteration + 1}[/dim]")
        console.print("[bold blue]Thinking...[/bold blue] ", end="")

        # Use streaming context manager
        tool_use_blocks = []
        current_tool = None
        streamed_text = []

        with self.client.messages.stream(
            model="claude-opus-4-5",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=self.messages,
        ) as stream:
            for event in stream:
                event_type = type(event).__name__

                if event_type == "RawContentBlockStartEvent":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool = {"id": block.id, "name": block.name, "input_json": ""}
                        console.print(f"\n[yellow]→ Calling tool: {block.name}[/yellow]")

                elif event_type == "RawContentBlockDeltaEvent":
                    delta = event.delta
                    if delta.type == "text_delta":
                        console.print(delta.text, end="", style="blue")
                        streamed_text.append(delta.text)
                    elif delta.type == "input_json_delta" and current_tool:
                        current_tool["input_json"] += delta.partial_json

                elif event_type == "RawContentBlockStopEvent":
                    if current_tool:
                        try:
                            current_tool["input"] = json.loads(current_tool["input_json"] or "{}")
                        except json.JSONDecodeError:
                            current_tool["input"] = {}
                        tool_use_blocks.append(current_tool)
                        current_tool = None

            # Get the final message for history
            final_message = stream.get_final_message()

        console.print()  # newline after streaming

        full_text = "".join(streamed_text)
        if full_text:
            full_text_parts.append(full_text)

        self.messages.append({"role": "assistant", "content": final_message.content})

        if final_message.stop_reason == "end_turn" or not tool_use_blocks:
            return "\n".join(full_text_parts)

        # Execute tools
        tool_results = []
        for tool in tool_use_blocks:
            result = run_tool(tool["name"], tool["input"])
            console.print(f"[green]Tool result ({tool['name']}):[/green] {result[:200]}...")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool["id"],
                "content": result,
            })

        self.messages.append({"role": "user", "content": tool_results})

    return "\n".join(full_text_parts)
```

Update `main.py` to use streaming:

```python
# Change in main.py:
result = agent.run_streaming(full_task)   # streaming version
```

---

## Expected Output

When you run the agent, you should see output like this:

```
══════════════════════════ Research Task ══════════════════════════
What are the top 3 ML compilers in 2025 and their key features?

─────────────────── Iteration 1 ────────────────────
╭─ Thinking ─────────────────────────────────────────╮
│ Let me search for information about ML compilers    │
│ to give you an accurate and current answer.        │
╰─────────────────────────────────────────────────────╯
╭─ Tool Call ─────────────────────────────────────────╮
│ Tool: web_search                                    │
│ Inputs: {                                           │
│   "query": "top ML compilers 2025 XLA TVM MLIR",   │
│   "max_results": 5                                  │
│ }                                                   │
╰─────────────────────────────────────────────────────╯
╭─ Tool Result ───────────────────────────────────────╮
│ Top ML Compilers in 2025:                           │
│ 1. XLA — whole-program optimization via HLO IR...  │
╰─────────────────────────────────────────────────────╯
[... more iterations ...]

╭─ Final Answer ──────────────────────────────────────╮
│ Based on my research, the top 3 ML compilers in     │
│ 2025 are:                                           │
│                                                     │
│ 1. XLA (Google) — key for JAX/TF, TPU support      │
│ 2. Apache TVM — widest hardware coverage            │
│ 3. MLIR — compiler infrastructure backbone          │
│                                                     │
│ I've saved a detailed note to notes/...ml_comp.md  │
╰─────────────────────────────────────────────────────╯

[Memory] Session 1 saved.
Research complete!
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `anthropic.AuthenticationError` | Bad/missing API key | Check `ANTHROPIC_API_KEY` in `.env` |
| `ToolsBetaNotAvailableError` | Wrong model | Use `claude-opus-4-5` or later |
| Agent loops forever | Tool always errors | Check `run_tool` dispatches correctly; add `print` in each tool |
| `json.JSONDecodeError` in streaming | Partial JSON accumulation bug | Ensure `input_json_delta` is accumulated before stop event |
| Mock search always returns default | Query doesn't match mock keys | Add your query keyword to `mock_data` dict in `tools.py` |
| Notes not saved | `notes/` dir missing | Check `NOTES_DIR.mkdir(exist_ok=True)` ran |
| Rich not installed | Missing dep | `pip install rich` |

---

## Extensions

Once the basic agent works, try these improvements:

1. **Web UI** — Wrap `run_streaming` in a FastAPI endpoint with SSE (see Lecture 30).
2. **More tools** — Add `calculate(expression: str)`, `read_pdf(path: str)`, or `run_python(code: str)`.
3. **Parallel tool calls** — The current loop executes tools sequentially. Try executing independent tool calls with `asyncio.gather`.
4. **Smarter memory** — Instead of prepending all sessions, embed past summaries and retrieve only the most relevant ones (RAG over your own notes).

---

*Next: [Lab 02 — Multi-Agent Code Review Pipeline](Lab-02-Multi-Agent-Pipeline.md)*
