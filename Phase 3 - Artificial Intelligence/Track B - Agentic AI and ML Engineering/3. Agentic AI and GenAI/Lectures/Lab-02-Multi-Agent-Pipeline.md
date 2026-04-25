# Lab 02 — Multi-Agent Code Review Pipeline

**Track B · Agentic AI & GenAI** | [← Index](README.md) | [Next → Lab 03](Lab-03-Production-RAG.md)

---

## Overview

In this lab you build a four-agent pipeline that takes a coding task and produces reviewed, documented code. Each agent has a single responsibility, and the pipeline includes a QA gate that loops the Coder and Reviewer if critical bugs are found.

```
User Task
    │
    ▼
┌──────────┐
│ Planner  │  Breaks the task into a spec & implementation plan
└──────────┘
    │ plan
    ▼
┌──────────┐
│  Coder   │  Writes Python code from the plan          ◄──┐
└──────────┘                                               │
    │ code                                                  │ retry (max 2x)
    ▼                                                       │
┌──────────┐   critical bugs?                              │
│ Reviewer │  ─────────────────────────────────────────────┘
└──────────┘
    │ approved code
    ▼
┌───────────┐
│ DocWriter │  Adds docstrings + README
└───────────┘
    │
    ▼
 output.md
```

**Estimated time:** 60–90 minutes
**Difficulty:** Intermediate

**What you will build:**

```
code_review_pipeline/
├── agents.py       ← individual agent functions
├── pipeline.py     ← orchestrator with QA gate
├── main.py         ← entry point
├── output/         ← saved results (created at runtime)
└── requirements.txt
```

---

## Step 1 — Project Setup

```bash
mkdir code_review_pipeline && cd code_review_pipeline
python -m venv .venv && source .venv/bin/activate
pip install openai python-dotenv rich
```

Create `requirements.txt`:

```
openai>=1.30.0
python-dotenv>=1.0.0
rich>=13.0.0
```

Create `.env`:

```
OPENAI_API_KEY=sk-...
```

---

## Step 2 — Implement Each Agent

Create `agents.py`:

```python
# agents.py
"""
Individual agent functions for the code review pipeline.
Each agent:
  - Accepts a specific input (task description, plan, code, etc.)
  - Returns a structured string output
  - Has a focused system prompt
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-fake"))

DEFAULT_MODEL = "your-fast-model-id"  # swap to "your-smart-model-id" for higher quality


def _call(system: str, user: str, model: str = DEFAULT_MODEL, temperature: float = 0.2) -> str:
    """Shared helper for single-turn LLM calls."""
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


# ── Agent 1: Planner ──────────────────────────────────────────────────────────
PLANNER_SYSTEM = """You are a senior software architect. Given a coding task, produce a
concise implementation plan.

Your output must follow this exact format:
## Task Summary
<one sentence summary>

## Requirements
- <requirement 1>
- <requirement 2>
...

## Implementation Plan
1. <step 1>
2. <step 2>
...

## Edge Cases to Handle
- <edge case 1>
- <edge case 2>
...

## Expected Interface
```python
<show the public API — class name, method signatures, and docstrings>
```
"""

def planner_agent(task: str) -> str:
    """
    Breaks a coding task into a detailed implementation spec.
    Returns: structured plan string.
    """
    return _call(PLANNER_SYSTEM, f"Task: {task}", temperature=0.1)


# ── Agent 2: Coder ────────────────────────────────────────────────────────────
CODER_SYSTEM = """You are an expert Python developer. Given an implementation plan,
write clean, correct, production-quality Python code.

Rules:
1. Follow the plan exactly.
2. Include proper type hints.
3. Add brief inline comments for non-obvious logic.
4. Handle all edge cases mentioned in the plan.
5. Do NOT include docstrings yet — the Doc Writer will add those.
6. Output ONLY a Python code block — no explanation text outside the block.

Output format:
```python
# Your code here
```
"""

def coder_agent(plan: str, feedback: str = "") -> str:
    """
    Writes Python code from a plan. Accepts optional reviewer feedback for fixes.
    Returns: Python code block string.
    """
    user_message = f"Implementation Plan:\n{plan}"
    if feedback:
        user_message += f"\n\nReviewer Feedback (must fix):\n{feedback}"
    return _call(CODER_SYSTEM, user_message, temperature=0.1)


# ── Agent 3: Reviewer ─────────────────────────────────────────────────────────
REVIEWER_SYSTEM = """You are a strict code reviewer. Analyze Python code for bugs,
security issues, and correctness. Be specific and actionable.

Output your review in this exact format:

## Verdict
APPROVED | NEEDS_FIXES

## Critical Bugs (must fix before approval)
- <bug description and line reference if possible>
(or "None" if no critical bugs)

## Minor Issues (nice to fix)
- <issue>
(or "None")

## Security Concerns
- <concern>
(or "None")

## Positive Aspects
- <what was done well>

A verdict of NEEDS_FIXES requires at least one critical bug listed.
A verdict of APPROVED means the code is correct and safe to ship.
"""

def reviewer_agent(code: str, plan: str) -> dict:
    """
    Reviews code against the original plan.
    Returns: dict with 'verdict' ('APPROVED'|'NEEDS_FIXES'), 'full_review', 'feedback'.
    """
    user_message = (
        f"Original Plan:\n{plan}\n\n"
        f"Code to Review:\n{code}"
    )
    review_text = _call(REVIEWER_SYSTEM, user_message, temperature=0.0)

    # Parse verdict
    verdict = "NEEDS_FIXES"
    if "## Verdict" in review_text:
        verdict_line = review_text.split("## Verdict")[1].split("\n")[1].strip().upper()
        if "APPROVED" in verdict_line:
            verdict = "APPROVED"

    # Extract critical bugs as feedback for the coder
    feedback = ""
    if "## Critical Bugs" in review_text:
        section = review_text.split("## Critical Bugs")[1].split("##")[0].strip()
        if section.lower() != "none":
            feedback = section

    return {
        "verdict": verdict,
        "full_review": review_text,
        "feedback": feedback,
    }


# ── Agent 4: Doc Writer ───────────────────────────────────────────────────────
DOC_WRITER_SYSTEM = """You are a technical writer and Python expert. Given Python code,
add comprehensive documentation.

Your job:
1. Add Google-style docstrings to every class and public method.
2. Add module-level docstring at the top.
3. Do NOT change any logic — only add/improve comments and docstrings.
4. After the code, add a ## Usage Example section showing how to use the class.

Output format:
```python
# fully documented code
```

## Usage Example
```python
# example usage
```
"""

def doc_writer_agent(code: str, task: str) -> str:
    """
    Adds docstrings and a usage example to approved code.
    Returns: documented code + usage example string.
    """
    user_message = f"Original Task: {task}\n\nCode:\n{code}"
    return _call(DOC_WRITER_SYSTEM, user_message, temperature=0.1)
```

---

## Step 3 — Build the Pipeline Orchestrator

Create `pipeline.py`:

```python
# pipeline.py
"""
Orchestrator for the four-agent code review pipeline.
Manages the QA gate that loops coder → reviewer up to MAX_RETRIES times.
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from agents import planner_agent, coder_agent, reviewer_agent, doc_writer_agent

console = Console()
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_RETRIES = 2  # maximum coder → reviewer loops


@dataclass
class PipelineResult:
    task: str
    plan: str
    final_code: str
    review: dict
    documented_output: str
    iterations: int
    total_time_s: float
    output_file: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


def extract_code_block(text: str) -> str:
    """Extract Python code from a markdown code block."""
    pattern = r"```python\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return raw text if no block found
    return text.strip()


def _print_agent_output(agent_name: str, output: str, style: str = "white"):
    """Print an agent's output in a styled panel."""
    styles = {
        "Planner":    "blue",
        "Coder":      "yellow",
        "Reviewer":   "magenta",
        "DocWriter":  "cyan",
    }
    color = styles.get(agent_name, style)
    console.print(Panel(
        output[:1500] + ("..." if len(output) > 1500 else ""),
        title=f"[bold {color}]{agent_name}[/bold {color}]",
        border_style=color,
    ))


def run_pipeline(task: str, save_output: bool = True) -> PipelineResult:
    """
    Run the complete four-agent pipeline on a coding task.

    Args:
        task: The coding task description.
        save_output: If True, save the final output to a markdown file.

    Returns:
        PipelineResult with all intermediate outputs and metadata.
    """
    start_time = time.perf_counter()
    console.rule(f"[bold]Pipeline Started[/bold]")
    console.print(f"[bold white]Task:[/bold white] {task}\n")

    try:
        # ── Stage 1: Planner ───────────────────────────────────────────────────
        console.print("[bold blue]Stage 1/4: Planning...[/bold blue]")
        t0 = time.perf_counter()
        plan = planner_agent(task)
        _print_agent_output("Planner", plan)
        console.print(f"[dim]  Planner took {time.perf_counter() - t0:.1f}s[/dim]\n")

        # ── Stage 2+3: Coder → Reviewer QA loop ───────────────────────────────
        code_raw = ""
        review_result = {}
        iteration = 0
        feedback = ""

        for attempt in range(MAX_RETRIES + 1):
            iteration = attempt + 1
            console.print(f"[bold yellow]Stage 2/4: Coding (attempt {iteration}/{MAX_RETRIES + 1})...[/bold yellow]")
            t0 = time.perf_counter()
            code_raw = coder_agent(plan, feedback=feedback)
            _print_agent_output("Coder", code_raw)
            console.print(f"[dim]  Coder took {time.perf_counter() - t0:.1f}s[/dim]\n")

            console.print(f"[bold magenta]Stage 3/4: Reviewing (attempt {iteration})...[/bold magenta]")
            t0 = time.perf_counter()
            review_result = reviewer_agent(code_raw, plan)
            _print_agent_output("Reviewer", review_result["full_review"])
            console.print(f"[dim]  Reviewer took {time.perf_counter() - t0:.1f}s[/dim]\n")

            verdict = review_result["verdict"]
            console.print(
                f"[bold]Verdict:[/bold] "
                + (f"[green]APPROVED[/green]" if verdict == "APPROVED" else f"[red]NEEDS_FIXES[/red]")
            )

            if verdict == "APPROVED":
                console.print(f"[green]Code approved on attempt {iteration}![/green]\n")
                break
            elif attempt < MAX_RETRIES:
                feedback = review_result["feedback"]
                console.print(
                    f"[yellow]Sending feedback to Coder (retry {attempt + 1}/{MAX_RETRIES})...[/yellow]\n"
                )
            else:
                console.print(
                    f"[red]Max retries ({MAX_RETRIES}) reached. Proceeding with best effort code.[/red]\n"
                )

        # ── Stage 4: Doc Writer ────────────────────────────────────────────────
        console.print("[bold cyan]Stage 4/4: Writing documentation...[/bold cyan]")
        t0 = time.perf_counter()
        documented_output = doc_writer_agent(code_raw, task)
        _print_agent_output("DocWriter", documented_output)
        console.print(f"[dim]  DocWriter took {time.perf_counter() - t0:.1f}s[/dim]\n")

        total_time = time.perf_counter() - start_time

        # ── Save output ────────────────────────────────────────────────────────
        output_file = None
        if save_output:
            output_file = _save_output(task, plan, code_raw, review_result, documented_output)

        result = PipelineResult(
            task=task,
            plan=plan,
            final_code=extract_code_block(code_raw),
            review=review_result,
            documented_output=documented_output,
            iterations=iteration,
            total_time_s=round(total_time, 1),
            output_file=output_file,
            success=True,
        )

        _print_summary(result)
        return result

    except Exception as e:
        total_time = time.perf_counter() - start_time
        console.print(f"[bold red]Pipeline Error:[/bold red] {e}")
        return PipelineResult(
            task=task,
            plan="",
            final_code="",
            review={},
            documented_output="",
            iterations=0,
            total_time_s=round(total_time, 1),
            success=False,
            error=str(e),
        )


def _save_output(
    task: str,
    plan: str,
    code: str,
    review: dict,
    documented: str,
) -> str:
    """Save the pipeline output as a markdown file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_task = re.sub(r"[^\w\s-]", "", task)[:30].strip().replace(" ", "_").lower()
    filename = f"{timestamp}_{safe_task}.md"
    filepath = OUTPUT_DIR / filename

    verdict = review.get("verdict", "UNKNOWN")
    verdict_badge = "APPROVED" if verdict == "APPROVED" else "NEEDS_FIXES"

    content = f"""# Code Review Pipeline Output

**Task:** {task}
**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Review Verdict:** {verdict_badge}

---

## Implementation Plan

{plan}

---

## Review

{review.get('full_review', 'N/A')}

---

## Final Documented Code

{documented}
"""
    filepath.write_text(content, encoding="utf-8")
    console.print(f"[dim]Output saved → {filepath}[/dim]")
    return str(filepath)


def _print_summary(result: PipelineResult):
    """Print a summary table after the pipeline completes."""
    table = Table(title="Pipeline Summary", show_header=True)
    table.add_column("Stage", style="bold")
    table.add_column("Result")

    verdict = result.review.get("verdict", "N/A")
    table.add_row("Planner", "Done")
    table.add_row("Coder", f"{result.iterations} iteration(s)")
    table.add_row("Reviewer", f"[green]{verdict}[/green]" if verdict == "APPROVED" else f"[red]{verdict}[/red]")
    table.add_row("DocWriter", "Done")
    table.add_row("Total time", f"{result.total_time_s}s")
    if result.output_file:
        table.add_row("Output file", result.output_file)

    console.print(table)
```

---

## Step 4 — QA Gate Logic Explained

The QA gate is the loop in `run_pipeline`:

```python
for attempt in range(MAX_RETRIES + 1):          # 0, 1, 2 → up to 3 total tries
    code = coder_agent(plan, feedback=feedback)  # code improves each attempt
    review = reviewer_agent(code, plan)

    if review["verdict"] == "APPROVED":
        break                                    # exit early if approved
    elif attempt < MAX_RETRIES:
        feedback = review["feedback"]            # pass bugs back to coder
    else:
        pass  # proceed with best-effort code after MAX_RETRIES
```

Key design decisions:
- `feedback` is extracted as a clean bullet list of critical bugs (not the full review).
- The `plan` is always passed to the `reviewer_agent` so it can check conformance to spec.
- `MAX_RETRIES = 2` prevents infinite loops while allowing meaningful correction cycles.

---

## Step 5 — Test with the LRU Cache Task

Create `main.py`:

```python
# main.py
import os
from dotenv import load_dotenv
from pipeline import run_pipeline

load_dotenv()

TASKS = {
    "lru_cache": (
        "Write a Python LRU (Least Recently Used) cache class. "
        "Requirements: O(1) get and put operations, configurable capacity, "
        "evict the least recently used item when capacity is exceeded, "
        "thread-safe using a lock, and a stats() method that returns "
        "hit_count, miss_count, and hit_rate."
    ),
    "rate_limiter": (
        "Write a Python token bucket rate limiter class. "
        "Requirements: configurable requests_per_second, is_allowed(client_id) method "
        "that returns True/False, automatic bucket refill over time, "
        "and a reset(client_id) method."
    ),
    "binary_search_tree": (
        "Write a Python binary search tree with insert, search, "
        "in-order traversal, and delete operations. Handle all edge cases."
    ),
}

if __name__ == "__main__":
    task = TASKS["lru_cache"]
    result = run_pipeline(task, save_output=True)

    if result.success:
        print(f"\nFinal code extracted ({len(result.final_code)} chars)")
        print(f"Review verdict: {result.review.get('verdict')}")
    else:
        print(f"Pipeline failed: {result.error}")
```

Run it:

```bash
python main.py
```

---

## Step 6 — View the Saved Output

After running, the pipeline saves a markdown file in `output/`. It looks like this:

```markdown
# Code Review Pipeline Output

**Task:** Write a Python LRU cache class...
**Generated:** 2025-04-03 14:22:01
**Review Verdict:** APPROVED

---

## Implementation Plan

## Task Summary
Implement an O(1) LRU cache with thread safety and statistics tracking.

## Requirements
- O(1) get and put operations using OrderedDict or doubly-linked list + hashmap
...

---

## Review

## Verdict
APPROVED

## Critical Bugs
None

## Minor Issues
- Consider adding a __repr__ method for easier debugging

---

## Final Documented Code

```python
"""LRU Cache implementation with O(1) operations and thread safety."""

from collections import OrderedDict
from threading import Lock
from typing import Any, Optional


class LRUCache:
    """
    A thread-safe Least Recently Used (LRU) cache with O(1) operations.

    Uses an OrderedDict to maintain insertion/access order.
    The least recently used item is at the front (beginning) of the dict.

    Args:
        capacity: Maximum number of items to store.

    Example:
        >>> cache = LRUCache(capacity=3)
        >>> cache.put("a", 1)
        >>> cache.get("a")
        1
    """
    ...
```
```

---

## Complete Working Code Reference

Here is the complete, minimal working version with no external dependencies beyond OpenAI:

```python
# standalone_pipeline.py — complete version in a single file

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-fake"))
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def llm(system: str, user: str) -> str:
    r = client.chat.completions.create(
        model="your-fast-model-id",
        temperature=0.1,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content.strip()

def extract_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def planner(task: str) -> str:
    return llm(
        "You are a software architect. Create a detailed implementation plan for the task.",
        f"Task: {task}"
    )

def coder(plan: str, feedback: str = "") -> str:
    msg = f"Plan:\n{plan}"
    if feedback:
        msg += f"\n\nFix these bugs:\n{feedback}"
    return llm(
        "Write clean Python code from the plan. Output ONLY a ```python ... ``` block.",
        msg
    )

def reviewer(code: str, plan: str) -> dict:
    review = llm(
        "Review this Python code. Reply with: VERDICT: APPROVED or NEEDS_FIXES. "
        "Then list CRITICAL_BUGS: (or 'none'). Be concise.",
        f"Plan:\n{plan}\n\nCode:\n{code}"
    )
    verdict = "APPROVED" if "APPROVED" in review.upper() else "NEEDS_FIXES"
    feedback = ""
    if "CRITICAL_BUGS:" in review:
        feedback = review.split("CRITICAL_BUGS:")[1].strip()
        if feedback.lower().startswith("none"):
            feedback = ""
    return {"verdict": verdict, "full_review": review, "feedback": feedback}

def doc_writer(code: str, task: str) -> str:
    return llm(
        "Add Google-style docstrings to this code. Output ```python ... ``` then a ## Usage Example.",
        f"Task: {task}\n\nCode:\n{code}"
    )

def run(task: str, max_retries: int = 2) -> dict:
    print(f"\n{'='*60}\nTask: {task}\n{'='*60}")
    plan = planner(task)
    print(f"\n[Planner]\n{plan[:300]}...")

    code, review, iteration = "", {}, 0
    feedback = ""
    for attempt in range(max_retries + 1):
        iteration = attempt + 1
        code = coder(plan, feedback)
        print(f"\n[Coder — attempt {iteration}]\n{code[:300]}...")
        review = reviewer(code, plan)
        print(f"\n[Reviewer]\nVerdict: {review['verdict']}")
        if review["verdict"] == "APPROVED":
            break
        feedback = review["feedback"]
        print(f"Feedback: {feedback[:200]}")

    documented = doc_writer(extract_code(code), task)
    print(f"\n[DocWriter]\n{documented[:300]}...")

    # Save output
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"{ts}_output.md"
    out.write_text(
        f"# Pipeline Output\n\n## Task\n{task}\n\n"
        f"## Plan\n{plan}\n\n## Review\n{review['full_review']}\n\n"
        f"## Documented Code\n{documented}\n"
    )
    print(f"\nSaved: {out}")
    return {"plan": plan, "code": extract_code(code), "review": review, "documented": documented}

if __name__ == "__main__":
    run(
        "Write a Python LRU cache class with O(1) get/put, "
        "configurable capacity, thread safety, and a stats() method."
    )
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `AuthenticationError` | Bad API key | Check `OPENAI_API_KEY` in `.env` |
| Agent always returns `NEEDS_FIXES` | Reviewer too strict | Add "Be lenient on style issues" to reviewer system prompt |
| Code block not extracted | Model didn't wrap code in ``` | Improve coder system prompt: "You MUST wrap code in a ```python block" |
| Pipeline runs too long | Slow model | Use `your-fast-model-id` for all agents; switch planner to `your-smart-model-id` only |
| Output file not created | `output/` dir missing | `OUTPUT_DIR.mkdir(exist_ok=True)` should handle this |

---

*Next: [Lab 03 — Production RAG System](Lab-03-Production-RAG.md)*
