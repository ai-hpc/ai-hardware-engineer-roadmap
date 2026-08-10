#!/usr/bin/env python3
"""Render star-history charts from the GitHub stargazers API.

GitHub restricted the stargazers endpoint to a repository's own admins and
collaborators on 2026-06-30, which is why the api.star-history.com embed now
returns a "GitHub restricted access to star data" placeholder instead of a
chart. This script rebuilds the chart in CI using the repository's own token
and writes light/dark SVGs into the repo, so the README embeds a static asset
we control rather than a live third-party call.

Standard library only — no pip install needed in the workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

API = "https://api.github.com"
PER_PAGE = 100
MAX_PAGES = 400  # GitHub caps this endpoint at 40k stargazers

WIDTH, HEIGHT = 800, 400
PAD_L, PAD_R, PAD_T, PAD_B = 64, 28, 40, 46
PLOT_W = WIDTH - PAD_L - PAD_R
PLOT_H = HEIGHT - PAD_T - PAD_B

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"

THEMES = {
    "light": {
        "bg": "#ffffff",
        "grid": "#eaeef2",
        "axis": "#d0d7de",
        "text": "#57606a",
        "title": "#1f2328",
        "line": "#0969da",
    },
    "dark": {
        "bg": "#0d1117",
        "grid": "#21262d",
        "axis": "#30363d",
        "text": "#8b949e",
        "title": "#e6edf3",
        "line": "#58a6ff",
    },
}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def fetch_stargazers(repo: str, token: str) -> list[datetime]:
    """Return every star timestamp for `repo`, oldest first."""
    stamps: list[datetime] = []
    for page in range(1, MAX_PAGES + 1):
        url = f"{API}/repos/{repo}/stargazers?per_page={PER_PAGE}&page={page}"
        request = urllib.request.Request(
            url,
            headers={
                # This media type is what makes the API return `starred_at`.
                "Accept": "application/vnd.github.star+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "generate_star_history.py",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                batch = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            if exc.code in (403, 404):
                sys.exit(
                    f"GitHub returned {exc.code} for {repo} stargazers.\n"
                    "Since 2026-06-30 this endpoint is readable only by the "
                    "repo's admins and collaborators — the token needs "
                    "read access to this repository.\n"
                    f"Response: {body}"
                )
            sys.exit(f"GitHub API error {exc.code} on page {page}: {body}")

        if not batch:
            break
        for entry in batch:
            starred_at = entry.get("starred_at")
            if starred_at:
                stamps.append(
                    datetime.strptime(starred_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                        tzinfo=timezone.utc
                    )
                )
        if len(batch) < PER_PAGE:
            break

    stamps.sort()
    return stamps


# --------------------------------------------------------------------------
# scales
# --------------------------------------------------------------------------

def nice_step(span: float, target_ticks: int = 5) -> float:
    """Round `span / target_ticks` up to a human-readable 1/2/2.5/5 x 10^n."""
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10 * magnitude


def add_months(moment: datetime, count: int) -> datetime:
    index = moment.year * 12 + (moment.month - 1) + count
    return moment.replace(year=index // 12, month=index % 12 + 1)


def month_ticks(start: datetime, end: datetime, target_ticks: int = 6):
    """Month boundaries between start and end, thinned to ~target_ticks."""
    total_months = (end.year - start.year) * 12 + (end.month - start.month)
    step = next(
        (s for s in (1, 2, 3, 4, 6, 12, 24) if total_months / s <= target_ticks), 36
    )

    boundaries = []
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    if cursor < start:
        cursor = add_months(cursor, 1)
    while cursor <= end:
        boundaries.append(cursor)
        cursor = add_months(cursor, 1)

    fmt = "%Y" if step >= 12 else "%b %Y"
    return [(tick, tick.strftime(fmt)) for tick in boundaries[::step]]


def format_count(value: float) -> str:
    count = int(round(value))
    if count >= 1000:
        return f"{count // 1000}k" if count % 1000 == 0 else f"{count / 1000:.1f}k"
    return str(count)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render(repo: str, stamps: list[datetime], theme: str, now: datetime) -> str:
    colors = THEMES[theme]
    total = len(stamps)

    start = stamps[0]
    end = max(now, stamps[-1])
    # A repo starred entirely within a few days would divide by ~zero below.
    span = max((end - start).total_seconds(), 1.0)

    def scale_x(moment: datetime) -> float:
        return PAD_L + (moment - start).total_seconds() / span * PLOT_W

    y_step = nice_step(total)
    y_max = max(math.ceil(total / y_step) * y_step, y_step)

    def scale_y(value: float) -> float:
        return PAD_T + PLOT_H - (value / y_max) * PLOT_H

    # Cumulative series: star i lands at count i+1. Anchor the line at the
    # first star and carry it flat to "now" so the chart reaches today.
    points = [(scale_x(start), scale_y(0))]
    points += [(scale_x(stamp), scale_y(i + 1)) for i, stamp in enumerate(stamps)]
    points.append((scale_x(end), scale_y(total)))

    # Keep the SVG small on very popular repos; endpoints always survive.
    if len(points) > 1500:
        stride = len(points) // 1500 + 1
        points = [points[0]] + points[1:-1:stride] + [points[-1]]

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{escape(FONT)}" role="img" '
        f'aria-label="Star history for {escape(repo)}: {total} stars">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{colors["bg"]}"/>',
        '<defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{colors["line"]}" stop-opacity="0.22"/>'
        f'<stop offset="100%" stop-color="{colors["line"]}" stop-opacity="0"/>'
        "</linearGradient></defs>",
    ]

    # horizontal grid + y labels
    ticks = int(round(y_max / y_step))
    for i in range(ticks + 1):
        value = y_step * i
        y = scale_y(value)
        out.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L + PLOT_W}" y2="{y:.1f}" '
            f'stroke="{colors["grid"]}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{PAD_L - 12}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="{colors["text"]}">{format_count(value)}</text>'
        )

    # vertical grid + x labels
    for moment, label in month_ticks(start, end):
        x = scale_x(moment)
        out.append(
            f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{PAD_T + PLOT_H}" '
            f'stroke="{colors["grid"]}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{PAD_T + PLOT_H + 22}" text-anchor="middle" '
            f'font-size="12" fill="{colors["text"]}">{escape(label)}</text>'
        )

    # baseline
    out.append(
        f'<line x1="{PAD_L}" y1="{PAD_T + PLOT_H}" x2="{PAD_L + PLOT_W}" '
        f'y2="{PAD_T + PLOT_H}" stroke="{colors["axis"]}" stroke-width="1"/>'
    )

    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    baseline = PAD_T + PLOT_H
    out.append(
        f'<polygon points="{points[0][0]:.1f},{baseline:.1f} {path} '
        f'{points[-1][0]:.1f},{baseline:.1f}" fill="url(#area)"/>'
    )
    out.append(
        f'<polyline points="{path}" fill="none" stroke="{colors["line"]}" '
        'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    out.append(
        f'<circle cx="{points[-1][0]:.1f}" cy="{points[-1][1]:.1f}" r="3.5" '
        f'fill="{colors["line"]}"/>'
    )

    out.append(
        f'<text x="{PAD_L}" y="24" font-size="13" font-weight="600" '
        f'fill="{colors["title"]}">{escape(repo)}</text>'
    )
    out.append(
        f'<text x="{PAD_L + PLOT_W}" y="24" text-anchor="end" font-size="13" '
        f'fill="{colors["text"]}">{total} stars</text>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_empty(repo: str, theme: str) -> str:
    colors = THEMES[theme]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{escape(FONT)}" role="img" '
        f'aria-label="No stars yet for {escape(repo)}">'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{colors["bg"]}"/>'
        f'<text x="{WIDTH // 2}" y="{HEIGHT // 2}" text-anchor="middle" '
        f'font-size="15" fill="{colors["text"]}">No stars yet</text>'
        "</svg>\n"
    )


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--out-dir", default="Assets/star-history")
    parser.add_argument(
        "--token",
        default=os.environ.get("STAR_HISTORY_TOKEN") or os.environ.get("GITHUB_TOKEN"),
        help="defaults to $STAR_HISTORY_TOKEN, then $GITHUB_TOKEN",
    )
    args = parser.parse_args()

    if not args.token:
        sys.exit("No token: set STAR_HISTORY_TOKEN or GITHUB_TOKEN.")

    stamps = fetch_stargazers(args.repo, args.token)
    now = datetime.now(timezone.utc)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        svg = render(args.repo, stamps, theme, now) if stamps else render_empty(args.repo, theme)
        (out_dir / f"star-history-{theme}.svg").write_text(svg, encoding="utf-8")

    print(f"Wrote {len(stamps)} stars to {out_dir}/star-history-{{light,dark}}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
