"""Render collected PPO / GRPO rollout traces as lightweight HTML plots.

The training pipeline can already write rollout JSONL via
``save_training_batch_jsonl``.  This module turns those records into a
shareable, screenshot-friendly trace view without extra plotting dependencies.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


_STYLE = """
:root {
  color-scheme: dark;
  --bg: #282d35;
  --fg: #cfd6df;
  --muted: #9aa6b2;
  --tag: #e06c75;
  --value: #d5dee9;
  --panel: #20242b;
  --border: #3a414c;
}
body {
  margin: 0;
  background: #111318;
  color: var(--fg);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.trace {
  background: var(--bg);
  border: 1px solid var(--border);
  padding: 16px;
  line-height: 1.45;
  font-size: 13px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.meta { color: var(--muted); font-weight: 700; }
.tag { color: var(--tag); font-weight: 700; }
.value { color: var(--value); }
.obs { color: var(--fg); }
.record {
  max-width: 1280px;
  margin: 24px auto;
  padding: 0 16px;
}
.summary {
  background: var(--panel);
  border: 1px solid var(--border);
  border-bottom: 0;
  padding: 10px 16px;
  color: var(--muted);
}
"""


def load_rollout_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records produced by ``save_training_batch_jsonl``."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _tag(name: str, content: str) -> str:
    escaped_name = html.escape(name)
    escaped_content = html.escape(content)
    return (
        f'<span class="tag">&lt;{escaped_name}&gt;</span>'
        f'<span class="value">{escaped_content}</span>'
        f'<span class="tag">&lt;/{escaped_name}&gt;</span>'
    )


def _observation(text: str | None) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        return html.escape(stripped)
    return _tag("information", stripped)


def rollout_record_to_html(record: dict[str, Any]) -> str:
    """Render one saved rollout record to an HTML fragment."""
    trajectory = record.get("trajectory") or {}
    question = trajectory.get("question") or record.get("question") or ""
    reward = record.get("reward")
    advantage = record.get("advantage")
    final_answer = trajectory.get("final_answer")

    lines: list[str] = [
        f'<span class="meta">Question:</span> {html.escape(str(question))}',
    ]
    ground_truth = record.get("ground_truth") or record.get("ground_truths")
    if ground_truth:
        lines.append(
            f'<span class="meta">Ground Truth:</span> {html.escape(str(ground_truth))}'
        )
    if reward is not None:
        lines.append(
            f'<span class="meta">Reward:</span> {html.escape(f"{float(reward):.3f}")}'
        )
    if advantage is not None:
        lines.append(
            '<span class="meta">Advantage:</span> '
            f"{html.escape(f'{float(advantage):+.3f}')}"
        )
    lines.append("")

    for step in trajectory.get("steps", []):
        action_type = str(step.get("action_type") or "invalid")
        action_value = str(step.get("action_value") or "")
        if action_type != "invalid":
            lines.append(_tag(action_type, action_value))
        else:
            lines.append(_tag("invalid", action_value or "invalid action"))
        obs = _observation(step.get("observation"))
        if obs:
            lines.append(f'<span class="obs">{obs}</span>')
        lines.append("")

    if final_answer:
        lines.append(_tag("answer", str(final_answer)))
    return "\n".join(lines).strip()


def render_rollout_html(
    records: list[dict[str, Any]],
    *,
    title: str = "Rollout Trace",
    max_records: int | None = None,
) -> str:
    """Render rollout JSONL records to a complete HTML document."""
    selected = records[:max_records] if max_records is not None else records
    rendered_records = []
    for idx, record in enumerate(selected):
        group_id = record.get("group_id", "")
        rollout_index = record.get("rollout_index", idx)
        summary = (
            f"Record {idx}"
            f" | group={html.escape(str(group_id))}"
            f" | rollout={html.escape(str(rollout_index))}"
        )
        rendered_records.append(
            '<section class="record">'
            f'<div class="summary">{summary}</div>'
            f'<div class="trace">{rollout_record_to_html(record)}</div>'
            "</section>"
        )

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n"
        "<body>\n" + "\n".join(rendered_records) + "\n</body>\n</html>\n"
    )


def save_rollout_plot(
    jsonl_path: str | Path,
    output_path: str | Path,
    *,
    max_records: int | None = 1,
    title: str = "Rollout Trace",
) -> int:
    """Load rollout JSONL and save an HTML trace plot."""
    records = load_rollout_jsonl(jsonl_path)
    html_text = render_rollout_html(records, title=title, max_records=max_records)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    return len(records if max_records is None else records[:max_records])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="Path to rollout JSONL.")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    parser.add_argument(
        "--max_records",
        type=int,
        default=1,
        help="Number of rollout records to render. Use 0 for all records.",
    )
    parser.add_argument("--title", default="Rollout Trace")
    args = parser.parse_args()
    max_records = None if args.max_records == 0 else args.max_records
    count = save_rollout_plot(
        args.jsonl,
        args.out,
        max_records=max_records,
        title=args.title,
    )
    print(f"Rendered {count} rollout record(s) to {args.out}")


if __name__ == "__main__":
    main()
