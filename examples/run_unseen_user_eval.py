"""Run the unseen-user evaluation against a simulated cohort.

    python -m examples.run_unseen_user_eval --output docs/benchmarks/unseen-user-evaluation.md

The cohort is generated, not observed. This repository holds two users and zero
feedback rows, so the report below is evidence about the *pipeline* -- that it
detects an effect of the configured size, on held-out users, at the reported
power -- and not evidence that any model converts real users. The report says so
itself; do not quote a number from it without that sentence.

``--allowed-tools`` and ``--max-search-rounds`` are analysis constants, not
cosmetics: they must match the population being measured. Pointed at a record
set whose tool calls name this repository's real tools, the defaults would score
every record non-compliant and report 0.0 with no error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model.post_training.eval import (
    CohortConfig,
    evaluate_unseen_users,
    format_report,
    generate_cohort,
)
from src.model.post_training.eval.cohort import effect_size_summary
from src.model.post_training.eval.unseen_users import achieved_power


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=CohortConfig.num_users)
    parser.add_argument("--sessions", type=int, default=CohortConfig.sessions_per_user)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--power-replications", type=int, default=200)
    parser.add_argument(
        "--allowed-tools",
        default="search,fetch",
        help="Comma-separated tool names a parseable tool call may name.",
    )
    parser.add_argument(
        "--max-search-rounds",
        type=int,
        default=5,
        help="Round budget the round_budget_respected constraint checks against.",
    )
    parser.add_argument(
        "--baseline-label",
        default="the baseline arm of the same simulated cohort",
        help="What the trained policy is being compared against, for the report.",
    )
    parser.add_argument("--output", help="Markdown file to write.")
    return parser


def _regeneration_command(args: argparse.Namespace) -> str:
    """The exact command that reproduces this report, for the report itself."""
    parts = [
        "python -m examples.run_unseen_user_eval",
        f"--users {args.users}",
        f"--sessions {args.sessions}",
        f"--holdout {args.holdout}",
        f"--seed {args.seed}",
        f"--resamples {args.resamples}",
        f"--power-replications {args.power_replications}",
        f"--allowed-tools {args.allowed_tools}",
        f"--max-search-rounds {args.max_search_rounds}",
        f"--baseline-label '{args.baseline_label}'",
    ]
    if args.output:
        parts.append(f"--output {args.output}")
    return " \\\n  ".join(parts)


def main() -> None:
    args = _build_parser().parse_args()
    config = CohortConfig(
        num_users=args.users, sessions_per_user=args.sessions, seed=args.seed
    )
    allowed_tools = frozenset(
        name.strip() for name in args.allowed_tools.split(",") if name.strip()
    )
    report = evaluate_unseen_users(
        generate_cohort(config),
        holdout_fraction=args.holdout,
        seed=args.seed,
        resamples=args.resamples,
        allowed_tools=allowed_tools,
        max_search_rounds=args.max_search_rounds,
        baseline_label=args.baseline_label,
        provenance=(
            f"simulated cohort ({args.users} users x {args.sessions} sessions, "
            f"seed {args.seed}) -- NOT real users; {effect_size_summary(config)}"
        ),
    )
    text = format_report(report)

    if args.power_replications > 0:
        power = achieved_power(
            config,
            replications=args.power_replications,
            resamples=min(args.resamples, 200),
            seed=args.seed,
            holdout_fraction=args.holdout,
            allowed_tools=allowed_tools,
            max_search_rounds=args.max_search_rounds,
        )
        skipped = round(power.pop("skipped_replication_rate") * args.power_replications)
        rows = "\n".join(
            f"| `{name}` | {rate:.2f} |" for name, rate in sorted(power.items())
        )
        note = (
            f" ({skipped} further replications skipped: the split left no "
            "held-out user)"
            if skipped
            else ""
        )
        text += (
            "\n## Achieved power\n\n"
            f"Rejection rate against the effect sizes above, over "
            f"{args.power_replications - skipped} freshly seeded cohorts"
            f"{note}.\n\n"
            "| measurement | power |\n| --- | ---: |\n" + rows + "\n"
        )

    text += (
        "\n## Reproducing this report\n\n"
        "```bash\n" + _regeneration_command(args) + "\n```\n"
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Wrote {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
