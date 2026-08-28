"""Run the unseen-user evaluation against a simulated cohort.

    python -m examples.run_unseen_user_eval --output docs/benchmarks/unseen-users.md

The cohort is generated, not observed. This repository holds two users and zero
feedback rows, so the report below is evidence about the *pipeline* -- that it
detects an effect of the configured size, on held-out users, at the reported
power -- and not evidence that any model converts real users. The report says so
itself; do not quote a number from it without that sentence.
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
from src.model.post_training.eval.unseen_users import achieved_power


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--power-replications", type=int, default=200)
    parser.add_argument("--output", help="Markdown file to write.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = CohortConfig(
        num_users=args.users, sessions_per_user=args.sessions, seed=args.seed
    )
    report = evaluate_unseen_users(
        generate_cohort(config),
        holdout_fraction=args.holdout,
        seed=args.seed,
        resamples=args.resamples,
        provenance=(
            f"simulated cohort ({args.users} users x {args.sessions} sessions, "
            f"seed {args.seed}) -- NOT real users"
        ),
    )
    text = format_report(report)

    if args.power_replications > 0:
        power = achieved_power(
            config,
            replications=args.power_replications,
            resamples=min(args.resamples, 200),
            seed=args.seed,
        )
        rows = "\n".join(
            f"| `{name}` | {rate:.2f} |" for name, rate in sorted(power.items())
        )
        text += (
            "\n## Achieved power\n\n"
            f"Rejection rate over {args.power_replications} freshly seeded cohorts.\n\n"
            "| measurement | power |\n| --- | ---: |\n" + rows + "\n"
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
