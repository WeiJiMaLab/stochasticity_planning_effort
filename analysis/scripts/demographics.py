#!/usr/bin/env python3
"""
Extract per-experiment participant demographics (N, gender breakdown, age) from
the raw survey dumps, for the Methods section of the manuscript.

The input data is NOT in this repository and is not on OSF. Demographic survey
responses are identifiable, so the anonymization step that produces
analysis/data/raw/ strips them, and the un-anonymized dumps this script reads
are withheld. They are not needed to reproduce any figure or analysis -- only
this one summary. Contact the authors if you need access.

Expects RAW_<TYPE>.json dumps (one per experiment: R, V, T) in
analysis/data/demographic/, or another directory via --raw-json.

Reproduces the reported demographics with no manual exclusion list: the
pilot / incomplete-session filter alone yields exactly N = 100 per experiment.

Writes analysis/workflows/figures/empirical/demographics.json (--out) and the
\\pgfval macros the Methods cites (--tex).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

DEFAULT_RAW_JSON_DIR = Path(__file__).resolve().parent.parent / "data" / "demographic"
DEFAULT_OUT_PATH = (
    Path(__file__).resolve().parent.parent / "workflows" / "figures" / "empirical" / "demographics.json"
)
# \pgfval macros for the manuscript, written next to the other generated .tex
# inputs (results.tex, diagnostics.tex, loggers.tex) so the Methods never hardcodes
# a demographic number.
DEFAULT_TEX_PATH = (
    Path(__file__).resolve().parent.parent / "workflows" / "figures" / "demographics.tex"
)
# No participant IDs are hardcoded here. The pilot / incomplete-session filter in
# get_valid_users() already yields exactly the analysed N = 100 per experiment, so
# no manual exclusion list is needed to reproduce the reported demographics; the
# --exclude flag remains available if a future dataset needs one.

EXPERIMENTS = [
    ("R", "p_unreliable"),
    ("V", "p_volatile"),
    ("T", "p_transition"),
]

GENDER_OPTIONS = ["Male", "Female", "Nonbinary", "Prefer Not to Answer"]


def fetch_raw_data(raw_json: Path) -> dict:
    with open(raw_json) as f:
        return json.load(f)


def get_valid_users(rawdata: dict, exclude_participants: set[str]) -> list[str]:
    """Same filter as the preprocessing notebook: drop pilots, incomplete
    sessions (no 'earnings' key), and manually excluded participants."""
    return [
        i
        for i in rawdata.keys()
        if "pilot" not in i
        and "earnings" in rawdata[i].keys()
        and i not in exclude_participants
    ]


def extract_survey(rawdata: dict, user: str) -> dict:
    """Merge every 'survey' frame's response dict from the trial log, so the
    demographics page (age/race/ethnicity/gender) and the consent page end up
    in one dict. Deliberately scans `data` rather than trusting a top-level
    `survey_data` field some dumps carry: in RAW_R.json, 14/100 valid users
    have an empty top-level `survey_data` even though their demographics
    answers are present in the trial log (same accumulation logic as the
    original preprocessing notebook)."""
    survey: dict = {}
    for frame in rawdata[user]["data"]:
        if frame.get("trial_type") == "survey":
            survey.update(frame["response"])
    return survey


def summarize(records: list[dict]) -> dict:
    ages = []
    for r in records:
        raw_age = r.get("P0_Q1")
        try:
            ages.append(float(raw_age))
        except (TypeError, ValueError):
            pass

    gender_counts = {g: 0 for g in GENDER_OPTIONS}
    for r in records:
        g = r.get("gender", "Missing")
        gender_counts[g] = gender_counts.get(g, 0) + 1

    return {
        "n": len(records),
        "n_age_reported": len(ages),
        "mean_age": statistics.mean(ages) if ages else None,
        "sd_age": statistics.stdev(ages) if len(ages) > 1 else None,
        "gender_counts": gender_counts,
    }


def format_summary(exp_type: str, summary: dict) -> str:
    lines = [f"Experiment {exp_type}: N = {summary['n']}"]
    if summary["mean_age"] is None:
        lines.append("  age: no ages reported")
    elif summary["n_age_reported"] < summary["n"]:
        lines.append(
            f"  age: mean {summary['mean_age']:.1f} +/- {summary['sd_age']:.1f} years "
            f"({summary['n_age_reported']}/{summary['n']} reported)"
        )
    else:
        lines.append(f"  age: mean {summary['mean_age']:.1f} +/- {summary['sd_age']:.1f} years")
    gender_str = ", ".join(f"{v} {k}" for k, v in summary["gender_counts"].items() if v > 0)
    lines.append(f"  gender: {gender_str}")
    return "\n".join(lines)


# Experiment letter -> the name the manuscript uses for that experiment.
VARIANT_LABEL = {"R": "Reliability", "V": "Volatility", "T": "Controllability", "pooled": "Pooled"}


def write_pgfvals(summaries: dict, out_path: Path) -> None:
    """Emit \\pgfkeyssetvalue lines for the Methods, mirroring results.tex's style.

    Keys are Demographics.<Variant>.<field>, e.g. Demographics.R.mean_age. Values
    are pre-rendered so the manuscript never formats a number itself.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for variant, s in summaries.items():
            label = VARIANT_LABEL.get(variant, variant)
            prefix = f"Demographics.{variant}"
            f.write(f"% {label}\n")
            f.write(f"\\pgfkeyssetvalue{{{prefix}.n}}{{{s['n']}}}\n")
            if s["mean_age"] is not None:
                f.write(f"\\pgfkeyssetvalue{{{prefix}.mean_age}}{{{s['mean_age']:.1f}}}\n")
            if s["sd_age"] is not None:
                f.write(f"\\pgfkeyssetvalue{{{prefix}.sd_age}}{{{s['sd_age']:.1f}}}\n")
            for gender, count in s["gender_counts"].items():
                key = gender.lower().replace(" ", "_").replace("-", "_")
                f.write(f"\\pgfkeyssetvalue{{{prefix}.n_{key}}}{{{count}}}\n")
            # Ready-made phrase so the Methods can cite one macro per experiment.
            parts = [f"{v} {k.lower()}" for k, v in s["gender_counts"].items() if v > 0]
            if len(parts) > 1:
                parts[-1] = "and " + parts[-1]
            f.write(
                f"\\pgfkeyssetvalue{{{prefix}.gender_breakdown}}"
                f"{{{', '.join(parts)}}}\n"
            )
    print(f"Wrote {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Participant IDs to exclude beyond the automatic pilot/incomplete filter. "
        "Not needed to reproduce the reported demographics -- that filter already "
        "yields exactly N = 100 per experiment on the shared data.",
    )
    parser.add_argument(
        "--raw-json",
        type=Path,
        default=DEFAULT_RAW_JSON_DIR,
        help="Directory containing RAW_<TYPE>.json dumps (default: "
        f"{DEFAULT_RAW_JSON_DIR}). These dumps are not distributed with the "
        "repository -- see the module docstring.",
    )
    parser.add_argument(
        "--types",
        nargs="*",
        default=[t for t, _ in EXPERIMENTS],
        choices=[t for t, _ in EXPERIMENTS],
        help="Which experiment types to summarize (default: all of R V T).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Where to write the JSON summary (default: {DEFAULT_OUT_PATH}). "
        "Pass an empty string to skip writing.",
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=DEFAULT_TEX_PATH,
        help=f"Where to write the \\pgfval macros for the manuscript "
        f"(default: {DEFAULT_TEX_PATH}). Pass an empty string to skip writing.",
    )
    args = parser.parse_args()

    exclude_participants = set(args.exclude or [])
    if exclude_participants:
        print(f"Excluding {len(exclude_participants)} participant(s) by ID", file=sys.stderr)

    all_records: dict[str, list[dict]] = {}
    for exp_type in args.types:
        candidate = args.raw_json / f"RAW_{exp_type}.json"
        if not candidate.exists():
            print(
                f"Skipping {exp_type}: {candidate} not found. The raw demographic "
                "dumps are not distributed with this repository (see the module "
                "docstring); contact the authors for access.",
                file=sys.stderr,
            )
            continue
        rawdata = fetch_raw_data(candidate)

        valid_users = get_valid_users(rawdata, exclude_participants)
        records = [extract_survey(rawdata, u) for u in valid_users]
        all_records[exp_type] = records

    print("=" * 60)
    print("Demographics summary")
    print("=" * 60)
    for exp_type, records in all_records.items():
        summary = summarize(records)
        print(format_summary(exp_type, summary))
        print()

    summaries = {exp_type: summarize(records) for exp_type, records in all_records.items()}

    if len(all_records) > 1:
        pooled = [r for records in all_records.values() for r in records]
        pooled_summary = summarize(pooled)
        print("-" * 60)
        print(format_summary("all (pooled)", pooled_summary))
        summaries["pooled"] = pooled_summary

    if str(args.out):
        out_path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWrote {out_path}", file=sys.stderr)

    if str(args.tex):
        write_pgfvals(summaries, args.tex)


if __name__ == "__main__":
    main()
