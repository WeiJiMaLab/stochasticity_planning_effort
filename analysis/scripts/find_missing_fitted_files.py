#!/usr/bin/env python3
"""
Verify that fit/ mirrors data_split/ for simulated data (same paths as fitter.py writes).

fitter.py glob: ../data_split/*/*/* (every folder); this checker audits only the simulated* subset
Output path:    each file -> ../fit/... with identical relative path under data_split/ vs fit/

This script lists missing outputs, invalid JSON, empty files, and optional orphan files in fit/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def analysis_dirs(script_path: Path) -> tuple[Path, Path]:
    analysis = script_path.resolve().parent.parent
    return analysis / "data_split", analysis / "fit"


def iter_simulated_split_files(split_root: Path):
    """Same coverage as glob('simulated*/*/*') but stable sorted order."""
    if not split_root.is_dir():
        return
    for data_folder in sorted(split_root.glob("simulated*")):
        if not data_folder.is_dir():
            continue
        for type_folder in sorted(data_folder.iterdir()):
            if not type_folder.is_dir():
                continue
            for fp in sorted(type_folder.glob("*_data.json")):
                yield fp


def expected_fit_path(split_root: Path, fit_root: Path, split_file: Path) -> Path:
    return fit_root / split_file.relative_to(split_root)


def collect_orphans(fit_root: Path, expected_relatives: set[Path]) -> list[Path]:
    """Fit files under simulated* that do not correspond to any data_split file."""
    orphans: list[Path] = []
    if not fit_root.is_dir():
        return orphans
    for data_folder in sorted(fit_root.glob("simulated*")):
        if not data_folder.is_dir():
            continue
        for type_folder in sorted(data_folder.iterdir()):
            if not type_folder.is_dir():
                continue
            for fp in sorted(type_folder.glob("*_data.json")):
                try:
                    rel = fp.relative_to(fit_root)
                except ValueError:
                    continue
                if rel not in expected_relatives:
                    orphans.append(fp)
    return orphans


def check_fit_file(fit_path: Path, validate_json: bool) -> str | None:
    """Return an issue description if something is wrong, else None."""
    if not fit_path.is_file():
        return "missing"
    if fit_path.stat().st_size == 0:
        return "empty"
    if validate_json:
        try:
            with open(fit_path, encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            return f"invalid JSON ({e.msg})"
    return None


def run_check(
    split_root: Path,
    fit_root: Path,
    *,
    validate_json: bool,
    report_orphans: bool,
    verbose: bool,
) -> int:
    if not split_root.is_dir():
        print(f"Error: data_split directory does not exist: {split_root}", file=sys.stderr)
        return 2

    split_files = list(iter_simulated_split_files(split_root))
    if not split_files:
        print(f"Warning: no simulated split files under {split_root}", file=sys.stderr)
        return 0

    expected_relatives = {p.relative_to(split_root) for p in split_files}

    missing: list[tuple[Path, Path]] = []
    bad: list[tuple[Path, str]] = []

    for sf in split_files:
        fit_path = expected_fit_path(split_root, fit_root, sf)
        issue = check_fit_file(fit_path, validate_json)
        if issue == "missing":
            missing.append((sf, fit_path))
        elif issue is not None:
            bad.append((fit_path, issue))

    orphans: list[Path] = []
    if report_orphans:
        orphans = collect_orphans(fit_root, expected_relatives)

    # --- output ---
    print(f"Data split root: {split_root}")
    print(f"Fit root:        {fit_root}")
    print(f"Expected fit outputs (from data_split): {len(split_files)}")
    print("=" * 60)

    if verbose:
        n_ok = len(split_files) - len(missing) - len(bad)
        print(f"OK (present and valid): {n_ok}")
        print(f"Missing:                {len(missing)}")
        print(f"Present but faulty:     {len(bad)}")
        if report_orphans:
            print(f"Orphan fit files:       {len(orphans)}")

    if missing:
        print(f"\nMissing ({len(missing)}):")
        for _, fp in missing[:50]:
            print(f"  {fp}")
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")

    if bad:
        print(f"\nPresent but faulty ({len(bad)}):")
        for fp, reason in bad[:50]:
            print(f"  {fp}  ({reason})")
        if len(bad) > 50:
            print(f"  ... and {len(bad) - 50} more")

    if report_orphans and orphans:
        print(f"\nOrphan files in fit/ (no matching data_split file) ({len(orphans)}):")
        for fp in orphans[:50]:
            print(f"  {fp}")
        if len(orphans) > 50:
            print(f"  ... and {len(orphans) - 50} more")

    print("\n" + "=" * 60)
    if missing or bad:
        print("SUMMARY: Incomplete or invalid fit outputs.")
        return 1
    if report_orphans and orphans:
        print("SUMMARY: Orphan files in fit/ (exit 1).")
        return 1
    print("SUMMARY: All expected fit files present" + (" and valid JSON." if validate_json else "."))
    return 0


def main() -> None:
    default_split, default_fit = analysis_dirs(Path(__file__))

    parser = argparse.ArgumentParser(
        description="Check that fit/ contains one output per simulated data_split *_data.json (fitter.py layout)."
    )
    parser.add_argument(
        "analysis_dir",
        nargs="?",
        default=None,
        help="Analysis directory containing data_split/ and fit/ (default: parent of scripts/)",
    )
    parser.add_argument(
        "--data-split",
        type=Path,
        default=None,
        help="Override data_split directory",
    )
    parser.add_argument(
        "--fit",
        type=Path,
        default=None,
        help="Override fit directory",
    )
    parser.add_argument(
        "--validate-json",
        action="store_true",
        help="Verify each present fit file parses as JSON",
    )
    parser.add_argument(
        "--no-orphans",
        action="store_true",
        help="Do not report extra *_data.json files under fit/ with no data_split counterpart",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print counts per category",
    )
    args = parser.parse_args()

    if args.analysis_dir:
        base = Path(args.analysis_dir).resolve()
        split_root = args.data_split or (base / "data_split")
        fit_root = args.fit or (base / "fit")
    else:
        split_root = args.data_split or default_split
        fit_root = args.fit or default_fit

    split_root = split_root.resolve()
    fit_root = fit_root.resolve()

    code = run_check(
        split_root,
        fit_root,
        validate_json=args.validate_json,
        report_orphans=not args.no_orphans,
        verbose=args.verbose,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
