#!/usr/bin/env python3
"""Run repeatable local ordinal-proxy experiments for Track 3 optimizers."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path


TRACK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TRACK_DIR.parents[1]
TRAIN_SCRIPT = TRACK_DIR / "train_gpt_local_proxy.py"

CANDIDATES = {
    "muon-020-010": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.020",
        "--muon-wd", "0.010",
    ],
    "muon-025-0125": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.025",
        "--muon-wd", "0.0125",
    ],
    "muon-030-0125": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.030",
        "--muon-wd", "0.0125",
    ],
    "muon-030-025": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.030",
        "--muon-wd", "0.025",
    ],
    "muon-035-025": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.035",
        "--muon-wd", "0.025",
    ],
    "muon-0375-025": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.0375",
        "--muon-wd", "0.025",
    ],
    "pmuon": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "pmuon",
        "--muon-lr", "0.035",
        "--muon-wd", "0.025",
        "--pmuon-beta", "0.95",
        "--pmuon-gamma", "0.3",
        "--train-steps", "3250",
    ],
    "adamh": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "adamh",
    ],
    "kfac-adamh": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "kfac-adamh",
        "--kfac-damping", "0.03",
        "--kfac-refresh-steps", "1",
    ],
    "kfac-muon": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "kfac-muon",
        "--muon-lr", "0.035",
        "--muon-wd", "0.025",
        "--kfac-damping", "0.03",
        "--kfac-refresh-steps", "1",
    ],
    "kfac": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "kfac",
        "--kfac-lr", "0.001",
        "--kfac-wd", "0.025",
        "--kfac-damping", "0.03",
        "--kfac-refresh-steps", "1",
    ],
    "poprisk-aux-0001": [
        "--optimizer", "poprisk-adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.035",
        "--muon-wd", "0.025",
        "--pop-gate", "snr",
        "--pop-lambda", "0.001",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-aux-0003": [
        "--optimizer", "poprisk-adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.035",
        "--muon-wd", "0.025",
        "--pop-gate", "snr",
        "--pop-lambda", "0.003",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-aux-001": [
        "--optimizer", "poprisk-adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.035",
        "--muon-wd", "0.025",
        "--pop-gate", "snr",
        "--pop-lambda", "0.01",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-aux-003": [
        "--optimizer", "poprisk-adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.035",
        "--muon-wd", "0.025",
        "--pop-gate", "snr",
        "--pop-lambda", "0.03",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-aux-001-w50": [
        "--optimizer", "poprisk-adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.035",
        "--muon-wd", "0.025",
        "--pop-gate", "snr",
        "--pop-lambda", "0.01",
        "--pop-warmup-steps", "50",
    ],
    "poprisk-aux-001-w200": [
        "--optimizer", "poprisk-adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.035",
        "--muon-wd", "0.025",
        "--pop-gate", "snr",
        "--pop-lambda", "0.01",
        "--pop-warmup-steps", "200",
    ],
    "poprisk-aux-adaptive-q067": [
        "--optimizer", "poprisk-adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.035",
        "--muon-wd", "0.025",
        "--pop-gate", "snr",
        "--pop-lambda-mode", "target-median-q",
        "--pop-target-q", "0.67",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-aux-cosine-003-zero": [
        "--optimizer", "poprisk-adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.035",
        "--muon-wd", "0.025",
        "--pop-gate", "snr",
        "--pop-lambda", "0.03",
        "--pop-lambda-final", "0.0",
        "--pop-lambda-mode", "cosine-decay",
        "--pop-lambda-decay-start-frac", "0.0",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-adamh-001": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.01",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-adamh-0001": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.001",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-adamh-0003": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.003",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-adamh-003": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.03",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-adamh-hard": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "hard",
        "--pop-warmup-steps", "0",
    ],
    "poprisk-adamh-soft-003": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "soft",
        "--pop-lambda", "0.03",
        "--pop-warmup-steps", "0",
    ],
    "poprisk-adamh-01": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.1",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-adamh-001-w50": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.01",
        "--pop-warmup-steps", "50",
    ],
    "poprisk-adamh-001-w200": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.01",
        "--pop-warmup-steps", "200",
    ],
    "poprisk-adamh-003-w50": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.03",
        "--pop-warmup-steps", "50",
    ],
    "poprisk-adamh-003-w200": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.03",
        "--pop-warmup-steps", "200",
    ],
    "poprisk-adamh-003-w0": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.03",
        "--pop-warmup-steps", "0",
    ],
    "poprisk-adamh-snr-wiener": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr-wiener",
        "--pop-warmup-steps", "0",
    ],
    "poprisk-adamh-snr-var": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr-var",
        "--pop-warmup-steps", "0",
    ],
    "poprisk-adamh-adaptive-q050": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda-mode", "target-median-q",
        "--pop-target-q", "0.50",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-adamh-adaptive-q067": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda-mode", "target-median-q",
        "--pop-target-q", "0.67",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-adamh-adaptive-q080": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda-mode", "target-median-q",
        "--pop-target-q", "0.80",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-adamh-cosine-003-zero": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.03",
        "--pop-lambda-final", "0.0",
        "--pop-lambda-mode", "cosine-decay",
        "--pop-lambda-decay-start-frac", "0.0",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-adamh-cosine-003-half": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.03",
        "--pop-lambda-final", "0.0",
        "--pop-lambda-mode", "cosine-decay",
        "--pop-lambda-decay-start-frac", "0.5",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-aux-adamh-001": [
        "--optimizer", "poprisk-adamw",
        "--matrix-optimizer", "adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.01",
        "--pop-warmup-steps", "100",
    ],
    "poprisk-both-adamh-001": [
        "--optimizer", "poprisk-adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.01",
        "--pop-warmup-steps", "100",
    ],
}

CAMPAIGNS = {
    "realbatch-calibration-100": [
        ("adamh", 0),
        ("muon-035-025", 0),
        ("pmuon", 0),
        ("kfac-muon", 0),
        ("poprisk-adamh-snr-wiener", 0),
    ],
    "kfac-proxy-3": [
        ("adamh", 6),
        ("muon-035-025", 6),
        ("kfac-adamh", 6),
        ("kfac-muon", 6),
        ("kfac", 6),
    ],
    "adamh-poprisk-review-8": [
        ("adamh", 5),
        ("poprisk-adamh-003", 5),
        ("poprisk-adamh-hard", 5),
        ("poprisk-adamh-soft-003", 5),
        ("poprisk-adamh-adaptive-q050", 5),
        ("poprisk-adamh-adaptive-q067", 5),
        ("poprisk-adamh-cosine-003-zero", 5),
        ("poprisk-adamh-snr-wiener", 5),
    ],
    "ordinal-calibration-poprisk-6h": [
        ("muon-025-0125", 3),
        ("muon-035-025", 3),
        ("adamh", 3),
        ("poprisk-adamh-003-w50", 3),
        ("poprisk-adamh-003-w0", 3),
        ("poprisk-adamh-snr-wiener", 3),
        ("poprisk-adamh-snr-var", 3),
        ("muon-025-0125", 4),
        ("muon-035-025", 4),
        ("adamh", 4),
        ("poprisk-adamh-003-w50", 4),
        ("poprisk-adamh-003-w0", 4),
        ("poprisk-adamh-snr-wiener", 4),
        ("poprisk-adamh-snr-var", 4),
    ],
    "adamh-poprisk-principled-4h": [
        ("adamh", 3),
        ("poprisk-adamh-003-w50", 3),
        ("poprisk-adamh-003-w0", 3),
        ("poprisk-adamh-snr-wiener", 3),
        ("poprisk-adamh-snr-var", 3),
        ("adamh", 4),
        ("poprisk-adamh-003-w50", 4),
        ("poprisk-adamh-003-w0", 4),
        ("poprisk-adamh-snr-wiener", 4),
        ("poprisk-adamh-snr-var", 4),
    ],
    "adamh-poprisk-confirm-4h": [
        ("adamh", 1),
        ("poprisk-adamh-003-w50", 1),
        ("poprisk-adamh-01", 1),
        ("poprisk-adamh-001-w50", 1),
        ("poprisk-adamh-adaptive-q050", 1),
        ("adamh", 2),
        ("poprisk-adamh-003-w50", 2),
        ("poprisk-adamh-01", 2),
        ("poprisk-adamh-001-w50", 2),
        ("poprisk-adamh-adaptive-q050", 2),
    ],
    "adamh-poprisk-8h": [
        ("adamh", 0),
        ("poprisk-adamh-0001", 0),
        ("poprisk-adamh-0003", 0),
        ("poprisk-adamh-001", 0),
        ("poprisk-adamh-003", 0),
        ("poprisk-adamh-01", 0),
        ("poprisk-adamh-001-w50", 0),
        ("poprisk-adamh-001-w200", 0),
        ("poprisk-adamh-003-w50", 0),
        ("poprisk-adamh-003-w200", 0),
        ("poprisk-adamh-adaptive-q050", 0),
        ("poprisk-adamh-adaptive-q067", 0),
        ("poprisk-adamh-adaptive-q080", 0),
        ("poprisk-adamh-cosine-003-zero", 0),
        ("poprisk-aux-adamh-001", 0),
        ("poprisk-both-adamh-001", 0),
    ],
    "ordinal-8h": [
        ("muon-020-010", 0),
        ("muon-025-0125", 0),
        ("muon-030-0125", 0),
        ("muon-030-025", 0),
        ("muon-035-025", 0),
        ("muon-0375-025", 0),
        ("adamh", 0),
        ("poprisk-aux-0001", 0),
        ("poprisk-aux-0003", 0),
        ("poprisk-aux-001", 0),
        ("poprisk-aux-003", 0),
        ("poprisk-aux-001-w50", 0),
        ("poprisk-aux-001-w200", 0),
        ("poprisk-aux-adaptive-q067", 0),
        ("poprisk-aux-cosine-003-zero", 0),
        ("poprisk-adamh-001", 0),
    ],
}

VAL_RE = re.compile(r"step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>[0-9.]+)")
LOG_RE = re.compile(r"(?:^| )logfile: (?P<path>\S+)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="*", help="Candidate names to run, or 'list'")
    parser.add_argument("--campaign", choices=sorted(CAMPAIGNS), default=None,
                        help="Named run list; cannot be combined with positional candidates")
    parser.add_argument("--preset", default="ordinal-3090",
                        help="train_gpt_local_proxy.py proxy preset")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override train steps; defaults to the train script or preset default")
    parser.add_argument("--stop-after-step", type=int, default=None,
                        help="Stop early while preserving --steps/--train-steps for schedules")
    parser.add_argument("--val-interval", type=int, default=50)
    parser.add_argument("--dense-val-start", type=int, default=-1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--estimated-minutes-per-run", type=float, default=30.0,
                        help="Only used to print campaign dry-run/runtime estimates")
    parser.add_argument("--batch-tokens", type=int, default=None)
    parser.add_argument("--reference-batch-tokens", type=int, default=None)
    parser.add_argument("--val-tokens", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER,
                        help="Arguments appended to every train_gpt_local_proxy.py run")
    return parser.parse_args()


def candidate_names(requested: list[str]) -> list[tuple[str, int]]:
    if not requested:
        requested = ["muon-035-025", "adamh"]
    if requested == ["list"]:
        print("Candidates:")
        for name in sorted(CANDIDATES):
            print(f"  {name}")
        print("\nCampaigns:")
        for name, entries in sorted(CAMPAIGNS.items()):
            print(f"  {name}: {len(entries)} runs")
        raise SystemExit(0)
    unknown = sorted(set(requested) - set(CANDIDATES))
    if unknown:
        raise SystemExit(f"unknown candidate(s): {', '.join(unknown)}")
    return [(name, 0) for name in requested]


def run_plan(args) -> list[dict]:
    if args.campaign and args.candidates:
        raise SystemExit("--campaign cannot be combined with positional candidates")
    entries = CAMPAIGNS[args.campaign] if args.campaign else candidate_names(args.candidates)
    return [
        {
            "candidate": name,
            "seed": args.seed + seed_offset,
            "seed_offset": seed_offset,
            "run_index": i,
        }
        for i, (name, seed_offset) in enumerate(entries, 1)
    ]


def build_command(args, run: dict) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--proxy-preset", args.preset,
        "--val-interval", str(args.val_interval),
        "--dense-val-start", str(args.dense_val_start),
        "--log-interval", str(args.log_interval),
        "--seed", str(run["seed"]),
    ]
    if args.steps is not None:
        cmd.extend(["--train-steps", str(args.steps)])
    if args.stop_after_step is not None:
        cmd.extend(["--stop-after-step", str(args.stop_after_step)])
    for flag, value in [
        ("--batch-tokens", args.batch_tokens),
        ("--reference-batch-tokens", args.reference_batch_tokens),
        ("--val-tokens", args.val_tokens),
        ("--data-dir", args.data_dir),
    ]:
        if value is not None:
            cmd.extend([flag, str(value)])
    cmd.extend(CANDIDATES[run["candidate"]])
    if args.extra:
        cmd.extend(args.extra)
    return cmd


def run_candidate(args, run: dict, total_runs: int) -> dict:
    name = run["candidate"]
    cmd = build_command(args, run)
    result = {
        "candidate": name,
        "run_index": run["run_index"],
        "total_runs": total_runs,
        "seed": run["seed"],
        "seed_offset": run["seed_offset"],
        "command": cmd,
        "returncode": None,
        "logfile": None,
        "final_val_step": None,
        "final_val_loss": None,
        "wall_time_sec": None,
    }
    print(f"\n=== {run['run_index']}/{total_runs} {name} seed={run['seed']} ===")
    print("+ " + shlex.join(cmd))
    if args.dry_run:
        return result

    started = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        if (match := LOG_RE.search(line)):
            result["logfile"] = match.group("path")
        if (match := VAL_RE.search(line)):
            result["final_val_step"] = int(match.group("step"))
            result["final_val_loss"] = float(match.group("loss"))
    result["returncode"] = proc.wait()
    result["wall_time_sec"] = time.perf_counter() - started
    if result["returncode"] != 0:
        raise SystemExit(f"{name} failed with exit code {result['returncode']}")
    return result


def write_summary(results: list[dict]) -> Path:
    summary_path = REPO_ROOT / "logs" / f"ordinal_proxy_summary_{uuid.uuid4()}.json"
    summary_path.parent.mkdir(exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    return summary_path


def main():
    args = parse_args()
    plan = run_plan(args)
    estimated_hours = len(plan) * args.estimated_minutes_per_run / 60
    if args.campaign or args.dry_run:
        label = args.campaign or "custom"
        print(f"Run plan: {label}, {len(plan)} runs, estimated {estimated_hours:.1f} hours")
    results = [run_candidate(args, run, len(plan)) for run in plan]
    if args.dry_run:
        return

    ranked = sorted(
        [row for row in results if row["final_val_loss"] is not None],
        key=lambda row: row["final_val_loss"],
    )
    print("\nOrdinal proxy summary")
    for i, row in enumerate(ranked, 1):
        minutes = row["wall_time_sec"] / 60
        print(
            f"{i}. {row['candidate']}: step {row['final_val_step']} "
            f"val_loss {row['final_val_loss']:.8f} ({minutes:.1f} min) "
            f"log {row['logfile']}"
        )
    print(f"summary: {write_summary(results)}")


if __name__ == "__main__":
    main()
