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
    "muon-025-0125": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.025",
        "--muon-wd", "0.0125",
    ],
    "muon-035-025": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "muon",
        "--muon-lr", "0.035",
        "--muon-wd", "0.025",
    ],
    "adamh": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "adamh",
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
    "poprisk-adamh-001": [
        "--optimizer", "adamw",
        "--matrix-optimizer", "poprisk-adamh",
        "--pop-gate", "snr",
        "--pop-lambda", "0.01",
        "--pop-warmup-steps", "100",
    ],
}

VAL_RE = re.compile(r"step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>[0-9.]+)")
LOG_RE = re.compile(r"(?:^| )logfile: (?P<path>\S+)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="*", help="Candidate names to run, or 'list'")
    parser.add_argument("--preset", default="ordinal-3090",
                        help="train_gpt_local_proxy.py proxy preset")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--val-interval", type=int, default=50)
    parser.add_argument("--dense-val-start", type=int, default=-1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-tokens", type=int, default=None)
    parser.add_argument("--reference-batch-tokens", type=int, default=None)
    parser.add_argument("--val-tokens", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER,
                        help="Arguments appended to every train_gpt_local_proxy.py run")
    return parser.parse_args()


def candidate_names(requested: list[str]) -> list[str]:
    if not requested:
        return ["muon-035-025", "adamh"]
    if requested == ["list"]:
        for name in sorted(CANDIDATES):
            print(name)
        raise SystemExit(0)
    unknown = sorted(set(requested) - set(CANDIDATES))
    if unknown:
        raise SystemExit(f"unknown candidate(s): {', '.join(unknown)}")
    return requested


def build_command(args, name: str) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--proxy-preset", args.preset,
        "--train-steps", str(args.steps),
        "--val-interval", str(args.val_interval),
        "--dense-val-start", str(args.dense_val_start),
        "--log-interval", str(args.log_interval),
        "--seed", str(args.seed),
    ]
    for flag, value in [
        ("--batch-tokens", args.batch_tokens),
        ("--reference-batch-tokens", args.reference_batch_tokens),
        ("--val-tokens", args.val_tokens),
        ("--data-dir", args.data_dir),
    ]:
        if value is not None:
            cmd.extend([flag, str(value)])
    cmd.extend(CANDIDATES[name])
    if args.extra:
        cmd.extend(args.extra)
    return cmd


def run_candidate(args, name: str) -> dict:
    cmd = build_command(args, name)
    result = {
        "candidate": name,
        "command": cmd,
        "returncode": None,
        "logfile": None,
        "final_val_step": None,
        "final_val_loss": None,
        "wall_time_sec": None,
    }
    print(f"\n=== {name} ===")
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
    names = candidate_names(args.candidates)
    results = [run_candidate(args, name) for name in names]
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
