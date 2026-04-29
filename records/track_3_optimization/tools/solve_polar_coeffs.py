#!/usr/bin/env python3
"""Generate Polar-Express-style Muon coefficient schedules.

The online training script consumes JSON files with either:
  {"coeffs": [[a, b, c], ...]}
or:
  {"step_coeffs": {"1": [[a, b, c], ...], ...}}

This tool implements the offline quintic minimax solver from the Polar Express
paper's reference implementation, plus helpers for choosing the initial lower
singular-value bound from spectrum logs collected during Track 3 runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


PUBLISHED_PE_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]


def apply_safety(coeffs: list[tuple[float, float, float]], safety: float) -> list[list[float]]:
    if safety == 1:
        return [list(row) for row in coeffs]
    scaled = []
    for i, (a, b, c) in enumerate(coeffs):
        if i == len(coeffs) - 1:
            scaled.append([a, b, c])
        else:
            scaled.append([a / safety, b / safety**3, c / safety**5])
    return scaled


def published_pe(steps: int, safety: float) -> list[list[float]]:
    coeffs = PUBLISHED_PE_COEFFS[:steps]
    if steps > len(coeffs):
        coeffs += [PUBLISHED_PE_COEFFS[-1]] * (steps - len(coeffs))
    return apply_safety(coeffs, safety)


def optimal_quintic(lower: float, upper: float) -> tuple[float, float, float]:
    assert 0 <= lower <= upper
    if upper <= 0:
        raise ValueError("upper bound must be positive")
    if lower / upper >= 1 - 5e-6:
        return (15 / 8) / upper, (-10 / 8) / upper**3, (3 / 8) / upper**5

    q = (3 * lower + upper) / 4
    r = (lower + 3 * upper) / 4
    error = math.inf
    old_error = None
    while old_error is None or abs(old_error - error) > 1e-15:
        old_error = error
        lhs = np.array(
            [
                [lower, lower**3, lower**5, 1],
                [q, q**3, q**5, -1],
                [r, r**3, r**5, 1],
                [upper, upper**3, upper**5, -1],
            ],
            dtype=np.float64,
        )
        a, b, c, error = np.linalg.solve(lhs, np.ones(4, dtype=np.float64))
        roots = (-3 * b + np.array([-1, 1]) * math.sqrt(9 * b**2 - 20 * a * c)) / (10 * c)
        q, r = np.sqrt(roots)
    return float(a), float(b), float(c)


def optimal_composition(
    lower: float,
    steps: int,
    *,
    cushion: float,
    safety: float,
) -> list[list[float]]:
    upper = 1.0
    coeffs: list[tuple[float, float, float]] = []
    for _ in range(steps):
        a, b, c = optimal_quintic(max(lower, cushion * upper), upper)
        p_lower = a * lower + b * lower**3 + c * lower**5
        p_upper = a * upper + b * upper**3 + c * upper**5
        rescale = 2 / (p_lower + p_upper)
        a *= rescale
        b *= rescale
        c *= rescale
        coeffs.append((a, b, c))
        lower = a * lower + b * lower**3 + c * lower**5
        upper = 2 - lower
    return apply_safety(coeffs, safety)


def read_spectrum_rows(path: Path) -> list[dict]:
    files = []
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    elif path.exists():
        files = [path]
    rows = []
    for file in files:
        with file.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no spectrum rows found under {path}")
    return rows


def quantile(values: list[float], q: float) -> float:
    arr = np.array([x for x in values if x > 0 and math.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no positive finite values available")
    return float(np.quantile(arr, q))


def lower_from_rows(rows: list[dict], field: str, aggregate_quantile: float, floor: float) -> float:
    return max(floor, min(1.0, quantile([float(row[field]) for row in rows], aggregate_quantile)))


def step_lowers(
    rows: list[dict],
    *,
    field: str,
    aggregate_quantile: float,
    floor: float,
    train_steps: int,
) -> dict[int, float]:
    by_step: dict[int, list[dict]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    observed = sorted((step, lower_from_rows(step_rows, field, aggregate_quantile, floor)) for step, step_rows in by_step.items())
    if not observed:
        raise ValueError("no observed steps")

    result: dict[int, float] = {}
    obs_steps = [x[0] for x in observed]
    obs_logs = [math.log(x[1]) for x in observed]
    for step in range(1, train_steps + 1):
        right = np.searchsorted(obs_steps, step, side="right")
        if right == 0:
            lower_log = obs_logs[0]
        elif right >= len(obs_steps):
            lower_log = obs_logs[-1]
        else:
            s0, s1 = obs_steps[right - 1], obs_steps[right]
            y0, y1 = obs_logs[right - 1], obs_logs[right]
            t = (step - s0) / (s1 - s0)
            lower_log = y0 + t * (y1 - y0)
        result[step] = math.exp(lower_log)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["published-pe", "fixed", "adaptive"], required=True)
    parser.add_argument("--spectrum", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lower", type=float)
    parser.add_argument("--field", default="q010")
    parser.add_argument("--aggregate-quantile", type=float, default=0.1)
    parser.add_argument("--floor", type=float, default=1e-5)
    parser.add_argument("--cushion", type=float, default=0.02407327424182761)
    parser.add_argument("--safety", type=float, default=1.01)
    parser.add_argument("--train-steps", type=int, default=3500)
    args = parser.parse_args()

    rows = read_spectrum_rows(args.spectrum) if args.spectrum else None

    if args.mode == "published-pe":
        payload = {
            "name": f"published_pe_steps{args.steps}",
            "mode": args.mode,
            "steps": args.steps,
            "safety": args.safety,
            "coeffs": published_pe(args.steps, args.safety),
        }
    elif args.mode == "fixed":
        lower = args.lower
        if lower is None:
            if rows is None:
                raise ValueError("--lower or --spectrum is required for fixed mode")
            lower = lower_from_rows(rows, args.field, args.aggregate_quantile, args.floor)
        payload = {
            "name": f"fixed_lower{lower:.6g}_steps{args.steps}",
            "mode": args.mode,
            "steps": args.steps,
            "lower": lower,
            "field": args.field,
            "aggregate_quantile": args.aggregate_quantile,
            "floor": args.floor,
            "cushion": args.cushion,
            "safety": args.safety,
            "coeffs": optimal_composition(lower, args.steps, cushion=args.cushion, safety=args.safety),
        }
    else:
        if rows is None:
            raise ValueError("--spectrum is required for adaptive mode")
        lowers = step_lowers(
            rows,
            field=args.field,
            aggregate_quantile=args.aggregate_quantile,
            floor=args.floor,
            train_steps=args.train_steps,
        )
        cache: dict[float, list[list[float]]] = {}
        step_coeffs = {}
        for step, lower in lowers.items():
            key = round(lower, 10)
            if key not in cache:
                cache[key] = optimal_composition(lower, args.steps, cushion=args.cushion, safety=args.safety)
            step_coeffs[str(step)] = cache[key]
        payload = {
            "name": f"adaptive_{args.field}_q{args.aggregate_quantile}_steps{args.steps}",
            "mode": args.mode,
            "steps": args.steps,
            "field": args.field,
            "aggregate_quantile": args.aggregate_quantile,
            "floor": args.floor,
            "cushion": args.cushion,
            "safety": args.safety,
            "train_steps": args.train_steps,
            "step_coeffs": step_coeffs,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: payload[k] for k in payload if k != "coeffs" and k != "step_coeffs"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
