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
        discriminant = 9 * b**2 - 20 * a * c
        if discriminant < 0:
            if discriminant < -1e-8:
                raise ValueError(f"negative quintic discriminant {discriminant}")
            discriminant = 0.0
        roots = (-3 * b + np.array([-1, 1]) * math.sqrt(discriminant)) / (10 * c)
        q, r = np.sqrt(np.maximum(roots, 0))
    return float(a), float(b), float(c)


def optimal_composition(
    lower: float,
    initial_upper: float,
    steps: int,
    *,
    cushion: float,
    safety: float,
) -> list[list[float]]:
    if not 0 <= lower <= initial_upper <= 1.0:
        raise ValueError(f"expected 0 <= lower <= upper <= 1, got lower={lower} upper={initial_upper}")
    upper = initial_upper
    coeffs: list[tuple[float, float, float]] = []
    for _ in range(steps):
        if lower > upper:
            if lower - upper > 1e-8:
                raise ValueError(f"composition bounds inverted: lower={lower} upper={upper}")
            lower = upper = (lower + upper) / 2
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
        return 0.0
    return float(np.quantile(arr, q))


def has_positive_finite(rows: list[dict], field: str) -> bool:
    return any(float(row[field]) > 0 and math.isfinite(float(row[field])) for row in rows)


def bound_from_rows(rows: list[dict], field: str, aggregate_quantile: float, floor: float, ceil: float) -> float:
    return max(floor, min(ceil, quantile([float(row[field]) for row in rows], aggregate_quantile)))


def valid_bounds(lower: float, upper: float, floor: float) -> tuple[float, float]:
    lower = max(floor, min(1.0, lower))
    upper = max(lower * (1 + 1e-6), min(1.0, upper))
    return lower, upper


def step_bounds(
    rows: list[dict],
    *,
    lower_field: str,
    lower_aggregate_quantile: float,
    upper_field: str,
    upper_aggregate_quantile: float,
    upper_override: float | None,
    floor: float,
    ceil: float,
    train_steps: int,
) -> dict[int, tuple[float, float]]:
    by_step: dict[int, list[dict]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    observed = []
    for step, step_rows in by_step.items():
        if upper_override is None and not has_positive_finite(step_rows, upper_field):
            continue
        lower = bound_from_rows(step_rows, lower_field, lower_aggregate_quantile, floor, ceil)
        if upper_override is None:
            upper = bound_from_rows(step_rows, upper_field, upper_aggregate_quantile, floor, ceil)
        else:
            upper = upper_override
        observed.append((step, *valid_bounds(lower, upper, floor)))
    observed.sort()
    if not observed:
        raise ValueError("no observed steps")

    result: dict[int, tuple[float, float]] = {}
    obs_steps = [x[0] for x in observed]
    lower_logs = [math.log(x[1]) for x in observed]
    upper_logs = [math.log(x[2]) for x in observed]
    for step in range(1, train_steps + 1):
        right = np.searchsorted(obs_steps, step, side="right")
        if right == 0:
            lower_log = lower_logs[0]
            upper_log = upper_logs[0]
        elif right >= len(obs_steps):
            lower_log = lower_logs[-1]
            upper_log = upper_logs[-1]
        else:
            s0, s1 = obs_steps[right - 1], obs_steps[right]
            lower_y0, lower_y1 = lower_logs[right - 1], lower_logs[right]
            upper_y0, upper_y1 = upper_logs[right - 1], upper_logs[right]
            t = (step - s0) / (s1 - s0)
            lower_log = lower_y0 + t * (lower_y1 - lower_y0)
            upper_log = upper_y0 + t * (upper_y1 - upper_y0)
        result[step] = valid_bounds(math.exp(lower_log), math.exp(upper_log), floor)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["published-pe", "fixed", "adaptive"], required=True)
    parser.add_argument("--spectrum", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lower", type=float)
    parser.add_argument("--upper", type=float)
    parser.add_argument("--field", "--lower-field", dest="field", default="q010")
    parser.add_argument("--aggregate-quantile", "--lower-aggregate-quantile", dest="aggregate_quantile", type=float, default=0.1)
    parser.add_argument("--upper-field", default="q1000")
    parser.add_argument("--upper-aggregate-quantile", type=float, default=1.0)
    parser.add_argument("--floor", type=float, default=1e-5)
    parser.add_argument("--ceil", type=float, default=1.0)
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
            lower = bound_from_rows(rows, args.field, args.aggregate_quantile, args.floor, args.ceil)
        upper = args.upper
        if upper is None:
            if rows is None:
                upper = 1.0
            else:
                upper = bound_from_rows(rows, args.upper_field, args.upper_aggregate_quantile, args.floor, args.ceil)
        lower, upper = valid_bounds(lower, upper, args.floor)
        payload = {
            "name": f"fixed_lower{lower:.6g}_upper{upper:.6g}_steps{args.steps}",
            "mode": args.mode,
            "steps": args.steps,
            "lower": lower,
            "upper": upper,
            "field": args.field,
            "aggregate_quantile": args.aggregate_quantile,
            "upper_field": args.upper_field,
            "upper_aggregate_quantile": args.upper_aggregate_quantile,
            "floor": args.floor,
            "ceil": args.ceil,
            "cushion": args.cushion,
            "safety": args.safety,
            "coeffs": optimal_composition(lower, upper, args.steps, cushion=args.cushion, safety=args.safety),
        }
    else:
        if rows is None:
            raise ValueError("--spectrum is required for adaptive mode")
        bounds = step_bounds(
            rows,
            lower_field=args.field,
            lower_aggregate_quantile=args.aggregate_quantile,
            upper_field=args.upper_field,
            upper_aggregate_quantile=args.upper_aggregate_quantile,
            upper_override=args.upper,
            floor=args.floor,
            ceil=args.ceil,
            train_steps=args.train_steps,
        )
        cache: dict[tuple[float, float], list[list[float]]] = {}
        step_coeffs = {}
        step_bounds_payload = {}
        for step, (lower, upper) in bounds.items():
            key = (round(lower, 10), round(upper, 10))
            if key not in cache:
                cache[key] = optimal_composition(lower, upper, args.steps, cushion=args.cushion, safety=args.safety)
            step_coeffs[str(step)] = cache[key]
            step_bounds_payload[str(step)] = [lower, upper]
        payload = {
            "name": f"adaptive_{args.field}_q{args.aggregate_quantile}_steps{args.steps}",
            "mode": args.mode,
            "steps": args.steps,
            "field": args.field,
            "aggregate_quantile": args.aggregate_quantile,
            "upper_field": args.upper_field,
            "upper_aggregate_quantile": args.upper_aggregate_quantile,
            "floor": args.floor,
            "ceil": args.ceil,
            "cushion": args.cushion,
            "safety": args.safety,
            "train_steps": args.train_steps,
            "step_bounds": step_bounds_payload,
            "step_coeffs": step_coeffs,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    hidden_fields = {"coeffs", "step_coeffs", "step_bounds"}
    print(json.dumps({k: payload[k] for k in payload if k not in hidden_fields}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
