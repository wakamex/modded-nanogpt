# Track 3 Local Proxy Ranking Snapshot

Date: 2026-05-06

This note summarizes the local ordinal proxy evidence for AdamH/PopRisk variants, with the latest upstream Track 3 ordering included for calibration. The proxy is useful for triage, but it is not yet a faithful ordinal model of official 3.28-step results.

Sources:

- `logs/ordinal_proxy_summary_7ed92aba-9138-4481-b63b-03f43e76a32f.json`
- `logs/ordinal_proxy_summary_63d9459b-01d2-4dc3-92e0-d06a54cbd50c.json`
- `.codex_logs/ordinal-calibration-poprisk-6h_20260506_133502.out`
- `upstream/master:records/track_3_optimization/README.md`

## Official Upstream Ordering

Current upstream Track 3 ordering by accepted steps to 3.28:

| Rank | Method | Steps |
|---:|---|---:|
| 1 | Contra-Muon | 3225 |
| 2 | NorMuon / NorMuonH family | 3250 |
| 3 | Muon `.035/.025` | 3325 |
| 4 | Muon `.025/.0125` | 3500 |
| 5 | AdamH | 4875 |
| 6 | AdamW | 5625 |

Our local proxy gets the broad class ordering right: Muon is far ahead of AdamH. But on seed `1340`, it ranks Muon `.025/.0125` ahead of `.035/.025`, which is backward relative to upstream. Treat the proxy as a useful filter, not a final judge.

## Latest Calibration Seed

Completed candidates from the current calibration campaign, seed `1340`, at proxy step `1000`:

| Rank | Candidate | Proxy val loss |
|---:|---|---:|
| 1 | Muon `.025/.0125` | `4.34916973` |
| 2 | Muon `.035/.025` | `4.36258984` |
| 3 | PopRisk-AdamH `snr-wiener` | `4.68850851` |
| 4 | PopRisk-AdamH `.003-w0` | `4.70422554` |
| 5 | PopRisk-AdamH `snr-var` | `4.70970201` |
| 6 | PopRisk-AdamH `.003-w50` | `4.72950554` |
| 7 | AdamH | `4.74654675` |

On this seed, every tested PopRisk-AdamH variant beats AdamH. The best one-seed result is `snr-wiener`.

## Previous Proxy Runs

The more reliable comparison is paired delta against same-seed AdamH, because raw proxy loss moves noticeably by seed.

| Candidate | Seeds | Mean delta vs AdamH | Interpretation |
|---|---:|---:|---|
| `poprisk-adamh-003-w50` | 4 | `-0.01123` | strongest replicated result; beat AdamH 4/4 |
| `poprisk-adamh-001-w50` | 3 | `-0.00538` | smaller but fairly consistent |
| `poprisk-adamh-adaptive-q050` | 3 | `-0.00244` | marginal/noisy |
| `poprisk-adamh-01` | 3 | `+0.00081` | not reliably helpful |

New one-seed results from seed `1340`:

| Candidate | Seed | Delta vs AdamH |
|---|---:|---:|
| `poprisk-adamh-snr-wiener` | 1340 | `-0.05804` |
| `poprisk-adamh-003-w0` | 1340 | `-0.04232` |
| `poprisk-adamh-snr-var` | 1340 | `-0.03684` |

## Current Triage

1. Most reliable: `poprisk-adamh-003-w50`
2. Most promising upside: `poprisk-adamh-snr-wiener`
3. Also worth confirming: `poprisk-adamh-003-w0`, `poprisk-adamh-snr-var`
4. Lower priority: `poprisk-adamh-001-w50`
5. Probably drop for now: `poprisk-adamh-01`, adaptive median-q variants, cosine decay

Practical read: previous proxy runs say `.003-w50` is real. The latest calibration seed says `snr-wiener` may be much better, but it only has one completed seed so far. Treat `snr-wiener` as the next thing to replicate, not yet as the champion.

## Reviewer-Priority Campaign

The main missing comparison is against the exact PopRisk threshold. Most variants above use continuous SNR shrinkage, not the hard or thresholded-soft PopRisk gate. The review campaign added for this gap is:

```bash
uv run python records/track_3_optimization/tools/run_ordinal_proxy.py \
  --campaign adamh-poprisk-review-8 \
  --estimated-minutes-per-run 24.5
```

It runs a self-contained fresh-seed block:

| Order | Candidate | Purpose |
|---:|---|---|
| 1 | `adamh` | same-seed baseline |
| 2 | `poprisk-adamh-003` | fixed-lambda SNR sanity check |
| 3 | `poprisk-adamh-hard` | exact binary PopRisk threshold |
| 4 | `poprisk-adamh-soft-003` | thresholded soft PopRisk gate, `lambda=.03` |
| 5 | `poprisk-adamh-adaptive-q050` | self-calibrating SNR gate, median `q=.50` |
| 6 | `poprisk-adamh-adaptive-q067` | self-calibrating SNR gate, median `q=.67` |
| 7 | `poprisk-adamh-cosine-003-zero` | time-varying SNR gate, `.03 -> 0` |
| 8 | `poprisk-adamh-snr-wiener` | parameter-free SNR shrinker |
