# Track 3 Local Proxy Ranking Snapshot

Date: 2026-05-06

This note summarizes the local ordinal proxy evidence for AdamH/PopRisk variants, with the latest upstream Track 3 ordering included for calibration. The proxy is useful for triage, but it is not yet a faithful ordinal model of official 3.28-step results.

Sources:

- `logs/ordinal_proxy_summary_7ed92aba-9138-4481-b63b-03f43e76a32f.json`
- `logs/ordinal_proxy_summary_63d9459b-01d2-4dc3-92e0-d06a54cbd50c.json`
- `logs/ordinal_proxy_summary_6deeda97-3f35-4539-b2c8-e9a6821b70c4.json`
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

Our local proxy gets the broad class ordering right: Muon is far ahead of AdamH. But across seeds `1340` and `1341`, it ranks Muon `.025/.0125` ahead of `.035/.025`, which is backward relative to upstream. Treat the proxy as a useful filter, not a final judge.

## Latest Calibration Campaign

Completed candidates from `ordinal-calibration-poprisk-6h`, seeds `1340` and `1341`, at proxy step `1000`:

| Rank | Candidate | Mean proxy val loss | Mean delta vs AdamH |
|---:|---|---:|---:|
| 1 | Muon `.025/.0125` | `4.35095048` | `-0.40175891` |
| 2 | Muon `.035/.025` | `4.36343908` | `-0.38927031` |
| 3 | PopRisk-AdamH `snr-wiener` | `4.69672513` | `-0.05598426` |
| 4 | PopRisk-AdamH `.003-w0` | `4.71144605` | `-0.04126334` |
| 5 | PopRisk-AdamH `snr-var` | `4.71458721` | `-0.03812218` |
| 6 | PopRisk-AdamH `.003-w50` | `4.73618412` | `-0.01652527` |
| 7 | AdamH | `4.75270939` | `0.00000000` |

Across both seeds, every tested PopRisk-AdamH variant beats AdamH. The best replicated result in this calibration block is `snr-wiener`.

## All Paired Proxy Evidence

The more reliable comparison is paired delta against same-seed AdamH, because raw proxy loss moves noticeably by seed.

| Candidate | Seeds | Mean delta vs AdamH | Interpretation |
|---|---:|---:|---|
| `poprisk-adamh-snr-wiener` | 2 | `-0.05598` | strongest replicated result so far; parameter-free SNR shrinker |
| `poprisk-adamh-003-w0` | 2 | `-0.04126` | strong replicated no-warmup fixed-lambda SNR |
| `poprisk-adamh-snr-var` | 2 | `-0.03812` | strong replicated variance-scaled SNR |
| `poprisk-adamh-003-w50` | 5 | `-0.01219` | most replicated fixed-lambda result; beat AdamH 5/5 |
| `poprisk-adamh-001-w50` | 3 | `-0.00538` | smaller but fairly consistent |
| `poprisk-adamh-adaptive-q050` | 3 | `-0.00244` | marginal/noisy |
| `poprisk-adamh-01` | 3 | `+0.00081` | not reliably helpful |

## Current Triage

1. Current top proxy candidate: `poprisk-adamh-snr-wiener`
2. Also strong: `poprisk-adamh-003-w0`, `poprisk-adamh-snr-var`
3. Most replicated but smaller effect: `poprisk-adamh-003-w50`
4. Lower priority: `poprisk-adamh-001-w50`
5. Probably drop for now: `poprisk-adamh-01`, adaptive median-q variants, cosine decay

Practical read: `.003-w50` is real, but the two-seed calibration block now says `snr-wiener` is much stronger on this proxy. The major missing comparison remains against exact hard/soft PopRisk threshold gates.

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
