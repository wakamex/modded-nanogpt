# Track 3 Local Proxy Ranking Snapshot

Date: 2026-05-07

This note summarizes the local ordinal proxy evidence for AdamH/PopRisk variants, with the latest upstream Track 3 ordering included for calibration. The proxy is useful for triage, but it is not yet a faithful ordinal model of official 3.28-step results.

Sources:

- `logs/ordinal_proxy_summary_7ed92aba-9138-4481-b63b-03f43e76a32f.json`
- `logs/ordinal_proxy_summary_63d9459b-01d2-4dc3-92e0-d06a54cbd50c.json`
- `logs/ordinal_proxy_summary_6deeda97-3f35-4539-b2c8-e9a6821b70c4.json`
- `logs/ordinal_proxy_summary_2c88c58d-86ef-4c5e-b945-bac7a8f40e56.json`
- `.codex_logs/ordinal-calibration-poprisk-6h_20260506_133502.out`
- `.codex_logs/adamh-poprisk-review-8_20260506_204951.out`
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

## PopRisk Theory Review Campaign

Completed candidates from `adamh-poprisk-review-8`, seed `1342`, at proxy step `1000`:

| Rank | Candidate | Proxy val loss | Delta vs AdamH |
|---:|---|---:|---:|
| 1 | `poprisk-adamh-snr-wiener` | `4.69129467` | `-0.08177662` |
| 2 | `poprisk-adamh-cosine-003-zero` | `4.76331377` | `-0.00975752` |
| 3 | `poprisk-adamh-adaptive-q067` | `4.77107906` | `-0.00199223` |
| 4 | `adamh` | `4.77307129` | `0.00000000` |
| 5 | `poprisk-adamh-adaptive-q050` | `4.77556753` | `+0.00249624` |
| 6 | `poprisk-adamh-003` | `4.77799892` | `+0.00492763` |
| 7 | `poprisk-adamh-soft-003` | `5.23263121` | `+0.45955992` |
| 8 | `poprisk-adamh-hard` | `5.24000216` | `+0.46693087` |

This review block resolves the main missing comparison: the exact hard PopRisk threshold and thresholded soft gate are much too aggressive in this proxy. The parameter-free continuous SNR shrinker remains the strongest candidate.

## All Paired Proxy Evidence

The more reliable comparison is paired delta against same-seed AdamH, because raw proxy loss moves noticeably by seed.

| Candidate | Seeds | Mean delta vs AdamH | Interpretation |
|---|---:|---:|---|
| `poprisk-adamh-snr-wiener` | 3 | `-0.06458` | strongest replicated result so far; parameter-free SNR shrinker |
| `poprisk-adamh-003-w0` | 2 | `-0.04126` | strong replicated no-warmup fixed-lambda SNR |
| `poprisk-adamh-snr-var` | 2 | `-0.03812` | strong replicated variance-scaled SNR |
| `poprisk-adamh-003-w50` | 5 | `-0.01219` | most replicated fixed-lambda result; beat AdamH 5/5 |
| `poprisk-adamh-cosine-003-zero` | 2 | `-0.00788` | mild positive signal; weaker than no-warmup SNR |
| `poprisk-adamh-001-w50` | 3 | `-0.00538` | smaller but fairly consistent |
| `poprisk-adamh-adaptive-q067` | 2 | `-0.00344` | marginal/noisy |
| `poprisk-adamh-adaptive-q050` | 4 | `-0.00121` | marginal/noisy |
| `poprisk-adamh-01` | 3 | `+0.00081` | not reliably helpful |
| `poprisk-adamh-003` | 2 | `+0.00233` | default warmup-100 version is not useful |
| `poprisk-adamh-soft-003` | 1 | `+0.45956` | thresholded soft PopRisk gate failed badly |
| `poprisk-adamh-hard` | 1 | `+0.46693` | exact hard PopRisk threshold failed badly |

## Current Triage

1. Current top proxy candidate: `poprisk-adamh-snr-wiener`
2. Also strong: `poprisk-adamh-003-w0`, `poprisk-adamh-snr-var`
3. Most replicated but smaller effect: `poprisk-adamh-003-w50`
4. Mild but lower priority: `poprisk-adamh-cosine-003-zero`, `poprisk-adamh-001-w50`
5. Probably drop for now: `poprisk-adamh-01`, adaptive median-q variants, warmup-100 `.003`
6. Reject in this proxy: exact hard/soft PopRisk threshold gates

Practical read: `.003-w50` is real, but the calibration and review blocks now say `snr-wiener` is much stronger on this proxy. The direct threshold tests suggest the theory-faithful hard/soft gates are over-filtering the AdamH matrix updates in this setting.

## Reviewer-Priority Campaign Definition

The review campaign that tested the exact PopRisk threshold was:

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
