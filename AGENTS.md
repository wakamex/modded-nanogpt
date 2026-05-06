# Repository Instructions

- Use `api.fxtwitter.com` to read tweets.
- Access Reddit with curl using a browser User-Agent:
  `curl -s -A "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0" "https://old.reddit.com/r/{sub}/comments/{post_id}/.json"`
  This returns post + comments as JSON.

## GitHub Activity

Stateless GitHub activity summary with compact, LLM-friendly output:

```bash
gh-pulse [-h] [--since TIME] [--json] [--me HANDLE] [--repo OWNER/REPO] [--limit LIMIT]
```

Options:

- `--since TIME`: show deltas since TIME, ISO 8601 or relative such as `30m`, `2h`, `1d`, `1w`.
- `--json`: output JSON.
- `--me HANDLE`: highlight mentions of this handle, for example `@clod`.
- `--repo OWNER/REPO`: GitHub repo, default is auto-detect from git remote.
- `--limit LIMIT`: max items per category, default `30`.

## Vast Track 3 Helper

Use `/code/scripts/vast_track3.py` for Vast.ai searches and rentals for Track 3 experiments. It wraps the official Vast CLI through `uvx`, loads `VAST_AI_KEY`/`VAST_API_KEY` from `.env`, and writes rent/destroy manifests under `.codex_logs/vast`.

Search cheap unverified/p2p 4xH100 offers:

```bash
/code/scripts/vast_track3.py search --gpu h100 --gpus 4 --verified false --limit 10 --order dph
```

Search 2x3090 offers:

```bash
/code/scripts/vast_track3.py search --gpu 3090 --gpus 2 --verified false --limit 10 --order dph
```

Dry-run a rental before spending money:

```bash
/code/scripts/vast_track3.py rent OFFER_ID --label track3-4xh100 --track3-onstart --dry-run
```

Rent with Track 3 setup on startup:

```bash
/code/scripts/vast_track3.py rent OFFER_ID --label track3-4xh100 --track3-onstart
```

Common instance operations:

```bash
/code/scripts/vast_track3.py instances
/code/scripts/vast_track3.py ssh INSTANCE_ID
/code/scripts/vast_track3.py logs INSTANCE_ID --tail 200
/code/scripts/vast_track3.py destroy INSTANCE_ID --yes
```

The `--track3-onstart` script clones upstream `modded-nanogpt`, creates a uv venv, installs `torch==2.11` plus `huggingface_hub`, and downloads FineWeb chunks. Add `--run-after-setup` only when the training run should start automatically after setup.

Always check `instances` after a run and destroy unused Vast instances promptly.
