# LeoBook Lite

A stripped-down version of LeoBook containing **only the RL training system**. Run training on GitHub Codespace, then download the model files and drop them into the main LeoBook.

---

## What's included

| Path | Purpose |
|---|---|
| `Leo_Lite.py` | CLI orchestrator (train, pull, sync) |
| `Core/Intelligence/rl/` | PPO trainer, model, feature encoder, adapters |
| `Core/Intelligence/rule_engine.py` + deps | Expert signal for Phase 1 imitation learning |
| `Data/Access/` | SQLite helpers + Supabase sync manager |
| `Data/Store/models/` | **Training output — files to download** |

## What's excluded

Everything else: Playwright, browser scrapers, betting automation, enrichment scripts, scheduling, guardrails.

---

## Setup (Codespace or any Linux machine)

```bash
# 1. Clone repo and install deps
pip install -r requirements-rl.txt

# 2. Set Supabase credentials
cp .env.example .env
# Edit .env → fill SUPABASE_URL and SUPABASE_KEY

# 3. Pull training data (schedules, teams, leagues)
python Leo_Lite.py --pull
#   → Writes Data/Store/leobook.db (may be ~150MB depending on data)

# 4. Train — Phase 1: Imitation Learning
python Leo_Lite.py --train-rl --phase 1 --resume
#   → Prints daily training stats

# 5. Sync Models to Supabase Storage
python Leo_Lite.py --push-models            # Upload trained models
python Leo_Lite.py --push-models --skip-large # Upload ONLY final trunk + adapters
```

---

## After training — copy models to main LeoBook

You can use the automated sync:
1. In Codespace: `python Leo_Lite.py --push-models`
2. Locally: `python Leo.py --pull-models`

Or download manually from **`Data/Store/models/`**:
- `leobook_base.pth`
- `adapter_registry.json`
- `phase1_latest.pth`

---

## Troubleshooting

- **ModuleNotFoundError: No module named 'playwright'**: This is fixed. LeoBook Lite is now independent of Playwright for training.
- **Sync hanging**: If a checkpoint is >500MB, it may take a few minutes. Use the progress indicators to monitor speed.


---

## Training phases

| Phase | Trigger | Method | Signal |
|---|---|---|---|
| 1 | Default (< 100 odds rows) | Imitation Learning | Rule Engine + Poisson probs |
| 2 | Auto (≥ 100 odds rows, ≥ 7 days live) | PPO + KL penalty | Live odds reward |
| 3 | Auto (≥ 500 odds rows, ≥ 14 days live) | Adapter fine-tuning | Frozen trunk |

The trainer auto-detects the active phase based on your data. For a fresh Codespace with historical data only, Phase 1 will run.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | ✓ | Your Supabase project URL |
| `SUPABASE_KEY` | ✓ | Anon or service-role key |
