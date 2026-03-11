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
#   → Prints daily: Rule Acc / RL Acc / KL / ImitLoss / GradNorm per day
#   → Checkpoints saved per day to Data/Store/models/checkpoints/
#   → Latest always at Data/Store/models/phase1_latest.pth

# (Optional) Resume after interruption
python Leo_Lite.py --train-rl --phase 1 --resume

# (Optional) Walk-forward backtest
python Leo_Lite.py --backtest-rl --bt-start 2024-06-01
```

---

## After training — copy models to main LeoBook

Download the following files from Codespace and copy into **`LeoBook/Data/Store/models/`**:

```
Data/Store/models/leobook_base.pth        ← final trained model
Data/Store/models/phase1_latest.pth       ← latest phase 1 checkpoint  
Data/Store/models/adapter_registry.json   ← league/team adapter index
Data/Store/models/checkpoints/            ← daily checkpoints (optional)
```

These filenames are identical to what main LeoBook expects — it's a drop-in replace.

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
