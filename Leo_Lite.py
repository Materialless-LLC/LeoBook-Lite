# Leo_Lite.py — LeoBook Lite Orchestrator
# Purpose: RL training on Codespace. Only three commands:
#   --pull    → Supabase → local SQLite (get training data)
#   --train-rl → Run Phase 1/2/3 RL training
#   --sync    → Push local SQLite → Supabase (optional, rarely needed in Lite)
#
# After training: download Data/Store/models/ and drop into main LeoBook.

import asyncio
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR   = PROJECT_ROOT / "Data" / "Store" / "models"


def parse_args():
    p = argparse.ArgumentParser(
        description="LeoBook Lite — RL training and sync only"
    )
    # Training
    p.add_argument("--train-rl",    action="store_true", help="Run RL training")
    p.add_argument("--phase",       type=int, default=1, help="Training phase (1/2/3)")
    p.add_argument("--resume",      action="store_true", help="Resume from latest checkpoint")
    p.add_argument("--cold",        action="store_true", help="Start from random weights")
    p.add_argument("--limit-days",  type=int, default=None, help="Limit training days (debug)")

    # Data sync
    p.add_argument("--pull",  action="store_true", help="Pull training tables from Supabase → SQLite")
    p.add_argument("--sync",  action="store_true", help="Push local SQLite → Supabase")

    # Model sync (Supabase Storage)
    p.add_argument("--push-models", action="store_true", help="Upload trained models → Supabase Storage")
    p.add_argument("--pull-models", action="store_true", help="Download models from Supabase Storage → local")
    p.add_argument("--skip-large",  action="store_true", help="Skip files > 50 MB during --push-models")
    p.add_argument("--all-checkpoints", action="store_true", help="Force sync all files in checkpoints/ folder (default: False)")

    # Backtest
    p.add_argument("--backtest-rl",    action="store_true", help="Walk-forward RL backtest")
    p.add_argument("--bt-start",       default="2024-01-01")
    p.add_argument("--bt-end",         default=None)
    p.add_argument("--bt-train-days",  type=int, default=180)
    p.add_argument("--bt-output",      default="backtest_report.json")

    return p.parse_args()


# ── Init DB ──────────────────────────────────────────────────────────────────

def init_db():
    from Data.Access.db_helpers import init_csvs
    from Data.Access.league_db import init_db as _init_db
    init_csvs()
    return _init_db()


# ── Pull: Supabase → SQLite (training tables only) ───────────────────────────

# Tables the trainer actually needs:
TRAINING_TABLES = ["schedules", "teams", "leagues"]

async def cmd_pull():
    print("\n" + "=" * 60)
    print("  PULL: Supabase → local SQLite (training tables)")
    print("=" * 60)
    init_db()

    from Data.Access.sync_manager import SyncManager, TABLE_CONFIG
    sync = SyncManager()
    total = 0
    for tbl in TRAINING_TABLES:
        if tbl not in TABLE_CONFIG:
            print(f"  [!] '{tbl}' not in TABLE_CONFIG — skipped")
            continue
        print(f"\n  Pulling: {tbl}")
        pulled = await sync.batch_pull(tbl)
        total += pulled
        print(f"  ✓ {tbl}: {pulled:,} rows")

    print(f"\n  [DONE] Total pulled: {total:,} rows across {len(TRAINING_TABLES)} tables")
    print("  → Run: python Leo_Lite.py --train-rl --phase 1 --resume\n")


# ── Sync: local SQLite → Supabase (optional) ─────────────────────────────────

async def cmd_sync():
    print("\n" + "=" * 60)
    print("  SYNC: local SQLite → Supabase")
    print("=" * 60)
    init_db()
    from Data.Access.sync_manager import run_full_sync
    ok = await run_full_sync(session_name="Lite Manual Sync")
    if ok:
        print("  ✓ Sync complete.")
    else:
        print("  [!] Sync completed with warnings — check output above.")


# ── Train RL ─────────────────────────────────────────────────────────────────

def cmd_train_rl(args):
    print("\n" + "=" * 60)
    print(f"  TRAIN RL — Phase {args.phase} {'[RESUME]' if args.resume else ''} {'[COLD]' if args.cold else ''}")
    print("=" * 60)
    init_db()

    from Core.Intelligence.rl.trainer import RLTrainer
    trainer = RLTrainer()

    if args.phase > 1 and not args.cold:
        trainer.load()

    trainer.train_from_fixtures(
        phase=args.phase,
        cold=args.cold,
        limit_days=args.limit_days,
        resume=args.resume,
    )

    print(f"\n  [DONE] Model saved to: {MODELS_DIR}")
    print("  → Run: python Leo_Lite.py --push-models   (to upload to Supabase Storage)")
    print("  → Or download Data/Store/models/ manually and place into LeoBook/Data/Store/models/\n")


# ── Backtest RL ───────────────────────────────────────────────────────────────

def cmd_backtest_rl(args):
    from Core.Intelligence.rl.backtest import WalkForwardBacktester
    from datetime import datetime
    conn = init_db()
    bt_end = args.bt_end or datetime.now().strftime("%Y-%m-%d")
    bt = WalkForwardBacktester(conn, train_days=args.bt_train_days, eval_days=1)
    bt.run(args.bt_start, bt_end)
    bt._write_report(args.bt_output)
    print(f"  [Backtest] Report → {args.bt_output}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.pull:
        asyncio.run(cmd_pull())

    elif args.sync:
        asyncio.run(cmd_sync())

    elif args.push_models:
        from Data.Access.model_sync import ModelSync
        ModelSync(
            skip_large=getattr(args, 'skip_large', False),
            all_checkpoints=getattr(args, 'all_checkpoints', False)
        ).push()

    elif args.pull_models:
        from Data.Access.model_sync import ModelSync
        ModelSync().pull()

    elif args.train_rl:
        cmd_train_rl(args)

    elif args.backtest_rl:
        cmd_backtest_rl(args)

    else:
        print("\nLeoBook Lite — Usage:")
        print("  python Leo_Lite.py --pull                         # Fetch training data from Supabase")
        print("  python Leo_Lite.py --train-rl --phase 1 --resume  # Train RL (Phase 1)")
        print("  python Leo_Lite.py --train-rl --phase 1 --cold    # Cold start")
        print("  python Leo_Lite.py --push-models                  # Upload models → Supabase Storage")
        print("  python Leo_Lite.py --pull-models                  # Download models ← Supabase Storage")
        print("  python Leo_Lite.py --backtest-rl                  # Walk-forward backtest")
        print("  python Leo_Lite.py --sync                         # Push local → Supabase\n")
