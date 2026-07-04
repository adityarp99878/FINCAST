"""
setup_kaggle_data.py
---------------------
One-click script to download all Kaggle datasets for FinCast.

Usage:
    python setup_kaggle_data.py

Prerequisites:
    1. pip install kaggle  (already done if you ran requirements)
    2. Place kaggle.json at C:\\Users\\<you>\\.kaggle\\kaggle.json
       Get it from: kaggle.com → Account → API → Create New Token
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("setup")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("\n" + "="*60)
    print("  FinCast — Kaggle Dataset Setup")
    print("="*60 + "\n")

    # ── Step 1: Check credentials ────────────────────────────────
    from utils.kaggle_downloader import check_credentials, download_all, status

    print("Step 1: Checking Kaggle credentials...")
    if not check_credentials():
        print("\n❌ Setup failed — kaggle.json not found.")
        print("\nTo fix:")
        print("  1. Go to https://www.kaggle.com → Account → API")
        print("  2. Click 'Create New Token'")
        print(f"  3. Move kaggle.json to: {os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')}")
        sys.exit(1)

    print("  ✅ Credentials OK\n")

    # ── Step 2: Show current status ──────────────────────────────
    print("Step 2: Current dataset status:")
    s = status()
    any_missing = False
    for name, info in s.items():
        icon = "✅" if info["present"] else "❌"
        print(f"  {icon}  {name:18s}  {info['desc']}")
        if not info["present"]:
            any_missing = True
    print()

    if not any_missing:
        print("All datasets already downloaded! Nothing to do.\n")
        print("To re-download, delete the data/external/ folder and run again.")
        sys.exit(0)

    # ── Step 3: Download missing datasets ───────────────────────
    print("Step 3: Downloading missing datasets...")
    print("  (This may take a few minutes depending on your internet speed)\n")

    results = download_all(force=False)

    # ── Step 4: Summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  Download Summary")
    print("="*60)
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        desc = s[name]["desc"]
        print(f"  {icon}  {name:18s}  {desc}")

    passed = sum(results.values())
    print(f"\n  {passed}/{len(results)} datasets ready.\n")

    if passed == len(results):
        print("✅ All datasets downloaded successfully!")
        print("   Restart the Flask app — the ML pipeline will now use the enriched data.\n")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"⚠️  Failed: {failed}")
        print("   The pipeline will fall back to yfinance for missing datasets.\n")

    # ── Step 5: Quick test ───────────────────────────────────────
    print("Step 4: Verifying external data loads correctly...")
    try:
        from utils.external_data import load_all_external
        ext = load_all_external(start="2020-01-01")
        for name, df in ext.items():
            if df.empty:
                print(f"  ⚠️   {name:10s}  — empty (will use yfinance fallback)")
            else:
                print(f"  ✅  {name:10s}  {len(df):6d} rows   "
                      f"[{df.index[0].date()} → {df.index[-1].date()}]")
    except Exception as e:
        print(f"  ⚠️  Verification failed: {e}")

    print("\n✅ Setup complete. Restart app.py to apply.\n")


if __name__ == "__main__":
    main()
