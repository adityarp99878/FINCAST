"""
utils/kaggle_downloader.py
--------------------------
Downloads external datasets from Kaggle into data/external/.
Requires kaggle.json at C:\\Users\\<user>\\.kaggle\\kaggle.json

Datasets downloaded:
  - lbronchal/gold-and-silver-prices-dataset   → gold_silver/
  - meehau/EURUSD                               → forex/eurusd/
  - sid321axn/cboe-volatility-index-vix-time-series-data → vix/

Usage:
    python utils/kaggle_downloader.py
    python utils/kaggle_downloader.py --dataset gold_silver
    python utils/kaggle_downloader.py --check   (just verify credentials)
"""

import os
import sys
import logging
import argparse
import shutil

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.join(_HERE, "..")
_EXT_DIR = os.path.join(_ROOT, "data", "external")

# ── Dataset registry ─────────────────────────────────────────────────────────
DATASETS = {
    "gold_silver": {
        "slug":    "lbronchal/gold-and-silver-prices-dataset",
        "dest":    os.path.join(_EXT_DIR, "gold_silver"),
        "desc":    "Daily Gold & Silver spot prices (1978–2023)",
        "key_file": "gold_price.csv",
    },
    "eurusd": {
        "slug":    "meehau/EURUSD",
        "dest":    os.path.join(_EXT_DIR, "forex", "eurusd"),
        "desc":    "EUR/USD hourly forex data",
        "key_file": "EURUSD_H1.csv",
    },
    "vix": {
        "slug":    "sid321axn/cboe-volatility-index-vix-time-series-data",
        "dest":    os.path.join(_EXT_DIR, "vix"),
        "desc":    "CBOE VIX Fear Index (daily, 1990–present)",
        "key_file": "vix-daily.csv",
    },
    "india_stocks": {
        "slug":    "rohanrao/nifty50-stock-data",
        "dest":    os.path.join(_EXT_DIR, "india"),
        "desc":    "Nifty 50 historical daily OHLCV",
        "key_file": "RELIANCE.csv",
    },
    "us_stocks_sp500": {
        "slug":    "andrewmvd/sp-500-stocks",
        "dest":    os.path.join(_EXT_DIR, "us"),
        "desc":    "S&P 500 individual stock OHLCV (2010–2023)",
        "key_file": "sp500_stocks.csv",
    },
    "us_index_sp500": {
        "slug":    "arashnic/mahindra-and-mahindra-stock-data",  # placeholder
        "dest":    os.path.join(_EXT_DIR, "us"),
        "desc":    "S&P 500 index level (2010–2023)",
        "key_file": "sp500_index.csv",
    },
}


def _write_kaggle_json_from_token(token: str) -> bool:
    """
    Write the KGAT_ token to ~/.kaggle/access_token (new Kaggle format)
    and also set KAGGLE_API_TOKEN in the environment for subprocess calls.
    """
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    access_token_path = os.path.join(kaggle_dir, "access_token")
    try:
        with open(access_token_path, "w") as f:
            f.write(token)
        try:
            os.chmod(access_token_path, 0o600)
        except Exception:
            pass
        # Set env var so subprocess kaggle CLI picks it up
        os.environ["KAGGLE_API_TOKEN"] = token
        log.info("Saved KGAT token to %s", access_token_path)
        return True
    except Exception as e:
        log.error("Failed to write access_token: %s", e)
        return False



def check_credentials() -> bool:
    """
    Return True if valid Kaggle credentials are found.
    Checks (in order):
      1. KAGGLE_API_TOKEN env var  (new KGAT_... format)
      2. KAGGLE_TOKEN env var  (alias)
      3. KAGGLE_USERNAME + KAGGLE_KEY env vars  (old format)
      4. ~/.kaggle/access_token  (new file format)
      5. ~/.kaggle/kaggle.json  (old file format)
    """
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.environ["KAGGLE_CONFIG_DIR"] = kaggle_dir

    # ── 1. New token format via env var ──────────────────────────
    token = os.environ.get("KAGGLE_TOKEN", "").strip()
    if token.startswith("KGAT_"):
        log.info("Found KAGGLE_TOKEN env var (new KGAT format)")
        _write_kaggle_json_from_token(token)
        return True

    # ── 2. Old env var style ──────────────────────────────────────
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key      = os.environ.get("KAGGLE_KEY", "").strip()
    if username and key:
        log.info("Found KAGGLE_USERNAME + KAGGLE_KEY env vars")
        import json
        creds_path = os.path.join(kaggle_dir, "kaggle.json")
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(creds_path, "w") as f:
            json.dump({"username": username, "key": key}, f)
        return True

    # ── 3. kaggle.json file ───────────────────────────────────────
    creds_path = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(creds_path):
        log.error("No Kaggle credentials found!")
        log.error("")
        log.error("You have two options:")
        log.error("")
        log.error("  OPTION A — Use your KGAT token (PowerShell):")
        log.error('    $env:KAGGLE_TOKEN = "KGAT_dc2f05bced3da2a..."')
        log.error('    python setup_kaggle_data.py')
        log.error("")
        log.error("  OPTION B — Create kaggle.json manually:")
        log.error('    mkdir C:\\Users\\%s\\.kaggle' % os.environ.get("USERNAME", "YOU"))
        log.error('    Then create the file with content:')
        log.error('    {"token": "KGAT_your_token_here"}')
        return False

    try:
        import json
        creds = json.loads(open(creds_path).read())
        # Accept both old {"username","key"} and new {"token"} formats
        if "token" in creds and creds["token"].startswith("KGAT_"):
            log.info("Kaggle credentials OK (new KGAT token format)")
            os.environ["KAGGLE_TOKEN"] = creds["token"]
            return True
        elif "username" in creds and "key" in creds:
            log.info("Kaggle credentials OK (classic format, user: %s)", creds["username"])
            return True
        else:
            log.error("kaggle.json is malformed — need {token} or {username, key}")
            return False
    except Exception as e:
        log.error("Failed to parse kaggle.json: %s", e)
        return False


def download_dataset(name: str, force: bool = False) -> bool:
    """
    Download a single Kaggle dataset using the kaggle CLI.
    The CLI supports new KGAT_ tokens via KAGGLE_TOKEN env var.
    """
    if name not in DATASETS:
        log.error("Unknown dataset '%s'. Available: %s", name, list(DATASETS.keys()))
        return False

    ds   = DATASETS[name]
    dest = ds["dest"]
    key  = os.path.join(dest, ds["key_file"])

    if not force and os.path.exists(key):
        log.info("[%s] Already downloaded - skipping", name)
        return True

    os.makedirs(dest, exist_ok=True)
    log.info("[%s] Downloading: %s", name, ds["desc"])
    log.info("[%s] Slug: %s", name, ds["slug"])

    # Build subprocess env with KAGGLE_TOKEN set
    import subprocess
    env = os.environ.copy()

    # Ensure KAGGLE_TOKEN is set from kaggle.json if not already in env
    if not env.get("KAGGLE_TOKEN"):
        creds_path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
        if os.path.exists(creds_path):
            try:
                import json as _j
                creds = _j.loads(open(creds_path, encoding="utf-8").read())
                if "token" in creds:
                    env["KAGGLE_TOKEN"] = creds["token"]
            except Exception:
                pass

    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "--dataset", ds["slug"],
        "--path",    dest,
        "--unzip",
    ]
    if force:
        cmd.append("--force")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=300
        )
        if result.returncode == 0:
            files = os.listdir(dest) if os.path.isdir(dest) else []
            log.info("[%s] Downloaded OK. Files: %s", name, files)
            return True
        else:
            log.error("[%s] CLI error (exit %d):", name, result.returncode)
            if result.stdout:
                log.error("  stdout: %s", result.stdout.strip())
            if result.stderr:
                log.error("  stderr: %s", result.stderr.strip())
            return False
    except subprocess.TimeoutExpired:
        log.error("[%s] Timed out after 5 minutes", name)
        return False
    except Exception as e:
        log.error("[%s] Unexpected error: %s", name, e)
        return False


def download_all(force: bool = False) -> dict:
    """Download all registered datasets. Returns {name: success} dict."""
    if not check_credentials():
        return {k: False for k in DATASETS}

    results = {}
    for name in DATASETS:
        results[name] = download_dataset(name, force=force)

    passed = sum(results.values())
    log.info("=== Download complete: %d/%d datasets ===", passed, len(results))
    return results


def status() -> dict:
    """
    Check which datasets are already present locally.
    Returns {name: {"present": bool, "path": str, "desc": str}}
    """
    out = {}
    for name, ds in DATASETS.items():
        key     = os.path.join(ds["dest"], ds["key_file"])
        present = os.path.exists(key)
        out[name] = {
            "present": present,
            "path":    ds["dest"],
            "desc":    ds["desc"],
            "slug":    ds["slug"],
        }
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinCast Kaggle Dataset Downloader")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"],
                        default="all", help="Which dataset to download")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--check", action="store_true", help="Only check credentials")
    parser.add_argument("--status", action="store_true", help="Show download status")
    args = parser.parse_args()

    if args.check:
        ok = check_credentials()
        sys.exit(0 if ok else 1)

    if args.status:
        s = status()
        print("\n── Dataset Status ──────────────────────────────")
        for name, info in s.items():
            icon = "✅" if info["present"] else "❌"
            print(f"  {icon}  {name:15s}  {info['desc']}")
        print()
        sys.exit(0)

    if not check_credentials():
        sys.exit(1)

    if args.dataset == "all":
        download_all(force=args.force)
    else:
        ok = download_dataset(args.dataset, force=args.force)
        sys.exit(0 if ok else 1)
