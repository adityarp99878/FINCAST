"""
download_datasets.py
--------------------
Downloads the proper Gold/Silver, VIX, EUR/USD Forex, Nifty 50 and S&P 500
datasets directly from the Kaggle REST API.
Public datasets do not require authentication.

Usage:
    python download_datasets.py
"""

import os, sys, zipfile, time
import requests

HERE    = os.path.dirname(os.path.abspath(__file__))
EXT_DIR = os.path.join(HERE, "data", "external")

DATASETS = [
    {
        "name":  "Gold & Silver Prices",
        "owner": "lbronchal",
        "slug":  "gold-and-silver-prices-dataset",
        "dest":  os.path.join(EXT_DIR, "gold_silver"),
    },
    {
        "name":  "VIX Fear Index",
        "owner": "sid321axn",
        "slug":  "cboe-volatility-index-vix-time-series-data",
        "dest":  os.path.join(EXT_DIR, "vix"),
    },
    {
        "name":  "EUR/USD Forex",
        "owner": "meehau",
        "slug":  "EURUSD",
        "dest":  os.path.join(EXT_DIR, "forex", "eurusd"),
    },
    {
        "name":  "Nifty 50 Indian Stocks",
        "owner": "rohanrao",
        "slug":  "nifty50-stock-data",
        "dest":  os.path.join(EXT_DIR, "india"),
    },
    {
        "name":  "S&P 500 US Stocks",
        "owner": "andrewmvd",
        "slug":  "sp-500-stocks",
        "dest":  os.path.join(EXT_DIR, "us"),
    },
]

# Public datasets — no auth needed
HEADERS = {"User-Agent": "Mozilla/5.0"}


def download_dataset(ds: dict) -> bool:
    name  = ds["name"]
    owner = ds["owner"]
    slug  = ds["slug"]
    dest  = ds["dest"]

    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "_download.zip")

    url = f"https://www.kaggle.com/api/v1/datasets/{owner}/{slug}/download"
    print(f"\n[{name}] Downloading from Kaggle...")
    print(f"  URL: {url}")

    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=300) as r:
            if r.status_code == 401:
                print(f"  ERROR: Token rejected (401). Token may be expired.")
                return False
            if r.status_code == 403:
                print(f"  ERROR: Access denied (403). Accept dataset rules on Kaggle first.")
                return False
            if r.status_code != 200:
                print(f"  ERROR: HTTP {r.status_code}: {r.text[:200]}")
                return False

            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"  {pct:.1f}%  ({downloaded//1024//1024} MB / {total//1024//1024} MB)", end="\r")
            print(f"\n  Downloaded {downloaded//1024//1024} MB")

        print(f"  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        os.remove(zip_path)

        files = [f for f in os.listdir(dest) if f.endswith(".csv")]
        print(f"  OK — {len(files)} CSV files in {dest}")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  FinCast — Kaggle Dataset Downloader")
    print("=" * 60)

    results = {}
    for ds in DATASETS:
        ok = download_dataset(ds)
        results[ds["name"]] = ok
        time.sleep(1)  # be polite to Kaggle servers

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, ok in results.items():
        icon = "OK" if ok else "FAIL"
        print(f"  [{icon}]  {name}")
    print()
