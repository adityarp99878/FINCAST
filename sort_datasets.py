"""
sort_datasets.py
-----------------
STEP 1: Extract all your archive ZIPs into:
    data\external\inbox\

STEP 2: Run this script:
    python sort_datasets.py

It reads every CSV, identifies the dataset by column names,
and moves it to the correct data\external\ subfolder automatically.
"""

import os, sys, glob, shutil, zipfile
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("sorter")

ROOT    = os.path.dirname(os.path.abspath(__file__))
EXT     = os.path.join(ROOT, "data", "external")
INBOX   = os.path.join(EXT, "inbox")

# ── Where each category of file should land ──────────────────────────────────
DESTINATIONS = {
    "gold":         os.path.join(EXT, "gold_silver"),
    "silver":       os.path.join(EXT, "gold_silver"),
    "vix":          os.path.join(EXT, "vix"),
    "eurusd":       os.path.join(EXT, "forex", "eurusd"),
    "forex":        os.path.join(EXT, "forex", "eurusd"),
    "india":        os.path.join(EXT, "india"),
    "nse":          os.path.join(EXT, "india"),
    "nifty":        os.path.join(EXT, "india"),
}

# ── Column-header fingerprints for each dataset ───────────────────────────────
FINGERPRINTS = [
    # (dataset_key, required_keywords_in_any_column)
    ("gold",   ["gold"]),
    ("silver", ["silver"]),
    ("vix",    ["vix"]),
    ("eurusd", ["eur", "eurusd"]),
    ("india",  ["open", "high", "low", "close", "volume"]),   # generic OHLCV maps to india
    ("nifty",  ["symbol", "series", "prev"]),                 # NSE format
    ("india",  ["nifty", "sensex", "bse", "nse"]),
]

# Additional: if the filename contains these keywords
FILENAME_HINTS = {
    "gold":   ["gold"],
    "silver": ["silver"],
    "vix":    ["vix", "volatility", "cboe"],
    "eurusd": ["eurusd", "eur_usd", "eur-usd"],
    "forex":  ["forex"],
    "nifty":  ["nifty", "nse", "india"],
}


def identify_csv(path: str):
    """Return the dataset key for a CSV file, or None if unrecognised."""
    fname = os.path.basename(path).lower()

    # 1. Filename hint
    for key, hints in FILENAME_HINTS.items():
        if any(h in fname for h in hints):
            return key

    # 2. Column header scan
    try:
        df = pd.read_csv(path, nrows=5, encoding="utf-8", low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(path, nrows=5, encoding="latin-1", low_memory=False)
        except Exception:
            return None

    cols_lower = " ".join(df.columns.str.lower().tolist())

    for key, kws in FINGERPRINTS:
        if all(k in cols_lower for k in kws):
            return key

    return None


def extract_zips(src_folder: str, dest_folder: str):
    """Extract every ZIP found in src_folder into dest_folder."""
    zips = glob.glob(os.path.join(src_folder, "*.zip"))
    if not zips:
        log.info("No ZIPs found in %s", src_folder)
        return
    os.makedirs(dest_folder, exist_ok=True)
    for z in zips:
        log.info("Extracting %s ...", os.path.basename(z))
        try:
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(dest_folder)
            log.info("  Extracted %d files", len(zipfile.ZipFile(z).namelist()))
        except Exception as e:
            log.error("  Failed: %s", e)


def sort_csvs(inbox: str):
    """Scan inbox for CSVs, identify each, move to correct subfolder."""
    csvs = glob.glob(os.path.join(inbox, "**", "*.csv"), recursive=True)
    if not csvs:
        log.warning("No CSV files found in %s", inbox)
        return

    moved, skipped = 0, 0
    for path in csvs:
        key = identify_csv(path)
        if key is None:
            log.warning("  Could not identify: %s", os.path.basename(path))
            skipped += 1
            continue

        dest_dir = DESTINATIONS[key]
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(path))

        if os.path.exists(dest_path):
            log.info("  [SKIP-EXISTS] %s", os.path.basename(path))
            continue

        shutil.move(path, dest_path)
        log.info("  [%s] %s  ->  %s", key.upper(), os.path.basename(path),
                 dest_dir.replace(ROOT, "."))
        moved += 1

    log.info("Done: %d moved, %d skipped/unrecognised", moved, skipped)


def main():
    print("\n" + "="*60)
    print("  FinCast Dataset Sorter")
    print("="*60)

    # ── If ZIPs passed as arguments, extract them to inbox ───────
    zip_args = [a for a in sys.argv[1:] if a.endswith(".zip")]
    if zip_args:
        os.makedirs(INBOX, exist_ok=True)
        for z in zip_args:
            log.info("Extracting %s ...", z)
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(INBOX)
        print()

    # ── Also extract any ZIPs already in inbox ───────────────────
    extract_zips(INBOX, INBOX)

    print()
    print("Scanning and sorting CSVs...")
    print()
    sort_csvs(INBOX)

    print()
    print("Verifying final layout:")
    for name, folder in {
        "gold_silver": os.path.join(EXT, "gold_silver"),
        "vix":         os.path.join(EXT, "vix"),
        "forex/eurusd":os.path.join(EXT, "forex", "eurusd"),
        "india":       os.path.join(EXT, "india"),
    }.items():
        csvs = glob.glob(os.path.join(folder, "*.csv"))
        status = f"{len(csvs)} CSV(s)" if csvs else "EMPTY"
        names  = [os.path.basename(c) for c in csvs[:3]]
        print(f"  [{status:12s}] data/external/{name}/  {names}")

    print()
    print("Run the pipeline now to use the new datasets.")
    print()


if __name__ == "__main__":
    main()
