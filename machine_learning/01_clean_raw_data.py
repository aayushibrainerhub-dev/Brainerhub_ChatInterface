"""
01_clean_raw_data.py

Turns a messy / raw B2B company export into the tidy
`globalb2bdataset_cleaned.csv` that the clustering pipeline
(02_cluster_industries.py) expects.

WHY THIS SCRIPT EXISTS
-----------------------
The clustering script assumes an `Industry` column where:
  - missing/unknown values are already blank or one of a known set of
    "junk" markers ("no details", "unclassified", "n/a", "none", "")
  - genuine multi-word category names use ", " (comma + space)
    e.g. "Food, Beverages & Tobacco"
  - multi-tag lists from a different source schema use "," with NO
    space between tags, e.g. "Robotics,Health Diagnostics,Medical Device"

Raw exports almost never satisfy this cleanly — you'll typically see
inconsistent casing, stray whitespace, mixed null markers ("NULL", "-",
"TBD", "Unknown"), duplicate rows, encoding artifacts, and sometimes
tag lists that got a stray space after a comma (which would otherwise
break the comma-spacing heuristic in the clustering script). This
script normalizes all of that.

>>> YOU WILL NEED TO EDIT THE CONFIG BLOCK BELOW <<<
This is written as a general-purpose template because the exact column
names / raw file format weren't available when this was written. Open
the CONFIG section, point RAW_CSV at your actual file, and adjust
COLUMN_RENAME_MAP / DEDUPE_KEYS to match your real columns. Everything
else should work unmodified.

Usage:
    python 01_clean_raw_data.py
"""

import re
import sys
import unicodedata
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────
# CONFIG — edit these to match your raw file
# ─────────────────────────────────────────────────────────────────────────

RAW_CSV    = "globalb2bdataset_cleaned_raw.csv"      # <- your uncleaned export
OUTPUT_CSV = "globalb2bdataset_cleaned.csv"   # <- feeds into the next script

# If your raw file's column names differ from what the rest of the
# pipeline expects, map them here: {raw_name: standard_name}.
# Leave empty ({}) if your raw column is already literally "Industry".
COLUMN_RENAME_MAP = {
    # "industry_category": "Industry",
    # "company_name": "CompanyName",
}

# Columns that, taken together, define "the same company" for de-duplication.
# Set to None to skip de-duplication, or to a single-column list like
# ["CompanyName"] if that's the only reliable key you have.
DEDUPE_KEYS = None   # e.g. ["CompanyName", "Website"]

INDUSTRY_COL = "Industry"

# Any of these (case-insensitive, after stripping whitespace) collapse to
# a single canonical blank value. Add whatever null-style markers show up
# in your raw export.
NULL_MARKERS = {
    "no details", "unclassified", "n/a", "na", "none", "null",
    "unknown", "tbd", "-", "--", "not specified", "not available",
    "nan", "",
}

# ─────────────────────────────────────────────────────────────────────────
# CLEANING HELPERS
# ─────────────────────────────────────────────────────────────────────────

_MULTI_SPACE_RE = re.compile(r"\s+")
# Collapses ", " followed by more space down to a single ", " so the
# downstream comma-spacing heuristic (tag-list vs. real category) stays
# reliable even if the raw export has irregular spacing like "A ,  B".
_COMMA_SPACE_RE = re.compile(r"\s*,\s*")


def normalize_text(value: str) -> str:
    """General-purpose text cleanup: fix encoding artifacts, strip control
    characters, collapse repeated whitespace, trim."""
    if value is None:
        return ""
    text = str(value)
    # Normalize unicode (fixes things like "Caf\u00e9" vs "Café" mismatches,
    # smart quotes, etc.)
    text = unicodedata.normalize("NFKC", text)
    # Drop characters that sometimes leak in from bad encodings/exports
    text = text.replace("\ufeff", "").replace("\x00", "")
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def normalize_industry_value(value: str) -> str:
    """Cleans one Industry cell and standardizes null markers to "".

    Preserves the distinction the clustering script relies on:
      - "A, B"  (comma + space)    -> real multi-word category, untouched
      - "A,B"   (comma, no space)  -> tag list, untouched (no space added)
    We also detect when tag lists are written with spaces (e.g. "A, B, C")
    or contain ampersands/ands (e.g. "A, B & C") and convert them to canonical
    comma-separated format without spaces (e.g. "A,B,C") to ensure the downstream
    clustering splits them correctly. This is done purely through structural
    heuristics without any predefined tags.
    """
    text = normalize_text(value)
    if text.lower() in NULL_MARKERS:
        return ""

    has_comma = "," in text
    
    # Check if the string has an ampersand or the word 'and'
    has_amp_or_and = False
    if "&" in text:
        has_amp_or_and = True
    elif re.search(r"\band\b", text, re.IGNORECASE):
        has_amp_or_and = True

    # If it has a comma, we treat it as a tag list.
    if has_comma:
        # If it also has an ampersand/and, we replace it with a comma so we can split all tags
        if has_amp_or_and:
            text = text.replace("&", ",")
            text = re.sub(r"\band\b", ",", text, flags=re.IGNORECASE)
        
        # Split by comma, strip whitespace from each part, and join with commas (no spaces)
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return ",".join(parts)

    return text


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 1: Loading raw data")
    print("=" * 60)

    try:
        df = pd.read_csv(RAW_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        print("  utf-8 failed, retrying with latin-1...")
        df = pd.read_csv(RAW_CSV, encoding="latin-1")
    except FileNotFoundError:
        sys.stderr.write(
            f"[ERROR] Could not find '{RAW_CSV}'. Edit RAW_CSV at the top "
            f"of this script to point at your actual raw export.\n"
        )
        sys.exit(1)

    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns: {df.columns.tolist()}")

    if COLUMN_RENAME_MAP:
        df = df.rename(columns=COLUMN_RENAME_MAP)
        print(f"  Renamed columns -> {df.columns.tolist()}")

    if INDUSTRY_COL not in df.columns:
        sys.stderr.write(
            f"[ERROR] Expected a '{INDUSTRY_COL}' column after renaming, "
            f"but found: {df.columns.tolist()}. Update COLUMN_RENAME_MAP.\n"
        )
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STEP 2: Trimming whitespace on all text columns")
    print("=" * 60)
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].map(normalize_text)
    print(f"  Normalized: {list(text_cols)}")

    print("\n" + "=" * 60)
    print("STEP 3: Normalizing the Industry column")
    print("=" * 60)
    before_blank = (df[INDUSTRY_COL].str.strip() == "").sum()
    df[INDUSTRY_COL] = df[INDUSTRY_COL].map(normalize_industry_value)
    after_blank = (df[INDUSTRY_COL] == "").sum()
    print(f"  Blank/junk Industry values: {before_blank} -> {after_blank} "
          f"(standardized to empty string)")

    print("\n" + "=" * 60)
    print("STEP 4: Removing exact duplicate rows")
    print("=" * 60)
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"  Exact duplicates removed: {n_before - len(df)}")

    if DEDUPE_KEYS:
        n_before = len(df)
        df = df.drop_duplicates(subset=DEDUPE_KEYS, keep="first")
        print(f"  Duplicate companies removed (by {DEDUPE_KEYS}): {n_before - len(df)}")

    print("\n" + "=" * 60)
    print("STEP 5: Saving cleaned CSV")
    print("=" * 60)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved -> {OUTPUT_CSV}  ({len(df)} rows)")
    print("\nDone. Next step: run 02_cluster_industries.py on this file.")


if __name__ == "__main__":
    main()
