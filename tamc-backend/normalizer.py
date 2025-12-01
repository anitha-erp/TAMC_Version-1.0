# normalizer.py  (ADVANCED VERSION)
# ----------------------------------------------------------
# Smart Normalization for AMC, Commodity, District, Mandal
# Auto-learns from db_debug_dump.json
# ----------------------------------------------------------

import json
import re
import os
from difflib import SequenceMatcher

# ----------------------------------------------------------
# 1. Load DB dump for learning
# ----------------------------------------------------------
DATA_PATH = "db_debug_dump.json"

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        DB_DATA = json.load(f)
        RAW_COMMODITIES = [x["commodity_name"] for x in DB_DATA["unique_commodities"]]
        RAW_AMCS = [x["amc_name"] for x in DB_DATA["unique_amcs"]]
        RAW_DISTRICTS = [x["district"] for x in DB_DATA["unique_amcs"]]
        RAW_MANDALS = [x["mandal"] for x in DB_DATA["unique_amcs"]]
else:
    RAW_COMMODITIES = []
    RAW_AMCS = []
    RAW_DISTRICTS = []
    RAW_MANDALS = []


# ----------------------------------------------------------
# 2. Utility helpers
# ----------------------------------------------------------
def _clean_basic(s):
    if not s:
        return None
    return s.strip().lower()


def _key(s):
    """Remove all non-alphanumeric chars."""
    return re.sub(r"[^a-z0-9]", "", s.lower()) if s else ""


def _similar(a, b):
    """Fuzzy ratio"""
    return SequenceMatcher(None, a, b).ratio()


def _best_match(raw, candidates, threshold=0.68):
    """Pick nearest matching phrase from DB clusters"""
    k = _key(raw)
    best = None
    best_ratio = 0

    for cand in candidates:
        r = _similar(k, _key(cand))
        if r > best_ratio:
            best_ratio = r
            best = cand

    return best if best_ratio >= threshold else raw


# ----------------------------------------------------------
# 3. Chilli Variant Extraction from real DB
# ----------------------------------------------------------
CHILLI_ROOT = "chilli"

# extract all variants like:
#   CHILLI-MALLISHWARA COLD
#   CHILLI DEEPIKA
#   CHILLI-1048
# ----------------------------------------------------------
def _extract_chilli_variants():
    variants = set()

    for c in RAW_COMMODITIES:
        if not c:
            continue

        s = c.lower()

        if "chilli" in s or "chili" in s:
            # remove chilli prefix
            v = re.sub(r"(chil+?i\s*[-]*)", "", s)
            v = v.replace("cold", "")
            v = v.replace("hi-tech", "")
            v = v.replace("(cover)", "")
            v = v.replace("bags", "")

            v = v.strip(" -_/()")

            if len(v) > 0:
                variants.add(v)

    # clean variants like "teja", "malleshwari cold", etc.
    cleaned = set()
    for v in variants:
        base = re.sub(r"[^a-z0-9]", "", v)
        if len(base) == 0:
            continue
        cleaned.add(base)

    return cleaned


CHILLI_VARIANTS_DB = _extract_chilli_variants()


def _normalize_chilli(raw):
    k = _key(raw)

    # detect chilli
    if not (k.startswith("chilli") or k.startswith("chili")):
        return None

    # extract variant part
    variant = k.replace("chilli", "").replace("chili", "")
    variant = variant.strip()

    # match against DB-learned variants
    best = _best_match(variant, CHILLI_VARIANTS_DB, threshold=0.55)

    if best and best != variant:
        return f"chilli-{best}"

    # fallback
    return "chilli"


# ----------------------------------------------------------
# 4. AMC, District, Mandal Normalization
# ----------------------------------------------------------
def clean_amc(raw):
    if not raw:
        return None

    # fuzzy match against AMC names in DB
    best = _best_match(raw, RAW_AMCS)
    return best.strip().title()


def clean_district(raw):
    if not raw:
        return None

    best = _best_match(raw, RAW_DISTRICTS)
    return best.strip().title()


def clean_mandal(raw):
    if not raw:
        return None

    best = _best_match(raw, RAW_MANDALS)
    return best.strip().title()


# ----------------------------------------------------------
# 5. Commodity Normalization
# ----------------------------------------------------------
def clean_commodity(raw):
    if not raw:
        return None

    raw_clean = raw.strip()

    # 1) Chilli auto-detect
    chilli = _normalize_chilli(raw_clean)
    if chilli:
        return chilli

    # 2) Remove (COVER), etc.
    tmp = raw_clean.lower()
    tmp = re.sub(r"\(.*?\)", "", tmp)
    tmp = tmp.replace("bags", "")
    tmp = tmp.replace("cold", "")
    tmp = tmp.replace("hi-tech", "")
    tmp = tmp.replace("/", " ")
    tmp = tmp.replace("  ", " ")

    # 3) Title case
    tmp = tmp.strip().title()

    # 4) Extract base commodity names from DB (remove variant suffixes)
    # For example: "COTTON-BAGS" -> "Cotton", "Groundnut-Dry" -> "Groundnut"
    base_commodities = set()
    for commodity in RAW_COMMODITIES:
        if commodity:
            # Split on hyphen and take the first part as the base commodity
            base = commodity.split('-')[0].strip().title()
            if base:
                base_commodities.add(base)
    
    # 5) Fuzzy match with base commodity names only
    best = _best_match(tmp, list(base_commodities))
    return best


# ----------------------------------------------------------
# 6. Exported API
# ----------------------------------------------------------
def normalize_all(record):
    """Normalize all fields in a dict with keys district, mandal, amc_name, commodity"""
    out = {}
    if "district" in record:
        out["district"] = clean_district(record["district"])

    if "mandal" in record:
        out["mandal"] = clean_mandal(record["mandal"])

    if "amc_name" in record:
        out["amc_name"] = clean_amc(record["amc_name"])

    if "commodity" in record:
        out["commodity"] = clean_commodity(record["commodity"])

    return out
