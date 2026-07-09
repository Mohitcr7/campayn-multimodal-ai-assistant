"""
profile_filter.py
=================
Campayn AI Assistant — Module 1: Pandas Profile Filter

Responsibilities:
  1. Load & normalise the 56k creator profiles CSV (done once at startup)
  2. Accept a structured FilterSpec (parsed by the Intent Parser)
  3. Apply exact + range filters using pandas (fast, zero embedding cost)
  4. Derive follower tier + estimated CPV range from knowledge-base benchmarks
  5. Score and rank the filtered profiles
  6. Return top-N profiles as a list of dicts for the Prompt Constructor

Author : Mohit Chauhan — Campayn Multimodal AI Assistant Project
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0.  CONSTANTS & KNOWLEDGE-BASE-DERIVED LOOKUP TABLES
# ---------------------------------------------------------------------------

CSV_PATH = Path("data/raw/creator_profiles.csv")

# ── Follower tier thresholds ────────────────────────────────────────────────
TIER_BINS   = [0, 10_000, 50_000, 100_000, 500_000, 1_000_000, float("inf")]
TIER_LABELS = ["Nano", "Micro", "Mid", "Macro", "Mega", "Celebrity"]

# ── CPV ranges derived from the Campayn knowledge base (rupees / view) ──────
# Keyed by (normalised_category, tier)
# Falls back to category-level default if tier-specific entry missing
CPV_TABLE: dict[tuple[str, str], tuple[float, float]] = {
    # Beauty & Skincare
    ("beauty",            "Nano"):      (0.10, 0.15),
    ("beauty",            "Micro"):     (0.10, 0.18),
    ("beauty",            "Mid"):       (0.12, 0.20),
    ("beauty",            "Macro"):     (0.15, 0.25),
    ("beauty",            "Mega"):      (0.20, 0.35),
    ("beauty",            "Celebrity"): (0.25, 0.40),
    # Fashion
    ("fashion",           "Nano"):      (0.10, 0.15),
    ("fashion",           "Micro"):     (0.10, 0.18),
    ("fashion",           "Mid"):       (0.12, 0.20),
    ("fashion",           "Macro"):     (0.15, 0.25),
    ("fashion",           "Mega"):      (0.20, 0.35),
    ("fashion",           "Celebrity"): (0.25, 0.40),
    # Food
    ("food",              "Nano"):      (0.10, 0.15),
    ("food",              "Micro"):     (0.10, 0.18),
    ("food",              "Mid"):       (0.10, 0.18),
    ("food",              "Macro"):     (0.12, 0.20),
    ("food",              "Mega"):      (0.15, 0.25),
    ("food",              "Celebrity"): (0.18, 0.30),
    # Fitness
    ("fitness",           "Nano"):      (0.15, 0.20),
    ("fitness",           "Micro"):     (0.15, 0.25),
    ("fitness",           "Mid"):       (0.18, 0.28),
    ("fitness",           "Macro"):     (0.20, 0.30),
    ("fitness",           "Mega"):      (0.25, 0.35),
    ("fitness",           "Celebrity"): (0.30, 0.45),
    # Health
    ("health",            "Nano"):      (0.20, 0.28),
    ("health",            "Micro"):     (0.20, 0.30),
    ("health",            "Mid"):       (0.22, 0.32),
    ("health",            "Macro"):     (0.25, 0.35),
    ("health",            "Mega"):      (0.30, 0.45),
    ("health",            "Celebrity"): (0.35, 0.50),
    # Education / Learning
    ("education",         "Nano"):      (0.25, 0.32),
    ("education",         "Micro"):     (0.25, 0.35),
    ("education",         "Mid"):       (0.28, 0.38),
    ("education",         "Macro"):     (0.30, 0.40),
    ("education",         "Mega"):      (0.35, 0.45),
    ("education",         "Celebrity"): (0.38, 0.50),
    # Finance
    ("finance",           "Nano"):      (0.15, 0.22),
    ("finance",           "Micro"):     (0.18, 0.25),
    ("finance",           "Mid"):       (0.20, 0.28),
    ("finance",           "Macro"):     (0.22, 0.30),
    ("finance",           "Mega"):      (0.25, 0.35),
    ("finance",           "Celebrity"): (0.30, 0.45),
    # Travel
    ("travel",            "Nano"):      (0.15, 0.22),
    ("travel",            "Micro"):     (0.15, 0.25),
    ("travel",            "Mid"):       (0.18, 0.28),
    ("travel",            "Macro"):     (0.20, 0.30),
    ("travel",            "Mega"):      (0.25, 0.35),
    ("travel",            "Celebrity"): (0.28, 0.40),
    # Tech
    ("tech",              "Nano"):      (0.15, 0.22),
    ("tech",              "Micro"):     (0.15, 0.25),
    ("tech",              "Mid"):       (0.18, 0.28),
    ("tech",              "Macro"):     (0.22, 0.30),
    ("tech",              "Mega"):      (0.25, 0.35),
    ("tech",              "Celebrity"): (0.30, 0.45),
    # Comedy & Memes
    ("comedy",            "Nano"):      (0.08, 0.12),
    ("comedy",            "Micro"):     (0.08, 0.15),
    ("comedy",            "Mid"):       (0.10, 0.15),
    ("comedy",            "Macro"):     (0.10, 0.18),
    ("comedy",            "Mega"):      (0.12, 0.20),
    ("comedy",            "Celebrity"): (0.15, 0.25),
    # Lifestyle
    ("lifestyle",         "Nano"):      (0.10, 0.15),
    ("lifestyle",         "Micro"):     (0.10, 0.18),
    ("lifestyle",         "Mid"):       (0.12, 0.20),
    ("lifestyle",         "Macro"):     (0.15, 0.25),
    ("lifestyle",         "Mega"):      (0.18, 0.28),
    ("lifestyle",         "Celebrity"): (0.20, 0.35),
    # Family / Parenting / Kids
    ("family",            "Nano"):      (0.20, 0.28),
    ("family",            "Micro"):     (0.20, 0.30),
    ("family",            "Mid"):       (0.22, 0.32),
    ("family",            "Macro"):     (0.25, 0.35),
    ("family",            "Mega"):      (0.28, 0.40),
    ("family",            "Celebrity"): (0.30, 0.45),
    # Gaming & Anime
    ("gaming",            "Nano"):      (0.10, 0.15),
    ("gaming",            "Micro"):     (0.10, 0.18),
    ("gaming",            "Mid"):       (0.12, 0.20),
    ("gaming",            "Macro"):     (0.15, 0.25),
    ("gaming",            "Mega"):      (0.18, 0.28),
    ("gaming",            "Celebrity"): (0.22, 0.35),
    # Automotive
    ("automotive",        "Nano"):      (0.20, 0.28),
    ("automotive",        "Micro"):     (0.20, 0.30),
    ("automotive",        "Mid"):       (0.22, 0.32),
    ("automotive",        "Macro"):     (0.25, 0.35),
    ("automotive",        "Mega"):      (0.28, 0.40),
    ("automotive",        "Celebrity"): (0.30, 0.45),
    # Business & Startups
    ("business",          "Nano"):      (0.20, 0.28),
    ("business",          "Micro"):     (0.22, 0.30),
    ("business",          "Mid"):       (0.25, 0.33),
    ("business",          "Macro"):     (0.28, 0.35),
    ("business",          "Mega"):      (0.30, 0.42),
    ("business",          "Celebrity"): (0.35, 0.50),
    # Arts / Entertainment (broad awareness)
    ("arts",              "Nano"):      (0.08, 0.14),
    ("arts",              "Micro"):     (0.08, 0.15),
    ("arts",              "Mid"):       (0.10, 0.18),
    ("arts",              "Macro"):     (0.12, 0.22),
    ("arts",              "Mega"):      (0.15, 0.28),
    ("arts",              "Celebrity"): (0.18, 0.35),
    # Catchall default
    ("default",           "Nano"):      (0.10, 0.15),
    ("default",           "Micro"):     (0.12, 0.18),
    ("default",           "Mid"):       (0.15, 0.22),
    ("default",           "Macro"):     (0.18, 0.28),
    ("default",           "Mega"):      (0.22, 0.35),
    ("default",           "Celebrity"): (0.28, 0.45),
}

# ── Category normalisation map ───────────────────────────────────────────────
# Maps raw CSV category strings → canonical key used in CPV_TABLE & filters
RAW_TO_CANONICAL: dict[str, str] = {
    "arts":                        "arts",
    "sports":                      "sports",
    "comedy & memes":              "comedy",
    "fashion":                     "fashion",
    "fitness":                     "fitness",
    "family, kids & pets":         "family",
    "lifestyle & living":          "lifestyle",
    "entertainment":               "arts",
    "tech":                        "tech",
    "beauty":                      "beauty",
    "photography / videography":   "photography",
    "food":                        "food",
    "food ":                       "food",           # trailing-space fix
    "home & decor":                "home",
    "general":                     "general",
    "news, media & magazines":     "media",
    "community pages":             "community",
    "travel & places":             "travel",
    "law, rights & activism":      "advocacy",
    "automotive":                  "automotive",
    "education":                   "education",
    "learning":                    "education",
    "finance & investments":       "finance",
    "health":                      "health",
    "gaming & anime":              "gaming",
    "business & startups":         "business",
    "acting, pro (tv / series)":   "arts",
    "personal vlogs":              "lifestyle",
    "politics":                    "advocacy",
    "publisher":                   "media",
    "meme":                        "comedy",
}

# ── Natural language → canonical category aliases ────────────────────────────
# Used by the intent parser to interpret free-text category requests
NL_ALIASES: dict[str, str] = {
    # beauty
    "beauty": "beauty", "skincare": "beauty", "skin care": "beauty",
    "makeup": "beauty", "cosmetics": "beauty",
    # fashion
    "fashion": "fashion", "style": "fashion", "apparel": "fashion",
    "clothing": "fashion", "outfit": "fashion",
    # food
    "food": "food", "recipe": "food", "cooking": "food",
    "chef": "food", "restaurant": "food", "cuisine": "food",
    # fitness
    "fitness": "fitness", "gym": "fitness", "workout": "fitness",
    "bodybuilding": "fitness", "yoga": "fitness", "exercise": "fitness",
    "athlete": "fitness",
    # health
    "health": "health", "wellness": "health", "supplement": "health",
    "nutrition": "health", "diet": "health", "ayurveda": "health",
    # travel
    "travel": "travel", "tourism": "travel", "destination": "travel",
    "hospitality": "travel", "hotel": "travel",
    # tech
    "tech": "tech", "technology": "tech", "gadget": "tech",
    "software": "tech", "app": "tech", "startup": "tech",
    # finance
    "finance": "finance", "fintech": "finance", "investing": "finance",
    "stock": "finance", "crypto": "finance", "money": "finance",
    # family
    "parenting": "family", "baby": "family", "kids": "family",
    "family": "family", "mom": "family", "dad": "family",
    # gaming
    "gaming": "gaming", "game": "gaming", "esports": "gaming",
    "anime": "gaming",
    # comedy
    "comedy": "comedy", "meme": "comedy", "humor": "comedy",
    "funny": "comedy",
    # lifestyle
    "lifestyle": "lifestyle", "vlog": "lifestyle",
    "daily life": "lifestyle",
    # education
    "education": "education", "edtech": "education", "study": "education",
    "teaching": "education", "upsc": "education", "jee": "education",
    # business
    "business": "business", "entrepreneur": "business", "startup": "business",
    "founder": "business",
    # automotive
    "automotive": "automotive", "car": "automotive", "bike": "automotive",
    "ev": "automotive", "motorcycle": "automotive",
    # arts / entertainment
    "entertainment": "arts", "bollywood": "arts", "actor": "arts",
    "music": "arts", "singer": "arts", "art": "arts",
}


# ---------------------------------------------------------------------------
# 1.  DATA LOADING & NORMALISATION  (called once at app startup)
# ---------------------------------------------------------------------------

def load_and_normalise(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """
    Load creator_profiles.csv, clean and enrich it.
    Returns a DataFrame stored in memory for the lifetime of the app.
    """
    df = pd.read_csv(csv_path)

    # ── 1a. Strip whitespace from all string columns ─────────────────────
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # ── 1b. Normalise category to canonical key ──────────────────────────
    df["category_norm"] = (
        df["category"]
        .str.lower()
        .str.strip()
        .map(RAW_TO_CANONICAL)
        .fillna("general")
    )

    # ── 1c. Normalise subcategory for display ────────────────────────────
    df["subcategory_clean"] = df["subcategory"].str.strip()

    # ── 1d. Derive follower tier ─────────────────────────────────────────
    df["tier"] = pd.cut(
        df["followers"],
        bins=TIER_BINS,
        labels=TIER_LABELS,
        right=False,
    ).astype(str)

    # ── 1e. Derive CPV range from knowledge-base table ───────────────────
    def _get_cpv(row: pd.Series) -> tuple[float, float]:
        key = (row["category_norm"], row["tier"])
        if key in CPV_TABLE:
            return CPV_TABLE[key]
        # fall back to category default across all tiers
        cat_key = (row["category_norm"], "Micro")
        if cat_key in CPV_TABLE:
            return CPV_TABLE[cat_key]
        return CPV_TABLE[("default", row["tier"])]

    cpv_vals = df.apply(_get_cpv, axis=1)
    df["cpv_min"] = cpv_vals.apply(lambda x: x[0])
    df["cpv_max"] = cpv_vals.apply(lambda x: x[1])
    df["cpv_mid"] = ((df["cpv_min"] + df["cpv_max"]) / 2).round(3)

    # ── 1f. Relevance score placeholder (updated per query in Module 1) ──
    df["relevance_score"] = 0.0

    print(f"[Module 1] Loaded {len(df):,} profiles. Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# 2.  FILTER SPEC  (produced by the Intent Parser — Module 0)
# ---------------------------------------------------------------------------

@dataclass
class FilterSpec:
    """
    Structured filter criteria extracted from natural language by the
    intent parser.  All fields are optional; None means "no constraint".
    """
    # Category / niche
    categories:         Optional[list[str]] = None   # canonical keys e.g. ["beauty", "fitness"]
    subcategories:      Optional[list[str]] = None   # partial-match strings

    # Follower range
    followers_min:      Optional[int]       = None
    followers_max:      Optional[int]       = None

    # Follower tier shorthand (alternative to min/max)
    tiers:              Optional[list[str]] = None   # e.g. ["Micro", "Mid"]

    # CPV budget (brand-side: what they can afford per view)
    cpv_budget_max:     Optional[float]     = None   # e.g. 0.20

    # Source filter
    sources:            Optional[list[str]] = None   # e.g. ["meme_pages"]

    # Result controls
    top_n:              int                 = 10
    sort_by:            str                 = "followers"   # "followers" | "cpv_mid" | "relevance"
    sort_ascending:     bool                = False

    # Free-text keyword (searched in name + subcategory)
    keyword:            Optional[str]       = None


# ---------------------------------------------------------------------------
# 3.  INTENT PARSER  (lightweight — converts NL query → FilterSpec)
# ---------------------------------------------------------------------------

def parse_intent(query: str, role: str = "brand") -> FilterSpec:
    """
    Lightweight rule-based + regex intent parser.
    In production this is replaced by an LLM call that returns JSON;
    this version handles common patterns without an extra LLM round-trip
    and serves as the fallback.

    Args:
        query : raw user query string
        role  : "brand" | "creator"

    Returns:
        FilterSpec
    """
    q = query.lower()
    spec = FilterSpec()

    # ── 3a. Category detection ───────────────────────────────────────────
    matched_cats: set[str] = set()
    for alias, canonical in NL_ALIASES.items():
        if alias in q:
            matched_cats.add(canonical)
    if matched_cats:
        spec.categories = list(matched_cats)

    # ── 3b. Follower range detection ─────────────────────────────────────
    # Patterns: "50k-200k", "above 100k", "below 500k", "10k to 1m"
    range_pat = re.search(
        r"(\d+\.?\d*)\s*([km]?)\s*(?:to|-)\s*(\d+\.?\d*)\s*([km]?)",
        q,
    )
    if range_pat:
        spec.followers_min = _parse_follower_str(range_pat.group(1) + range_pat.group(2))
        spec.followers_max = _parse_follower_str(range_pat.group(3) + range_pat.group(4))
    else:
        above_pat = re.search(r"(?:above|over|more than|>)\s*(\d+\.?\d*)\s*([km]?)", q)
        below_pat = re.search(r"(?:below|under|less than|<)\s*(\d+\.?\d*)\s*([km]?)", q)
        if above_pat:
            spec.followers_min = _parse_follower_str(above_pat.group(1) + above_pat.group(2))
        if below_pat:
            spec.followers_max = _parse_follower_str(below_pat.group(1) + below_pat.group(2))

    # ── 3c. Tier detection ───────────────────────────────────────────────
    tier_map = {
        "nano": "Nano", "micro": "Micro", "mid": "Mid",
        "mid-tier": "Mid", "macro": "Macro",
        "mega": "Mega", "celebrity": "Celebrity",
    }
    found_tiers = [v for k, v in tier_map.items() if k in q]
    if found_tiers:
        spec.tiers = found_tiers

    # ── 3d. CPV budget detection ─────────────────────────────────────────
    cpv_pat = re.search(r"(?:cpv|budget|rate)\s*(?:of|:)?\s*(?:₹|rs\.?)?\s*(\d+\.?\d*)", q)
    if cpv_pat:
        spec.cpv_budget_max = float(cpv_pat.group(1))

    # ── 3e. Source detection ─────────────────────────────────────────────
    if "meme" in q:
        spec.sources = ["meme_pages"]

    # ── 3f. Top-N detection ──────────────────────────────────────────────
    topn_pat = re.search(r"top\s*(\d+)|(\d+)\s*creators?", q)
    if topn_pat:
        n = int(topn_pat.group(1) or topn_pat.group(2))
        spec.top_n = min(n, 50)   # cap at 50 for prompt safety

    # ── 3g. Sort preference ──────────────────────────────────────────────
    if "cheapest" in q or "lowest cpv" in q:
        spec.sort_by = "cpv_mid"
        spec.sort_ascending = True
    elif "largest" in q or "most followers" in q or "biggest" in q:
        spec.sort_by = "followers"
        spec.sort_ascending = False

    # ── 3h. Keyword passthrough ──────────────────────────────────────────
    kw_pat = re.search(r"(?:keyword|named|called|about)\s+['\"]?(\w[\w\s]*?)['\"]?(?:\s|$)", q)
    if kw_pat:
        spec.keyword = kw_pat.group(1).strip()

    return spec


def _parse_follower_str(s: str) -> int:
    """Convert '100k', '1m', '50000' → int."""
    s = s.strip().lower()
    if s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    else:
        try:
            return int(float(s))
        except ValueError:
            return 0


# ---------------------------------------------------------------------------
# 4.  MAIN FILTER FUNCTION
# ---------------------------------------------------------------------------

def filter_profiles(
    df: pd.DataFrame,
    spec: FilterSpec,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply FilterSpec to the loaded DataFrame using pandas.
    Returns (filtered_df, metadata_dict).

    Args:
        df   : the normalised DataFrame from load_and_normalise()
        spec : a FilterSpec (from parse_intent or direct construction)

    Returns:
        top_df   : DataFrame of top-N results, scored and ranked
        metadata : dict with filter summary, match count, confidence
    """
    mask = pd.Series(True, index=df.index)

    # ── 4a. Category filter ──────────────────────────────────────────────
    if spec.categories:
        canonical_cats = [c.lower() for c in spec.categories]
        mask &= df["category_norm"].isin(canonical_cats)

    # ── 4b. Subcategory partial match ────────────────────────────────────
    if spec.subcategories:
        sub_mask = pd.Series(False, index=df.index)
        for sub in spec.subcategories:
            sub_mask |= df["subcategory_clean"].str.lower().str.contains(
                sub.lower(), na=False, regex=False
            )
        mask &= sub_mask

    # ── 4c. Follower range ───────────────────────────────────────────────
    if spec.followers_min is not None:
        mask &= df["followers"] >= spec.followers_min
    if spec.followers_max is not None:
        mask &= df["followers"] <= spec.followers_max

    # ── 4d. Tier filter ──────────────────────────────────────────────────
    if spec.tiers:
        mask &= df["tier"].isin(spec.tiers)

    # ── 4e. CPV budget filter (brand can afford up to spec.cpv_budget_max) ──
    if spec.cpv_budget_max is not None:
        mask &= df["cpv_min"] <= spec.cpv_budget_max

    # ── 4f. Source filter ────────────────────────────────────────────────
    if spec.sources:
        mask &= df["source"].isin(spec.sources)

    # ── 4g. Keyword filter (name or subcategory) ─────────────────────────
    if spec.keyword:
        kw = spec.keyword.lower()
        kw_mask = (
            df["name"].str.lower().str.contains(kw, na=False, regex=False)
            | df["subcategory_clean"].str.lower().str.contains(kw, na=False, regex=False)
        )
        mask &= kw_mask

    filtered = df[mask].copy()
    total_matches = len(filtered)

    # ── 4h. Relevance scoring ────────────────────────────────────────────
    # Score = weighted combination of tier alignment + CPV efficiency
    # Normalise followers to 0–1 log scale
    if total_matches > 0:
        log_followers = np.log1p(filtered["followers"])
        filtered["relevance_score"] = (
            0.6 * (log_followers / log_followers.max())            # reach weight
            + 0.4 * (1 - (filtered["cpv_mid"] / filtered["cpv_mid"].max()))  # CPV efficiency
        ).round(4)

    # ── 4i. Sort ─────────────────────────────────────────────────────────
    sort_col = spec.sort_by if spec.sort_by in filtered.columns else "followers"
    filtered = filtered.sort_values(sort_col, ascending=spec.sort_ascending)

    # ── 4j. Top-N selection ──────────────────────────────────────────────
    top_df = filtered.head(spec.top_n)

    # ── 4k. Confidence signal ────────────────────────────────────────────
    confidence = "high" if total_matches >= 10 else ("medium" if total_matches >= 3 else "low")

    metadata = {
        "total_matches":  total_matches,
        "returned":       len(top_df),
        "confidence":     confidence,
        "filters_applied": {
            "categories":    spec.categories,
            "subcategories": spec.subcategories,
            "followers_min": spec.followers_min,
            "followers_max": spec.followers_max,
            "tiers":         spec.tiers,
            "cpv_budget":    spec.cpv_budget_max,
            "sources":       spec.sources,
            "keyword":       spec.keyword,
        },
    }

    return top_df, metadata


# ---------------------------------------------------------------------------
# 5.  OUTPUT FORMATTER  (for Prompt Constructor / LLM context)
# ---------------------------------------------------------------------------

DISPLAY_COLS = [
    "name", "ig_handle", "followers", "tier",
    "category", "subcategory_clean",
    "cpv_min", "cpv_max", "cpv_mid",
    "source", "relevance_score",
]


def format_for_prompt(top_df: pd.DataFrame, metadata: dict) -> str:
    """
    Converts the filtered DataFrame into a compact, token-efficient
    string block for injection into the LLM prompt.

    Format:
        [Module 1 — Influencer Profiles]
        Total matches: 847  |  Returned: 10  |  Confidence: high

        1. Shraddha Kapoor | @shraddhakapoor | 85M followers | Celebrity | Lifestyle & Living
           CPV est: ₹0.20–0.35 | Source: celebrities

        ...
    """
    lines = ["[Module 1 — Influencer Profiles]"]
    lines.append(
        f"Total matches: {metadata['total_matches']:,}  |  "
        f"Returned: {metadata['returned']}  |  "
        f"Confidence: {metadata['confidence']}"
    )
    lines.append("")

    for i, (_, row) in enumerate(top_df.iterrows(), start=1):
        followers_fmt = _fmt_followers(row["followers"])
        lines.append(
            f"{i}. {row['name']} | @{row['ig_handle']} | "
            f"{followers_fmt} followers | {row['tier']} | {row['category']}"
        )
        lines.append(
            f"   Niche: {row['subcategory_clean']} | "
            f"CPV est: ₹{row['cpv_min']}–{row['cpv_max']} | "
            f"Source: {row['source']} | Score: {row['relevance_score']}"
        )

    if metadata["confidence"] == "low":
        lines.append(
            "\n⚠ Low match count — consider relaxing follower range or category filters."
        )

    return "\n".join(lines)


def _fmt_followers(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# 6.  PUBLIC API  (called by LangGraph orchestrator)
# ---------------------------------------------------------------------------

def run_module1(
    query: str,
    df: pd.DataFrame,
    role: str = "brand",
    spec_override: Optional[FilterSpec] = None,
) -> tuple[str, dict, pd.DataFrame]:
    """
    Single entry-point for the LangGraph node that calls Module 1.

    Args:
        query         : raw user query string
        df            : pre-loaded normalised DataFrame
        role          : "brand" | "creator"
        spec_override : pass a FilterSpec directly (bypasses NL parser)

    Returns:
        prompt_block  : formatted string for the Prompt Constructor
        metadata      : dict with match stats and filter summary
        top_df        : raw DataFrame for downstream use
    """
    spec = spec_override if spec_override else parse_intent(query, role)
    top_df, metadata = filter_profiles(df, spec)
    prompt_block = format_for_prompt(top_df, metadata)
    return prompt_block, metadata, top_df


# ---------------------------------------------------------------------------
# 7.  DEMO / SMOKE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # ── Load data ──────────────────────────────────────────────────────────
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    df_global = load_and_normalise(csv)

    test_queries = [
        # Brand side
        ("brand", "Find me top 10 fitness creators with 50k to 200k followers"),
        ("brand", "I need beauty micro-influencers for a skincare launch, CPV budget 0.20"),
        ("brand", "Show me meme pages for a food delivery campaign"),
        ("brand", "Top 5 macro fashion creators"),
        # Creator side
        ("creator", "I am a fitness creator with 80k followers, who are my peers?"),
        ("creator", "Show me top education creators so I can benchmark my growth"),
    ]

    for role, query in test_queries:
        print("\n" + "=" * 70)
        print(f"ROLE: {role.upper()}")
        print(f"QUERY: {query}")
        print("-" * 70)
        block, meta, _ = run_module1(query, df_global, role)
        print(block)
        print(f"\nFilters applied: {meta['filters_applied']}")