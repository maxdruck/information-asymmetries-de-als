"""
Bundles the paths and names of the DE/ALS InfoGap pipeline.

The module provides all scripts with consistent project-relative paths,
file names and model parameters.

By default, the pipeline uses the bundled gold dataset and the
few-shot output directories:

    outputs/goldstandard_fewshot
    outputs/alignments
    outputs/verified
    outputs/cache

Alternative variants can be selected via environment variables, e.g.:

    INFOGAP_INPUT_MODE=gold
    INFOGAP_GOLD_DIRNAME=goldstandard_oneshot
    INFOGAP_ALIGNMENT_DIRNAME=alignments_oneshot
    INFOGAP_VERIFIED_DIRNAME=verified_oneshot
    INFOGAP_CACHE_DIRNAME=cache_oneshot

The directory names are always interpreted within ``outputs/``.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable



# Project paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CONFIG_DIR = ROOT_DIR / "configs"
OUTPUT_DIR = ROOT_DIR / "outputs"


# All configurable outputs deliberately stay within the project folder
def output_subdir_from_env(env_name: str, default_name: str) -> Path:
    """Determine an output directory within ``outputs/``.

    This allows variants such as ``outputs/goldstandard_oneshot`` or
    ``outputs/verified_oneshot`` to be stored separately. The value must
    be relative and must be neither an absolute path nor a navigation
    via ``..``.
    """
    value = os.getenv(env_name, default_name).strip() or default_name
    candidate = Path(value)

    if candidate.is_absolute():
        raise ValueError(
            f"{env_name} must be a directory name inside outputs/, not an absolute path: "
            f"{value!r}"
        )

    if any(part == ".." for part in candidate.parts):
        raise ValueError(
            f"{env_name} must not contain '..': {value!r}"
        )

    return OUTPUT_DIR / candidate


GOLD_DIR = output_subdir_from_env("INFOGAP_GOLD_DIRNAME", "goldstandard_fewshot")
ALIGNMENT_DIR = output_subdir_from_env("INFOGAP_ALIGNMENT_DIRNAME", "alignments")
VERIFIED_DIR = output_subdir_from_env("INFOGAP_VERIFIED_DIRNAME", "verified")
CACHE_DIR = output_subdir_from_env("INFOGAP_CACHE_DIRNAME", "cache")



# Input mode
# ---------------------------------------------------------------------------
# full = complete external dataset
# gold = bundled gold texts, prepared with:
#        python prepare_articles.py --mode gold
#
# Usage in gold mode:
#   INFOGAP_INPUT_MODE=gold python decompose_facts.py --pair-key gold__wahrscheinlichkeit
# ---------------------------------------------------------------------------

INPUT_MODE = os.getenv("INFOGAP_INPUT_MODE", "gold").strip().lower()

if INPUT_MODE not in {"full", "gold"}:
    raise ValueError(
        f"Invalid INFOGAP_INPUT_MODE={INPUT_MODE!r}. "
        "Use 'full' or 'gold'."
    )



# Input paths of the complete dataset
# ---------------------------------------------------------------------------

ALS_RAW_PATH = RAW_DIR / "als_filtered_matched.jsonl"
DE_RAW_PATH = RAW_DIR / "de_filtered_matched.jsonl"

MAPPING_BASENAME = "filtered_als_de_mapping"



# Cleaned inputs for decomposition and alignment
# ---------------------------------------------------------------------------
# In full mode:
#   data/clean/als_filtered_matched.clean.jsonl
#   data/clean/de_filtered_matched.clean.jsonl
#   configs/article_pairs.jsonl
#
# In gold mode:
#   data/gold_clean/gold_als.clean.jsonl
#   data/gold_clean/gold_de.clean.jsonl
#   configs/gold_article_pairs.jsonl
# ---------------------------------------------------------------------------

if INPUT_MODE == "gold":
    GOLD_CLEAN_DIR = DATA_DIR / "gold_clean"

    ALS_CLEAN_PATH = GOLD_CLEAN_DIR / "gold_als.clean.jsonl"
    DE_CLEAN_PATH = GOLD_CLEAN_DIR / "gold_de.clean.jsonl"

    PAIR_CONFIG_PATH = CONFIG_DIR / "gold_article_pairs.jsonl"

else:
    ALS_CLEAN_PATH = CLEAN_DIR / "als_filtered_matched.clean.jsonl"
    DE_CLEAN_PATH = CLEAN_DIR / "de_filtered_matched.clean.jsonl"

    PAIR_CONFIG_PATH = CONFIG_DIR / "article_pairs.jsonl"


# Model parameters of the pipeline
# ---------------------------------------------------------------------------

DECOMPOSITION_MODEL = "gemma-4-31b-it"
XFACTMATCH_MODEL = "gemma-4-31b-it"
BASE_URL = "https://chat-ai.academiccloud.de/v1"



# Expected clusters. If an explicit value is missing in the mapping or the article
# metadata, the pair receives UNASSIGNED; nothing is inferred from the content
# ---------------------------------------------------------------------------

CANONICAL_CLUSTERS = {
    "biografie": "Biografie",
    "geschichte": "Geschichte",
    "gesellschaft": "Gesellschaft",
}

UNASSIGNED_CLUSTER = "UNASSIGNED"


# Supported field names of different mapping exports
ALS_ID_KEYS = (
    "als_page_id",
    "als_pageid",
    "als_id",
    "source_page_id",
    "source_id",
    "page_id_als",
)

DE_ID_KEYS = (
    "de_page_id",
    "de_pageid",
    "de_id",
    "target_page_id",
    "target_id",
    "page_id_de",
)

CLUSTER_KEYS = (
    "cluster",
    "topic_cluster",
    "theme_cluster",
    "thematic_cluster",
    "themencluster",
    "topic",
    "theme",
)

TITLE_KEYS = (
    "title",
    "page_title",
    "de_title",
    "als_title",
    "label",
)


def resolve_mapping_path() -> Path:
    """Find the existing mapping file with or without a JSON/JSONL extension."""
    candidates = [
        RAW_DIR / MAPPING_BASENAME,
        RAW_DIR / f"{MAPPING_BASENAME}.jsonl",
        RAW_DIR / f"{MAPPING_BASENAME}.json",
    ]

    # The mapping file may exist without an extension or as JSON or JSONL
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Mapping file not found. Put it in data/raw/ as one of: "
        + ", ".join(str(path.relative_to(ROOT_DIR)) for path in candidates)
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    # Empty lines are allowed; malformed JSON objects are reported with their line number
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {error}"
                ) from error

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_number}"
                )

            yield value


def load_jsonl_dict(path: Path, key: str = "page_id") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for record in iter_jsonl(path):
        value = record.get(key)

        if value is not None:
            result[str(value)] = record

    return result


def first_present(record: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        value = record.get(key)

        if value not in (None, ""):
            return value

    return None


def normalize_cluster(value: Any) -> str:
    # Known English and German labels end up in the same main clusters
    if value is None:
        return UNASSIGNED_CLUSTER

    text = str(value).strip()

    if not text:
        return UNASSIGNED_CLUSTER

    key = text.lower()

    if key in CANONICAL_CLUSTERS:
        return CANONICAL_CLUSTERS[key]

    if key == "biography":
        return "Biografie"

    if key == "history":
        return "Geschichte"

    if key == "society":
        return "Gesellschaft"

    return text


def get_explicit_cluster(*records: dict[str, Any] | None) -> str:
    """Return an explicit cluster or otherwise ``UNASSIGNED``.

    The cluster is not inferred from running text, categories or weighted
    keywords.
    """
    # Multiple metadata sources are checked in the priority order in which they are passed
    for record in records:
        if not record:
            continue

        value = first_present(record, CLUSTER_KEYS)

        if value not in (None, ""):
            return normalize_cluster(value)

    return UNASSIGNED_CLUSTER


def slugify(value: Any, max_len: int = 80) -> str:
    # Slugs serve only as stable file names and contain only ASCII characters
    text = str(value) if value is not None else "untitled"
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    if not text:
        text = "untitled"

    return text[:max_len].strip("_") or "untitled"


def make_pair_key(cluster: str, als_page_id: Any, de_page_id: Any, title: Any) -> str:
    cluster_slug = slugify(cluster)
    title_slug = slugify(title)

    return f"{cluster_slug}__als{als_page_id}__de{de_page_id}__{title_slug}"


def facts_path(language: str, pair_key: str) -> Path:
    if language not in {"als", "de"}:
        raise ValueError(f"Invalid language: {language}")

    return GOLD_DIR / f"facts_{language}_{pair_key}.json"


def alignment_path(pair_key: str) -> Path:
    return ALIGNMENT_DIR / f"alignment_{pair_key}.json"


def verified_path(pair_key: str) -> Path:
    return VERIFIED_DIR / f"verified_{pair_key}.json"


def xfactmatch_cache_path(pair_key: str) -> Path:
    return CACHE_DIR / f"xfactmatch_{pair_key}_cache.json"


def load_pairs(path: Path = PAIR_CONFIG_PATH) -> list[dict[str, Any]]:
    # Depending on the mode, the error message names the appropriate preparation step
    if not path.exists():
        if INPUT_MODE == "gold":
            raise FileNotFoundError(
                f"Missing pair config: {path}. "
                "Run: python prepare_articles.py --mode gold"
            )

        raise FileNotFoundError(
            f"Missing pair config: {path}. "
            "Run build_article_pairs.py first."
        )

    return list(iter_jsonl(path))


def select_pairs(
    pairs: list[dict[str, Any]],
    cluster: str | None = None,
    pair_key: str | None = None,
    limit: int | None = None,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    # Filters are applied one after another; the order of the configuration is preserved
    selected = pairs

    if cluster:
        wanted = normalize_cluster(cluster)
        selected = [
            pair for pair in selected
            if normalize_cluster(pair.get("cluster")) == wanted
        ]

    if pair_key:
        selected = [
            pair for pair in selected
            if pair.get("pair_key") == pair_key
        ]

    if start_index:
        selected = selected[start_index:]

    if limit is not None:
        selected = selected[:limit]

    return selected
