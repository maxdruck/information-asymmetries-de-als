"""Selects suitable article pairs from the complete DE/ALS raw data.

The three input files can be specified via command-line arguments.
Without arguments, the ``files_out/als-de`` layout relative to the
project directory is used.
"""

import argparse
import json
import math
from pathlib import Path


# Project-relative default paths for the external raw data
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "files_out" / "als-de"
DEFAULT_MAPPING_PATH = DEFAULT_INPUT_DIR / "als-de_mapping.jsonl"
DEFAULT_ALS_PATH = DEFAULT_INPUT_DIR / "data_als.jsonl"
DEFAULT_DE_PATH = DEFAULT_INPUT_DIR / "data_de_subset-als.jsonl"


# Reference sections that are excluded via fuzzy title matching
# Dialect variants such as Weblingg, Litratuur or Fuessnoote are covered as well;
# "Ausgaben" (editions) additionally excludes ISBN and edition lists


# ---------------------------------------------------------------------------
SKIP_REFERENCE_TITLES = [
    "weblinks", "literatur", "einzelnachweise", "siehe auch",
    "fuessnoote", "kategorie", "references", "notes", "bibliography",
    "weblingg", "litratuur", "quellen", "anmerkungen", "ausgaben"
]

# Thresholds for the article selection
# ---------------------------------------------------------------------------
MIN_WORDS = 500
MIN_SECTIONS = 3
MAX_WORDS = 1000
MAX_LENGTH_RATIO = 0.5
MAX_SECTION_RATIO = 0.5

# Sections above this list share are considered list-heavy and are dropped
# ---------------------------------------------------------------------------
MAX_LIST_RATIO = 0.5



# Fuzzily checks section titles against the known reference labels
# ---------------------------------------------------------------------------
def should_skip_title(title):
    from thefuzz import fuzz

    title_normalized = title.lower().strip()
    if title_normalized == "lead_paragraph":
        return True
    for ref_title in SKIP_REFERENCE_TITLES:
        if fuzz.ratio(title_normalized, ref_title) > 80:
            return True
    return False


# Checks whether too many non-empty lines start with list markers, numbers or ISBN
# ---------------------------------------------------------------------------
def is_list_heavy(text):
    if not text.strip():
        return False
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    list_lines = sum(
        1 for l in lines
        if l.startswith("*")
        or l.startswith("•")
        or l.startswith("-")
        or l.startswith("#")
        or (len(l) > 1 and l[0].isdigit() and l[1] in ".):")
        or "ISBN" in l
    )
    return (list_lines / len(lines)) > MAX_LIST_RATIO



# Combines the title and content filters into a single exclusion decision
# ---------------------------------------------------------------------------
def should_skip(title, text=""):
    if should_skip_title(title):
        return True
    if text and is_list_heavy(text):
        return True
    return False



# Loads a JSONL file as a dictionary keyed by page_id for fast access
# ---------------------------------------------------------------------------
def load_as_dict(path, key):
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                data[str(record[key])] = record
    return data



# Disambiguation pages are not used as content article pairs
# ---------------------------------------------------------------------------
def is_disambiguation(article):
    return "Begriffsklärung" in article.get("category", [])



# Recursively collects the relevant text and removes references, lists and true duplicates
# ---------------------------------------------------------------------------
def get_all_text(sections, seen_title_texts=None):
    if seen_title_texts is None:
        seen_title_texts = {}
    total_text = ""
    for section in sections:
        title = section.get("section_title", "")
        text = section.get("text", "").strip()
        if not should_skip(title, text):
            if text:
                key = (title, text)
                if key not in seen_title_texts:
                    seen_title_texts[key] = True
                    total_text += " " + text
        subsections = section.get("subsections", [])
        if subsections:
            total_text += get_all_text(subsections, seen_title_texts)
    return total_text.strip()



# Counts the same relevant sections that also enter the text set
# ---------------------------------------------------------------------------
def count_content_sections(sections, seen_title_texts=None):
    if seen_title_texts is None:
        seen_title_texts = {}
    count = 0
    for section in sections:
        title = section.get("section_title", "")
        text = section.get("text", "").strip()
        if not should_skip(title, text):
            if text:
                key = (title, text)
                if key not in seen_title_texts:
                    seen_title_texts[key] = True
                    count += 1
        subsections = section.get("subsections", [])
        if subsections:
            count += count_content_sections(subsections, seen_title_texts)
    return count



# Reads the introductory paragraph for the minimum length check
# ---------------------------------------------------------------------------
def get_lead_text(article):
    for section in article.get("sections", []):
        if section.get("section_title") == "lead_paragraph":
            return section.get("text", "").strip()
    return ""


# Main selection: check files, filter article pairs and output candidates
# ---------------------------------------------------------------------------
def main(mapping_path: Path, als_path: Path, de_path: Path) -> None:
    missing_paths = [
        path for path in (mapping_path, als_path, de_path)
        if not path.is_file()
    ]
    if missing_paths:
        missing = "\n  ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Die folgenden Eingabedateien fehlen:\n  " + missing
            + "\nGib abweichende Pfade mit --mapping, --als und --de an."
        )

    print("Loading ALS articles...")
    als_articles = load_as_dict(als_path, "page_id")
    print(f"Loaded {len(als_articles)} ALS articles.")

    print("Loading DE articles...")
    de_articles = load_as_dict(de_path, "page_id")
    print(f"Loaded {len(de_articles)} DE articles.")

    results = []
    skipped_nolinks = 0
    skipped_missing = 0
    skipped_disambiguation = 0
    skipped_short_lead = 0
    skipped_few_sections = 0
    skipped_short_article = 0
    skipped_long_article = 0
    skipped_length_mismatch = 0
    skipped_section_mismatch = 0

    print("Processing mapping file...")
    with mapping_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            mapping = json.loads(line)

            als_id = str(mapping.get("als_pageid", ""))
            de_id = str(mapping.get("de_pageid", ""))

            if de_id == "nolinks" or not als_id or not de_id:
                skipped_nolinks += 1
                continue

            als_article = als_articles.get(als_id)
            de_article = de_articles.get(de_id)

            if not als_article or not de_article:
                skipped_missing += 1
                continue

            if is_disambiguation(als_article) or is_disambiguation(de_article):
                skipped_disambiguation += 1
                continue

            als_lead_words = len(get_lead_text(als_article).split())
            de_lead_words = len(get_lead_text(de_article).split())
            if als_lead_words < 50 or de_lead_words < 50:
                skipped_short_lead += 1
                continue

            als_sections = count_content_sections(als_article.get("sections", []))
            de_sections = count_content_sections(de_article.get("sections", []))

            if als_sections < MIN_SECTIONS or de_sections < MIN_SECTIONS:
                skipped_few_sections += 1
                continue

            als_total_words = len(get_all_text(als_article.get("sections", [])).split())
            de_total_words = len(get_all_text(de_article.get("sections", [])).split())

            if als_total_words < MIN_WORDS or de_total_words < MIN_WORDS:
                skipped_short_article += 1
                continue

            if als_total_words > MAX_WORDS or de_total_words > MAX_WORDS:
                skipped_long_article += 1
                continue

            length_ratio = abs(math.log(als_total_words / de_total_words))
            if length_ratio > MAX_LENGTH_RATIO:
                skipped_length_mismatch += 1
                continue

            section_ratio = abs(math.log(als_sections / de_sections))
            if section_ratio > MAX_SECTION_RATIO:
                skipped_section_mismatch += 1
                continue

            results.append({
                "als_title": als_article.get("page_title"),
                "de_title": de_article.get("page_title"),
                "als_id": als_id,
                "de_id": de_id,
                "als_sections": als_sections,
                "de_sections": de_sections,
                "als_lead_words": als_lead_words,
                "de_lead_words": de_lead_words,
                "als_total_words": als_total_words,
                "de_total_words": de_total_words,
                "length_ratio": round(length_ratio, 3),
                "section_ratio": round(section_ratio, 3),
            })

    # The remaining article pairs are sorted by ALS word count in descending order
    results.sort(key=lambda x: x["als_total_words"], reverse=True)

    print(f"\nFiltering summary:")
    print(f"  Skipped (no DE link):           {skipped_nolinks}")
    print(f"  Skipped (missing from data):    {skipped_missing}")
    print(f"  Skipped (disambiguation):       {skipped_disambiguation}")
    print(f"  Skipped (lead too short):       {skipped_short_lead}")
    print(f"  Skipped (too few sections):     {skipped_few_sections}")
    print(f"  Skipped (article too short):    {skipped_short_article}")
    print(f"  Skipped (article too long):     {skipped_long_article}")
    print(f"  Skipped (length too different): {skipped_length_mismatch}")
    print(f"  Skipped (sections too diff):    {skipped_section_mismatch}")
    print(f"  Remaining good pairs:           {len(results)}")

    print(f"\nTop 20 candidates (500-1000 words, comparable length and sections):")
    print(f"{'ALS Title':<35} {'DE Title':<35} {'ALS sec':<8} {'DE sec':<8} "
          f"{'ALS words':<12} {'DE words':<12} {'Len ratio':<10} {'Sec ratio':<10} "
          f"{'ALS ID':<12} {'DE ID':<12}")
    print("-" * 155)

    for r in results[:20]:
        print(
            f"{str(r['als_title']):<35} {str(r['de_title']):<35} "
            f"{r['als_sections']:<8} {r['de_sections']:<8} "
            f"{r['als_total_words']:<12} {r['de_total_words']:<12} "
            f"{r['length_ratio']:<10} {r['section_ratio']:<10} "
            f"{r['als_id']:<12} {r['de_id']:<12}"
        )


def parse_args() -> argparse.Namespace:
    """Read the optional input paths for the article selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
        help="JSONL-Datei mit der ALS-DE-Zuordnung.",
    )
    parser.add_argument(
        "--als",
        type=Path,
        default=DEFAULT_ALS_PATH,
        help="JSONL-Datei mit den alemannischen Artikeln.",
    )
    parser.add_argument(
        "--de",
        type=Path,
        default=DEFAULT_DE_PATH,
        help="JSONL-Datei mit den standarddeutschen Artikeln.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments.mapping, arguments.als, arguments.de)
