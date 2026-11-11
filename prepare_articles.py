
"""
Cleans the articles for the DE/ALS InfoGap pipeline.

Processing steps:
  1. Raw or text inputs are converted into clean paragraph blocks.
  2. Reference and navigation sections are removed.
  3. Section texts are split into individual paragraphs.
  4. Obvious markup blocks are discarded.
  5. Inline citation remnants are cleaned up carefully.
  6. Paragraphs with fewer than six words are removed.
  7. Exactly duplicated sections and cleaned paragraphs are removed.
  8. The ordered paragraphs are saved as a flat JSONL structure.

Modes:
  ``full`` reads the regular raw data from the JSONL files defined in
  ``pipeline_config.py`` and writes the regular cleaned files.

  ``gold`` reads the manually exported texts from ``gold_texts/`` and writes:
      data/gold_clean/gold_als.clean.jsonl
      data/gold_clean/gold_de.clean.jsonl
      configs/gold_article_pairs.jsonl

By default, all ten gold pairs are prepared.
"""


import argparse
import json
import re
from pathlib import Path

from pipeline_config import (
    ALS_RAW_PATH,
    DE_RAW_PATH,
    ALS_CLEAN_PATH,
    DE_CLEAN_PATH,
    CONFIG_DIR,
    DATA_DIR,
    ROOT_DIR,
)



# Default mode
# ---------------------------------------------------------------------------

DEFAULT_MODE = "gold"

GOLD_TEXT_DIR = ROOT_DIR / "gold_texts"
GOLD_CLEAN_DIR = DATA_DIR / "gold_clean"
GOLD_ALS_CLEAN_PATH = GOLD_CLEAN_DIR / "gold_als.clean.jsonl"
GOLD_DE_CLEAN_PATH = GOLD_CLEAN_DIR / "gold_de.clean.jsonl"
GOLD_PAIR_CONFIG_PATH = CONFIG_DIR / "gold_article_pairs.jsonl"



# Gold article pairs
# ---------------------------------------------------------------------------
# ``None`` enables all ten bundled gold pairs. For targeted tests,
# a set of individual pair_id values can be entered here instead

ACTIVE_GOLD_PAIR_KEYS: set[str] | None = None

GOLD_TEXT_PAIRS = [
    {
        "pair_id": "le_mans",
        "title": "Le Mans",
        "als_filename": "le_mans_als.txt",
        "de_filename": "le_mans_de.txt",
    },
    {
        "pair_id": "notschrei",
        "title": "Notschrei",
        "als_filename": "notschrei_als.txt",
        "de_filename": "notschrei_de.txt",
    },
    {
        "pair_id": "ernst_tappolet",
        "title": "Ernst Tappolet",
        "als_filename": "ernst_tappolet_als.txt",
        "de_filename": "ernst_tappolet_de.txt",
    },
    {
        "pair_id": "ethnie",
        "title": "Ethnie",
        "als_filename": "ethnie_als.txt",
        "de_filename": "ethnie_de.txt",
    },
    {
        "pair_id": "kalender",
        "title": "Kalender",
        "als_filename": "kalender_als.txt",
        "de_filename": "kalender_de.txt",
    },
    {
        "pair_id": "rheinbruecke_schaffhausen_feuerthalen",
        "title": "Rheinbrücke Schaffhausen–Feuerthalen",
        "als_filename": "facts_rheinbrücke_schaffhausen_feuerthalen_als.txt",
        "de_filename": "facts_rheinbrücke_schaffhausen_feuerthalen_de.txt",
    },
    {
        "pair_id": "leo_der_grosse",
        "title": "Leo der Große",
        "als_filename": "facts_leo_der_große_als.txt",
        "de_filename": "facts_leo_der_große_de.txt",
    },
    {
        "pair_id": "eichener_see",
        "title": "Eichener See",
        "als_filename": "facts_eichener_see_als.txt",
        "de_filename": "facts_eichener_see_de.txt",
    },
    {
        "pair_id": "wahrscheinlichkeit",
        "title": "Wahrscheinlichkeit",
        "als_filename": "facts_wahrscheinlichkeit_als.txt",
        "de_filename": "facts_wahrscheinlichkeit_de.txt",
    },
    {
        "pair_id": "oliver_kahn",
        "title": "Oliver Kahn",
        "als_filename": "facts_oliver_kahn_als.txt",
        "de_filename": "facts_oliver_kahn_de.txt",
    },
]



# Reference and navigation sections to exclude
# ---------------------------------------------------------------------------

SKIP_REFERENCE_TITLES = {
    "weblinks",
    "weblingg",
    "literatur",
    "litratuur",
    "einzelnachweise",
    "fuessnoote",
    "fußnoten",
    "fussnoten",
    "quellen",
    "anmerkungen",
    "siehe auch",
    "kategorie",
    "categories",
    "references",
    "notes",
    "bibliography",
    "external links",
    "web links",
    "see also",
}

MIN_WORDS = 6



# Markup filtering
# ---------------------------------------------------------------------------

IMAGE_MAP_PREFIXES = ("Datei:", "Bild:", "File:", "Image:")

IMAGE_MAP_LINE = re.compile(
    r"^\s*(?:rect|poly|circle|default|desc)\b[\d\s.,;|-]*$",
    re.IGNORECASE,
)

HEADER_RE = re.compile(r"^\s*==\s*(.*?)\s*==\s*$")


def is_markup_unit(unit: str) -> bool:
    """Detect units consisting of flattened image or file markup.

    List markers are not discarded here. If a list line contains
    actual article content, ``clean_unit()`` only removes the marker.
    """
    # Only clearly technical markup is removed; actual list content is preserved
    u = unit.strip()

    if not u:
        return True

    if u.startswith(IMAGE_MAP_PREFIXES):
        return True

    if IMAGE_MAP_LINE.match(u):
        return True

    return False



# Removal of embedded source citations
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
BRACKET_NOTE_RE = re.compile(r"\[(?:\d+|[a-z])\]")

ATTACHED_DOMAIN_RE = re.compile(
    r"(?<=[A-Za-zÄÖÜäöüß])\.(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b"
)

CITATION_RE = re.compile(
    r"(?:"
    r"(?<=[\w)\]\"'\u201c\u201d\u00bb][.,])\s*"
    r"|(?<=[a-zäöüß])(?=[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+){0,4}:)"
    r")"
    r"(?=[^\n]{0,900}(?:In:|\(Hrsg\.\)|\(Begr\.\)|\(Red\.\)|\(Hg\.\)|\(ed\.\)"
    r"|Verlag|ISBN|doi:|Zugriff am|abgerufen am|Auflage|Press|Journal|Bd\.|Nr\.|S\.\s?[IVXLCDM\d]))"
    r".*?"
    r"(?:"
    r"\)\.?"
    r"|ISBN\s[\dXx][\dXx\s-]*?\.?"
    r"|doi:[^\s)]+\)?\.?"
    r"|S\.\s?[IVXLCDM\d]+(?:\s*[\u2013-]\s*[IVXLCDM\d]+)?(?:,\s?\d+)*(?:\s?ff?\.?)?\.?"
    r"|abgerufen am [^.)]+\.?"
    r")"
    r"(?=\s+[A-Za-zÄÖÜäöüß]|\s*$)",
    re.DOTALL | re.IGNORECASE,
)

KNOWN_INLINE_NOTE_RE = re.compile(
    r"\s+Der erste Teil der Ars conjectandi.*?(?=\s+und Jakob|\s+\.\s*|$)",
    re.DOTALL,
)


def strip_footnotes(text: str) -> str:
    # The patterns remove citation remnants without rewriting normal running text
    cleaned = URL_RE.sub("", text)
    cleaned = ATTACHED_DOMAIN_RE.sub("", cleaned)
    cleaned = BRACKET_NOTE_RE.sub("", cleaned)
    cleaned = KNOWN_INLINE_NOTE_RE.sub("", cleaned)

    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = CITATION_RE.sub("", cleaned)

    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip()


# General helper functions
# ---------------------------------------------------------------------------

def normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_title(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower().strip(":"))


def should_skip_title(title: str) -> bool:
    return normalize_title(title) in SKIP_REFERENCE_TITLES


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n+", text) if p.strip()]


def clean_unit(unit: str) -> str | None:
    # A unit is normalized, stripped of list markers and checked against the minimum length
    lines = []

    for line in unit.splitlines():
        line = re.sub(r"^\s*[*#•]+\s*", "", line.strip())
        if line:
            lines.append(line)

    cleaned = " ".join(lines)
    cleaned = strip_footnotes(cleaned)
    return cleaned or None



# Cleaning of the JSONL articles
# ---------------------------------------------------------------------------

def collect_paragraphs_from_sections(
    sections: list[dict],
    seen_sections: set[tuple[str, str]] | None = None,
    seen_paragraphs: set[str] | None = None,
) -> list[str]:
    # Sections are traversed recursively in the order of the input data
    if seen_sections is None:
        seen_sections = set()

    if seen_paragraphs is None:
        seen_paragraphs = set()

    paragraphs = []

    for section in sections:
        title = section.get("section_title", "") or ""

        if should_skip_title(title):
            continue

        text = (section.get("text") or "").strip()

        section_key = (
            normalize_for_dedupe(title),
            normalize_for_dedupe(text),
        )

        section_body_is_duplicate = bool(text and section_key in seen_sections)

        if text and not section_body_is_duplicate:
            seen_sections.add(section_key)

            for unit in split_paragraphs(text):
                if is_markup_unit(unit):
                    continue

                cleaned = clean_unit(unit)
                if not cleaned:
                    continue

                if len(cleaned.split()) < MIN_WORDS:
                    continue

                paragraph_key = normalize_for_dedupe(cleaned)

                if paragraph_key in seen_paragraphs:
                    continue

                seen_paragraphs.add(paragraph_key)
                paragraphs.append(cleaned)

        subsections = section.get("subsections", []) or []
        if subsections:
            paragraphs.extend(
                collect_paragraphs_from_sections(
                    subsections,
                    seen_sections=seen_sections,
                    seen_paragraphs=seen_paragraphs,
                )
            )

    return paragraphs


def clean_jsonl_article(article: dict) -> dict:
    # The pipeline's paragraph list is built from the raw structure
    paragraphs = collect_paragraphs_from_sections(article.get("sections", []) or [])

    cleaned_article = dict(article)
    cleaned_article["paragraphs"] = paragraphs
    cleaned_article.pop("sections", None)

    return cleaned_article



# Cleaning of the gold texts
# ---------------------------------------------------------------------------

def parse_gold_text_file(path: Path, page_id: str, language: str, fallback_title: str) -> dict:
    # Headings structure the text file; the text before them forms the introductory paragraph
    if not path.exists():
        raise FileNotFoundError(f"Missing gold text file: {path}")

    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    title = None
    body_start = 0

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not HEADER_RE.match(stripped):
            title = stripped
            body_start = index + 1
            break

    if title is None:
        title = fallback_title

    sections: list[tuple[str, str]] = []
    current_title = "lead_paragraph"
    buffer: list[str] = []

    def flush_section() -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if text:
            sections.append((current_title, text))
        buffer = []

    for line in lines[body_start:]:
        match = HEADER_RE.match(line)
        if match:
            flush_section()
            current_title = match.group(1).strip() or "lead_paragraph"
        else:
            buffer.append(line)

    flush_section()

    seen_sections: set[tuple[str, str]] = set()
    seen_paragraphs: set[str] = set()
    paragraphs: list[str] = []

    for section_title, section_text in sections:
        if should_skip_title(section_title):
            continue

        section_key = (
            normalize_title(section_title),
            normalize_for_dedupe(section_text),
        )

        if section_key in seen_sections:
            continue

        seen_sections.add(section_key)

        for unit in split_paragraphs(section_text):
            if is_markup_unit(unit):
                continue

            if should_skip_title(unit):
                continue

            if normalize_title(unit).startswith("siehe auch"):
                continue

            cleaned = clean_unit(unit)
            if not cleaned:
                continue

            if len(cleaned.split()) < MIN_WORDS:
                continue

            paragraph_key = normalize_for_dedupe(cleaned)

            if paragraph_key in seen_paragraphs:
                continue

            seen_paragraphs.add(paragraph_key)
            paragraphs.append(cleaned)

    return {
        "page_id": page_id,
        "page_title": title,
        "language": language,
        "source_path": str(path.relative_to(ROOT_DIR)),
        "paragraphs": paragraphs,
    }



# Reading and writing JSONL
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {error}"
                ) from error


def write_jsonl(path: Path, records) -> None:
    # Each line remains a self-contained UTF-8 JSON object without ASCII escaping
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Mode for the complete dataset
# ---------------------------------------------------------------------------

def clean_file(input_path: Path, output_path: Path, language_label: str) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    total_articles = 0
    total_paragraphs = 0
    empty_articles = 0

    cleaned_records = []

    print(f"\nCleaning {language_label}:")
    print(f"  input:  {input_path}")
    print(f"  output: {output_path}")

    for article in iter_jsonl(input_path):
        total_articles += 1
        cleaned = clean_jsonl_article(article)

        n_paragraphs = len(cleaned.get("paragraphs", []))
        total_paragraphs += n_paragraphs

        if n_paragraphs == 0:
            empty_articles += 1

        cleaned_records.append(cleaned)

    write_jsonl(output_path, cleaned_records)

    print(f"  articles written:       {total_articles}")
    print(f"  total clean paragraphs: {total_paragraphs}")
    print(f"  articles with 0 paras:  {empty_articles}")


def run_full_mode() -> None:
    # Full mode processes external raw data outside the project folder
    clean_file(ALS_RAW_PATH, ALS_CLEAN_PATH, "ALS")
    clean_file(DE_RAW_PATH, DE_CLEAN_PATH, "DE")

    print("\nDone.")
    print("Clean files created:")
    print(f"  {ALS_CLEAN_PATH}")
    print(f"  {DE_CLEAN_PATH}")



# Mode for the bundled gold dataset
# ---------------------------------------------------------------------------

def selected_gold_pairs() -> list[dict]:
    if ACTIVE_GOLD_PAIR_KEYS is None:
        return GOLD_TEXT_PAIRS

    return [
        pair for pair in GOLD_TEXT_PAIRS
        if pair["pair_id"] in ACTIVE_GOLD_PAIR_KEYS
    ]


def run_gold_mode() -> None:
    # Gold mode generates the cleaned texts and their pair configuration
    pairs_to_prepare = selected_gold_pairs()

    if not pairs_to_prepare:
        raise ValueError("No gold pairs selected. Check ACTIVE_GOLD_PAIR_KEYS.")

    als_records: list[dict] = []
    de_records: list[dict] = []
    pair_records: list[dict] = []

    print("\nPreparing gold texts:")
    print(f"  input folder: {GOLD_TEXT_DIR}")
    print(f"  active pairs: {len(pairs_to_prepare)}")

    for pair in pairs_to_prepare:
        pair_id = pair["pair_id"]
        title = pair["title"]

        pair_key = f"gold__{pair_id}"
        als_page_id = f"gold_{pair_id}_als"
        de_page_id = f"gold_{pair_id}_de"

        als_path = GOLD_TEXT_DIR / pair["als_filename"]
        de_path = GOLD_TEXT_DIR / pair["de_filename"]

        als_article = parse_gold_text_file(
            path=als_path,
            page_id=als_page_id,
            language="als",
            fallback_title=title,
        )

        de_article = parse_gold_text_file(
            path=de_path,
            page_id=de_page_id,
            language="de",
            fallback_title=title,
        )

        als_records.append(als_article)
        de_records.append(de_article)

        pair_records.append({
            "pair_key": pair_key,
            "cluster": "Gold",
            "als_page_id": als_page_id,
            "de_page_id": de_page_id,
            "title": title,
            "als_title": als_article.get("page_title"),
            "de_title": de_article.get("page_title"),
            "als_source_path": str(als_path.relative_to(ROOT_DIR)),
            "de_source_path": str(de_path.relative_to(ROOT_DIR)),
        })

        als_words = sum(len(p.split()) for p in als_article["paragraphs"])
        de_words = sum(len(p.split()) for p in de_article["paragraphs"])

        print(f"\n{pair_key}")
        print(f"  ALS: {len(als_article['paragraphs'])} paragraphs, {als_words} words")
        print(f"  DE:  {len(de_article['paragraphs'])} paragraphs, {de_words} words")

    write_jsonl(GOLD_ALS_CLEAN_PATH, als_records)
    write_jsonl(GOLD_DE_CLEAN_PATH, de_records)
    write_jsonl(GOLD_PAIR_CONFIG_PATH, pair_records)

    print("\nDone.")
    print("Gold clean files created:")
    print(f"  {GOLD_ALS_CLEAN_PATH}")
    print(f"  {GOLD_DE_CLEAN_PATH}")
    print(f"  {GOLD_PAIR_CONFIG_PATH}")



# CLI: without an argument, the bundled, ready-to-use gold mode is used
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("full", "gold"),
        default=DEFAULT_MODE,
        help="full = normal raw JSONL preparation; gold = prepare gold_texts/*.txt",
    )
    args = parser.parse_args()

    if args.mode == "full":
        run_full_mode()
    elif args.mode == "gold":
        run_gold_mode()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
