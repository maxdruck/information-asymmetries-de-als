"""
Prepares the manually decomposed facts for the control run.

The module converts ``manual_paragraph_fact_annotation_max.xlsx`` into the fact
structure used by alignment and X-FACTMATCH.

Expected workbook structure
---------------------------
Sheet: ``IAA_Facts_Clean``
Required columns:
  page_title, lang, paragraph_id, fact_order, fact_id, fact_text
Optional columns, if present:
  page_id, section_title, paragraph_text, span_evidence, annotator_id

Output schema
-------------
Each fact receives a numeric, article-wide ``fact_index`` for
processing. The ID from the manual annotation is kept as
``manual_fact_id``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MANUAL_XLSX = ROOT_DIR / "manual_paragraph_fact_annotation_max.xlsx"
MANUAL_FACTS_DIR = ROOT_DIR / "outputs" / "manual_facts"
MANUAL_ALIGNMENT_DIR = ROOT_DIR / "outputs" / "manual_alignments"
MANUAL_VERIFIED_DIR = ROOT_DIR / "outputs" / "manual_verified"
MANUAL_CACHE_DIR = ROOT_DIR / "outputs" / "cache" / "manual_xfactmatch"
MANUAL_CONTROL_PAIR_KEYS = (
    "gold__eichener_see",
    "gold__ernst_tappolet",
    "gold__kalender",
)


@dataclass(frozen=True)
class ManualFactRow:
    page_id: Any
    page_title: str
    lang: str
    section_title: str | None
    paragraph_id: str
    paragraph_index: int
    paragraph_text: str | None
    fact_order: int
    manual_fact_id: str
    fact_text: str
    span_evidence: str | None
    annotator_id: str | None
    row_number: int


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def portable_path(path: Path) -> str:
    """Represent project-internal paths for stored metadata as relative paths."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR))
    except ValueError:
        return str(resolved)


# These path functions keep the file names of all manual stages consistent
def manual_facts_path(language: str, pair_key: str) -> Path:
    return MANUAL_FACTS_DIR / f"manual_facts_{language}_{pair_key}.json"


def manual_alignment_path(pair_key: str) -> Path:
    return MANUAL_ALIGNMENT_DIR / f"manual_alignment_{pair_key}.json"


def manual_verified_path(pair_key: str) -> Path:
    return MANUAL_VERIFIED_DIR / f"manual_verified_{pair_key}.json"


def manual_xfactmatch_cache_path(pair_key: str) -> Path:
    return MANUAL_CACHE_DIR / f"manual_xfactmatch_cache_{pair_key}.json"


def normalize_title(value: str | None) -> str:
    """Normalize titles for a robust mapping between configuration and Excel."""
    if value is None:
        return ""

    # Accents, dashes and ß are unified for matching
    text = str(value).strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("ß", "ss")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify_title(value: str | None) -> str:
    return normalize_title(value).replace(" ", "_")


def parse_paragraph_index(paragraph_id: Any) -> int:
    """Convert a paragraph ID such as P0 or P1 into an integer index."""
    if paragraph_id is None:
        raise ValueError("Missing paragraph_id in manual facts sheet")

    text = str(paragraph_id).strip()
    match = re.search(r"(\d+)$", text)
    if not match:
        raise ValueError(f"Cannot parse paragraph_id as P<number>: {paragraph_id!r}")
    return int(match.group(1))


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_manual_rows(xlsx_path: Path, sheet_name: str = "IAA_Facts_Clean") -> list[ManualFactRow]:
    """Load all usable fact rows from the manual workbook."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Manual facts workbook not found: {xlsx_path}")

    workbook = load_workbook(xlsx_path, read_only=True, data_only=True)
    if sheet_name not in workbook.sheetnames:
        raise ValueError(
            f"Sheet {sheet_name!r} not found in {xlsx_path}. "
            f"Available sheets: {workbook.sheetnames}"
        )

    worksheet = workbook[sheet_name]
    header_values = next(worksheet.iter_rows(min_row=1, max_row=1, values_only=True))
    headers = {str(name).strip(): index for index, name in enumerate(header_values) if name is not None}

    required = ["page_title", "lang", "paragraph_id", "fact_order", "fact_id", "fact_text"]
    missing = [name for name in required if name not in headers]
    if missing:
        raise ValueError(f"Missing required columns in {sheet_name}: {missing}")

    def get(row: tuple, column: str, default: Any = None) -> Any:
        index = headers.get(column)
        if index is None or index >= len(row):
            return default
        return row[index]

    # Only fully identifiable fact rows are passed into the pipeline
    rows: list[ManualFactRow] = []
    for row_number, row in enumerate(worksheet.iter_rows(min_row=2, values_only=True), start=2):
        page_title = get(row, "page_title")
        lang = get(row, "lang")
        paragraph_id = get(row, "paragraph_id")
        fact_text = get(row, "fact_text")
        manual_fact_id = get(row, "fact_id")

        if not page_title or not lang or not paragraph_id or not fact_text or not manual_fact_id:
            continue

        rows.append(
            ManualFactRow(
                page_id=get(row, "page_id"),
                page_title=str(page_title).strip(),
                lang=str(lang).strip().lower(),
                section_title=str(get(row, "section_title", "")).strip() or None,
                paragraph_id=str(paragraph_id).strip(),
                paragraph_index=parse_paragraph_index(paragraph_id),
                paragraph_text=str(get(row, "paragraph_text", "")).strip() or None,
                fact_order=as_int(get(row, "fact_order"), default=0),
                manual_fact_id=str(manual_fact_id).strip(),
                fact_text=str(fact_text).strip(),
                span_evidence=str(get(row, "span_evidence", "")).strip() or None,
                annotator_id=str(get(row, "annotator_id", "")).strip() or None,
                row_number=row_number,
            )
        )

    if not rows:
        raise ValueError(f"No usable manual fact rows found in {xlsx_path}:{sheet_name}")

    return rows


def available_manual_titles(rows: list[ManualFactRow]) -> list[str]:
    return sorted({row.page_title for row in rows})


def resolve_manual_title(pair: dict, manual_titles: list[str]) -> str:
    """Map a configured pair to the title used in the manual Excel file.

    First, ``pair['title']``, ``pair['de_title']`` and ``pair['als_title']``
    are matched exactly. If that fails, it is checked whether the
    normalized title occurs in the ``pair_key``. This helps with
    pair IDs that additionally contain page IDs.
    """
    # First an exact normalized comparison is made; the pair_key only serves as a fallback

    normalized_to_original = {normalize_title(title): title for title in manual_titles}

    candidates = [
        pair.get("title"),
        pair.get("de_title"),
        pair.get("als_title"),
        pair.get("page_title"),
    ]

    for candidate in candidates:
        key = normalize_title(candidate)
        if key in normalized_to_original:
            return normalized_to_original[key]

    pair_key = str(pair.get("pair_key", "")).lower()
    for title in manual_titles:
        slug = slugify_title(title)
        if slug and slug in pair_key:
            return title

    raise ValueError(
        "Could not match config pair to a manual Excel title.\n"
        f"pair_key={pair.get('pair_key')!r}, title={pair.get('title')!r}\n"
        f"Available manual titles: {manual_titles}"
    )


def facts_for_title(rows: list[ManualFactRow], manual_title: str, language: str) -> list[dict]:
    """Create article-wide numbered facts for one title and one language."""
    language = language.lower()
    selected = [
        row for row in rows
        if row.page_title == manual_title and row.lang.lower() == language
    ]

    # The stable sort produces the same article-wide fact_index values on every run
    selected.sort(key=lambda row: (row.paragraph_index, row.fact_order, row.row_number))

    facts: list[dict] = []
    for fact_index, row in enumerate(selected):
        facts.append({
            "fact_index": fact_index,
            "manual_fact_id": row.manual_fact_id,
            "paragraph_index": row.paragraph_index,
            "paragraph_id": row.paragraph_id,
            "fact_order_in_paragraph": row.fact_order,
            "fact": row.fact_text,
            "span_evidence": row.span_evidence,
            "paragraph_text": row.paragraph_text,
            "section_title": row.section_title,
            "source": "manual_paragraph_fact_annotation_max.xlsx",
            "annotator_id": row.annotator_id,
        })

    return facts


def load_manual_pair_facts(xlsx_path: Path, pair: dict) -> tuple[str, dict[str, dict]]:
    """Load the manual DE and ALS facts of a selected pair.

    Returns:
      (manual_title, {"de": de_data, "als": als_data})
    Each ``*_data`` object corresponds to the JSON structure of the Gemma facts.
    """
    rows = load_manual_rows(xlsx_path)
    manual_title = resolve_manual_title(pair, available_manual_titles(rows))
    pair_key = pair["pair_key"]

    output: dict[str, dict] = {}
    for language in ("de", "als"):
        facts = facts_for_title(rows, manual_title, language)
        if not facts:
            raise ValueError(f"No manual {language.upper()} facts found for title {manual_title!r}")

        output[language] = {
            "pair_key": pair_key,
            "pair_title": manual_title,
            "language": language,
            "fact_source": "manual",
            "manual_xlsx": portable_path(xlsx_path),
            "facts": facts,
        }

    return manual_title, output


def write_manual_pair_facts(xlsx_path: Path, pair: dict, overwrite: bool = False) -> tuple[str, dict[str, dict]]:
    """Load manual facts and save them in the unified schema under ``outputs/``."""
    manual_title, facts_by_language = load_manual_pair_facts(xlsx_path, pair)

    # DE and ALS are saved in the same JSON schema as the automatically generated facts
    for language, data in facts_by_language.items():
        path = manual_facts_path(language, pair["pair_key"])
        if overwrite or not path.exists():
            write_json(path, data)

    return manual_title, facts_by_language
