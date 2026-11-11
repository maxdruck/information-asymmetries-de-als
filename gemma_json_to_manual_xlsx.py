"""
Transfers the Gemma decompositions into the schema of the manual Excel annotation.

The ``facts_*.json`` files from ``Gemma_sample_gold`` are formatted so that
they match the structure of ``manual_paragraph_fact_annotation_max.xlsx`` in
the ``IAA_Facts_Clean`` sheet.

For an identical mapping, the script takes the following from the manual table:
  - the exact column order (incl. the empty column at the end)
  - the mapping ``page_title`` -> slug for the ``fact_id``
    (e.g. Ernst Tappolet -> Tappolet)
  - the (page_title, lang) -> page_id mapping (the real Wikipedia IDs)

The ``fact_id`` follows the pattern ``LANG_Slug_P{paragraph}_F{position}``;
``fact_order`` starts at 1 in each paragraph.

Not reconstructable from the Gemma JSON and therefore left empty:
  section_title, span_evidence
(Gemma provides neither section headings nor evidence spans.)

The ``paragraph_id`` (P0, P1, ...) follows Gemma's own paragraph segmentation.
It therefore does not have to match the paragraphs of the manual annotation
exactly.

Usage:
    python gemma_json_to_manual_xlsx.py \
        --gemma-dir Gemma_sample_gold \
        --manual manual_paragraph_fact_annotation_max.xlsx \
        --out gemma_facts_formatiert.xlsx
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import openpyxl

from pipeline_config import OUTPUT_DIR, ROOT_DIR


DEFAULT_GEMMA_DIR = OUTPUT_DIR / "goldstandard_oneshot"
DEFAULT_MANUAL_PATH = ROOT_DIR / "manual_paragraph_fact_annotation_max.xlsx"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "goldstandard_oneshot_formatiert.xlsx"


# Tolerant JSON loader (handles null padding and BOM)
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, "rb") as f:
        raw = f.read()
    raw = raw.rstrip(b"\x00").strip().lstrip(b"\xef\xbb\xbf")
    return json.loads(raw.decode("utf-8"))


def slugify_fallback(title: str) -> str:
    # This fallback only applies if the manual table does not provide a known slug.
    t = title
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"),
                 ("Ä", "Ae"), ("Ö", "Oe"), ("Ü", "Ue"), ("ß", "ss")):
        t = t.replace(a, b)
    t = t.replace("–", "_").replace("-", "_")
    t = re.sub(r"\s+", "_", t.strip())
    t = re.sub(r"[^A-Za-z0-9_]", "", t)
    return t



# Learn the schema, slug and page_id mapping from the manual table
# ---------------------------------------------------------------------------
def learn_from_manual(path: str, sheet: str):
    # Schema and IDs come from the manual annotation so that both files are directly comparable.
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    data = list(ws.iter_rows(values_only=True))
    header = list(data[0])
    idx = {h: i for i, h in enumerate(header) if h is not None}

    slug_map = {}
    pageid_map = {}
    for r in data[1:]:
        if not any(c is not None for c in r):
            continue
        pt = r[idx["page_title"]]
        lang = r[idx["lang"]]
        fid = r[idx["fact_id"]]
        pid = r[idx["page_id"]]
        if pt and fid:
            core = re.sub(r"^(DE|ALS)_", "", str(fid))
            slug = re.sub(r"_P\d+_F\d+$", "", core)
            slug_map.setdefault(str(pt), slug)
        if pt and lang and pid is not None:
            pageid_map.setdefault((str(pt), str(lang).upper()), pid)
    return header, slug_map, pageid_map



# Convert one Gemma JSON into rows (dicts) in the manual schema
# ---------------------------------------------------------------------------
def rows_from_gemma(d: dict, slug_map: dict, pageid_map: dict, annotator: str):
    # The nested paragraph facts are converted into a flat Excel row structure.
    page_title = d.get("page_title", "")
    lang = str(d.get("language_variety", "")).upper()
    paragraphs = d.get("paragraphs", [])
    model = d.get("decomposition_model", "")
    ann = annotator or model

    slug = slug_map.get(page_title) or slugify_fallback(page_title)
    page_id = pageid_map.get((page_title, lang), d.get("page_id"))

    facts = sorted(
        d.get("facts", []),
        key=lambda f: (f.get("paragraph_index", 0), f.get("fact_index", 0)),
    )

    per_para_counter = {}
    rows = []
    for f in facts:
        p = f.get("paragraph_index", 0)
        per_para_counter[p] = per_para_counter.get(p, 0) + 1
        order = per_para_counter[p]
        para_text = paragraphs[p] if 0 <= p < len(paragraphs) else ""
        rows.append({
            "page_id": page_id,
            "page_title": page_title,
            "lang": lang,
            "section_title": "",
            "paragraph_id": f"P{p}",
            "paragraph_text": para_text,
            "fact_order": order,
            "fact_id": f"{lang}_{slug}_P{p}_F{order}",
            "fact_text": f.get("fact", ""),
            "span_evidence": "",
            "annotator_id": ann,
        })
    return rows



# Entry point: collect all Gemma files and write them together into one workbook
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gemma-dir", type=Path, default=DEFAULT_GEMMA_DIR,
                    help="Ordner mit facts_*.json")
    ap.add_argument("--manual", type=Path, default=DEFAULT_MANUAL_PATH,
                    help="manuelle Annotation (xlsx)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    ap.add_argument("--sheet", default="IAA_Facts_Clean")
    ap.add_argument("--glob", default="facts_*.json")
    ap.add_argument("--annotator", default="",
                    help="Wert fuer annotator_id (leer = Modellname aus JSON)")
    args = ap.parse_args()

    header, slug_map, pageid_map = learn_from_manual(args.manual, args.sheet)
    col_names = [h if h is not None else "" for h in header]

    paths = sorted(glob.glob(os.path.join(args.gemma_dir, args.glob)))
    if not paths:
        print("Keine Gemma-JSONs gefunden.")
        return

    out = openpyxl.Workbook()
    ws = out.active
    ws.title = args.sheet
    ws.append(col_names)

    total = 0
    for p in paths:
        try:
            d = load_json(p)
        except Exception as e:
            print(f"  WARN: {os.path.basename(p)} uebersprungen ({e})")
            continue
        rows = rows_from_gemma(d, slug_map, pageid_map, args.annotator)
        for r in rows:
            ws.append([r.get(name, "") for name in col_names])
        total += len(rows)
        print(f"  {os.path.basename(p)}: {len(rows)} Fakten "
              f"({d.get('page_title')} / {str(d.get('language_variety')).upper()})")

    for cells in ws.columns:
        w = max((len(str(c.value)) if c.value is not None else 0) for c in cells) + 2
        ws.column_dimensions[cells[0].column_letter].width = min(max(w, 8), 60)

    out.save(args.out)
    print(f"\n{total} Fakten aus {len(paths)} Dateien geschrieben: {args.out}")


if __name__ == "__main__":
    main()
