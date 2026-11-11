"""
Aggregates the X-FACTMATCH results of all article pairs.

The input consists of the ``verified_*.json`` files from ``outputs/verified``.
From these, a result matrix and the corresponding aggregates are produced.

Both directions are evaluated for each pair:
DE -> ALS : share of DE facts that occur in ALS
ALS -> DE : share of ALS facts that occur in DE

Metric following the paper (InfoGap, Fig. 3, "Num. entailed facts over total"):
overlap = shared_yes / total_source_facts
``unknown`` cases are counted in the denominator. This corresponds to
``raw_yes_share`` from the summary of ``run_xfactmatch.py``.

Additional columns:
de_to_als_ver / als_to_de_ver = overlap_ratio_verified = yes/(yes+no),
i.e. the overlap computed only over successfully verified facts. If there are
no ``unknown`` cases, the value is identical to the paper metric.

Output:
    1. Matrix table (one row per article pair) on the console
    2. Aggregate per cluster and overall (mean, median, min, max)
    3. CSV export, optional XLSX export

Usage:
python aggregate_xfactmatch.py --dir outputs/verified
python aggregate_xfactmatch.py --dir outputs/verified --xlsx overlap_matrix.xlsx
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from typing import Optional

from pipeline_config import OUTPUT_DIR, VERIFIED_DIR


DEFAULT_CSV_PATH = OUTPUT_DIR / "xfactmatch_aggregate_fewshot.csv"
DEFAULT_XLSX_PATH = OUTPUT_DIR / "xfactmatch_aggregate_fewshot.xlsx"



# Tolerant loader (handles null padding and broken BOMs)
# ---------------------------------------------------------------------------
def load_verified(path: str) -> Optional[dict]:
    try:
        with open(path, "rb") as f:
            raw = f.read()
        raw = raw.rstrip(b"\x00").strip().lstrip(b"\xef\xbb\xbf")
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        print(f"  WARN: {os.path.basename(path)} nicht lesbar ({e})")
        return None



def paper_overlap(summary: dict) -> Optional[float]:
    """
    Computes ``shared_yes / total_source_facts``.

    The computation uses the counters directly and therefore does not need
    a separate ``raw_yes_share`` field.
    """
    total = summary.get("total_source_facts")
    yes = summary.get("shared_yes")
    if total and yes is not None:
        return yes / total
    return summary.get("raw_yes_share")


def classify(overlap: Optional[float]) -> str:
    if overlap is None:
        return "?"
    if overlap < 0.35:
        return "niedrig"
    if overlap < 0.70:
        return "mittel"
    return "hoch"


# Each verified pair file yields exactly one row of the overlap matrix
def row_from(d: dict) -> dict:
    de = d.get("de_to_als_summary", {}) or {}
    als = d.get("als_to_de_summary", {}) or {}
    title = d.get("title") or d.get("pair_key") or "?"

    #  yes / total (unknown in the denominator)
    de_ov = paper_overlap(de)
    als_ov = paper_overlap(als)

    # Additional metric over all conclusively judged facts: yes / (yes + no)
    de_ov_ver = de.get("overlap_ratio_verified")
    als_ov_ver = als.get("overlap_ratio_verified")

    mean_ov = None
    if de_ov is not None and als_ov is not None:
        mean_ov = (de_ov + als_ov) / 2

    unknown_total = (de.get("unknown", 0) or 0) + (als.get("unknown", 0) or 0)
    facts_total = (de.get("total_source_facts") or 0) + (als.get("total_source_facts") or 0)
    unknown_share = (unknown_total / facts_total) if facts_total else None

    return {
        "titel": title,
        "cluster": d.get("cluster", ""),
        "modell": d.get("verification_model", ""),
        "de_fakten": de.get("total_source_facts"),
        "als_fakten": als.get("total_source_facts"),
        "de_to_als": de_ov,
        "als_to_de": als_ov,
        "mittel": mean_ov,
        "klasse": classify(mean_ov),
        "de_to_als_ver": de_ov_ver,
        "als_to_de_ver": als_ov_ver,
        "de_geteilt": de.get("shared_yes"),
        "de_exklusiv": de.get("source_only_no"),
        "als_geteilt": als.get("shared_yes"),
        "als_exklusiv": als.get("source_only_no"),
        "unknown": unknown_total,
        "unknown_anteil": unknown_share,
        "coverage_min": _mincov(de.get("verified_coverage"), als.get("verified_coverage")),
    }


def _mincov(a, b):
    vals = [x for x in (a, b) if x is not None]
    return min(vals) if vals else None


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)



# Output
# ---------------------------------------------------------------------------
FLOAT_KEYS = {
    "de_to_als", "als_to_de", "mittel",
    "de_to_als_ver", "als_to_de_ver",
    "unknown_anteil", "coverage_min",
}

COLS = [
    ("titel", "Artikel", 24, "l"),
    ("cluster", "Cluster", 11, "l"),
    ("de_fakten", "#DE", 5, "r"),
    ("als_fakten", "#ALS", 5, "r"),
    ("de_to_als", "DE->ALS", 8, "r"),
    ("als_to_de", "ALS->DE", 8, "r"),
    ("mittel", "Mittel", 7, "r"),
    ("klasse", "Klasse", 8, "l"),
    ("de_exklusiv", "DE-ex", 6, "r"),
    ("als_exklusiv", "ALS-ex", 6, "r"),
    ("unknown", "unkn", 5, "r"),
    ("unknown_anteil", "u-ant", 6, "r"),
]

# Additional columns that end up in the CSV/XLSX
EXPORT_ONLY = ["de_to_als_ver", "als_to_de_ver", "coverage_min",
               "de_geteilt", "als_geteilt", "modell"]


def print_table(rows):
    header = " | ".join(
        (name.ljust(w) if al == "l" else name.rjust(w)) for _, name, w, al in COLS
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for key, _, w, al in COLS:
            v = r[key]
            s = fmt(v) if key in FLOAT_KEYS else fmt(v, 0)
            s = str(s)[:w]
            cells.append(s.ljust(w) if al == "l" else s.rjust(w))
        print(" | ".join(cells))


# The aggregation first combines articles and then computes cluster values


def aggregate(rows, label):
    des = [r["de_to_als"] for r in rows if r["de_to_als"] is not None]
    alss = [r["als_to_de"] for r in rows if r["als_to_de"] is not None]
    if not des and not alss:
        return
    def block(vals):
        if not vals:
            return "  -"
        return (f"Mittel {statistics.mean(vals):.2f}  Median {statistics.median(vals):.2f}  "
                f"Min {min(vals):.2f}  Max {max(vals):.2f}")
    print(f"\n=== {label} ({len(rows)} Paare) ===")
    print(f"  DE->ALS : {block(des)}")
    print(f"  ALS->DE : {block(alss)}")
    tot_de = sum(r["de_fakten"] or 0 for r in rows)
    tot_als = sum(r["als_fakten"] or 0 for r in rows)
    tot_unk = sum(r["unknown"] or 0 for r in rows)
    tot_facts = tot_de + tot_als
    unk_share = f" ({tot_unk / tot_facts:.1%} aller Fakten)" if tot_facts else ""
    print(f"  Fakten gesamt: {tot_de} DE, {tot_als} ALS  |  unknown gesamt: {tot_unk}{unk_share}")


def export_csv(rows, path):
    keys = [c[0] for c in COLS] + EXPORT_ONLY
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"\nCSV geschrieben: {path}")


# The optional Excel export uses the same values and the same order as the CSV

def export_xlsx(rows, path):
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        print("  (openpyxl fehlt, XLSX uebersprungen)")
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Overlap-Matrix"
    export_headers = ["DE->ALS (yes/yes+no)", "ALS->DE (yes/yes+no)", "cov min",
                      "DE geteilt", "ALS geteilt", "Modell"]
    headers = [c[1] for c in COLS] + export_headers
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(bold=True)
    keys = [c[0] for c in COLS] + EXPORT_ONLY
    tints = {"niedrig": "F7C1C1", "mittel": "FAC775", "hoch": "C0DD97"}
    for r in rows:
        ws.append([r.get(k) for k in keys])
        cell = ws.cell(row=ws.max_row, column=[c[0] for c in COLS].index("klasse") + 1)
        col = tints.get(r["klasse"])
        if col:
            cell.fill = PatternFill("solid", fgColor=col)
    for col_cells in ws.columns:
        width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells) + 2
        ws.column_dimensions[col_cells[0].column_letter].width = min(width, 28)
    wb.save(path)
    print(f"XLSX geschrieben: {path}")


# The entry point loads all matching JSON files, sorts them and writes the outputs
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=VERIFIED_DIR, help="Ordner mit verified_*.json")
    ap.add_argument("--glob", default="verified_*.json", help="Dateimuster")
    ap.add_argument("--csv", default=DEFAULT_CSV_PATH)
    ap.add_argument("--xlsx", default=DEFAULT_XLSX_PATH, help="optionaler XLSX-Pfad")
    ap.add_argument("--sort", default="mittel", help="Sortierspalte (z. B. mittel, titel)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not paths:
        print("Keine Dateien gefunden.")
        return
    rows = []
    for p in paths:
        d = load_verified(p)
        if d:
            rows.append(row_from(d))

    rows.sort(key=lambda r: (r.get(args.sort) is None, r.get(args.sort)))

    print("Kennzahl: overlap = shared_yes / total_source_facts "
          "(papergetreu, unknown im Nenner)\n")
    print_table(rows)

    clusters = sorted(set(r["cluster"] for r in rows))
    if len(clusters) > 1:
        for cl in clusters:
            aggregate([r for r in rows if r["cluster"] == cl], f"Cluster {cl}")
    aggregate(rows, "GESAMT")

    export_csv(rows, args.csv)
    if args.xlsx:
        export_xlsx(rows, args.xlsx)


if __name__ == "__main__":
    main()
