"""
Computes statistics on the fact decomposition of the manually processed articles.

The fact counts of the manual and the Gemma decomposition are compared per
article and language. In addition, the granularity ratio (Gemma/manual) and
the fact density of the manual decomposition are reported.

Inputs:
    --manual     Excel file with sheet 'IAA_Facts_Clean'
                 (columns: page_title, lang, paragraph_id, paragraph_text,
                  fact_id, fact_text)
    --aggregate  overlap matrix from aggregate_xfactmatch.py as XLSX
                 (columns: Artikel, #DE, #ALS) for the Gemma fact counts

Metrics per article and language:
    manual       number of manual facts
    gemma        number of Gemma facts (one-shot main run)
    ratio        Gemma / manual  (> 1: Gemma decomposes more finely,
                                  < 1: Gemma decomposes more coarsely)
    density      manual facts per 100 words of the decomposed paragraphs

Set P/R/F1 and gamma agreement for matching manual and Gemma facts via text
similarity are not part of this computation, since they would require a
separate similarity threshold.

Usage:
    python decomposition_stats.py --manual manual_paragraph_fact_annotation.xlsx \
        --aggregate xfactmatch_aggregate_oneshot.xlsx --csv decomposition_stats.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_config import OUTPUT_DIR, ROOT_DIR


DEFAULT_MANUAL_PATH = ROOT_DIR / "manual_paragraph_fact_annotation_max.xlsx"
DEFAULT_AGGREGATE_PATH = OUTPUT_DIR / "xfactmatch_aggregate_oneshot.xlsx"


# The manual workbook is reduced to valid fact rows and the required columns
def load_manual(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="IAA_Facts_Clean")
    missing = df["fact_text"].isna() | df["fact_id"].isna()
    if missing.any():
        print(f"Hinweis: {int(missing.sum())} Zeile(n) ohne fact_id/"
              f"fact_text entfernt.")
        df = df[~missing]
    return df


def word_counts(df: pd.DataFrame) -> pd.Series:
    """Count the words of the decomposed paragraphs per article and language."""
    # Paragraph texts are counted only once, even if several facts belong to a paragraph
    paragraphs = df.drop_duplicates(
        subset=["page_title", "lang", "paragraph_id"])
    words = paragraphs["paragraph_text"].fillna("").str.split().str.len()
    return words.groupby(
        [paragraphs["page_title"], paragraphs["lang"]]).sum()


# The main routine combines the manual fact counts with the Gemma counts from the aggregate
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL_PATH)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE_PATH)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    manual = load_manual(args.manual)
    counts = manual.groupby(["page_title", "lang"])["fact_id"].count()
    words = word_counts(manual)

    agg = pd.read_excel(args.aggregate).set_index("Artikel")

    rows = []
    for artikel in sorted(manual["page_title"].unique()):
        if artikel not in agg.index:
            print(f"WARN: '{artikel}' nicht in der Aggregat-Matrix; "
                  f"Gemma-Zahlen fehlen.")
        row = {"artikel": artikel}
        for lang, col in (("DE", "#DE"), ("ALS", "#ALS")):
            man = int(counts.get((artikel, lang), 0))
            gem = int(agg.loc[artikel, col]) if artikel in agg.index else None
            wc = int(words.get((artikel, lang), 0))
            row[f"man_{lang}"] = man
            row[f"gemma_{lang}"] = gem
            row[f"ratio_{lang}"] = round(gem / man, 2) if man and gem else None
            row[f"dichte_{lang}"] = round(man / wc * 100, 1) if wc else None
        rows.append(row)

    frame = pd.DataFrame(rows)

    header = (f"{'Artikel':<38} {'man DE':>7} {'Gemma DE':>9} {'V DE':>5} "
              f"{'man ALS':>8} {'Gemma ALS':>10} {'V ALS':>6} "
              f"{'D DE':>5} {'D ALS':>6}")
    print("\nDekomposition: manuell vs. Gemma (One-Shot)   "
          "V = Gemma/manuell, D = man. Fakten je 100 Woerter")
    print(header)
    print("-" * len(header))
    for _, r in frame.iterrows():
        print(f"{r['artikel']:<38} {r['man_DE']:>7} {r['gemma_DE']:>9} "
              f"{r['ratio_DE']:>5} {r['man_ALS']:>8} {r['gemma_ALS']:>10} "
              f"{r['ratio_ALS']:>6} {r['dichte_DE']:>5} {r['dichte_ALS']:>6}")

    tot_man_de, tot_man_als = frame["man_DE"].sum(), frame["man_ALS"].sum()
    tot_gem_de = frame["gemma_DE"].sum()
    tot_gem_als = frame["gemma_ALS"].sum()
    print("-" * len(header))
    print(f"{'GESAMT':<38} {tot_man_de:>7} {tot_gem_de:>9} "
          f"{tot_gem_de / tot_man_de:>5.2f} {tot_man_als:>8} "
          f"{tot_gem_als:>10} {tot_gem_als / tot_man_als:>6.2f}")
    print(f"\nVerhaeltnis (Median ueber Artikel): "
          f"DE {frame['ratio_DE'].median():.2f}, "
          f"ALS {frame['ratio_ALS'].median():.2f}")

    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"CSV geschrieben: {args.csv}")


if __name__ == "__main__":
    main()
