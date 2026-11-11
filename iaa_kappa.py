"""
Computes Cohen's kappa between two annotators.

The input is the Excel file of the fact alignment annotation with two
blocks per sheet.

Fully automatic:
    - Sheets are detected via the naming scheme <Condition>_<Annotator>
      (e.g. Manual_Gold_A/Manual_Gold_B, Gemma_Gold_A/Gemma_Gold_B);
      every condition with two filled sheets is computed. The rest is
      reported and skipped.
    - The direction of a block is detected from the ``fact_id`` prefixes
      (DE_* -> de_to_als, ALS_* -> als_to_de). The block order in the
      sheet therefore does not matter (Manual: DE on the left; Gemma: ALS on the left).
    - Items are joined via fact_id, not via row order.

Robustness:
    - label normalization (upper/lower case, de_only/als_only -> only)
    - pairwise complete items; missing labels are reported
    - bootstrap confidence interval (percentile method, items resampled)
    - p_o, p_e, confusion matrix and marginal distributions in the report
    - binarized, paper-comparable variant (shared + partial -> yes, only -> no)
    - final overview across all conditions

Usage:
    python iaa_kappa.py --xlsx annotationen.xlsx
    python iaa_kappa.py --xlsx annotationen.xlsx --csv iaa_ergebnisse.csv
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import OUTPUT_DIR


DEFAULT_XLSX_PATH = OUTPUT_DIR / "annotationen_iaa.xlsx"
DEFAULT_CSV_PATH = OUTPUT_DIR / "iaa_ergebnisse.csv"

BOOTSTRAP_SAMPLES = 10_000
SEED = 42
MIN_LABELED_ITEMS = 10   # below this, a sheet is considered not (yet) annotated

BLOCK_COLUMNS = [("fact_id", "label"), ("fact_id.1", "label.1")]

PREFIX_TO_DIRECTION = {"DE": "de_to_als", "ALS": "als_to_de"}

LABEL_MAP = {
    "shared": "shared",
    "partial": "partial",
    "de_only": "only",
    "als_only": "only",
    "only": "only",
}

BINARY_MAP = {"shared": "yes", "partial": "yes", "only": "no"}


# Sheet detection
# ---------------------------------------------------------------------------
def discover_conditions(path: str) -> dict[str, dict[str, str]]:
    """Group sheets by ``<Condition>_<Annotator>``."""
    conditions: dict[str, dict[str, str]] = defaultdict(dict)

    for sheet in pd.ExcelFile(path).sheet_names:
        if "_" not in sheet:
            print(f"Hinweis: Sheet '{sheet}' passt nicht ins Schema "
                  f"<Bedingung>_<Annotator> und wird ignoriert.")
            continue
        condition, annotator = sheet.rsplit("_", 1)
        conditions[condition][annotator] = sheet

    return dict(conditions)



# Loading and normalization
# ---------------------------------------------------------------------------
def normalize_label(value) -> str | None:
    # Spelling variants are mapped to the three shared evaluation categories
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text or text == "label":
        return None
    if text not in LABEL_MAP:
        raise ValueError(f"Unbekanntes Label: {value!r}")
    return LABEL_MAP[text]


def load_blocks(path: str, sheet: str) -> dict[str, pd.DataFrame]:
    """
    Load both blocks of a sheet.

    The direction is detected from the ``fact_id`` prefixes ``DE_`` and ``ALS_``,
    not from the column position.
    """
    raw = pd.read_excel(path, sheet_name=sheet)
    blocks: dict[str, pd.DataFrame] = {}

    # The ID prefixes are more reliable than the position of the block in the sheet
    for id_col, label_col in BLOCK_COLUMNS:
        if id_col not in raw.columns or label_col not in raw.columns:
            raise ValueError(f"{sheet}: Spalten {id_col}/{label_col} fehlen.")

        frame = raw[[id_col, label_col]].copy()
        frame.columns = ["fact_id", "label"]
        frame = frame.dropna(subset=["fact_id"])
        frame["fact_id"] = frame["fact_id"].astype(str).str.strip()
        frame["label"] = frame["label"].map(normalize_label)

        prefixes = set(frame["fact_id"].str.split("_").str[0].str.upper())
        if len(prefixes) != 1 or prefixes.pop() not in PREFIX_TO_DIRECTION:
            raise ValueError(
                f"{sheet}/{id_col}: uneindeutige fact_id-Praefixe "
                f"{sorted(set(frame['fact_id'].str.split('_').str[0]))}")
        direction = PREFIX_TO_DIRECTION[
            frame["fact_id"].iloc[0].split("_")[0].upper()]

        duplicates = frame["fact_id"][frame["fact_id"].duplicated()].tolist()
        if duplicates:
            raise ValueError(f"Doppelte fact_ids in {sheet}/{direction}: "
                             f"{duplicates}")
        if direction in blocks:
            raise ValueError(f"{sheet}: beide Bloecke enthalten "
                             f"{direction}-IDs.")

        blocks[direction] = frame.set_index("fact_id")

    missing = set(PREFIX_TO_DIRECTION.values()) - set(blocks)
    if missing:
        raise ValueError(f"{sheet}: Richtung(en) {sorted(missing)} fehlen.")
    return blocks


def count_labeled(path: str, sheet: str) -> int:
    try:
        blocks = load_blocks(path, sheet)
    except ValueError:
        return 0
    return int(sum(frame["label"].notna().sum() for frame in blocks.values()))


def paired_labels(block_a: pd.DataFrame,
                  block_b: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Joining via fact_id prevents errors caused by differing row orders
    joined = block_a.join(block_b, how="outer", lsuffix="_a", rsuffix="_b")
    incomplete = joined[joined["label_a"].isna()
                        | joined["label_b"].isna()].index.tolist()
    complete = joined.dropna(subset=["label_a", "label_b"])
    return complete, incomplete


# Kappa
# ---------------------------------------------------------------------------
def cohen_kappa(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """
    Compute the unweighted Cohen's kappa:

        kappa = (p_o - p_e) / (1 - p_e)
        p_o   = (1/N) * number of agreeing items
        p_e   = sum_k  p_A(k) * p_B(k)
    """
    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))

    # Observed and chance-expected agreement are determined separately
    p_o = float(np.mean(labels_a == labels_b))
    freq_a = Counter(labels_a)
    freq_b = Counter(labels_b)
    p_e = sum((freq_a[k] / n) * (freq_b[k] / n) for k in categories)

    if p_e >= 1.0:
        kappa = 1.0 if p_o == 1.0 else float("nan")
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    return {"kappa": kappa, "p_o": p_o, "p_e": p_e, "n": n,
            "categories": categories}


def bootstrap_ci(labels_a: np.ndarray, labels_b: np.ndarray,
                 n_samples: int = BOOTSTRAP_SAMPLES,
                 seed: int = SEED) -> tuple[float, float, int]:
    # A fixed seed makes the percentile interval identical on every run
    rng = np.random.default_rng(seed)
    n = len(labels_a)
    values = []
    skipped = 0

    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        result = cohen_kappa(labels_a[idx], labels_b[idx])
        if np.isnan(result["kappa"]):
            skipped += 1
            continue
        values.append(result["kappa"])

    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper), skipped


def confusion(labels_a: np.ndarray, labels_b: np.ndarray,
              categories: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=categories, columns=categories, dtype=int)
    for x, y in zip(labels_a, labels_b):
        matrix.loc[x, y] += 1
    matrix.index.name = "A \\ B"
    return matrix



# Report
# ---------------------------------------------------------------------------
def article_of(fact_id: str) -> str:
    parts = fact_id.split("_")
    return "_".join(parts[1:-2]) if len(parts) >= 4 else fact_id


def report(condition: str, name: str, frame: pd.DataFrame,
           binary: bool = False) -> dict:
    labels_a = frame["label_a"].to_numpy()
    labels_b = frame["label_b"].to_numpy()

    if binary:
        labels_a = np.array([BINARY_MAP[x] for x in labels_a])
        labels_b = np.array([BINARY_MAP[x] for x in labels_b])

    stats = cohen_kappa(labels_a, labels_b)
    lower, upper, skipped = bootstrap_ci(labels_a, labels_b)

    print(f"\n=== {name} ===")
    print(f"  N = {stats['n']}   p_o = {stats['p_o']:.4f}   "
          f"p_e = {stats['p_e']:.4f}")
    print(f"  Cohen's kappa = {stats['kappa']:.4f}   "
          f"95%-Bootstrap-CI [{lower:.4f}, {upper:.4f}]"
          + (f"   ({skipped} degenerierte Resamples uebersprungen)"
             if skipped else ""))
    print("  Konfusionsmatrix (Zeilen: Annotator A, Spalten: Annotator B):")
    print(confusion(labels_a, labels_b, stats["categories"])
          .to_string().replace("\n", "\n    "))

    marginals = pd.DataFrame({
        "A": pd.Series(Counter(labels_a)),
        "B": pd.Series(Counter(labels_b)),
    }).fillna(0).astype(int)
    print("  Randverteilungen:")
    print(marginals.to_string().replace("\n", "\n    "))

    return {"condition": condition, "name": name,
            **{k: stats[k] for k in ("kappa", "p_o", "p_e", "n")},
            "ci_low": lower, "ci_high": upper}


def per_article_agreement(frame: pd.DataFrame) -> None:
    articles = frame.index.to_series().map(article_of)
    print("  Rohuebereinstimmung je Artikel (kappa dafuer zu instabil):")
    for article, group in frame.groupby(articles):
        p_o = float(np.mean(group["label_a"] == group["label_b"]))
        print(f"    {article:<16} n={len(group):>3}   p_o={p_o:.3f}")


def analyze_condition(path: str, condition: str, sheet_a: str,
                      sheet_b: str) -> list[dict]:
    # Each condition is evaluated per direction, binary and with three categories
    print(f"\n{'#' * 70}")
    print(f"# Bedingung: {condition}   (A: {sheet_a}, B: {sheet_b})")
    print(f"{'#' * 70}")

    blocks_a = load_blocks(path, sheet_a)
    blocks_b = load_blocks(path, sheet_b)

    results = []
    frames = {}

    for direction in ("de_to_als", "als_to_de"):
        frame, incomplete = paired_labels(blocks_a[direction],
                                          blocks_b[direction])
        frames[direction] = frame
        if incomplete:
            print(f"\nWARN {direction}: {len(incomplete)} Item(s) ohne "
                  f"beidseitiges Label entfernt: {incomplete}")

        results.append(report(condition,
                              f"{condition} · {direction} · dreikategorial "
                              f"(shared/partial/only)", frame))
        per_article_agreement(frame)
        results.append(report(condition,
                              f"{condition} · {direction} · binarisiert "
                              f"(yes = shared+partial, no = only)",
                              frame, binary=True))

    pooled = pd.concat(list(frames.values()))
    results.append(report(condition,
                          f"{condition} · gepoolt · dreikategorial", pooled))
    results.append(report(condition,
                          f"{condition} · gepoolt · binarisiert",
                          pooled, binary=True))
    return results


def print_overview(all_results: list[dict]) -> None:
    """Compact final overview: one row per condition and evaluation."""
    if not all_results:
        return

    print(f"\n{'=' * 78}")
    print("ÜBERSICHT (Cohen's kappa, 95%-Bootstrap-CI)")
    print(f"{'=' * 78}")
    header = (f"{'Bedingung':<13} {'Auswertung':<28} {'N':>4} "
              f"{'p_o':>6} {'kappa':>7}  {'95%-CI':<16}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        short = (r["name"].replace(f"{r['condition']} · ", "")
                 .replace(" (shared/partial/only)", "")
                 .replace(" (yes = shared+partial, no = only)", ""))
        print(f"{r['condition']:<13} {short:<28} {r['n']:>4} "
              f"{r['p_o']:>6.3f} {r['kappa']:>7.3f}  "
              f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


# The entry point automatically detects all valid annotator pairs and creates the overall report
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX_PATH)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH,
                        help="optionaler Pfad fuer Ergebnis-CSV")
    args = parser.parse_args()

    conditions = discover_conditions(args.xlsx)
    if not conditions:
        raise SystemExit("Keine Sheets im Schema <Bedingung>_<Annotator> gefunden.")

    all_results = []

    for condition, annotators in sorted(conditions.items()):
        if len(annotators) < 2:
            print(f"\nBedingung '{condition}': nur ein Annotator-Sheet "
                  f"({list(annotators.values())}); uebersprungen.")
            continue
        if len(annotators) > 2:
            print(f"\nBedingung '{condition}': mehr als zwei Annotatoren "
                  f"({sorted(annotators)}); Cohen's Kappa ist paarweise, "
                  f"bitte Sheets aufteilen. Uebersprungen.")
            continue

        (name_a, sheet_a), (name_b, sheet_b) = sorted(annotators.items())

        counts = {sheet: count_labeled(args.xlsx, sheet)
                  for sheet in (sheet_a, sheet_b)}
        empty = [sheet for sheet, count in counts.items()
                 if count < MIN_LABELED_ITEMS]
        if empty:
            print(f"\nBedingung '{condition}': noch nicht (vollstaendig) "
                  f"annotiert, uebersprungen. Gelabelte Items: {counts}")
            continue

        all_results.extend(
            analyze_condition(args.xlsx, condition, sheet_a, sheet_b))

    print_overview(all_results)

    if args.csv and all_results:
        pd.DataFrame(all_results).to_csv(args.csv, index=False)
        print(f"\nErgebnisse geschrieben: {args.csv}")


if __name__ == "__main__":
    main()
