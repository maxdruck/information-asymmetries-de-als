"""
Compares the X-FACTMATCH judgments with the human gold standard.

The evaluation uses the pipeline's ``verified_*.json`` files and the
``Gemma_Gold_*`` sheets of the annotation file. Reported are Cohen's kappa,
precision, recall, F1 and coverage.

Mapping pipeline <-> Excel:
    Excel ID = {LANG}_{Artikel}_P{source_paragraph}_F{running no. within the
    paragraph, 1-based, sorted by source_fact_index}.
    The article name is determined per JSON by text matching against the Excel
    facts; each mapping is additionally checked by text comparison.

Gold standard:
    - two filled ``Gemma_Gold_*`` sheets: agreement gold with the items on
      which both annotators agree; diverging judgments are listed and
      excluded)
    - only one filled sheet: this one serves as a provisional gold standard

Metrics (binary: shared+partial -> yes, only -> no):
    - Cohen's kappa with bootstrap 95% CI
    - precision/recall/F1 for ``yes`` and ``no``, macro F1 with bootstrap CI
    - coverage: share of gold items for which the pipeline has a ``yes`` or
      ``no`` judgment; ``unknown`` is reported and not counted as ``no``

Usage:
    python kappa3_system_vs_gold.py --xlsx annotationen.xlsx \
        --json verified_gold__*.json [--csv kappa3.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import OUTPUT_DIR


DEFAULT_XLSX_PATH = OUTPUT_DIR / "annotationen_iaa.xlsx"
DEFAULT_JSON_GLOB = str(
    OUTPUT_DIR / "verified_oneshot" / "verified_gold__*.json"
)
DEFAULT_CSV_PATH = OUTPUT_DIR / "kappa3_ergebnisse.csv"

BOOTSTRAP_SAMPLES = 10_000
SEED = 42
MIN_LABELED_ITEMS = 10

BLOCK_COLUMNS = [("fact_id", "label", "fact_text"),
                 ("fact_id.1", "label.1", "fact_text.1")]

LABEL_MAP = {"shared": "shared", "partial": "partial",
             "de_only": "only", "als_only": "only", "only": "only"}
BINARY_MAP = {"shared": "yes", "partial": "yes", "only": "no"}

DIRECTION_LANG = {"de_to_als": "DE", "als_to_de": "ALS"}


def norm_text(text) -> str:
    return " ".join(str(text).split())


def normalize_label(value) -> str | None:
    # Detailed human categories are mapped to yes/no for K3
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text or text == "label":
        return None
    if text not in LABEL_MAP:
        raise ValueError(f"Unbekanntes Label: {value!r}")
    return LABEL_MAP[text]


# Excel: build the gold standard
# ---------------------------------------------------------------------------
def load_annotator_sheet(path: str, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet)
    parts = []
    for id_col, label_col, text_col in BLOCK_COLUMNS:
        frame = raw[[id_col, label_col, text_col]].copy()
        frame.columns = ["fact_id", "label", "fact_text"]
        frame = frame.dropna(subset=["fact_id"])
        frame["fact_id"] = frame["fact_id"].astype(str).str.strip()
        frame["label"] = frame["label"].map(normalize_label)
        parts.append(frame)
    merged = pd.concat(parts)
    duplicates = merged["fact_id"][merged["fact_id"].duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Doppelte fact_ids in {sheet}: {duplicates}")
    return merged.set_index("fact_id")


def discover_gold_sheets(path: str, condition: str) -> dict[str, str]:
    # Only sheets of the selected condition are eligible as a gold source
    sheets = {}
    for sheet in pd.ExcelFile(path).sheet_names:
        if sheet.startswith(condition + "_"):
            sheets[sheet.rsplit("_", 1)[1]] = sheet
    return sheets


def build_gold(path: str, condition: str) -> tuple[pd.DataFrame, dict]:
    # With two annotations, only agreeing judgments are kept
    sheets = discover_gold_sheets(path, condition)
    if not sheets:
        raise SystemExit(f"Keine Sheets fuer Bedingung '{condition}' gefunden.")

    loaded = {}
    for annotator, sheet in sorted(sheets.items()):
        frame = load_annotator_sheet(path, sheet)
        if frame["label"].notna().sum() >= MIN_LABELED_ITEMS:
            loaded[annotator] = frame
        else:
            print(f"Hinweis: {sheet} ist (noch) nicht annotiert und wird "
                  f"nicht fuer das Gold verwendet.")

    if not loaded:
        raise SystemExit("Kein befuelltes Gold-Sheet gefunden.")

    info = {"annotators": sorted(loaded), "mode": None, "disagreements": []}

    if len(loaded) == 1:
        annotator, frame = next(iter(loaded.items()))
        print(f"\nWARN: Nur ein Annotator ({annotator}) befuellt; "
              f"vorlaeufiges Gold ohne Adjudikation.")
        info["mode"] = f"single ({annotator})"
        gold = frame.dropna(subset=["label"]).copy()
    else:
        annotators = sorted(loaded)
        first, second = loaded[annotators[0]], loaded[annotators[1]]
        joined = first.join(second, how="inner",
                            lsuffix="_a", rsuffix="_b")
        joined = joined.dropna(subset=["label_a", "label_b"])
        agree = joined["label_a"] == joined["label_b"]
        info["mode"] = f"agreement ({annotators[0]}, {annotators[1]})"
        info["disagreements"] = joined.index[~agree].tolist()
        if info["disagreements"]:
            print(f"\nAgreement-Gold: {int((~agree).sum())} Disagreement-"
                  f"Item(s) ausgeschlossen:")
            for fact_id in info["disagreements"]:
                print(f"    {fact_id}: "
                      f"{joined.loc[fact_id, 'label_a']} vs. "
                      f"{joined.loc[fact_id, 'label_b']}")
        gold = joined[agree][["label_a", "fact_text_a"]].copy()
        gold.columns = ["label", "fact_text"]

    gold["binary"] = gold["label"].map(BINARY_MAP)
    return gold, info



# JSON: pipeline judgments with reconstructed Excel IDs
# ---------------------------------------------------------------------------
def article_tokens_from_gold(gold: pd.DataFrame) -> set[str]:
    tokens = set()
    for fact_id in gold.index:
        match = re.match(r"(?:DE|ALS)_(.+)_P\d+_F\d+$", fact_id)
        if match:
            tokens.add(match.group(1))
    return tokens


def load_pipeline_judgements(json_paths: list[str],
                             gold: pd.DataFrame) -> pd.DataFrame:
    # Pipeline facts receive reconstructed Excel IDs and are additionally checked by text
    tokens = article_tokens_from_gold(gold)
    gold_texts = {fid: norm_text(t) for fid, t
                  in gold["fact_text"].items() if pd.notna(t)}

    rows = []
    for path in json_paths:
        data = json.loads(open(path, encoding="utf-8").read())
        for direction, lang in DIRECTION_LANG.items():
            facts = sorted(data[direction],
                           key=lambda r: int(r["source_fact_index"]))
            counter: dict[int, int] = defaultdict(int)
            candidates = []
            for record in facts:
                paragraph = int(record["source_paragraph"])
                counter[paragraph] += 1
                candidates.append((paragraph, counter[paragraph], record))

            # Determine the article token via a text sample
            token = None
            for probe_token in tokens:
                hits = 0
                for paragraph, number, record in candidates[:10]:
                    fid = f"{lang}_{probe_token}_P{paragraph}_F{number}"
                    if gold_texts.get(fid) == norm_text(record["source_fact"]):
                        hits += 1
                if hits >= 3:
                    token = probe_token
                    break
            if token is None:
                print(f"Hinweis: {path}/{direction}: kein Artikel-Match im "
                      f"Gold (Paar vermutlich nicht annotiert); "
                      f"uebersprungen.")
                continue

            for paragraph, number, record in candidates:
                fid = f"{lang}_{token}_P{paragraph}_F{number}"
                expected = gold_texts.get(fid)
                if expected is not None and \
                        expected != norm_text(record["source_fact"]):
                    raise ValueError(
                        f"Textmismatch bei {fid} in {path}: ID-Schema "
                        f"passt nicht. Excel: {expected[:60]!r} vs. "
                        f"JSON: {norm_text(record['source_fact'])[:60]!r}")
                verification = record.get("verification", {})
                rows.append({
                    "fact_id": fid,
                    "direction": direction,
                    "article": token,
                    "pipeline_label":
                        verification.get("intersection_label", "unknown"),
                    "status": verification.get("verification_status", ""),
                })

    frame = pd.DataFrame(rows).set_index("fact_id")
    duplicates = frame.index[frame.index.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Doppelte rekonstruierte IDs: {duplicates[:5]}")
    return frame



# Metrics
# ---------------------------------------------------------------------------
def cohen_kappa(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
   # K3 computes the unweighted Cohen’s kappa between the pipeline and the gold standard
    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))
    p_o = float(np.mean(labels_a == labels_b))
    freq_a, freq_b = Counter(labels_a), Counter(labels_b)
    p_e = sum((freq_a[k] / n) * (freq_b[k] / n) for k in categories)
    if p_e >= 1.0:
        kappa = 1.0 if p_o == 1.0 else float("nan")
    else:
        kappa = (p_o - p_e) / (1 - p_e)
    return {"kappa": kappa, "p_o": p_o, "p_e": p_e, "n": n,
            "categories": categories}


def prf(gold: np.ndarray, pred: np.ndarray, positive: str) -> dict:
    tp = int(np.sum((pred == positive) & (gold == positive)))
    fp = int(np.sum((pred == positive) & (gold != positive)))
    fn = int(np.sum((pred != positive) & (gold == positive)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(gold: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([prf(gold, pred, c)["f1"]
                          for c in ("yes", "no")]))


def bootstrap(gold: np.ndarray, pred: np.ndarray,
              statistic, n_samples: int = BOOTSTRAP_SAMPLES,
              seed: int = SEED) -> tuple[float, float]:
    # Resampling provides uncertainty intervals for kappa and macro F1
    rng = np.random.default_rng(seed)
    n = len(gold)
    values = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        value = statistic(gold[idx], pred[idx])
        if not np.isnan(value):
            values.append(value)
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)



# Report
# ---------------------------------------------------------------------------
def evaluate(name: str, frame: pd.DataFrame) -> dict | None:
    # One evaluation bundles coverage, confusion matrix and all computed metrics
    judged = frame[frame["pipeline_label"].isin(["yes", "no"])]
    coverage = len(judged) / len(frame) if len(frame) else float("nan")
    unknown_status = Counter(
        frame.loc[~frame["pipeline_label"].isin(["yes", "no"]), "status"])

    print(f"\n=== {name} ===")
    print(f"  Gold-Items: {len(frame)}   davon Pipeline-Urteil yes/no: "
          f"{len(judged)}   Coverage = {coverage:.3f}")
    if unknown_status:
        print(f"  unknown nach Status: {dict(unknown_status)}")
    if len(judged) < 5:
        print("  Zu wenige Items fuer Metriken.")
        return None

    gold = judged["binary"].to_numpy()
    pred = judged["pipeline_label"].to_numpy()

    stats = cohen_kappa(gold, pred)
    kappa_ci = bootstrap(gold, pred,
                         lambda g, p: cohen_kappa(g, p)["kappa"])
    macro = macro_f1(gold, pred)
    macro_ci = bootstrap(gold, pred, macro_f1)
    yes_metrics = prf(gold, pred, "yes")
    no_metrics = prf(gold, pred, "no")

    print(f"  p_o = {stats['p_o']:.4f}   kappa = {stats['kappa']:.4f}   "
          f"95%-CI [{kappa_ci[0]:.4f}, {kappa_ci[1]:.4f}]")
    print(f"  Macro-F1 = {macro:.4f}   95%-CI "
          f"[{macro_ci[0]:.4f}, {macro_ci[1]:.4f}]")
    print(f"  yes: P={yes_metrics['precision']:.3f} "
          f"R={yes_metrics['recall']:.3f} F1={yes_metrics['f1']:.3f}   |   "
          f"no: P={no_metrics['precision']:.3f} "
          f"R={no_metrics['recall']:.3f} F1={no_metrics['f1']:.3f}")

    matrix = pd.crosstab(pd.Series(gold, name="Gold"),
                         pd.Series(pred, name="Pipeline"))
    print("  Konfusionsmatrix (Zeilen: Gold, Spalten: Pipeline):")
    print(matrix.to_string().replace("\n", "\n    "))

    return {"name": name, "n_gold": len(frame), "n_judged": len(judged),
            "coverage": coverage, "p_o": stats["p_o"],
            "kappa": stats["kappa"], "kappa_ci_low": kappa_ci[0],
            "kappa_ci_high": kappa_ci[1], "macro_f1": macro,
            "macro_f1_ci_low": macro_ci[0], "macro_f1_ci_high": macro_ci[1],
            "yes_precision": yes_metrics["precision"],
            "yes_recall": yes_metrics["recall"], "yes_f1": yes_metrics["f1"],
            "no_f1": no_metrics["f1"]}


def print_overview(results: list[dict], gold_mode: str) -> None:
    print(f"\n{'=' * 84}")
    print(f"ÜBERSICHT κ3 · System vs. Gold ({gold_mode}) · binaer")
    print(f"{'=' * 84}")
    header = (f"{'Auswertung':<28} {'N':>4} {'Cov':>5} {'p_o':>6} "
              f"{'kappa':>6}  {'95%-CI':<15} {'MacroF1':>7}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<28} {r['n_judged']:>4} {r['coverage']:>5.2f} "
              f"{r['p_o']:>6.3f} {r['kappa']:>6.3f}  "
              f"[{r['kappa_ci_low']:.3f}, {r['kappa_ci_high']:.3f}] "
              f"{r['macro_f1']:>7.3f}")


# The entry point first builds the agreement gold standard and then maps the pipeline judgments to it
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX_PATH,
                        help="Annotations-Excel mit Gemma_Gold_*-Sheets")
    parser.add_argument("--json", nargs="+", default=[DEFAULT_JSON_GLOB],
                        help="verified_*.json (Globs erlaubt)")
    parser.add_argument("--condition", default="Gemma_Gold")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    args = parser.parse_args()

    json_paths = sorted({p for pattern in args.json
                         for p in glob.glob(pattern)})
    if not json_paths:
        raise SystemExit("Keine verified-JSONs gefunden.")
    print("Pipeline-Dateien:", ", ".join(json_paths))

    gold, info = build_gold(args.xlsx, args.condition)
    print(f"Gold: {len(gold)} Items ({info['mode']})")

    pipeline = load_pipeline_judgements(json_paths, gold)
    joined = gold.join(pipeline, how="inner")
    missing = gold.index.difference(pipeline.index).tolist()
    if missing:
        print(f"WARN: {len(missing)} Gold-Item(s) ohne Pipeline-Urteil "
              f"(nicht in den JSONs): {missing[:5]}")

    results = []
    for direction, group in joined.groupby("direction"):
        result = evaluate(f"{direction}", group)
        if result:
            results.append(result)
    for article, group in joined.groupby("article"):
        result = evaluate(f"Artikel {article}", group)
        if result:
            results.append(result)
    result = evaluate("gepoolt", joined)
    if result:
        results.append(result)

    print_overview(results, info["mode"])

    if args.csv and results:
        pd.DataFrame(results).to_csv(args.csv, index=False)
        print(f"\nErgebnisse geschrieben: {args.csv}")


if __name__ == "__main__":
    main()
