"""Runs all evaluation steps using the stored model data.

The script runs, one after another, both overlap aggregations, K1/K2, K3 as
well as the result tables and the overlap figure. No API key is required
for this.

LLM calls are not part of this offline evaluation and are started separately
via ``decompose_facts.py`` and ``run_xfactmatch.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def run_step(title: str, *arguments: str) -> None:
    """Run one evaluation step and abort on error."""
    print(f"\n{'=' * 72}", flush=True)
    print(title, flush=True)
    print(f"{'=' * 72}", flush=True)
    # sys.executable ensures that all sub-steps use the same Python environment
    subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT_DIR,
        check=True,
    )


def main() -> None:
    """Generate aggregates, reliability scores, tables and figures."""
    # The order reflects the data flow from the aggregates to the figures
    run_step(
        "One-shot-Overlap aggregieren",
        "aggregate_xfactmatch.py",
        "--dir",
        "outputs/verified_oneshot",
        "--csv",
        "outputs/xfactmatch_aggregate_oneshot.csv",
        "--xlsx",
        "outputs/xfactmatch_aggregate_oneshot.xlsx",
    )
    run_step(
        "Few-shot-Overlap aggregieren",
        "aggregate_xfactmatch.py",
        "--dir",
        "outputs/verified",
        "--csv",
        "outputs/xfactmatch_aggregate_fewshot.csv",
        "--xlsx",
        "outputs/xfactmatch_aggregate_fewshot.xlsx",
    )
    run_step(
        "K1 und K2 berechnen",
        "iaa_kappa.py",
        "--xlsx",
        "outputs/annotationen_iaa.xlsx",
        "--csv",
        "outputs/iaa_ergebnisse.csv",
    )
    run_step(
        "K3 berechnen",
        "kappa3_system_vs_gold.py",
        "--xlsx",
        "outputs/annotationen_iaa.xlsx",
        "--json",
        "outputs/verified_oneshot/verified_gold__*.json",
        "--csv",
        "outputs/kappa3_ergebnisse.csv",
    )
    run_step("Overlap-Abbildung erzeugen", "visualize_final_overlap_matrix.py")
    run_step("Ergebnistabellen erzeugen", "visualize_final_result_tables.py")
    print("\nDie vollständige Offline-Auswertung ist abgeschlossen.")


if __name__ == "__main__":
    main()
