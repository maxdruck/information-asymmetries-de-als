"""
Generates four tables from the stored result data.

The values are read from the configured Excel and CSV files.

Usage:
    python visualize_final_result_tables.py

Output folder:
    outputs/visualizations_final_tables/

For each table, PNG, SVG and the displayed values as CSV are written.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
MATPLOTLIB_CACHE_DIR = ROOT_DIR / "tmp" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

# Matplotlib's DejaVu Sans is used for a consistent appearance.
_original_check_output = subprocess.check_output


# The macOS font query receives an empty list; other calls remain unchanged.
def _check_output_without_macos_font_profiler(command, *args, **kwargs):
    if (
        isinstance(command, (list, tuple))
        and list(command) == ["system_profiler", "-xml", "SPFontsDataType"]
    ):
        return plistlib.dumps([{"_items": []}])
    return _original_check_output(command, *args, **kwargs)


subprocess.check_output = _check_output_without_macos_font_profiler
try:
    import matplotlib.pyplot as plt
finally:
    subprocess.check_output = _original_check_output

import pandas as pd

from decomposition_stats import load_manual, word_counts


DEFAULT_MANUAL = ROOT_DIR / "manual_paragraph_fact_annotation_max.xlsx"
DEFAULT_OVERLAP = ROOT_DIR / "outputs" / "xfactmatch_aggregate_oneshot.csv"
DEFAULT_IAA = ROOT_DIR / "outputs" / "iaa_ergebnisse.csv"
DEFAULT_K3 = ROOT_DIR / "outputs" / "kappa3_ergebnisse.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "visualizations_final_tables"

BLACK = "#000000"
WHITE = "#FFFFFF"


def decimal(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def integer(value) -> str:
    return str(int(value))


def percentage(value: float, digits: int = 1) -> str:
    return decimal(100 * float(value), digits) + " %"


def interval(low: float, high: float) -> str:
    return f"[{decimal(low, 3)}; {decimal(high, 3)}]"


def display_title(title: str) -> str:
    replacements = {
        "Rheinbrücke Schaffhausen–Feuerthalen":
            "Rheinbrücke Schaffhausen-\nFeuerthalen",
        "Rheinbrücke Schaffhausen-Feuerthalen":
            "Rheinbrücke Schaffhausen-\nFeuerthalen",
    }
    return replacements.get(str(title), str(title))


def save_table_data(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    frame.to_csv(output_dir / f"{stem}.csv", index=False, encoding="utf-8")


def render_table(
    frame: pd.DataFrame,
    output_dir: Path,
    stem: str,
    figsize: tuple[float, float],
    column_widths: list[float],
    font_size: float,
    row_scale: float = 1.55,
    footnote: str | None = None,
) -> None:
    """Render a table."""
    
    fig, ax = plt.subplots(figsize=figsize, facecolor=WHITE)
    ax.set_facecolor(WHITE)
    ax.axis("off")

    bottom = 0.08 if footnote else 0.02
    table = ax.table(
        cellText=frame.values.tolist(),
        colLabels=list(frame.columns),
        cellLoc="right",
        colLoc="center",
        colWidths=column_widths,
        bbox=[0.01, bottom, 0.98, 0.97 - bottom],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, row_scale)

    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor(WHITE)
        cell.set_edgecolor(BLACK)
        cell.set_linewidth(0.8)
        cell.PAD = 0.035
        text = cell.get_text()
        text.set_color(BLACK)
        text.set_fontfamily("DejaVu Sans")
        if row == 0:
            text.set_fontweight("bold")
            text.set_ha("left" if column == 0 else "center")
        else:
            text.set_ha("left" if column == 0 else "right")
        text.set_va("center")

    if footnote:
        fig.text(
            0.015,
            0.018,
            footnote,
            ha="left",
            va="bottom",
            fontsize=max(font_size - 1.5, 7),
            color=BLACK,
            family="DejaVu Sans",
        )

    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def build_decomposition_table(manual_path: Path, overlap_path: Path) -> pd.DataFrame:
    # Manual and automatic fact counts are compared per article.
    manual = load_manual(str(manual_path))
    counts = manual.groupby(["page_title", "lang"])["fact_id"].count()
    words = word_counts(manual)
    overlap = pd.read_csv(overlap_path).set_index("titel")

    rows: list[dict[str, str]] = []
    numeric_rows = []
    for title in sorted(manual["page_title"].unique()):
        man_de = int(counts.get((title, "DE"), 0))
        man_als = int(counts.get((title, "ALS"), 0))
        gemma_de = int(overlap.loc[title, "de_fakten"])
        gemma_als = int(overlap.loc[title, "als_fakten"])
        words_de = int(words.get((title, "DE"), 0))
        words_als = int(words.get((title, "ALS"), 0))

        numeric = {
            "man_de": man_de,
            "gemma_de": gemma_de,
            "ratio_de": gemma_de / man_de,
            "man_als": man_als,
            "gemma_als": gemma_als,
            "ratio_als": gemma_als / man_als,
            "density_de": man_de / words_de * 100,
            "density_als": man_als / words_als * 100,
        }
        numeric_rows.append(numeric)
        rows.append({
            "Artikel": display_title(title),
            "man. DE": integer(man_de),
            "Gemma DE": integer(gemma_de),
            "V DE": decimal(numeric["ratio_de"], 2),
            "man. ALS": integer(man_als),
            "Gemma ALS": integer(gemma_als),
            "V ALS": decimal(numeric["ratio_als"], 2),
            "Dichte DE": decimal(numeric["density_de"], 1),
            "Dichte ALS": decimal(numeric["density_als"], 1),
        })

    total_man_de = sum(row["man_de"] for row in numeric_rows)
    total_gemma_de = sum(row["gemma_de"] for row in numeric_rows)
    total_man_als = sum(row["man_als"] for row in numeric_rows)
    total_gemma_als = sum(row["gemma_als"] for row in numeric_rows)
    assert (total_man_de, total_gemma_de, total_man_als, total_gemma_als) == (
        1661, 1490, 1486, 1220
    )

    rows.append({
        "Artikel": "GESAMT",
        "man. DE": integer(total_man_de),
        "Gemma DE": integer(total_gemma_de),
        "V DE": decimal(total_gemma_de / total_man_de, 2),
        "man. ALS": integer(total_man_als),
        "Gemma ALS": integer(total_gemma_als),
        "V ALS": decimal(total_gemma_als / total_man_als, 2),
        "Dichte DE": "",
        "Dichte ALS": "",
    })
    return pd.DataFrame(rows)


def build_overlap_table(path: Path) -> pd.DataFrame:
    # The overlap data are reduced to the required columns and formats
    frame = pd.read_csv(path).copy()
    frame = frame.sort_values("mittel", ascending=True).reset_index(drop=True)
    if int(frame["unknown"].sum()) != 0:
        raise ValueError("Das gewählte Overlap-Aggregat enthält noch unknown-Fälle.")
    if len(frame) != 10:
        raise ValueError(f"Erwartet wurden 10 Artikelpaare, gefunden: {len(frame)}")

    return pd.DataFrame({
        "Artikel": frame["titel"].map(display_title),
        "#DE": frame["de_fakten"].map(integer),
        "#ALS": frame["als_fakten"].map(integer),
        "DE → ALS": frame["de_to_als"].map(lambda value: decimal(value, 3)),
        "ALS → DE": frame["als_to_de"].map(lambda value: decimal(value, 3)),
        "Mittel": frame["mittel"].map(lambda value: decimal(value, 3)),
        "Klasse": frame["klasse"].astype(str),
        "unknown": frame["unknown"].map(integer),
        "u-Anteil": frame["unknown_anteil"].map(lambda value: percentage(value, 1)),
    })


def evaluation_label(name: str) -> str:
    lowered = name.lower()
    if "de_to_als" in lowered:
        direction = "DE → ALS"
    elif "als_to_de" in lowered:
        direction = "ALS → DE"
    else:
        direction = "gepoolt"
    kind = "binarisiert" if "binarisiert" in lowered else "dreikategorial"
    return f"{direction}, {kind}"


def build_iaa_table(path: Path) -> pd.DataFrame:
    # K1 and K2 are taken from the evaluation data and labeled
    frame = pd.read_csv(path)
    order = {"Manual_Gold": 0, "Gemma_Gold": 1}
    direction_order = {
        "DE → ALS, dreikategorial": 0,
        "DE → ALS, binarisiert": 1,
        "ALS → DE, dreikategorial": 2,
        "ALS → DE, binarisiert": 3,
        "gepoolt, dreikategorial": 4,
        "gepoolt, binarisiert": 5,
    }
    frame = frame.copy()
    frame["Auswertung"] = frame["name"].map(evaluation_label)
    frame["condition_order"] = frame["condition"].map(order)
    frame["evaluation_order"] = frame["Auswertung"].map(direction_order)
    frame = frame.sort_values(["condition_order", "evaluation_order"])
    if len(frame) != 12:
        raise ValueError(f"Erwartet wurden 12 K1/K2-Zeilen, gefunden: {len(frame)}")

    label = frame["condition"].map({
        "Manual_Gold": "κ1 (Manual_Gold)",
        "Gemma_Gold": "κ2 (Gemma_Gold)",
    })
    return pd.DataFrame({
        "Kennzahl": label,
        "Auswertung": frame["Auswertung"],
        "N": frame["n"].map(integer),
        "p_o": frame["p_o"].map(lambda value: decimal(value, 3)),
        "κ": frame["kappa"].map(lambda value: decimal(value, 3)),
        "95%-CI": [interval(low, high) for low, high in zip(frame["ci_low"], frame["ci_high"])],
    })


def build_k3_table(path: Path) -> pd.DataFrame:
    # K3 is shown pooled and separately by comparison direction.
    frame = pd.read_csv(path).set_index("name")
    ordered_names = [
        "de_to_als",
        "als_to_de",
        "Artikel Eichener_See",
        "Artikel Tappolet",
        "Artikel Kalender",
        "gepoolt",
    ]
    missing = [name for name in ordered_names if name not in frame.index]
    if missing:
        raise ValueError(f"K3-Zeilen fehlen: {missing}")
    frame = frame.loc[ordered_names]
    if not (frame["n_gold"].eq(frame["n_judged"]).all() and frame["coverage"].eq(1.0).all()):
        raise ValueError("Die K3-Datei enthält nicht für alle Gold-Items ein Pipeline-Urteil.")

    labels = {
        "de_to_als": "DE → ALS",
        "als_to_de": "ALS → DE",
        "Artikel Eichener_See": "Eichener See",
        "Artikel Tappolet": "Tappolet",
        "Artikel Kalender": "Kalender",
        "gepoolt": "gepoolt",
    }
    return pd.DataFrame({
        "Auswertung": [labels[name] for name in frame.index],
        "N": frame["n_judged"].map(integer).values,
        "Coverage": frame["coverage"].map(lambda value: decimal(value, 2)).values,
        "p_o": frame["p_o"].map(lambda value: decimal(value, 3)).values,
        "κ": frame["kappa"].map(lambda value: decimal(value, 3)).values,
        "95%-CI": [interval(low, high) for low, high in zip(frame["kappa_ci_low"], frame["kappa_ci_high"])],
        "Macro-F1": frame["macro_f1"].map(lambda value: decimal(value, 3)).values,
        "yes-F1": frame["yes_f1"].map(lambda value: decimal(value, 3)).values,
    })


def parse_args() -> argparse.Namespace:
    # The default paths can be adjusted via command-line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--overlap", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--iaa", type=Path, default=DEFAULT_IAA)
    parser.add_argument("--k3", type=Path, default=DEFAULT_K3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


# The entry point generates and exports all four result tables
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables = [
        (
            "folie41_dekomposition_final",
            build_decomposition_table(args.manual, args.overlap),
            (16.0, 6.2),
            [0.18, 0.09, 0.10, 0.08, 0.09, 0.10, 0.08, 0.10, 0.10],
            9.2,
            1.45,
            "Anmerkung: 3.149 Excel-Einträge; 2 Zeilen ohne fact_id/fact_text ausgeschlossen, daher 3.147 gültige Fakten.",
        ),
        (
            "folie42_overlap_final",
            build_overlap_table(args.overlap),
            (16.0, 5.8),
            [0.20, 0.07, 0.07, 0.10, 0.10, 0.09, 0.10, 0.08, 0.09],
            9.5,
            1.50,
            None,
        ),
        (
            "folie44_k1_k2_final",
            build_iaa_table(args.iaa),
            (14.0, 6.2),
            [0.20, 0.29, 0.07, 0.10, 0.10, 0.18],
            9.3,
            1.48,
            None,
        ),
        (
            "folie45_k3_final",
            build_k3_table(args.k3),
            (14.0, 4.5),
            [0.18, 0.07, 0.11, 0.09, 0.09, 0.18, 0.14, 0.12],
            10.0,
            1.55,
            None,
        ),
    ]

    for stem, frame, figsize, widths, font_size, row_scale, footnote in tables:
        save_table_data(frame, args.output_dir, stem)
        render_table(
            frame=frame,
            output_dir=args.output_dir,
            stem=stem,
            figsize=figsize,
            column_widths=widths,
            font_size=font_size,
            row_scale=row_scale,
            footnote=footnote,
        )
        print(f"Geschrieben: {args.output_dir / stem}.{{csv,png,svg}}")


if __name__ == "__main__":
    main()
