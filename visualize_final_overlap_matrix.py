"""
Visualization of the directed overlap values of the ten gold pairs.

The values are read from the one-shot aggregate, validated and sorted by
their mean.

Default usage:
    python visualize_final_overlap_matrix.py

Custom input and output folders:
    python visualize_final_overlap_matrix.py \
        --csv outputs/xfactmatch_aggregate_oneshot.csv \
        --output-dir outputs/visualizations_final_overlap


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

# Matplotlib's DejaVu Sans is used for a consistent appearance
_original_check_output = subprocess.check_output


# The macOS font query receives an empty list; other calls remain unchanged
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

import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


DEFAULT_CSV = ROOT_DIR / "outputs" / "xfactmatch_aggregate_oneshot.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "visualizations_final_overlap"


COLOR_DE_TO_ALS = "#000099"
COLOR_ALS_TO_DE = "#FF9900"
COLOR_TEXT = "#000000"
COLOR_BACKGROUND = "#FFFFFF"

SHORT_TITLES = {
    "Rheinbrücke Schaffhausen–Feuerthalen": "Rheinbrücke",
    "Rheinbrücke Schaffhausen-Feuerthalen": "Rheinbrücke",
}

REQUIRED_COLUMNS = {"titel", "de_to_als", "als_to_de"}


def load_results(path: Path) -> pd.DataFrame:
    """Load, validate and sort the overlap values in ascending order."""
    # Required columns and value range are validated before plotting
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"In {path} fehlen erforderliche Spalten: {sorted(missing)}"
        )
    if frame.empty:
        raise ValueError(f"{path} enthält keine Ergebniszeilen.")

    frame = frame.copy()
    frame["de_to_als"] = pd.to_numeric(frame["de_to_als"], errors="raise")
    frame["als_to_de"] = pd.to_numeric(frame["als_to_de"], errors="raise")

    values = frame[["de_to_als", "als_to_de"]]
    if not values.apply(lambda column: column.between(0, 1).all()).all():
        raise ValueError("Overlap-Werte müssen zwischen 0 und 1 liegen.")

    frame["mittel"] = values.mean(axis=1)
    frame = frame.sort_values("mittel", ascending=True).reset_index(drop=True)
    frame["anzeigetitel"] = frame["titel"].map(
        lambda value: SHORT_TITLES.get(str(value), str(value))
    )
    return frame


def add_value_labels(ax, bars) -> None:
    """Write percentage values directly to the right of the bars."""
    for bar in bars:
        value = float(bar.get_width())
        ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}",
            ha="left",
            va="center",
            fontsize=10.5,
            color=COLOR_TEXT,
        )


def create_figure(frame: pd.DataFrame):
    """Create a horizontal bar chart."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
        "figure.facecolor": COLOR_BACKGROUND,
        "axes.facecolor": COLOR_BACKGROUND,
        "savefig.facecolor": COLOR_BACKGROUND,
    })

    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    y = np.arange(len(frame))
    # Two slightly offset bars per article pair, one for each direction (DE -> ALS and ALS -> DE)
    bar_height = 0.34

    bars_de = ax.barh(
        y - bar_height / 2,
        frame["de_to_als"],
        height=bar_height,
        color=COLOR_DE_TO_ALS,
        edgecolor=COLOR_TEXT,
        linewidth=0.8,
        label="DE → ALS",
    )
    bars_als = ax.barh(
        y + bar_height / 2,
        frame["als_to_de"],
        height=bar_height,
        color=COLOR_ALS_TO_DE,
        edgecolor=COLOR_TEXT,
        linewidth=0.8,
        label="ALS → DE",
    )

    add_value_labels(ax, bars_de)
    add_value_labels(ax, bars_als)

    ax.set_yticks(y)
    ax.set_yticklabels(frame["anzeigetitel"], fontsize=10.5)
    ax.invert_yaxis()

    # Some space on the right for the directly labeled values
    ax.set_xlim(0, 1.10)
    ax.set_xticks(np.arange(0, 1.01, 0.10))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Overlap", fontsize=12)

    # Reduced styling: no grid, no top/right spines, emphasized left and bottom axes
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_TEXT)
    ax.spines["bottom"].set_color(COLOR_TEXT)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="x", length=6, width=1.2, direction="out")
    ax.tick_params(axis="y", length=0, pad=8)

    ax.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=10.5,
        handlelength=1.7,
        columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.12, top=0.97)
    return fig


def parse_args() -> argparse.Namespace:
    # The default paths can be adjusted via command-line arguments


    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="One-shot-Aggregat als CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Zielordner für PNG, PDF und SVG.",
    )
    return parser.parse_args()


# The entry point loads the data once and exports the same figure in three formats
def main() -> None:
    args = parse_args()
    frame = load_results(args.csv)
    figure = create_figure(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "folie17_overlap_matrix"
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".svg"))
    plt.close(figure)

    print(f"Geschrieben: {stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
