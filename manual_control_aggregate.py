"""
Aggregates the verified results of the manual control run.

The script reads ``outputs/manual_verified/manual_verified_<pair_key>.json``
and writes:
  outputs/manual_verified/manual_overlap_summary.csv
  outputs/manual_verified/manual_overlap_summary.json

The overlap is ``raw_yes_share = yes / total_source_facts``. In addition,
``overlap_ratio_verified = yes / (yes + no)`` is kept so that remaining
``unknown`` cases can be traced.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

from manual_control_facts_io import (
    MANUAL_CONTROL_PAIR_KEYS,
    MANUAL_VERIFIED_DIR,
    manual_verified_path,
    portable_path,
    read_json,
    write_json,
)

try:
    from pipeline_config import load_pairs, select_pairs
except Exception:  # Enables the aggregation even without the complete project folder
    load_pairs = None
    select_pairs = None


def row_for_direction(data: dict, direction: str) -> dict:
    # Both language directions are brought into a common tabular schema
    summary = data[f"{direction}_summary"]
    return {
        "pair_key": data.get("pair_key"),
        "title": data.get("title"),
        "manual_title": data.get("manual_title"),
        "direction": direction,
        "total_source_facts": summary.get("total_source_facts"),
        "shared_yes": summary.get("shared_yes"),
        "source_only_no": summary.get("source_only_no"),
        "unknown": summary.get("unknown"),
        "coverage": summary.get("verified_coverage"),
        "overlap_yes_total": summary.get("raw_yes_share"),
        "overlap_yes_verified": summary.get("overlap_ratio_verified"),
    }


def load_selected_files(pair_key: str | None, cluster: str | None, limit: int | None, start_index: int) -> list[Path]:
    # Without a filter, only the three control pairs that actually exist are loaded
    if load_pairs is not None and select_pairs is not None:
        configured_pairs = load_pairs()
        if pair_key is None and cluster is None and limit is None and start_index == 0:
            manual_keys = set(MANUAL_CONTROL_PAIR_KEYS)
            pairs = [pair for pair in configured_pairs if pair["pair_key"] in manual_keys]
        else:
            pairs = select_pairs(
                configured_pairs,
                cluster=cluster,
                pair_key=pair_key,
                limit=limit,
                start_index=start_index,
            )
        return [manual_verified_path(pair["pair_key"]) for pair in pairs]

    files = sorted(MANUAL_VERIFIED_DIR.glob("manual_verified_*.json"))
    if pair_key:
        files = [path for path in files if path.name == f"manual_verified_{pair_key}.json"]
    if start_index:
        files = files[start_index:]
    if limit is not None:
        files = files[:limit]
    return files


def aggregate(files: list[Path]) -> dict:
    # Missing files are logged without discarding existing pairs
    rows: list[dict] = []
    missing: list[str] = []

    for path in files:
        if not path.exists():
            missing.append(portable_path(path))
            continue
        data = read_json(path)
        rows.append(row_for_direction(data, "de_to_als"))
        rows.append(row_for_direction(data, "als_to_de"))

    by_direction: dict[str, dict] = {}
    for direction in ("de_to_als", "als_to_de"):
        values = [row["overlap_yes_total"] for row in rows if row["direction"] == direction]
        values = [value for value in values if value is not None]
        by_direction[direction] = {
            "n_pairs": len(values),
            "mean_overlap_yes_total": round(mean(values), 4) if values else None,
            "median_overlap_yes_total": round(median(values), 4) if values else None,
            "min_overlap_yes_total": round(min(values), 4) if values else None,
            "max_overlap_yes_total": round(max(values), 4) if values else None,
        }

    return {
        "fact_source": "manual",
        "rows": rows,
        "by_direction": by_direction,
        "missing_files": missing,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    # The CSV contains the same direction rows as the more detailed JSON report
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# The entry point selects files, aggregates them and writes JSON and CSV together
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_key", nargs="?", help="Optional exact pair_key. Equivalent to --pair-key.")
    parser.add_argument("--pair-key", dest="pair_key_flag", help="Process one exact pair_key.")
    parser.add_argument("--cluster", help="Process only one cluster, if pipeline_config is available.")
    parser.add_argument("--limit", type=int, help="Process only the first N selected pairs.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N selected pairs.")
    args = parser.parse_args()

    pair_key = args.pair_key_flag or args.pair_key
    files = load_selected_files(pair_key, args.cluster, args.limit, args.start_index)
    result = aggregate(files)

    json_path = MANUAL_VERIFIED_DIR / "manual_overlap_summary.json"
    csv_path = MANUAL_VERIFIED_DIR / "manual_overlap_summary.csv"
    write_json(json_path, result)
    write_csv(csv_path, result["rows"])

    print("Manual overlap summary")
    print(json.dumps(result["by_direction"], ensure_ascii=False, indent=2))

    if result["missing_files"]:
        print("\nMissing verified files:")
        for missing in result["missing_files"]:
            print(f"  {missing}")

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
