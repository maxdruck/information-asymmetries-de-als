"""
Aligns the manually decomposed facts for the control run.

The fact source is ``manual_paragraph_fact_annotation_max.xlsx``. The facts
it contains are processed with X-FACTALIGN.

Procedure:
  1. The manual DE and ALS facts are loaded from the Excel file.
  2. LaBSE generates the multilingual embeddings once.
  3. The paragraphs are aligned in both directions and combined into an
     undirected neighborhood graph.
  4. The density-normalized distance is computed separately for each direction.
  5. Within neighboring paragraphs, the two best target facts are retrieved.
  6. Optionally, only the source side of each direction is limited to the first
     N facts, matching the manually evaluated sample of the first 50 facts.
  7. The candidates are saved for both directions.

Output:
  outputs/manual_facts/manual_facts_<de|als>_<pair_key>.json
  outputs/manual_alignments/manual_alignment_<pair_key>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from align_facts import MODEL_NAME, align_facts, get_model
from align_paragraphs import (
    compute_algn_strngths,
    force_directional_alignments,
    union_forced_alignment,
)
from manual_control_facts_io import (
    DEFAULT_MANUAL_XLSX,
    MANUAL_ALIGNMENT_DIR,
    MANUAL_CONTROL_PAIR_KEYS,
    manual_alignment_path,
    portable_path,
    write_manual_pair_facts,
)
from pipeline_config import load_pairs, select_pairs


TOP_K_FACTS = 2
ALIGNMENT_COLUMNS = ["paragraph_index", "fact", "fact_embed"]


# Only the source side is limited; on the target side the whole article remains searchable
def limit_source_frame(frame: pl.DataFrame, limit: int | None) -> pl.DataFrame:
    """Limit the source side to the first N facts or use all of them.

    The target side is deliberately kept complete. This matches the
    validation setup: only the first 50 source facts were evaluated manually,
    but a source fact can occur anywhere in the partner article.
    """
    frame = frame.sort(["paragraph_index", "fact_index"])
    if limit is None:
        return frame
    if limit <= 0:
        raise ValueError("--source-limit-per-direction must be a positive integer")
    return frame.head(limit)


def build_frame(facts: list[dict]) -> pl.DataFrame:
    """Keep stable fact IDs and add the corresponding LaBSE embeddings."""
    texts = [fact["fact"] for fact in facts]
    paragraph_indices = [int(fact["paragraph_index"]) for fact in facts]
    fact_indices = [int(fact.get("fact_index", position)) for position, fact in enumerate(facts)]
    manual_fact_ids = [str(fact.get("manual_fact_id", fact_indices[position])) for position, fact in enumerate(facts)]

    # A single shared encoding step avoids repeated LaBSE calls during alignment
    embeddings = np.asarray(get_model().encode(texts))

    return pl.DataFrame({
        "fact_index": fact_indices,
        "manual_fact_id": manual_fact_ids,
        "paragraph_index": paragraph_indices,
        "fact": texts,
        "fact_embed": [row.tolist() for row in embeddings],
    })


def alignment_view(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(ALIGNMENT_COLUMNS)


def sorted_unique_paragraphs(frame: pl.DataFrame) -> list[int]:
    return sorted(int(x) for x in frame["paragraph_index"].unique().to_list())


def make_adjacency_lookup(adjacency_pairs: list[dict], source_key: str, target_key: str) -> dict[int, set[int]]:
    lookup: dict[int, set[int]] = {}
    for edge in adjacency_pairs:
        lookup.setdefault(int(edge[source_key]), set()).add(int(edge[target_key]))
    return lookup


def retrieve_direction(
    source_frame: pl.DataFrame,
    target_frame: pl.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_lookup: dict[int, set[int]],
    source_language: str,
    target_language: str,
) -> list[dict]:
    """Determine the best candidates for each source fact.

    Only target facts from paragraphs adjacent to the source paragraph are
    considered. ``distance_matrix`` contains the density-normalized distances
    following the paper; smaller values are better.
    """
    source_frame = source_frame.sort(["paragraph_index", "fact_index"])
    target_frame = target_frame.sort(["paragraph_index", "fact_index"])

    source_indices = source_frame["fact_index"].to_list()
    source_manual_ids = source_frame["manual_fact_id"].to_list()
    source_paragraphs = source_frame["paragraph_index"].to_list()
    source_texts = source_frame["fact"].to_list()

    target_indices = target_frame["fact_index"].to_list()
    target_manual_ids = target_frame["manual_fact_id"].to_list()
    target_paragraphs = target_frame["paragraph_index"].to_list()
    target_texts = target_frame["fact"].to_list()

    if distance_matrix.shape != (len(source_frame), len(target_frame)):
        raise ValueError(
            "Distance-matrix shape does not match source/target frame sizes: "
            f"{distance_matrix.shape} vs {(len(source_frame), len(target_frame))}"
        )

    results = []

    # For each source fact, only candidates from linked target paragraphs are considered
    for source_position in range(distance_matrix.shape[0]):
        source_paragraph = int(source_paragraphs[source_position])
        allowed_target_paragraphs = adjacency_lookup.get(source_paragraph, set())

        candidate_positions = [
            target_position
            for target_position in range(distance_matrix.shape[1])
            if int(target_paragraphs[target_position]) in allowed_target_paragraphs
        ]

        candidate_scores = [
            (target_position, float(distance_matrix[source_position, target_position]))
            for target_position in candidate_positions
        ]

        # Sort the candidates by D(e_i, f_j) in ascending order (smaller distance = better match)
        candidate_scores.sort(key=lambda item: item[1])

        matches = [
            {
                "target_fact_index": int(target_indices[target_position]),
                "target_manual_fact_id": target_manual_ids[target_position],
                "target_fact": target_texts[target_position],
                "target_paragraph": int(target_paragraphs[target_position]),
                "density_normalized_distance": round(score, 4),
            }
            for target_position, score in candidate_scores[:TOP_K_FACTS]
        ]

        results.append({
            "source_language": source_language,
            "target_language": target_language,
            "source_fact_index": int(source_indices[source_position]),
            "source_manual_fact_id": source_manual_ids[source_position],
            "source_fact": source_texts[source_position],
            "source_paragraph": source_paragraph,
            "matches": matches,
        })

    return results


def align_pair(
    pair: dict,
    manual_xlsx: Path,
    overwrite: bool = False,
    source_limit_per_direction: int | None = None,
) -> None:
    # A pair goes through fact preparation, paragraph alignment and directed fact retrieval
    pair_key = pair["pair_key"]
    output_path = manual_alignment_path(pair_key)

    if output_path.exists() and not overwrite:
        print(f"Already exists, skipping: {output_path}")
        return

    manual_title, facts_by_language = write_manual_pair_facts(
        xlsx_path=manual_xlsx,
        pair=pair,
        overwrite=overwrite,
    )

    de_facts = facts_by_language["de"]["facts"]
    als_facts = facts_by_language["als"]["facts"]

    if not de_facts or not als_facts:
        print(f"  WARNING: empty manual facts for pair {pair_key}; skipping")
        return

    print(f"Loading MANUAL facts for '{manual_title}' ({pair_key})...")

    de_df = build_frame(de_facts).sort(["paragraph_index", "fact_index"])
    als_df = build_frame(als_facts).sort(["paragraph_index", "fact_index"])

    print(
        f"  de: {len(de_df)} manual facts / {de_df['paragraph_index'].n_unique()} paragraphs; "
        f"als: {len(als_df)} manual facts / {als_df['paragraph_index'].n_unique()} paragraphs"
    )

    de_alignment_df = alignment_view(de_df)
    als_alignment_df = alignment_view(als_df)

    # Paragraph alignment: directed maximum in both directions, followed by union
    de_to_als_strength, als_to_de_strength = compute_algn_strngths(
        de_alignment_df,
        als_alignment_df,
        "cosine",
    )
    forced_de_to_als = force_directional_alignments(de_to_als_strength)
    forced_als_to_de = force_directional_alignments(als_to_de_strength)
    adjacency = union_forced_alignment(forced_de_to_als, forced_als_to_de)

    de_paragraphs = sorted_unique_paragraphs(de_df)
    als_paragraphs = sorted_unique_paragraphs(als_df)

    adjacency_pairs = [
        {
            "de_paragraph": int(de_paragraphs[i]),
            "als_paragraph": int(als_paragraphs[j]),
            "weight": int(adjacency[i, j]),
        }
        for i in range(adjacency.shape[0])
        for j in range(adjacency.shape[1])
        if adjacency[i, j] > 0
    ]

    print(f"\nParagraph adjacency: {len(adjacency_pairs)} edges")
    for edge in adjacency_pairs:
        print(
            f"  de[{edge['de_paragraph']}] <-> "
            f"als[{edge['als_paragraph']}] "
            f"(w={edge['weight']})"
        )

    de_adj_lookup = make_adjacency_lookup(adjacency_pairs, "de_paragraph", "als_paragraph")
    als_adj_lookup = make_adjacency_lookup(adjacency_pairs, "als_paragraph", "de_paragraph")

    # The density-normalized distance is direction-dependent
    de_to_als_distance = align_facts(
        de_df,
        als_df,
        de_df["fact"].to_list(),
        als_df["fact"].to_list(),
        adjacency_lookup=de_adj_lookup,
    )
    als_to_de_distance = align_facts(
        als_df,
        de_df,
        als_df["fact"].to_list(),
        de_df["fact"].to_list(),
        adjacency_lookup=als_adj_lookup,
    )

    # For the comparison with the human sample, only the source side of each
    # direction is limited. Paragraph alignment and hubness distance are still computed
    # beforehand on the complete manual fact sets
    de_source_df = limit_source_frame(de_df, source_limit_per_direction)
    als_source_df = limit_source_frame(als_df, source_limit_per_direction)

    de_to_als_distance_for_output = de_to_als_distance[: len(de_source_df), :]
    als_to_de_distance_for_output = als_to_de_distance[: len(als_source_df), :]

    if source_limit_per_direction is not None:
        print(
            f"\nSource-side validation limit: first {source_limit_per_direction} facts per direction"
        )
        print(
            f"  DE -> ALS source facts: {len(de_source_df)} of {len(de_df)}; "
            f"ALS target facts: {len(als_df)}"
        )
        print(
            f"  ALS -> DE source facts: {len(als_source_df)} of {len(als_df)}; "
            f"DE target facts: {len(de_df)}"
        )

    de_to_als = retrieve_direction(
        source_frame=de_source_df,
        target_frame=als_df,
        distance_matrix=de_to_als_distance_for_output,
        adjacency_lookup=de_adj_lookup,
        source_language="de",
        target_language="als",
    )
    als_to_de = retrieve_direction(
        source_frame=als_source_df,
        target_frame=de_df,
        distance_matrix=als_to_de_distance_for_output,
        adjacency_lookup=als_adj_lookup,
        source_language="als",
        target_language="de",
    )

    MANUAL_ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "pair_key": pair_key,
        "cluster": pair.get("cluster"),
        "title": pair.get("title") or manual_title,
        "manual_title": manual_title,
        "als_page_id": pair.get("als_page_id"),
        "de_page_id": pair.get("de_page_id"),
        "als_title": pair.get("als_title"),
        "de_title": pair.get("de_title"),
        "fact_source": "manual",
        "manual_xlsx": portable_path(manual_xlsx),
        "source_limit_per_direction": source_limit_per_direction,
        "full_manual_fact_counts": {
            "de": len(de_df),
            "als": len(als_df),
        },
        "reported_source_fact_counts": {
            "de_to_als": len(de_to_als),
            "als_to_de": len(als_to_de),
        },
        "embed_model": MODEL_NAME,
        "xfactalign_version": "paper_style_density_normalized_distance",
        "top_k_facts": TOP_K_FACTS,
        "paragraph_alignment": "directed_argmax_union",
        "fact_retrieval_metric": "D(e_i,f_j)=1-cos(e_i,f_j)-hubness(f_j)",
        "paragraph_adjacency": adjacency_pairs,
        "de_to_als": de_to_als,
        "als_to_de": als_to_de,
    }

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDE -> ALS candidate mappings: {len(de_to_als)}")
    print(f"ALS -> DE candidate mappings: {len(als_to_de)}")
    print(f"Saved MANUAL alignment to {output_path}")


# Without a filter, the control run processes exactly the three manually annotated pairs
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pair_key",
        nargs="?",
        help="Optional exact pair_key. Equivalent to --pair-key.",
    )
    parser.add_argument(
        "--pair-key",
        dest="pair_key_flag",
        help="Process one exact pair_key from configs/article_pairs.jsonl.",
    )
    parser.add_argument("--cluster", help="Process only one cluster.")
    parser.add_argument("--limit", type=int, help="Process only the first N selected pairs.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N selected pairs.",
    )
    parser.add_argument(
        "--manual-xlsx",
        type=Path,
        default=DEFAULT_MANUAL_XLSX,
        help="Path to manual_paragraph_fact_annotation_max.xlsx.",
    )
    parser.add_argument(
        "--source-limit-per-direction",
        type=int,
        default=None,
        help=(
            "Restrict only the source side to the first N facts per direction. "
            "Use 50 for the manually aligned validation sample. The target side remains full."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs that already exist.",
    )
    args = parser.parse_args()

    pair_key = args.pair_key_flag or args.pair_key
    configured_pairs = load_pairs()
    if pair_key is None and args.cluster is None and args.limit is None and args.start_index == 0:
        manual_keys = set(MANUAL_CONTROL_PAIR_KEYS)
        pairs = [pair for pair in configured_pairs if pair["pair_key"] in manual_keys]
    else:
        pairs = select_pairs(
            configured_pairs,
            cluster=args.cluster,
            pair_key=pair_key,
            limit=args.limit,
            start_index=args.start_index,
        )

    if not pairs:
        raise ValueError(
            "No article pairs selected. Check pair_key/--cluster or configs/article_pairs.jsonl."
        )

    for index, pair in enumerate(pairs, start=1):
        print(f"\n{'=' * 70}")
        print(f"[{index}/{len(pairs)}] Aligning MANUAL facts: {pair.get('title')} ({pair['pair_key']})")
        print(f"{'=' * 70}")
        align_pair(
            pair,
            manual_xlsx=args.manual_xlsx,
            overwrite=args.overwrite,
            source_limit_per_direction=args.source_limit_per_direction,
        )


if __name__ == "__main__":
    main()
