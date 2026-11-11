"""
Aligns paragraphs for X-FactAlign

The procedure:
  - splits the facts in both languages into paragraphs,
  - represents each paragraph by the embeddings of its facts,
  - computes a MaxSim-based association score,
  - links each source paragraph (directed) to its best target paragraph,
  - repeats this in the opposite direction and
  - merges both directed matrices into one undirected matrix.

The resulting adjacency matrix contains the following values:
  0 = no neighborhood between the paragraphs
  1 = selected in one direction
  2 = selected in both directions

Each source paragraph receives at least one outgoing link.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import loguru
import numpy as np
import polars as pl

from gpt_query import FactParagraph  # noqa: F401
from align_facts import pairwise_cosine_matrix

logger = loguru.logger


def _ordered_paragraph_sizes(df: pl.DataFrame) -> list[int]:
    """Count the facts per paragraph in ascending paragraph order."""
    grouped = df.group_by("paragraph_index").len().sort("paragraph_index")
    return grouped["len"].to_list()


# MaxSim condenses the fact similarities into one value per paragraph pair
def compute_directional_paragraph_alignment(
    src_lang_fact_df: pl.DataFrame,
    tgt_lang_fact_df: pl.DataFrame,
    pairwise_fact_comparison_fn: Optional[Callable] = None,
) -> np.ndarray:
    """Compute association scores between source and target paragraphs.

    The MaxSim-based score is obtained in two steps:
      1. For each source fact, the highest similarity to any fact of the
         target paragraph is determined.
      2. The values of all source facts of a paragraph are combined via the
         median.

    Larger values indicate a stronger link between the paragraphs.
    """
    if pairwise_fact_comparison_fn is None:
        raise ValueError("pairwise_fact_comparison_fn must be provided")

    src_lang_fact_df = src_lang_fact_df.sort(["paragraph_index"])
    tgt_lang_fact_df = tgt_lang_fact_df.sort(["paragraph_index"])

    pairwise_score_matrix = pairwise_fact_comparison_fn(
        src_lang_fact_df,
        tgt_lang_fact_df,
    )

    if pairwise_score_matrix.shape != (len(src_lang_fact_df), len(tgt_lang_fact_df)):
        raise ValueError(
            "Pairwise score matrix has wrong shape: "
            f"{pairwise_score_matrix.shape} vs {(len(src_lang_fact_df), len(tgt_lang_fact_df))}"
        )

    tgt_sizes = _ordered_paragraph_sizes(tgt_lang_fact_df)
    src_sizes = _ordered_paragraph_sizes(src_lang_fact_df)

    src_tgt_paragraph_associations = []
    tgt_start = 0

    for tgt_size in tgt_sizes:
        # The maximum summarizes the fact scores of the current target paragraph
        target_paragraph_scores = np.max(
            pairwise_score_matrix[:, tgt_start:tgt_start + tgt_size],
            axis=1,
        )
        tgt_start += tgt_size

        # The median summarizes the scores within each source paragraph
        src_to_this_tgt = []
        src_start = 0
        for src_size in src_sizes:
            src_to_this_tgt.append(
                float(np.median(target_paragraph_scores[src_start:src_start + src_size]))
            )
            src_start += src_size

        src_tgt_paragraph_associations.append(np.asarray(src_to_this_tgt))

    return np.asarray(src_tgt_paragraph_associations).T


def compute_algn_strngths(
    lang_one_facts_df: pl.DataFrame,
    lang_two_facts_df: pl.DataFrame,
    comparison_fn_name: str,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the paragraph association strengths in both directions."""
    lang_one_facts_df = lang_one_facts_df.sort("paragraph_index", descending=False)
    lang_two_facts_df = lang_two_facts_df.sort("paragraph_index", descending=False)

    # For the main run, the already available LaBSE embeddings are compared
    if comparison_fn_name == "cosine":
        # The paragraph alignment uses the LaBSE cosine similarity directly
        # The direction-dependent hubness correction is only applied during fact retrieval,
        # once the paragraph neighborhood has been determined
        comparison_fn = pairwise_cosine_matrix

    elif comparison_fn_name == "bert_score":
        from importlib import import_module

        try:
            compute_all_pairs_bertscore = import_module(
                "bert_score_pairs"
            ).compute_all_pairs_bertscore
        except ImportError as error:
            raise ImportError(
                "bert_score_pairs is only required when comparison_fn_name='bert_score'. "
                "The current pipeline should use comparison_fn_name='cosine'."
            ) from error

        def comparison_fn(
            lang_one_facts_df: pl.DataFrame,
            lang_two_facts_df: pl.DataFrame,
        ) -> np.ndarray:
            return compute_all_pairs_bertscore(
                lang_one_facts_df["fact"].to_list(),
                lang_two_facts_df["fact"].to_list(),
            )

    else:
        raise ValueError(f"unknown comparison_fn_name: {comparison_fn_name}")

    lang_one_to_lang_two = compute_directional_paragraph_alignment(
        lang_one_facts_df,
        lang_two_facts_df,
        comparison_fn,
    )
    lang_two_to_lang_one = compute_directional_paragraph_alignment(
        lang_two_facts_df,
        lang_one_facts_df,
        comparison_fn,
    )

    return lang_one_to_lang_two, lang_two_to_lang_one


def force_directional_alignments(para_para_assoc_strength: np.ndarray) -> np.ndarray:
    """Create the directed paragraph alignment following the paper.

    For each source paragraph, exactly one edge is kept: the link to the
    target paragraph with the highest association score.
    """
    if para_para_assoc_strength.ndim != 2:
        raise ValueError("Paragraph association matrix must be two-dimensional.")

    forced = np.zeros_like(para_para_assoc_strength, dtype=int)
    if para_para_assoc_strength.shape[1] == 0:
        return forced

    # Each source paragraph is linked to exactly its strongest target paragraph
    for i in range(para_para_assoc_strength.shape[0]):
        forced[i, int(np.argmax(para_para_assoc_strength[i]))] = 1

    return forced


def prune_directional_alignments(para_para_assoc_strength: np.ndarray) -> np.ndarray:
    """Call the directed selection of the strongest paragraph edge via an alias."""
    return force_directional_alignments(para_para_assoc_strength)


def union_forced_alignment(
    dir_algn_mat_1: np.ndarray,
    dir_algn_mat_2: np.ndarray,
) -> np.ndarray:
    """Merge both directed paragraph matrices into one undirected matrix.

    ``dir_algn_mat_1`` has the shape paragraphs language 1 × paragraphs language 2,
    ``dir_algn_mat_2`` correspondingly the reverse shape.
    """
    if dir_algn_mat_1.shape != dir_algn_mat_2.T.shape:
        raise ValueError(
            "Directional paragraph-alignment matrices have incompatible shapes: "
            f"{dir_algn_mat_1.shape} and {dir_algn_mat_2.shape}"
        )

    # Transposition puts both directions into the same matrix orientation
    return dir_algn_mat_1.astype(int) + dir_algn_mat_2.T.astype(int)


def union_pruned_alignment(
    dir_algn_mat_1: np.ndarray,
    dir_algn_mat_2: np.ndarray,
) -> np.ndarray:
    """Call the union of both directions via an alias.

    The entries mean:
      0 = no edge
      1 = selected in one direction
      2 = selected in both directions
    """
    return union_forced_alignment(dir_algn_mat_1, dir_algn_mat_2)
