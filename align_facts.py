"""
Performs the fact-based retrieval of X-FactAlign.

The implementation follows this logic:


    cos(e_i, f_j) = (e_i · f_j) / (||e_i|| ||f_j||)
    d(e_i, f_j)   = 1 - cos(e_i, f_j)
    D(e_i, f_j)   = d(e_i, f_j) - hubness(f_j)

The hubness of f_j is estimated as the mean distance to its k nearest
neighbors (kNN = 5). For this, up to 50 target-language facts are used from
paragraphs that are not adjacent to the source paragraph of e_i.

Important:
    - A smaller value of D(e_i, f_j) means a better match.
    - DE -> ALS and ALS -> DE are computed separately.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer


HF_CACHE_DIR = None
MODEL_NAME = "sentence-transformers/LaBSE"
KNN_HUBNESS = 5
HUBNESS_BACKGROUND_SIZE = 50
RANDOM_SEED = 42

_model: SentenceTransformer | None = None


# The LaBSE embedding model is loaded lazily, i.e. only once embeddings are actually generated
def get_model() -> SentenceTransformer:
    """Load LaBSE on first use and cache the model instance."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, cache_folder=HF_CACHE_DIR)
    return _model


def _as_embedding_matrix(frame: pl.DataFrame) -> np.ndarray:
    """Create a row-normalized embedding matrix from a Polars DataFrame."""
    embeds = np.asarray(frame["fact_embed"].to_list(), dtype=float)
    if embeds.ndim != 2:
        raise ValueError("fact_embed column must contain one vector per row.")

    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeds / norms


def pairwise_cosine_matrix(
    src_fact_frame: pl.DataFrame,
    tgt_fact_frame: pl.DataFrame,
) -> np.ndarray:
    """Compute the pairwise cosine similarity of source and target facts.

    Larger values indicate more similar facts.
    """
    src_embeds = _as_embedding_matrix(src_fact_frame)
    tgt_embeds = _as_embedding_matrix(tgt_fact_frame)
    return np.dot(src_embeds, tgt_embeds.T)


def pairwise_distance_matrix(
    src_fact_frame: pl.DataFrame,
    tgt_fact_frame: pl.DataFrame,
) -> np.ndarray:
    """Compute the pairwise cosine distance:

        d(e_i, f_j) = 1 - cos(e_i, f_j)

    Smaller values indicate more similar facts.
    """
    return 1.0 - pairwise_cosine_matrix(src_fact_frame, tgt_fact_frame)


# A fixed random source yields the same background sample for the same input
def _sample_background_positions(
    positions: list[int],
    seed: int,
    max_size: int = HUBNESS_BACKGROUND_SIZE,
) -> list[int]:
    """Deterministically draw a sample of up to 50 background facts."""
    if len(positions) <= max_size:
        return positions

    rng = np.random.default_rng(seed)
    sampled = rng.choice(np.asarray(positions), size=max_size, replace=False)
    return sorted(int(x) for x in sampled.tolist())


def _target_hubness_for_source_paragraph(
    tgt_distance_matrix: np.ndarray,
    target_paragraphs: list[int],
    allowed_target_paragraphs: set[int],
    seed: int,
) -> np.ndarray:
    """Computes the target-side hubness for a source paragraph.
    """
    n_tgt = len(target_paragraphs)

    background_positions = [
        pos
        for pos, paragraph_index in enumerate(target_paragraphs)
        if int(paragraph_index) not in allowed_target_paragraphs
    ]

    if not background_positions:
        background_positions = list(range(n_tgt))

    background_positions = _sample_background_positions(
        background_positions,
        seed=seed,
    )

    hubness_values = np.zeros(n_tgt, dtype=float)

    for target_pos in range(n_tgt):
        usable_background = [
            pos for pos in background_positions
            if pos != target_pos
        ]

        if not usable_background:
            hubness_values[target_pos] = 0.0
            continue

        distances = tgt_distance_matrix[target_pos, usable_background]
        k = min(KNN_HUBNESS, len(distances))
        nearest_distances = np.partition(distances, k - 1)[:k]
        hubness_values[target_pos] = float(np.mean(nearest_distances))

    return hubness_values


# This is where the actual hubness correction is applied, separately for each search direction
def paper_density_normalized_distance(
    src_fact_frame: pl.DataFrame,
    tgt_fact_frame: pl.DataFrame,
    adjacency_lookup: Optional[dict[int, set[int]]] = None,
) -> np.ndarray:
    """Computes the density-normalized distance matrix.

    For a source fact e_i and a target fact f_j:

        d(e_i, f_j) = 1 - cos(e_i, f_j)
        D(e_i, f_j) = d(e_i, f_j) - hubness(f_j)

    A smaller value of D means a better match.
    """
    src_fact_frame = src_fact_frame.sort(["paragraph_index", "fact_index"])
    tgt_fact_frame = tgt_fact_frame.sort(["paragraph_index", "fact_index"])

    base_distance = pairwise_distance_matrix(src_fact_frame, tgt_fact_frame)

    # The target-target distance matrix provides the basis for the target-side hubness
    tgt_embeds = _as_embedding_matrix(tgt_fact_frame)
    tgt_distance = 1.0 - np.dot(tgt_embeds, tgt_embeds.T)

    source_paragraphs = [int(x) for x in src_fact_frame["paragraph_index"].to_list()]
    target_paragraphs = [int(x) for x in tgt_fact_frame["paragraph_index"].to_list()]

    if adjacency_lookup is None:
        adjacency_lookup = {}

    normalized_distance = np.zeros_like(base_distance)
    hubness_cache: dict[int, np.ndarray] = {}

    for src_pos, src_paragraph in enumerate(source_paragraphs):
        if src_paragraph not in hubness_cache:
            allowed_target_paragraphs = adjacency_lookup.get(src_paragraph, set())
            hubness_cache[src_paragraph] = _target_hubness_for_source_paragraph(
                tgt_distance_matrix=tgt_distance,
                target_paragraphs=target_paragraphs,
                allowed_target_paragraphs=allowed_target_paragraphs,
                seed=RANDOM_SEED + int(src_paragraph),
            )

        target_hubness = hubness_cache[src_paragraph]
        normalized_distance[src_pos, :] = base_distance[src_pos, :] - target_hubness

    return normalized_distance


def align_facts(
    src_lang_fact_df: pl.DataFrame,
    tgt_lang_fact_df: pl.DataFrame,
    src_sentences: list[str] | None = None,
    tgt_sentences: list[str] | None = None,
    adjacency_lookup: Optional[dict[int, set[int]]] = None,
) -> np.ndarray:
    """Returns the density-normalized distance matrix.

    Smaller values are better. ``src_sentences`` and ``tgt_sentences``
    remain optional parameters of the public interface and are not
    used for this computation.
    """
   
    return paper_density_normalized_distance(
        src_fact_frame=src_lang_fact_df,
        tgt_fact_frame=tgt_lang_fact_df,
        adjacency_lookup=adjacency_lookup,
    )
