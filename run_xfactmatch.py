"""
Verifies the DE/ALS fact candidates with X-FACTMATCH.

The script reads the two best candidates from ``run_alignment.py`` and asks
the KISSKI/GWDG model whether the source fact can be inferred from at least
one of the retrieved target contexts:
  - The context comprises the fact under consideration and up to two directly
    preceding facts of the same article.
  - The judgment is parsed as ``yes``, ``no`` or ``unknown``.
  - Successful model judgments are cached per article pair.

Output:
  outputs/verified/verified_<pair_key>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from openai import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from gpt_query import ask_gpt_about_fact_intersection, load_kisski_client
from pipeline_config import (
    CACHE_DIR,
    VERIFIED_DIR,
    XFACTMATCH_MODEL,
    alignment_path,
    facts_path,
    load_pairs,
    select_pairs,
    verified_path,
    xfactmatch_cache_path,
)

MODEL = os.getenv("XFACTMATCH_MODEL", XFACTMATCH_MODEL)
CONTEXT_SIZE = 3
MAX_RETRIES = 5
RETRY_DELAY = 10


# These I/O helpers write JSON as readable UTF-8 and create target folders automatically
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ordered_facts(facts: list[dict]) -> list[dict]:
    # Article-wide fact indices determine the order of the subsequent context windows
    return sorted(facts, key=lambda fact: int(fact["fact_index"]))


def build_fact_position_map(facts: list[dict]) -> dict[int, int]:
    return {int(fact["fact_index"]): position for position, fact in enumerate(facts)}


def get_fact_context(
    facts: list[dict],
    fact_positions: dict[int, int],
    fact_index: int,
    context_size: int = CONTEXT_SIZE,
) -> list[str]:
    """Return the current fact and up to two directly preceding facts."""
    if fact_index not in fact_positions:
        raise KeyError(f"Unknown fact_index: {fact_index}")

    # The fact being checked is at the end of the window, preceded by at most two predecessors
    position = fact_positions[fact_index]
    start = max(0, position - context_size + 1)

    return [fact["fact"] for fact in facts[start:position + 1]]


def parse_intersection_label(raw_response: str) -> str | None:
    """Map the model response to the labels yes and no."""
    text = raw_response.strip().lower()
    text = re.sub(r"^```(?:json|text)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip(" \t\r\n\"'`.,;:[]{}()")

    if re.search(r"\b(yes|ja)\b", text):
        return "yes"

    if re.search(r"\b(no|nein)\b", text):
        return "no"

    return None


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return load_json(path)
    except (json.JSONDecodeError, OSError):
        return {}


def make_cache_key(
    direction: str,
    source_fact_index: int,
    candidate_indices: list[int],
    source_context: list[str],
    target_contexts: list[list[str]],
) -> str:
    """Create the cache key for a model judgment.

    It comprises the direction, fact indices, model settings and the context
    texts actually used.
    """
    # Context texts in the key prevent hits from an input that has changed in the meantime
    payload = {
        "direction": direction,
        "source_fact_index": source_fact_index,
        "candidate_indices": candidate_indices,
        "source_context": source_context,
        "target_contexts": target_contexts,
        "model": MODEL,
        "context_size": CONTEXT_SIZE,
    }

    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def query_with_retries(
    client,
    title: str,
    source_context: list[str],
    target_contexts: list[list[str]],
    source_language: str,
    target_language: str,
) -> dict:
    #  Server problems are retried
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_response, total_tokens = ask_gpt_about_fact_intersection(
                client=client,
                model_name=MODEL,
                article_title=title,
                src_facts=source_context,
                tgt_fact_contexts=target_contexts,
                src_lang=source_language,
                tgt_lang=target_language,
            )

            label = parse_intersection_label(raw_response or "")

            if label is None:
                return {
                    "intersection_label": "unknown",
                    "raw_response": raw_response,
                    "total_tokens": total_tokens,
                    "verification_status": "unparseable_response",
                }

            return {
                "intersection_label": label,
                "raw_response": raw_response,
                "total_tokens": total_tokens,
                "verification_status": "model_verified",
            }

        except BadRequestError as error:
            return {
                "intersection_label": "unknown",
                "raw_response": None,
                "total_tokens": None,
                "verification_status": "bad_request",
                "error": str(error),
            }

        except RateLimitError as error:
            raise RuntimeError(
                "Rate limit reached during X-FACTMATCH. "
                "No unknown label was cached. Restart the script later; "
                "already completed judgements remain cached."
                
            ) from error

        except InternalServerError as error:
            wait = min(RETRY_DELAY * (2 ** (attempt - 1)), 120)

            if attempt < MAX_RETRIES:
                print(f"      Server error ({attempt}/{MAX_RETRIES}); waiting {wait}s...")
                time.sleep(wait)
            else:
                return {
                    "intersection_label": "unknown",
                    "raw_response": None,
                    "total_tokens": None,
                    "verification_status": "server_error",
                    "error": str(error),
                }

        except (NotFoundError, AuthenticationError, PermissionDeniedError) as error:
            raise RuntimeError(
                f"API configuration error ({type(error).__name__}): {error}. "
                f"Check XFACTMATCH_MODEL ('{MODEL}') and API_KEY."
            ) from error

        except Exception as error:
            return {
                "intersection_label": "unknown",
                "raw_response": None,
                "total_tokens": None,
                "verification_status": "unexpected_error",
                "error": str(error),
            }

    raise RuntimeError("Unreachable retry state")


def verify_direction(
    client,
    title: str,
    direction: str,
    alignments: list[dict],
    source_facts: list[dict],
    target_facts: list[dict],
    source_language: str,
    target_language: str,
    cache: dict,
    cache_path: Path,
) -> list[dict]:
    # Each fact is checked against the candidates provided by the alignment
    source_facts = ordered_facts(source_facts)
    target_facts = ordered_facts(target_facts)

    source_positions = build_fact_position_map(source_facts)
    target_positions = build_fact_position_map(target_facts)

    verified_rows = []

    for row_number, alignment in enumerate(alignments, start=1):
        source_fact_index = int(alignment["source_fact_index"])
        matches = alignment.get("matches", [])
        candidate_indices = [int(match["target_fact_index"]) for match in matches]

        source_context = get_fact_context(
            source_facts,
            source_positions,
            source_fact_index,
        )

        candidate_contexts = []

        for match in matches:
            target_fact_index = int(match["target_fact_index"])
            candidate_contexts.append({
                **match,
                "context": get_fact_context(
                    target_facts,
                    target_positions,
                    target_fact_index,
                ),
            })

        target_contexts = [candidate["context"] for candidate in candidate_contexts]

        cache_key = make_cache_key(
            direction=direction,
            source_fact_index=source_fact_index,
            candidate_indices=candidate_indices,
            source_context=source_context,
            target_contexts=target_contexts,
        )

        print(f"  [{direction} {row_number}/{len(alignments)}] fact {source_fact_index}")

        if cache_key in cache:
            judgement = cache[cache_key]
            print("      (cached)")

        elif not candidate_contexts:
            judgement = {
                "intersection_label": "unknown",
                "raw_response": None,
                "total_tokens": 0,
                "verification_status": "no_candidates",
            }
            cache[cache_key] = judgement
            write_json(cache_path, cache)

        else:
            judgement = query_with_retries(
                client=client,
                title=title,
                source_context=source_context,
                target_contexts=target_contexts,
                source_language=source_language,
                target_language=target_language,
            )
            cache[cache_key] = judgement
            write_json(cache_path, cache)

        verified_rows.append({
            **alignment,
            "source_context": source_context,
            "candidate_contexts": candidate_contexts,
            "verification": judgement,
        })

    return verified_rows


def summarize(rows: list[dict]) -> dict:
# The summary contains the overlap over all facts and
# additionally the yes share among the conclusively verified facts
    counts = Counter(row["verification"]["intersection_label"] for row in rows)

    total = len(rows)
    yes = counts.get("yes", 0)
    no = counts.get("no", 0)
    unknown = counts.get("unknown", 0)
    verified = yes + no

    return {
        "total_source_facts": total,
        "shared_yes": yes,
        "source_only_no": no,
        "unknown": unknown,
        "verified_coverage": round(verified / total, 4) if total else None,
        "overlap_ratio_verified": round(yes / verified, 4) if verified else None,
        "raw_yes_share": round(yes / total, 4) if total else None,
    }


def verify_pair(client, pair: dict, overwrite: bool = False) -> None:
    # Both language directions are verified separately and saved in one pair file
    pair_key = pair["pair_key"]

    de_path = facts_path("de", pair_key)
    als_path = facts_path("als", pair_key)
    aln_path = alignment_path(pair_key)
    output_path = verified_path(pair_key)

    if output_path.exists() and not overwrite:
        print(f"Already exists, skipping: {output_path}")
        return

    for path in (de_path, als_path, aln_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    de_data = load_json(de_path)
    als_data = load_json(als_path)
    alignment_data = load_json(aln_path)

    if "de_to_als" not in alignment_data or "als_to_de" not in alignment_data:
        raise ValueError(
            "The alignment JSON does not contain both directions. "
            "Rerun run_alignment.py first."
        )

    cache_path = xfactmatch_cache_path(pair_key)
    cache = load_cache(cache_path)

    title = (
        pair.get("title")
        or alignment_data.get("title")
        or de_data.get("pair_title")
        or pair_key
    )

    print(f"Verifying fact intersections for '{title}' with {MODEL}...")

    verified_de_to_als = verify_direction(
        client=client,
        title=title,
        direction="de_to_als",
        alignments=alignment_data["de_to_als"],
        source_facts=de_data["facts"],
        target_facts=als_data["facts"],
        source_language="de",
        target_language="als",
        cache=cache,
        cache_path=cache_path,
    )

    verified_als_to_de = verify_direction(
        client=client,
        title=title,
        direction="als_to_de",
        alignments=alignment_data["als_to_de"],
        source_facts=als_data["facts"],
        target_facts=de_data["facts"],
        source_language="als",
        target_language="de",
        cache=cache,
        cache_path=cache_path,
    )

    output = {
        "pair_key": pair_key,
        "cluster": pair.get("cluster"),
        "title": title,
        "als_page_id": pair.get("als_page_id"),
        "de_page_id": pair.get("de_page_id"),
        "verification_model": MODEL,
        "context_size": CONTEXT_SIZE,
        "de_to_als_summary": summarize(verified_de_to_als),
        "als_to_de_summary": summarize(verified_als_to_de),
        "de_to_als": verified_de_to_als,
        "als_to_de": verified_als_to_de,
    }

    write_json(output_path, output)

    print("\nDE -> ALS:", output["de_to_als_summary"])
    print("ALS -> DE:", output["als_to_de_summary"])
    print(f"\nSaved verified information gaps to {output_path}")


# The entry point selects pairs, initializes the API client and processes them one after another
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
    parser.add_argument(
        "--cluster",
        help="Process only one cluster.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N selected pairs.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N selected pairs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs that already exist.",
    )

    args = parser.parse_args()

    pair_key = args.pair_key_flag or args.pair_key

    pairs = select_pairs(
        load_pairs(),
        cluster=args.cluster,
        pair_key=pair_key,
        limit=args.limit,
        start_index=args.start_index,
    )

    if not pairs:
        raise ValueError(
            "No article pairs selected. "
            "Check pair_key/--cluster or configs/article_pairs.jsonl."
        )

    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    client = load_kisski_client()

    for index, pair in enumerate(pairs, start=1):
        print(f"\n{'=' * 70}")
        print(f"[{index}/{len(pairs)}] Verifying {pair.get('title')} ({pair['pair_key']})")
        print(f"{'=' * 70}")

        verify_pair(client, pair, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
