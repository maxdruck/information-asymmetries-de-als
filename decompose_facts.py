"""
Decomposes the cleaned DE/ALS articles into individual facts.

The script reads the configured article pairs and writes one fact file per
language and pair:

  outputs/goldstandard/facts_als_<pair_key>.json
  outputs/goldstandard/facts_de_<pair_key>.json

For the gold run:
  - The facts are produced with the decomposition prompt from ``gpt_query.py``.
  - The temperature is 0.
  - The model response must be parseable as a Python or JSON list.
  - Successfully decomposed paragraphs are cached by ``PROMPT_VERSION``.
  - The output file records the prompt version used.
  - On API or parsing errors, no substitute sentence facts are generated.
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from gpt_query import PROMPT_VERSION, ask_gpt_for_facts, load_kisski_client
from pipeline_config import (
    ALS_CLEAN_PATH,
    CACHE_DIR,
    DE_CLEAN_PATH,
    DECOMPOSITION_MODEL,
    ENV_PATH,
    GOLD_DIR,
    facts_path,
    load_jsonl_dict,
    load_pairs,
    select_pairs,
)

MODEL = DECOMPOSITION_MODEL
MAX_RETRIES = 5
RETRY_DELAY = 10
PARSE_RETRY_DELAY = 5


# The cache prevents repeated API calls for paragraphs that were already decomposed successfully
def load_fact_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    return {}


def write_fact_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    return cleaned


def _extract_bracketed(text: str) -> str | None:
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return None


# Model responses may be JSON or Python lists and are parsed defensively
def parse_fact_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None

    cleaned = _strip_code_fences(raw)

    for candidate in (cleaned, _extract_bracketed(cleaned)):
        if not candidate:
            continue

        for parser in (ast.literal_eval, json.loads):
            try:
                value = parser(candidate)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue

            if isinstance(value, list):
                return [
                    str(item).strip()
                    for item in value
                    if str(item).strip()
                ]

    return None


def decompose_paragraph(
    client,
    paragraph: str,
    language_variety: str,
) -> list[str]:
    """Decompose a paragraph into individual facts.

    Technical API errors or unparseable model responses terminate the
    gold run instead of falling back to whole sentences as facts. Paragraphs
    that were already processed successfully remain in the cache.
    """
    last_raw_response = None

    # Multiple attempts mainly catch incompletely formatted model responses
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                print(f"      Retry attempt {attempt}/{MAX_RETRIES}...")

            content, _tokens = ask_gpt_for_facts(
                client,
                MODEL,
                paragraph,
                language_variety,
            )

            last_raw_response = content
            facts = parse_fact_list(content)

            if facts is not None:
                return facts

            print(
                f"      WARNING: could not parse a list "
                f"(attempt {attempt}/{MAX_RETRIES})."
            )
            print(f"      Raw response: {(content or '')[:500]}")

            if attempt < MAX_RETRIES:
                print(
                    f"      Waiting {PARSE_RETRY_DELAY} seconds before retry..."
                )
                time.sleep(PARSE_RETRY_DELAY)
                continue

            raise RuntimeError(
                "The model response could not be parsed as a Python list "
                f"after {MAX_RETRIES} attempts. No sentence fallback was written. "
                f"Last raw response: {(last_raw_response or '')[:1000]}"
            )

        except RateLimitError as error:
            raise RuntimeError(
                "Rate limit reached during fact decomposition. "
                "No sentence fallback was written. Restart the script later; "
                "already completed paragraphs remain cached."
            ) from error

        except BadRequestError as error:
            raise RuntimeError(
                "Bad request during fact decomposition. "
                "No sentence fallback was written. Check model name, prompt size, "
                "request parameters, and API configuration."
            ) from error

        except InternalServerError as error:
            wait = min(RETRY_DELAY * (2 ** (attempt - 1)), 120)
            print(f"      Server error (attempt {attempt}/{MAX_RETRIES}): {error}")

            if attempt < MAX_RETRIES:
                print(f"      Waiting {wait} seconds before retry...")
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"Server error after {MAX_RETRIES} attempts. "
                "No sentence fallback was written. Restart the script later; "
                "already completed paragraphs remain cached."
            ) from error

        except (NotFoundError, AuthenticationError, PermissionDeniedError) as error:
            raise RuntimeError(
                f"Configuration error from the API ({type(error).__name__}): "
                f"{error}. Check MODEL ('{MODEL}') and the API keys in .env."
            ) from error

        except Exception as error:
            raise RuntimeError(
                "Fact decomposition failed before a valid model response was parsed. "
                "No sentence fallback was written. Check the API client, model name, "
                "prompt code, and .env configuration."
            ) from error

    raise RuntimeError(
        "Reached the end of the retry loop without a valid decomposition. "
        "No sentence fallback was written."
    )


def process_article(
    client,
    article: dict,
    language_variety: str,
    output_path: Path,
    pair: dict,
    overwrite: bool = False,
) -> None:
    # Each paragraph is decomposed independently so that progress can be cached at a fine-grained level
    if output_path.exists() and not overwrite:
        print(f"  Already exists, skipping: {output_path}")
        return

    paragraphs = article.get("paragraphs", [])

    if not paragraphs:
        print(f"  WARNING: No content found for {article.get('page_title')}")
        return

    word_count = sum(len(paragraph.split()) for paragraph in paragraphs)
    print(f"\n  {len(paragraphs)} paragraphs, {word_count} words total")

    cache_file = CACHE_DIR / f"{output_path.stem}_{PROMPT_VERSION}_cache.json"
    fact_cache = load_fact_cache(cache_file)

    all_facts: list[dict] = []
    fact_index = 0
    num_cached = 0
    num_fallback_paragraphs = 0

    for paragraph_index, paragraph in enumerate(paragraphs):
        preview = paragraph[:160] + ("..." if len(paragraph) > 160 else "")
        print(f"\n  [paragraph {paragraph_index + 1}/{len(paragraphs)}]")
        print(f"    {preview}")

        if paragraph in fact_cache:
            facts = fact_cache[paragraph]
            num_cached += 1
            print("      (cached)")
        else:
            facts = decompose_paragraph(
                client=client,
                paragraph=paragraph,
                language_variety=language_variety,
            )
            fact_cache[paragraph] = facts
            write_fact_cache(cache_file, fact_cache)

        for fact in facts:
            all_facts.append({
                "fact_index": fact_index,
                "paragraph_index": paragraph_index,
                "fact": fact,
                "fallback": False,
            })

            print(f"      {fact_index}. {fact}")
            fact_index += 1

    result = {
        "pair_key": pair["pair_key"],
        "cluster": pair.get("cluster"),
        "als_page_id": pair.get("als_page_id"),
        "de_page_id": pair.get("de_page_id"),
        "pair_title": pair.get("title"),
        "page_id": article.get("page_id"),
        "page_title": article.get("page_title"),
        "language_variety": language_variety,
        "decomposition_model": MODEL,
        "decomposition_prompt_version": PROMPT_VERSION,
        "num_paragraphs": len(paragraphs),
        "num_facts": len(all_facts),
        "num_fallback_paragraphs": num_fallback_paragraphs,
        "paragraphs": paragraphs,
        "facts": all_facts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"\n  Saved {len(all_facts)} facts to {output_path} "
        f"({num_cached} cached, {num_fallback_paragraphs} sentence fallback)"
    )


# The entry point selects article pairs, initializes the client and processes both languages
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster",
        help="Process only one cluster, e.g. Biografie, Geschichte, Gesellschaft.",
    )
    parser.add_argument(
        "--pair-key",
        help="Process one exact pair_key from configs/article_pairs.jsonl.",
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

    print("Starting INFOGAP FINAL fact decomposition pipeline...")

    if not ALS_CLEAN_PATH.exists() or not DE_CLEAN_PATH.exists():
        raise FileNotFoundError(
            "Cleaned input not found. Run prepare_articles.py first to create "
            f"{ALS_CLEAN_PATH} and {DE_CLEAN_PATH}."
        )

    pairs = select_pairs(
        load_pairs(),
        cluster=args.cluster,
        pair_key=args.pair_key,
        limit=args.limit,
        start_index=args.start_index,
    )

    if not pairs:
        raise ValueError(
            "No article pairs selected. "
            "Check --cluster/--pair-key or configs/article_pairs.jsonl."
        )

    print(
        f"Processing {len(pairs)} article pairs "
        f"({len(pairs) * 2} total articles)\n"
    )

    load_dotenv(ENV_PATH)
    client = load_kisski_client()

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    als_articles = load_jsonl_dict(ALS_CLEAN_PATH)
    de_articles = load_jsonl_dict(DE_CLEAN_PATH)

    for index, pair in enumerate(pairs, start=1):
        pair_key = pair["pair_key"]
        title = pair.get("title", pair_key)

        print(f"\n{'=' * 70}")
        print(f"[{index}/{len(pairs)}] {title}  ({pair_key})")
        print(f"Cluster: {pair.get('cluster')}")
        print(f"{'=' * 70}")

        als_id = str(pair["als_page_id"])
        de_id = str(pair["de_page_id"])

        als_article = als_articles.get(als_id)
        de_article = de_articles.get(de_id)

        if als_article is None or de_article is None:
            print(
                f"  WARNING: missing cleaned article for "
                f"ALS={als_id} or DE={de_id}; skipping"
            )
            continue

        print(f"\n[ALS] {als_article.get('page_title')} (page_id={als_id})")
        process_article(
            client=client,
            article=als_article,
            language_variety="als",
            output_path=facts_path("als", pair_key),
            pair=pair,
            overwrite=args.overwrite,
        )

        print(f"\n[DE] {de_article.get('page_title')} (page_id={de_id})")
        process_article(
            client=client,
            article=de_article,
            language_variety="de",
            output_path=facts_path("de", pair_key),
            pair=pair,
            overwrite=args.overwrite,
        )

    print(f"\n{'=' * 70}")
    print(f"Done. Fact outputs saved to {GOLD_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
