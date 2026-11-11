# InfoGap for Standard German and Alemannic
This project examines which facts occur in content-equivalent Wikipedia
articles in Standard German (`de`) and Alemannic (`als`), and which can only
be found on one of the two sides.

The evaluation uses the stored one-shot fact decompositions.
In addition, this project contains the few-shot comparison, the human
annotations, the inter-annotator agreement computations, the comparison
between the system and the gold standard, as well as the generated tables
and figures.

## Background

The method is an adaptation of InfoGap:
> Farhan Samir, Chan Young Park, Anjalie Field, Vered Shwartz, and Yulia
> Tsvetkov. 2024. “Locating Information Gaps and Narrative Inconsistencies
> Across Languages: A Case Study of LGBT People Portrayals on Wikipedia.” In
> *Proceedings of EMNLP 2024*, pp. 6747–6762. Association for Computational
> Linguistics. <https://doi.org/10.18653/v1/2024.emnlp-main.384>
Original repository: <https://github.com/smfsamir/infogap>

The scripts apply this method to the two language versions. The paper and the
original repository form the methodological basis for fact decomposition,
paragraph and fact alignment, hubness correction and fact matching. The
reference is also provided in `CITATION.bib`.

The code is an independent reimplementation of the method described in the
paper; some function names follow the original repository for comparability.

## Modes of execution

The pipeline can be run with stored or newly generated model data:

1. Offline evaluation of the stored results: The required model responses are
   included in the project. Aggregation, K1/K2, K3 as well as the tables and
   figures therefore do not require an API key.
2. Performing a new model run: Fact decomposition and X-FACTMATCH can be run
   with an AcademicCloud account. Even with `temperature=0`, external LLM
   calls may return slightly different responses.

LaBSE is loaded via `sentence-transformers` during the first actual alignment,
unless the model is already present in the local Hugging Face cache. A plain
`--help` call does not trigger a model download.

## Project structure

```text
gold_texts/                                fixed DE/ALS text versions
data/gold_clean/                           cleaned article inputs
configs/gold_article_pairs.jsonl           ten evaluated article pairs
files_out/als-de/                          complete DE/ALS raw data for pair selection (not included, see below)

outputs/goldstandard_oneshot/              one-shot fact decompositions of the main run
outputs/alignments_oneshot/                X-FACTALIGN candidates of the main run
outputs/verified_oneshot/                  X-FACTMATCH judgments of the main run
outputs/cache_oneshot/                     response cache of the main run

outputs/goldstandard_fewshot/              few-shot fact decompositions
outputs/alignments/                        few-shot X-FACTALIGN candidates
outputs/verified/                          few-shot X-FACTMATCH judgments
outputs/cache/                             few-shot response cache

outputs/annotationen_iaa.xlsx              human validation annotation
manual_paragraph_fact_annotation_max.xlsx  manual fact decomposition
outputs/manual_*                           artifacts of the manual control run

outputs/xfactmatch_aggregate_oneshot.*     one-shot overlap matrix
outputs/xfactmatch_aggregate_fewshot.*     few-shot comparison matrix
outputs/iaa_ergebnisse.csv                 K1/K2 results
outputs/kappa3_ergebnisse.csv              K3 results
outputs/visualizations_final_tables/       result tables
outputs/visualizations_final_overlap/      overlap figure

IAA_Guidelines.rtf                         annotation guidelines for the manual fact decomposition
```

The raw data in `files_out/` is not part of this repository because of its
size (about 1 GB). It is only needed to re-run the pair selection with
`select_article_pairs.py`. All later steps use the bundled texts in
`gold_texts/` and the derived files in `data/` and `configs/`.

## Installation

The package was tested with Python 3.14.4. For a clean environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The internal default paths are resolved relative to the project directory.
For the shell commands shown below, it is nevertheless clearest to start in
the project directory, because explicitly specified CLI paths remain relative
to the current terminal directory.

## Setting up the API key

No API key is needed for the offline evaluation of stored model data. Before
running a new fact decomposition or generating new X-FACTMATCH judgments,
create the environment file:

```bash
cp .env.example .env
```

Then enter either a single key as `API_KEY` or several keys as `API_KEY_1`,
`API_KEY_2` and `API_KEY_3` in `.env`. The file is always looked up in the
project directory, regardless of the working directory from which a script is
started.

The configured OpenAI-compatible endpoint is
`https://chat-ai.academiccloud.de/v1`, and the default model is
`gemma-4-31b-it`.

## Running from an IDE

For the offline evaluation, a single entry point is sufficient after
installation: open `run_reproduction.py` in the IDE and select "Run". The
script generates both overlap aggregates, K1/K2, K3 as well as the tables and
figures. Since it uses stored model data, no API key is needed.

The evaluation scripts also have project-relative default paths and can be
started directly via "Run". Command-line arguments remain available if other
inputs or output targets are to be used. The bundled gold dataset is the
default; the environment variable `INFOGAP_INPUT_MODE=gold` no longer needs to
be set for this. Existing pipeline results are skipped unless `--overwrite`
is passed.

The scripts have different tasks and, for a new run, must be started in this
order:

1. `prepare_articles.py`
2. `decompose_facts.py` – requires the AcademicCloud key
3. `run_alignment.py` – requires LaBSE, but no API key
4. `run_xfactmatch.py` – requires the AcademicCloud key
5. `aggregate_xfactmatch.py`

When started without arguments, the two manual scripts (manual validation)
automatically select the three pairs that were actually annotated: Eichener
See, Ernst Tappolet and Kalender. This prevents them from accidentally running
on the seven article pairs without a manual fact decomposition.

## Generating the result files

### 1. One-shot and few-shot aggregates

When started directly, `aggregate_xfactmatch.py` processes
`outputs/verified/` by default and writes the few-shot aggregate. To select
each of the two variants explicitly, use the following calls:

```bash
python aggregate_xfactmatch.py \
  --dir outputs/verified_oneshot \
  --csv outputs/xfactmatch_aggregate_oneshot.csv \
  --xlsx outputs/xfactmatch_aggregate_oneshot.xlsx

python aggregate_xfactmatch.py \
  --dir outputs/verified \
  --csv outputs/xfactmatch_aggregate_fewshot.csv \
  --xlsx outputs/xfactmatch_aggregate_fewshot.xlsx
```

### 2. K1 and K2

Without arguments, the script uses `outputs/annotationen_iaa.xlsx` and
writes `outputs/iaa_ergebnisse.csv`. The same run with explicit paths:

```bash
python iaa_kappa.py \
  --xlsx outputs/annotationen_iaa.xlsx \
  --csv outputs/iaa_ergebnisse.csv
```

Bootstrap percentile intervals with 10,000 resamples and seed 42 are used.
`Manual_Gold` contains 299 pairwise complete items, `Gemma_Gold` contains 300.

### 3. K3

Without arguments, the script uses the annotation file and the one-shot
verifications from `outputs/`. The same run with explicit paths:

```bash
python kappa3_system_vs_gold.py \
  --xlsx outputs/annotationen_iaa.xlsx \
  --json 'outputs/verified_oneshot/verified_gold__*.json' \
  --csv outputs/kappa3_ergebnisse.csv
```

K3 uses the 282 items on which both annotators agree. For the comparison, the
three human categories are binarized as `shared + partial -> yes` and
`only -> no`.

### 4. Figures and tables

```bash
python visualize_final_overlap_matrix.py
python visualize_final_result_tables.py
```

The scripts write PNG, SVG and CSV files, plus a PDF file for the overlap
figure, to `outputs/visualizations_final_overlap/` and
`outputs/visualizations_final_tables/` respectively.

### 5. Further evaluation scripts

Without arguments, the fact decomposition statistics use the manual
annotation and the one-shot aggregate:

```bash
python decomposition_stats.py
```

Different paths and a CSV target can be specified:

```bash
python decomposition_stats.py \
  --manual manual_paragraph_fact_annotation_max.xlsx \
  --aggregate outputs/xfactmatch_aggregate_oneshot.xlsx \
  --csv outputs/decomposition_stats.csv
```

Transferring the one-shot facts into the manual Excel schema also works
without arguments:

```bash
python gemma_json_to_manual_xlsx.py
```

The corresponding paths can be set explicitly if needed:

```bash
python gemma_json_to_manual_xlsx.py \
  --gemma-dir outputs/goldstandard_oneshot \
  --manual manual_paragraph_fact_annotation_max.xlsx \
  --out outputs/goldstandard_oneshot_formatiert.xlsx
```

## Generating the cleaned gold inputs

The text versions used are located in `gold_texts/`. The following call
creates the cleaned JSONL files and the pair configuration from them:

```bash
python prepare_articles.py
```

This writes `data/gold_clean/gold_de.clean.jsonl`,
`data/gold_clean/gold_als.clean.jsonl` and
`configs/gold_article_pairs.jsonl`.

## Writing the few-shot pipeline to new directories

This variant does not overwrite any bundled artifacts:

```bash
export INFOGAP_GOLD_DIRNAME=new_fewshot_facts
export INFOGAP_ALIGNMENT_DIRNAME=new_fewshot_alignments
export INFOGAP_VERIFIED_DIRNAME=new_fewshot_verified
export INFOGAP_CACHE_DIRNAME=new_fewshot_cache
export DECOMP_PROMPT_VERSION=iaa_atomic_balanced_fewshot_v2

python decompose_facts.py
python run_alignment.py
python run_xfactmatch.py
python aggregate_xfactmatch.py \
  --dir outputs/new_fewshot_verified \
  --csv outputs/new_fewshot_aggregate.csv
```

`decompose_facts.py` and `run_xfactmatch.py` require API access.
`run_alignment.py` requires LaBSE, but no API key.

## Computing X-FACTALIGN from the fixed one-shot facts

```bash
export INFOGAP_GOLD_DIRNAME=goldstandard_oneshot
export INFOGAP_ALIGNMENT_DIRNAME=new_oneshot_alignments
python run_alignment.py
```

This computes the alignment stage without overwriting the bundled alignment
files.

## Manual control run (manual validation)

In each direction, the control run considers the first 50 facts of the three
manually decomposed article pairs. The respective target article remains
complete. A complete alignment run can be started as follows:

```bash
python manual_control_align_first50.py \
  --source-limit-per-direction 50 --overwrite
```

Alternatively, a single pair can be selected:

```bash
python manual_control_align_first50.py \
  --pair-key gold__eichener_see \
  --source-limit-per-direction 50 --overwrite
```

The subsequent verification requires API access:

```bash
python manual_control_xfactmatch_first50.py
```

Existing verified results can be aggregated without API access:

```bash
python manual_control_aggregate.py
```

## Main scripts

| File | Purpose |
|---|---|
| `run_reproduction.py` | Runs the complete offline evaluation in a single start. |
| `prepare_articles.py` | Prepares the fixed DE/ALS gold texts. |
| `decompose_facts.py`, `gpt_query.py` | Decompose articles into facts via LLM and contain the prompts. |
| `align_facts.py`, `align_paragraphs.py`, `run_alignment.py` | Run X-FACTALIGN with LaBSE. |
| `run_xfactmatch.py` | Verifies fact candidates via LLM. |
| `aggregate_xfactmatch.py` | Aggregates the directed overlap. |
| `iaa_kappa.py` | Computes K1/K2 between the annotators. |
| `kappa3_system_vs_gold.py` | Computes K3 between the system and the agreement gold standard. |
| `manual_control_*.py` | Contain the robustness analysis with manual facts. |
| `visualize_final_*.py` | Generate the result tables and the overlap figure. |

`select_article_pairs.py` belongs to the preparatory selection from the
complete raw data. Without arguments, it reads the three files from
`files_out/als-de/`:

```bash
python select_article_pairs.py
```

If the files are stored elsewhere, the input paths can be specified:

```bash
python select_article_pairs.py \
  --mapping /path/to/als-de_mapping.jsonl \
  --als /path/to/data_als.jsonl \
  --de /path/to/data_de_subset-als.jsonl
```

## License and attribution

The source code (all `.py` files) is licensed under the MIT License, see
`LICENSE`.

The article texts in `gold_texts/` and all data derived from them (`data/`,
`configs/`, `outputs/` and the Excel annotation files) are based on articles
from the German Wikipedia (de.wikipedia.org) and the Alemannic Wikipedia
(als.wikipedia.org). They are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The authors
of the original articles are listed in the version history of each article.
The texts were cleaned and segmented for this project (see
`prepare_articles.py`).

| Article | German Wikipedia | Alemannic Wikipedia |
|---|---|---|
| Eichener See | [de](https://de.wikipedia.org/wiki?curid=1222039) | [als](https://als.wikipedia.org/wiki?curid=2271) |
| Ernst Tappolet | [de](https://de.wikipedia.org/wiki?curid=5683206) | [als](https://als.wikipedia.org/wiki?curid=77042) |
| Ethnie | [de](https://de.wikipedia.org/wiki?curid=297012) | [als](https://als.wikipedia.org/wiki?curid=23392) |
| Kalender | [de](https://de.wikipedia.org/wiki?curid=2838) | [als](https://als.wikipedia.org/wiki?curid=53949) |
| Le Mans | [de](https://de.wikipedia.org/wiki?curid=121313) | [als](https://als.wikipedia.org/wiki?curid=63185) |
| Leo der Große | [de](https://de.wikipedia.org/wiki?curid=56464) | [als](https://als.wikipedia.org/wiki?curid=75317) |
| Notschrei | [de](https://de.wikipedia.org/wiki?curid=2253006) | [als](https://als.wikipedia.org/wiki?curid=69110) |
| Oliver Kahn | [de](https://de.wikipedia.org/wiki?curid=100112) | [als](https://als.wikipedia.org/wiki?curid=81700) |
| Rheinbrücke Schaffhausen–Feuerthalen | [de](https://de.wikipedia.org/wiki?curid=5666380) | [als](https://als.wikipedia.org/wiki?curid=57553) |
| Wahrscheinlichkeit | [de](https://de.wikipedia.org/wiki?curid=9000) | [als](https://als.wikipedia.org/wiki?curid=34355) |

The human annotations in `outputs/annotationen_iaa.xlsx` were produced by two
annotators, referred to as A and B.
