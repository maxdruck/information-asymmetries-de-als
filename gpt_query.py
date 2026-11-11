"""
Bundles the prompts and API access of the DE/ALS InfoGap pipeline.

The module contains the client with key rotation, the prompt for fact
decomposition and the prompt for X-FACTMATCH.
To try the zero-shot or one-shot prompt instead, the prompt below has to be replaced.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

from api_client import load_rotating_kisski_client
from pipeline_config import ENV_PATH


load_dotenv(ENV_PATH)


@dataclass
class FactParagraph:
    facts: List[str]

    def __len__(self):
        return len(self.facts)

    def __iter__(self):
        return iter(self.facts)


def load_kisski_client():
    # All model calls use the same client with API key rotation
    return load_rotating_kisski_client()


def create_chat_completion(client, **kwargs):
    if hasattr(client, "create_chat_completion"):
        return client.create_chat_completion(**kwargs)

    return client.chat.completions.create(**kwargs)


PROMPT_VERSION = os.getenv(
    "DECOMP_PROMPT_VERSION",
    "iaa_atomic_balanced_fewshot_v2",
).strip()

NORMALIZE_ALS_TO_STANDARD_GERMAN = (
    os.getenv("NORMALIZE_ALS_TO_STANDARD_GERMAN", "1").strip() != "0"
)


def _language_note(src_lang: str) -> str:
    if src_lang == "als" and NORMALIZE_ALS_TO_STANDARD_GERMAN:
        return (
            "Ausgangssprache: Alemannisch. "
            "Zielform der Fakten: Standarddeutsch. "
            "Die im Absatz gemeinte Information bleibt erhalten."
        )

    if src_lang == "als":
        return (
            "Ausgangssprache: Alemannisch. "
            "Zielform der Fakten: möglichst nah am alemannischen Ausgangstext."
        )

    if src_lang == "de":
        return (
            "Ausgangssprache: Standarddeutsch. "
            "Zielform der Fakten: Standarddeutsch."
        )

    raise ValueError(f"Invalid language: {src_lang}")


def construct_fact_decomp_prompt(src_lang: str, paragraph: str) -> str:
    # The language hint and decomposition rules are combined into a single prompt
    if src_lang not in {"de", "als"}:
        raise ValueError(f"Invalid language: {src_lang}")

    return f"""
Aufgabe:
Zerlegung eines Wikipedia-Absatzes in eigenständige Fakten für ein späteres Alignment.

{_language_note(src_lang)}

Ausgabeformat:
- ausschließlich eine Python-Liste von Strings
- keine Markdown-Formatierung
- keine Erklärung
- kein Codeblock
- jeder String ist ein eigenständiger Fakt

Granularität:
Die Zerlegung ist atomar, aber nicht maximal kleinteilig.
Ein Fakt soll genau eine inhaltlich eigenständige Aussage enthalten.
Informationen werden getrennt, wenn sie unabhängig voneinander in einer anderen Artikelversion vorhanden oder nicht vorhanden sein könnten.
Reine sprachliche Zusätze, Modifikatoren oder nähere Umstände werden nicht als eigene Fakten isoliert.

Regeln:
1. Eine eigenständige Aussage wird als eigener Fakt ausgegeben.
2. Unterschiedliche Ereignisse, Rollen, Funktionen, Ämter, Werke, Auszeichnungen, Mitgliedschaften, Stationen, Statistiken und biografische Angaben werden getrennt.
3. Mehrere Eigenschaften derselben Person oder Sache werden getrennt, wenn jede Eigenschaft eine eigene sachliche Information enthält.
4. Appositionen und Einschübe werden getrennt, wenn sie eine eigene sachliche Information enthalten.
5. Geburtsdatum und Geburtsort bleiben zusammen, wenn sie im selben Ausdruck genannt werden.
6. Sterbedatum und Sterbeort bleiben zusammen, wenn sie im selben Ausdruck genannt werden.
7. Datum, Ort, Art und Weise oder Wirksamkeit bleiben mit der zugehörigen Handlung zusammen, wenn sie nur diese Handlung näher bestimmen.
8. Reine Modifikatoren werden nicht isoliert. Beispiele: „mit sofortiger Wirkung“, „ehemalig“, „später“, „zunächst“, „erneut“, „damals“, „insgesamt“.
9. Koordinationen werden getrennt, wenn die koordinierten Teile unterschiedliche inhaltliche Aussagen bilden.
10. Koordinationen bleiben zusammen, wenn sie ein festes Begriffspaar oder eine gemeinsame Formulierung bilden. Beispiele: „Urteile und Aussagen“, „Daten und Zufall“, „Art und Weise“.
11. Aufzählungen werden getrennt, wenn die einzelnen Elemente eigenständige alignierbare Informationen sind, etwa verschiedene Titel, Vereine, Auszeichnungen, Positionen oder Funktionen.
12. Aufzählungen werden nicht getrennt, wenn sie nur Beispiele, Varianten oder unselbständige Ergänzungen derselben Aussage sind.
13. Bei Zahlen, Jahreszahlen, Statistiken und Rekorden wird die konkrete Aussage als eigener Fakt ausgegeben.
14. Pronomen und elliptische Bezüge werden nur dann aufgelöst, wenn der Bezug im Absatz eindeutig ist.
15. Kein Weltwissen, keine Ergänzungen und keine Interpretation über den Absatz hinaus.
16. Literaturangaben, URLs, ISBNs, DOI-Angaben, Seitenzahlen, Weblinks, Navigationshinweise und „Siehe auch“-Angaben werden nicht als Fakten übernommen.
17. Zahlen, Daten, Formeln, Namen und Fachbegriffe bleiben so genau wie möglich erhalten.
18. Die Formulierung bleibt möglichst nah am Inhalt des Absatzes.

Nicht erwünscht:
- keine isolierten Fakten nur aus Adverbialen oder Modifikatoren
- keine Trennung von Datum und Ort, wenn beide zur gleichen Handlung gehören
- keine künstliche Aufspaltung fester Begriffspaare
- keine Fakten, die nur aus Literatur- oder Quellenangaben bestehen
- keine Zusammenfassung mehrerer unterschiedlicher Rollen, Stationen oder Leistungen in einem einzigen Sammelfakt

Beispiel 1:

Absatz:
Ernst Emanuel Tappolet, in französischsprachigen Publikationen Ernest Tappolet, wurde am 21. Oktober 1870 in Hittnau geboren und starb am 2. November 1939 in Basel. Er war ein Schweizer Romanist.

Ausgabe:
[
  "Ernst Emanuel Tappolet wurde am 21. Oktober 1870 in Hittnau geboren.",
  "Ernst Emanuel Tappolet starb am 2. November 1939 in Basel.",
  "Ernst Emanuel Tappolet war ein Schweizer Romanist.",
  "Ernst Emanuel Tappolet wurde in französischsprachigen Publikationen Ernest Tappolet genannt."
]

Beispiel 2:

Absatz:
Zusammen mit Louis Gauchat und Jules Jeanjaquet gründete Tappolet 1899 das Glossaire des patois de la Suisse romande.

Ausgabe:
[
  "Ernst Tappolet gründete 1899 das Glossaire des patois de la Suisse romande.",
  "Louis Gauchat war an der Gründung des Glossaire des patois de la Suisse romande beteiligt.",
  "Jules Jeanjaquet war an der Gründung des Glossaire des patois de la Suisse romande beteiligt."
]

Beispiel 3:

Absatz:
D Woorschiinligkäit isch en Iistuefig vo Urdäil und Ussaage, vor allem Brognoose, noch em Grad vo dr Gwüsshäit. In dr Mathematik het sich mit dr Woorschiinligkäitstheorii en äiges Fachgebiet entwicklet.

Ausgabe:
[
  "Wahrscheinlichkeit ist eine Einstufung von Urteilen und Aussagen nach dem Grad der Gewissheit.",
  "Wahrscheinlichkeit betrifft vor allem Prognosen.",
  "In der Mathematik entwickelte sich mit der Wahrscheinlichkeitstheorie ein eigenes Fachgebiet."
]

Beispiel 4:

Absatz:
Bei einem idealen, fairen Würfel wird kein Ergebnis durch unsymmetrische Massenverteilung oder Ähnliches bevorzugt; wegen der Symmetrie hat jede der sechs Seiten die gleiche Chance, nach dem Wurf oben zu liegen.

Ausgabe:
[
  "Bei einem idealen, fairen Würfel wird kein Ergebnis durch unsymmetrische Massenverteilung oder Ähnliches bevorzugt.",
  "Bei einem idealen, fairen Würfel hat jede der sechs Seiten wegen der Symmetrie die gleiche Chance, nach dem Wurf oben zu liegen."
]

Beispiel 5:

Absatz:
Oliver Kahn ist der jüngere Sohn des in Lettland geborenen Deutsch-Balten und ehemaligen Karlsruher Bundesliga-Fußballspielers Rolf Kahn.

Ausgabe:
[
  "Oliver Kahn ist der jüngere Sohn von Rolf Kahn.",
  "Rolf Kahn wurde in Lettland geboren.",
  "Rolf Kahn war Deutsch-Balte.",
  "Rolf Kahn war ehemaliger Karlsruher Bundesliga-Fußballspieler."
]

Beispiel 6:

Absatz:
Am 27. Mai 2023 wurde Kahn mit sofortiger Wirkung von seinen Aufgaben als Vorstandsvorsitzender entbunden.

Ausgabe:
[
  "Oliver Kahn wurde am 27. Mai 2023 mit sofortiger Wirkung von seinen Aufgaben als Vorstandsvorsitzender entbunden."
]

Beispiel 7:

Absatz:
Kahn gewann mit Bayern München achtmal die deutsche Meisterschaft, sechsmal den DFB-Pokal, 1996 den UEFA-Pokal, 2001 die Champions League und 2001 den Weltpokal.

Ausgabe:
[
  "Oliver Kahn gewann mit Bayern München achtmal die deutsche Meisterschaft.",
  "Oliver Kahn gewann mit Bayern München sechsmal den DFB-Pokal.",
  "Oliver Kahn gewann mit Bayern München 1996 den UEFA-Pokal.",
  "Oliver Kahn gewann mit Bayern München 2001 die Champions League.",
  "Oliver Kahn gewann mit Bayern München 2001 den Weltpokal."
]

Beispiel 8:

Absatz:
Er wurde 1999, 2001 und 2002 zum Welttorhüter des Jahres gewählt und erhielt 2002 den Goldenen Ball der Weltmeisterschaft.

Ausgabe:
[
  "Oliver Kahn wurde 1999 zum Welttorhüter des Jahres gewählt.",
  "Oliver Kahn wurde 2001 zum Welttorhüter des Jahres gewählt.",
  "Oliver Kahn wurde 2002 zum Welttorhüter des Jahres gewählt.",
  "Oliver Kahn erhielt 2002 den Goldenen Ball der Weltmeisterschaft."
]

Beispiel 9:

Absatz:
Die Stochastik ist die Mathematik des ZufallsUlrich Krengel: Einführung in die Wahrscheinlichkeitstheorie und Statistik. Vieweg Verlag, Braunschweig 1991, ISBN 3-528-27259-7, S. V. oder die Mathematik der Daten und des Zufalls.

Ausgabe:
[
  "Die Stochastik ist die Mathematik des Zufalls.",
  "Die Stochastik ist die Mathematik der Daten und des Zufalls."
]

Zu zerlegender Absatz:
{paragraph}
""".strip()


def ask_gpt_for_facts(
    client,
    model_name: str,
    paragraph: str,
    lang_code: str,
):
    # Keeps model parameters and the return format in one place
    if lang_code not in {"de", "als"}:
        raise ValueError(f"Invalid language: {lang_code}")

    message = [{
        "role": "user",
        "content": construct_fact_decomp_prompt(lang_code, paragraph),
    }]

    response = create_chat_completion(
        client,
        model=model_name,
        temperature=0,
        max_tokens=max(800, min(4000, len(paragraph) + 500)),
        messages=message,
    )

    response_content = response.choices[0].message.content
    response_total_tokens = response.usage.total_tokens if response.usage else None

    return response_content, response_total_tokens


def _article_version_label(lang: str) -> str:
    if lang == "de":
        return "standarddeutschen Wikipedia-Version"

    if lang == "als":
        if NORMALIZE_ALS_TO_STANDARD_GERMAN:
            return "alemannischen Wikipedia-Version, in Standarddeutsch normalisiert"
        return "alemannischen Wikipedia-Version"

    raise ValueError(f"Invalid language: {lang}")


def construct_fact_intersection_prompt(
    article_title: str,
    src_facts: List[str],
    tgt_fact_contexts: List[List[str]],
    src_lang: str,
    tgt_lang: str,
) -> str:
    # The source fact and candidate contexts are formatted for a binary entailment judgment
    if src_lang not in {"de", "als"}:
        raise ValueError(f"Invalid source language: {src_lang}")

    if tgt_lang not in {"de", "als"}:
        raise ValueError(f"Invalid target language: {tgt_lang}")

    if not src_facts:
        raise ValueError("src_facts must not be empty")

    src_name = _article_version_label(src_lang)
    tgt_name = _article_version_label(tgt_lang)

    return (
        f"Artikelthema: {article_title}\n\n"
        f"Fakten aus der {src_name}:\n"
        f"{repr(src_facts)}\n\n"
        "Prüffakt:\n"
        "Der letzte Fakt der vorherigen Liste.\n\n"
        f"Kandidatenkontexte aus der {tgt_name}:\n"
        f"{repr(tgt_fact_contexts)}\n\n"
        "Entscheidungsfrage:\n"
        "Ist die wesentliche Information des Prüffakts in mindestens einem "
        "Kandidatenkontext enthalten?\n\n"
        "Entscheidungsregel:\n"
        "Die Antwort ist yes, wenn die wesentliche Information des Prüffakts "
        "durch mindestens einen Kandidatenkontext vermittelt wird. "
        "Unwichtige ausgelassene Einzelheiten verhindern keine Übereinstimmung. "
        "Die Antwort ist no, wenn die wesentliche Information in keinem "
        "Kandidatenkontext enthalten ist.\n\n"
        "Zulässige Ausgabe: yes oder no."
    )


def ask_gpt_about_fact_intersection(
    client,
    model_name: str,
    article_title: str,
    src_facts: List[str],
    tgt_fact_contexts: List[List[str]],
    src_lang: str,
    tgt_lang: str,
):
    # X-FACTMATCH uses temperature 0 and returns the response text plus the token count
    prompt = construct_fact_intersection_prompt(
        article_title=article_title,
        src_facts=src_facts,
        tgt_fact_contexts=tgt_fact_contexts,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    response = create_chat_completion(
        client,
        model=model_name,
        temperature=0,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )

    response_content = response.choices[0].message.content
    response_total_tokens = response.usage.total_tokens if response.usage else None

    return response_content, response_total_tokens
