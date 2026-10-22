"""Build the generalisation corpus used for model choice and threshold calibration.

The supplied corpora cannot rank classifiers: their messages fall into two
sentence-sharing components that are one class each, so every internal resampling
scheme trains on one label and tests on the other. This module supplies a corpus
whose label is a function of declared communicative slots rather than of any single
surface pattern.

Run directly to regenerate data/generalisation_corpus.csv:

    python scripts/build_generalisation_corpus.py

Construction rules, frozen before any model is scored:

* No factor level may determine the label on its own.
* Every label is recoverable from the wording, because the rule reads only the three
  stated factors.
* Sentences may not be reused from the supplied corpora.
* Class counts stay within tolerance.
* No duplicate messages.

Destination is defined by the structural marks a domain actually shows rather than by
an invented institutional name, because a bag-of-words model cannot learn which
made-up host names the author decided were legitimate. Half the external hosts are
deliberately unremarkable so the corpus does not become a mirror of the feature list.

Corpus revisions are permitted only to satisfy these gates or to correct a declared
construct flaw, never to move an arm's score.
"""

from __future__ import annotations

import csv
import itertools
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "generalisation_corpus.csv"
SUPPLIED_CORPORA = (
    "dataset_classroom.csv",
    "dataset_homework.csv",
    "dataset_large.csv",
)

# --------------------------------------------------------------- label rule
# Three indicators, each weak on its own:
#   A  a sensitive request     credentials or money movement are asked for
#   B  an external destination anything that is not the recipient's own channel
#   C  deadline framing        urgency or loss is attached to the request
#
#   phishing  <=>  at least two of A, B and C hold
RULE = (
    "phishing iff at least two of {sensitive request, external destination, "
    "deadline framing} hold"
)
SEED = 20240905
PHRASINGS_PER_CELL = 6

DESTINATIONS = ("institutional", "external")

INSTITUTIONAL_LINKS = [
    "the staff portal at https://portal.northvale-university.ac.uk/login",
    "the finance workspace at https://finance.northvale-university.ac.uk/claims",
    "the service desk form at https://helpdesk.northvale-university.ac.uk/ticket",
    "the registry page at https://registry.northvale-university.ac.uk/record",
    "the payroll screen at https://payroll.northvale-university.ac.uk/profile",
    "the identity page at https://id.northvale-university.ac.uk/check",
]
# The flagged half carries the structural marks a hostile domain really shows, so URL
# evidence can reach it. The clean half is an ordinary-looking external domain, which
# keeps the hard part of the task and stops the corpus testing the feature list by name.
# Links are written as real URLs, because a bare domain in prose is invisible to URL
# extraction and would silently zero every destination feature.
EXTERNAL_FLAGGED_LINKS = [
    "the confirmation page at https://northvale-verify-account.click/confirm",
    "the secure form at http://198.51.100.77/handbook",
    "the review sheet at https://docs.northvale-verification.services/review",
    "the upload page at https://secure-portal-access.northvalepay.top/upload",
]
EXTERNAL_CLEAN_LINKS = [
    "the shared brief at https://files.northvale-partners.org/brief",
    "the document view at https://docs.northvalecollab.com/view",
    "the intake form at https://northvaleintake.net/intake",
    "the summary at https://records.northvaleconsulting.org/summary",
]
LINK_POOLS = {
    "institutional": INSTITUTIONAL_LINKS,
    "external": EXTERNAL_FLAGGED_LINKS + EXTERNAL_CLEAN_LINKS,
}
LINK_STRUCTURE = {
    "institutional": "institutional",
    "external": "flagged",
}

OPENERS = {
    "personal": [
        "Dear Dr Okonjo,",
        "Hello Priya,",
        "Good morning Tomas,",
        "Hi Marisol,",
        "Dear Aisha,",
        "Good afternoon Ben,",
    ],
    "impersonal": [
        "Dear colleague,",
        "To the member of staff concerned,",
        "Attention staff group,",
        "Hello there,",
        "Dear account holder,",
        "To whom it may concern,",
    ],
}

PREAMBLES = [
    "Following the registry update,",
    "As part of the termly record check,",
    "After the audit note circulated on Friday,",
    "Alongside the onboarding pack,",
    "While the archives are being moved,",
    "During the scheduled maintenance window,",
]

ACTION_LINES = {
    ("sensitive", "institutional"): [
        "Please sign in and confirm your staff password on {link}.",
        "We need you to re-enter your network credentials on {link}.",
        "Kindly set your payroll details on {link}.",
        "Could you approve the bank details listed on {link}?",
        "Please review the payment template saved on {link}.",
        "Would you confirm the security questions held on {link}?",
    ],
    ("sensitive", "external"): [
        "Please sign in and confirm your staff password on {link}.",
        "We need you to re-enter your network credentials on {link}.",
        "Kindly set your payroll details on {link}.",
        "Could you approve the bank details listed on {link}?",
        "Please review the payment template saved on {link}.",
        "Would you confirm the security questions held on {link}?",
    ],
    ("routine", "institutional"): [
        "Have a look at the room booking draft on {link}.",
        "The reading list for the induction sits on {link}.",
        "Comments on the timetable proposal are on {link}.",
        "Your seminar grouping is posted on {link}.",
        "The parking survey results are on {link}.",
        "Notes from the library committee are on {link}.",
    ],
    ("routine", "external"): [
        "Have a look at the room booking draft on {link}.",
        "The reading list for the induction sits on {link}.",
        "Comments on the timetable proposal are on {link}.",
        "Your seminar grouping is posted on {link}.",
        "The parking survey results are on {link}.",
        "Notes from the library committee are on {link}.",
    ],
}

CLOSERS = {
    "urgent": [
        "Please treat this as time critical; the window shuts this afternoon.",
        "This closes today, so reply before you leave work.",
        "We need this inside two hours to keep the record active.",
        "Action by end of day is required or the entry lapses.",
        "The record is suspended until this is done, so please act now.",
        "Access stops at five today unless the form is returned.",
    ],
    "neutral": [
        "Whenever you get a chance is fine.",
        "There is no rush with this.",
        "No need to hurry, the deadline is next month.",
        "Take your time over it.",
        "It can wait until the next committee meeting.",
        "Nothing is blocked while this is outstanding.",
    ],
}


def is_phishing(action: str, destination: str, framing: str) -> bool:
    indicators = (
        action == "sensitive",
        destination.startswith("external"),
        framing == "urgent",
    )
    return sum(indicators) >= 2


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    cells = list(itertools.product(("sensitive", "routine"), DESTINATIONS, ("urgent", "neutral")))
    for action, destination, framing in cells:
        phishing = is_phishing(action, destination, framing)
        links = LINK_POOLS[destination]
        lines = ACTION_LINES[(action, destination)]
        phrasings = random.Random(SEED).sample(
            list(itertools.product(range(6), repeat=5)), PHRASINGS_PER_CELL
        )
        for index, (line_i, preamble_i, opener_i, link_i, closer_i) in enumerate(phrasings):
            greeting = "personal" if index % 2 == 0 else "impersonal"
            if destination == "external":
                # alternate the hostile-structured and unremarkable external hosts
                pool = (
                    EXTERNAL_FLAGGED_LINKS
                    if index % 2 == 0
                    else EXTERNAL_CLEAN_LINKS
                )
                structure = "flagged" if index % 2 == 0 else "clean"
            else:
                pool = INSTITUTIONAL_LINKS
                structure = "institutional"
            text = " ".join(
                [
                    OPENERS[greeting][opener_i % len(OPENERS[greeting])],
                    PREAMBLES[preamble_i % len(PREAMBLES)],
                    lines[line_i % len(lines)].format(
                        link=pool[link_i % len(pool)]
                    ),
                    CLOSERS[framing][closer_i % len(CLOSERS[framing])],
                ]
            )
            rows.append(
                {
                    "text": text,
                    "label": "phishing" if phishing else "legitimate",
                    "cell": f"{action}+{destination}+{framing}+{greeting}",
                    "action": action,
                    "destination": destination,
                    "link_structure": structure,
                    "framing": framing,
                    "greeting": greeting,
                }
            )
    return rows


def sentences(text: str) -> set[str]:
    return {
        piece.strip().rstrip(".")
        for piece in text.replace(",", ". ").split(". ")
        if len(piece.strip()) > 20
    }


def gate(rows: list[dict[str, str]]) -> list[str]:
    """Check the pre-declared construction properties, return human-readable results."""
    failures: list[str] = []
    labels = [row["label"] for row in rows]
    counts = {label: labels.count(label) for label in set(labels)}
    if min(counts.values()) / max(counts.values()) < 0.8:
        failures.append(f"class balance outside tolerance: {counts}")
    if len({row["text"] for row in rows}) != len(rows):
        failures.append("duplicate messages present")

    supplied: set[str] = set()
    for name in SUPPLIED_CORPORA:
        path = ROOT / "data" / name
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for record in csv.DictReader(handle):
                    supplied |= sentences(record["text"])
    corpus_sentences = {s for row in rows for s in sentences(row["text"])}
    overlap = corpus_sentences & supplied
    if overlap:
        failures.append(
            f"{len(overlap)} sentences are reused from the supplied corpora, for example "
            f"{sorted(overlap)[:3]}"
        )

    for factor in ("action", "destination", "link_structure", "framing", "greeting"):
        levels = {row[factor] for row in rows}
        for level in levels:
            subset = [row["label"] for row in rows if row[factor] == level]
            share = subset.count("phishing") / max(len(subset), 1)
            if share in (0.0, 1.0):
                failures.append(
                    f"factor {factor}={level} alone determines the label ({share:.2f})"
                )

    cells = {
        (row["action"], row["destination"], row["framing"]): row["label"] for row in rows
    }
    expected = {
        combination: "phishing" if is_phishing(*combination) else "legitimate"
        for combination in itertools.product(
            ("sensitive", "routine"), DESTINATIONS, ("urgent", "neutral")
        )
    }
    if cells != expected:
        failures.append("stored labels do not follow the declared rule")
    return failures


def main() -> None:
    rows = build_rows()
    OUTPUT.parent.mkdir(exist_ok=True)
    fields = [
        "text",
        "label",
        "cell",
        "action",
        "destination",
        "link_structure",
        "framing",
        "greeting",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    labels = [row["label"] for row in rows]
    print("Rule:", RULE)
    print(f"Wrote {len(rows)} messages to {OUTPUT.name}")
    print(
        "  phishing %d | legitimate %d"
        % (labels.count("phishing"), labels.count("legitimate"))
    )
    for structure in ("institutional", "flagged", "clean"):
        part = [row for row in rows if row["link_structure"] == structure]
        print(
            f"  link structure {structure}: {len(part)} rows, "
            f"{sum(r['label'] == 'phishing' for r in part)} phishing"
        )
    failures = gate(rows)
    print("  gates:", "all passed" if not failures else "")
    for failure in failures:
        print("   FAIL:", failure)


if __name__ == "__main__":
    main()
