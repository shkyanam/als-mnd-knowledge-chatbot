"""Run the fixed retrieval task-completion smoke test.

The benchmark deliberately measures evidence retrieval rather than model prose:
the task is complete only when the top five excerpts contain the source and the
three values needed for the seeded SNR answer.

Usage:
    uv run python evaluate_retrieval.py
    uv run python evaluate_retrieval.py --mode hybrid
"""

import argparse
from dataclasses import dataclass

from rag_app import (
    DENSE_CANDIDATE_COUNT,
    RETRIEVAL_TOP_K,
    hybrid_search,
    vector_store,
)


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    question: str
    expected_source: str
    required_terms: tuple[str, ...]


CASES = (
    EvaluationCase(
        name="active-period SNR trajectory",
        question=(
            "In the ALS speech BCI study, how did active-period spectral "
            "signal-to-noise ratio (SNR) change over time?"
        ),
        expected_source="PMC13042181.xml",
        required_terms=("1 dB", "6 dB", "3 dB"),
    ),
)


def contains_expected_evidence(document, case: EvaluationCase) -> bool:
    """Return whether one excerpt is sufficient for the seeded task."""
    if document.metadata.get("source") != case.expected_source:
        return False
    content = document.page_content.lower()
    return all(term.lower() in content for term in case.required_terms)


def dense_search(question: str):
    """Return dense results, retaining enough ranks to report the baseline."""
    return [
        document
        for document, _score in vector_store().similarity_search_with_relevance_scores(
            question, k=DENSE_CANDIDATE_COUNT
        )
    ]


def evaluate(mode: str) -> None:
    completed = 0
    for case in CASES:
        if mode == "dense":
            ranked_documents = dense_search(case.question)
        else:
            ranked_documents = hybrid_search(case.question, k=RETRIEVAL_TOP_K)

        match_rank = next(
            (
                rank
                for rank, document in enumerate(ranked_documents, start=1)
                if contains_expected_evidence(document, case)
            ),
            None,
        )
        top_k_completed = match_rank is not None and match_rank <= RETRIEVAL_TOP_K
        completed += int(top_k_completed)
        rank_text = str(match_rank) if match_rank is not None else "not found"
        result_text = "PASS" if top_k_completed else "FAIL"
        print(f"{case.name}: {result_text} (evidence rank: {rank_text})")

    percentage = completed / len(CASES) * 100
    print(
        f"{mode} task completion: {completed}/{len(CASES)} "
        f"({percentage:.1f}%) at top-{RETRIEVAL_TOP_K}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "both"],
        default="both",
        help="Evaluate the baseline, the deployed retriever, or both.",
    )
    args = parser.parse_args()

    modes = ["dense", "hybrid"] if args.mode == "both" else [args.mode]
    for index, mode in enumerate(modes):
        if index:
            print()
        evaluate(mode)


if __name__ == "__main__":
    main()
