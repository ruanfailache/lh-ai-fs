"""Eval harness for the BS Detector pipeline.

Runs the real pipeline (real OpenAI calls -- this measures actual model quality, unlike the
free unit tests in tests/, which use FakeStructuredLLM) against the fixture documents and
scores it against a hand-curated golden set (evals/golden_set.json).

Usage:
    cd backend && python run_evals.py
"""

import json
from pathlib import Path

from document_loader import load_documents
from llm import OpenAIStructuredLLM
from models import AnalysisReport
from pipeline import run_pipeline

EVALS_DIR = Path(__file__).parent / "evals"
GOLDEN_SET_PATH = EVALS_DIR / "golden_set.json"
RESULTS_PATH = EVALS_DIR / "last_run_results.json"


def find_golden_match(golden_entries: list[dict], key_field: str, haystack: str) -> dict | None:
    haystack_lower = haystack.lower()
    for entry in golden_entries:
        if entry[key_field].lower() in haystack_lower:
            return entry
    return None


def score_category(produced_items: list, verdicts: list, golden_entries: list[dict], key_field: str, haystack_field: str, match_field: str) -> dict:
    """Match produced items+verdicts against the golden set and tally TP/FP/FN.

    - TP: golden entry matched, expected_flagged=True, predicted flagged=True
    - FN: golden entry matched but predicted flagged=False, OR golden entry never matched at all
    - FP: golden entry matched, expected_flagged=False, predicted flagged=True
    Unmatched produced items (no golden entry found) are not scored here -- they're handled by
    the hallucination check instead, since there's nothing to compare them against.
    """
    verdict_by_match_id = {getattr(v, match_field): v for v in verdicts}
    matched_golden_ids = set()
    tp = fp = 0
    details = []

    for item in produced_items:
        haystack = getattr(item, haystack_field)
        golden = find_golden_match(golden_entries, key_field, haystack)
        verdict = verdict_by_match_id.get(item.id)
        predicted_flag = bool(verdict.flagged) if verdict else False

        if golden is None:
            details.append({"item": haystack[:80], "golden_match": None, "predicted_flagged": predicted_flag})
            continue

        matched_golden_ids.add(id(golden))
        expected_flag = golden["expected_flagged"]
        if expected_flag and predicted_flag:
            tp += 1
        elif not expected_flag and predicted_flag:
            fp += 1
        details.append(
            {
                "item": haystack[:80],
                "golden_match": golden[key_field],
                "expected_flagged": expected_flag,
                "predicted_flagged": predicted_flag,
                "confidence": golden.get("confidence"),
            }
        )

    fn = sum(
        1
        for entry in golden_entries
        if entry["expected_flagged"] and id(entry) not in matched_golden_ids
    )
    # Golden entries that were matched but the model flagged=False also count as FN.
    for d in details:
        if d.get("golden_match") and d.get("expected_flagged") and not d.get("predicted_flagged"):
            fn += 1

    return {"tp": tp, "fp": fp, "fn": fn, "details": details}


def hallucination_rate(produced_items: list, source_text: str, haystack_field: str, min_overlap_words: int = 4) -> float:
    """Fraction of produced items whose key text isn't actually findable in the source document.

    Uses a loose n-gram containment check rather than exact substring match, since the model
    may legitimately paraphrase slightly. An item is "grounded" if any run of
    `min_overlap_words` consecutive words from it appears in the source text.
    """
    if not produced_items:
        return 0.0
    source_lower = source_text.lower()
    ungrounded = 0
    for item in produced_items:
        text = getattr(item, haystack_field).lower()
        words = text.split()
        grounded = any(
            " ".join(words[i : i + min_overlap_words]) in source_lower
            for i in range(max(1, len(words) - min_overlap_words + 1))
        )
        if not grounded:
            ungrounded += 1
    return ungrounded / len(produced_items)


def main():
    documents = load_documents()
    golden_set = json.loads(GOLDEN_SET_PATH.read_text())

    print("Running pipeline against real OpenAI API (this costs a small number of real calls)...")
    report: AnalysisReport = run_pipeline(documents, llm=OpenAIStructuredLLM())

    citation_scores = score_category(
        report.citations, report.verdicts, golden_set["citations"], "case_name_contains", "case_name", "citation_id"
    )
    fact_scores = score_category(
        report.facts, report.fact_checks, golden_set["facts"], "fact_contains", "raw_text", "fact_id"
    )

    citation_hallucination = hallucination_rate(report.citations, documents["motion_for_summary_judgment"], "case_name", min_overlap_words=2)
    fact_hallucination = hallucination_rate(report.facts, documents["motion_for_summary_judgment"], "raw_text")

    total_tp = citation_scores["tp"] + fact_scores["tp"]
    total_fp = citation_scores["fp"] + fact_scores["fp"]
    total_fn = citation_scores["fn"] + fact_scores["fn"]
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else None
    total_items = len(report.citations) + len(report.facts)
    overall_hallucination = (
        (citation_hallucination * len(report.citations) + fact_hallucination * len(report.facts)) / total_items
        if total_items
        else 0.0
    )

    results = {
        "precision": precision,
        "recall": recall,
        "hallucination_rate": overall_hallucination,
        "pipeline_errors": report.errors,
        "citations": {**citation_scores, "hallucination_rate": citation_hallucination, "extracted_count": len(report.citations)},
        "facts": {**fact_scores, "hallucination_rate": fact_hallucination, "extracted_count": len(report.facts)},
    }

    print("\n=== BS Detector Eval Results ===")
    print(f"Overall precision: {precision:.0%}" if precision is not None else "Overall precision: n/a (no flags raised)")
    print(f"Overall recall:    {recall:.0%}" if recall is not None else "Overall recall: n/a (no known flaws)")
    print(f"Hallucination rate: {overall_hallucination:.0%}")
    if report.errors:
        print(f"\n⚠ {len(report.errors)} pipeline node failure(s) occurred -- scores above are computed on partial results:")
        for err in report.errors:
            print(f"  - {err}")
    print(f"\nCitations: extracted {len(report.citations)}, TP={citation_scores['tp']} FP={citation_scores['fp']} FN={citation_scores['fn']}, hallucination={citation_hallucination:.0%}")
    print(f"Facts:     extracted {len(report.facts)}, TP={fact_scores['tp']} FP={fact_scores['fp']} FN={fact_scores['fn']}, hallucination={fact_hallucination:.0%}")

    print("\n--- Per-citation detail ---")
    for d in citation_scores["details"]:
        print(f"  {d}")
    print("\n--- Per-fact detail ---")
    for d in fact_scores["details"]:
        print(f"  {d}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
