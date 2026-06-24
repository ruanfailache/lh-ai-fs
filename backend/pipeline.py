import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from llm import StructuredLLM
from models import (
    AnalysisReport,
    Citation,
    CitationList,
    CitationVerdict,
    ConfidenceAssessment,
    ConsistencyAssessment,
    FactCheckResult,
    FactClaim,
    FactClaimList,
    JudicialMemo,
    VerdictAssessment,
)
from prompts import (
    CONFIDENCE_SCORER_SYSTEM_PROMPT,
    EXTRACTOR_SYSTEM_PROMPT,
    FACT_CHECKER_SYSTEM_PROMPT,
    FACT_EXTRACTOR_SYSTEM_PROMPT,
    JUDICIAL_MEMO_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    confidence_scorer_user_prompt,
    fact_checker_user_prompt,
    judicial_memo_user_prompt,
    verifier_user_prompt,
)


class PipelineState(TypedDict):
    msj_text: str
    police_report: str
    medical_records: str
    witness_statement: str
    citations: list[Citation]
    verdicts: Annotated[list[CitationVerdict], operator.add]
    facts: list[FactClaim]
    fact_checks: Annotated[list[FactCheckResult], operator.add]
    errors: Annotated[list[str], operator.add]
    report: AnalysisReport


class VerifyCitationState(TypedDict):
    citation: Citation | None


class CheckFactState(TypedDict):
    fact: FactClaim | None
    police_report: str
    medical_records: str
    witness_statement: str


def build_pipeline(llm: StructuredLLM):
    def score_confidence(finding_description: str, reasoning: str) -> tuple[float | None, str | None, str | None]:
        """Run the Confidence Scoring Agent for one already-flagged finding.

        Returns (confidence, confidence_reasoning, error) -- on failure, the first two are None
        and `error` carries a message instead of raising, so a scorer failure never costs the
        verdict/fact-check that's already been produced.
        """
        try:
            assessment = llm.call_structured(
                messages=[
                    {"role": "system", "content": CONFIDENCE_SCORER_SYSTEM_PROMPT},
                    {"role": "user", "content": confidence_scorer_user_prompt(finding_description, reasoning)},
                ],
                schema=ConfidenceAssessment,
            )
            return assessment.confidence, assessment.reasoning, None
        except Exception as exc:
            return None, None, f"Confidence scoring failed for finding ({finding_description[:60]}...): {exc}"

    def extract_citations(state: PipelineState) -> dict:
        try:
            result = llm.call_structured(
                messages=[
                    {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": state["msj_text"]},
                ],
                schema=CitationList,
            )
            return {"citations": result.citations}
        except Exception as exc:
            return {"citations": [], "errors": [f"Citation extraction failed: {exc}"]}

    def route_to_verifiers(state: PipelineState) -> list[Send]:
        # Always fan out through verify_citation, even with zero citations (a single sentinel
        # task that's a no-op) -- both this branch and the facts branch must reach `aggregate`
        # at the *same* graph depth, or aggregate can fire twice (once prematurely) and race
        # with whatever runs after it. Skipping the hop for the empty case caused exactly that.
        if not state["citations"]:
            return [Send("verify_citation", {"citation": None})]
        return [Send("verify_citation", {"citation": citation}) for citation in state["citations"]]

    def verify_citation(state: VerifyCitationState) -> dict:
        citation = state["citation"]
        if citation is None:
            return {}
        try:
            assessment = llm.call_structured(
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": verifier_user_prompt(
                            citation.case_name,
                            citation.citation_string,
                            citation.proposition,
                            citation.quoted_text,
                        ),
                    },
                ],
                schema=VerdictAssessment,
            )
            verdict = CitationVerdict(citation_id=citation.id, **assessment.model_dump())
            errors = []
            if verdict.flagged:
                confidence, confidence_reasoning, error = score_confidence(
                    f"Citation '{citation.case_name}' cited for: {citation.proposition}", verdict.reasoning
                )
                verdict.confidence = confidence
                verdict.confidence_reasoning = confidence_reasoning
                if error:
                    errors.append(error)
            return {"verdicts": [verdict], "errors": errors}
        except Exception as exc:
            return {"errors": [f"Verification failed for {citation.id} ({citation.case_name}): {exc}"]}

    def extract_facts(state: PipelineState) -> dict:
        try:
            result = llm.call_structured(
                messages=[
                    {"role": "system", "content": FACT_EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": state["msj_text"]},
                ],
                schema=FactClaimList,
            )
            return {"facts": result.facts}
        except Exception as exc:
            return {"facts": [], "errors": [f"Fact extraction failed: {exc}"]}

    def route_to_fact_checkers(state: PipelineState) -> list[Send]:
        # Same reasoning as route_to_verifiers: always fan out through check_fact, even with a
        # sentinel no-op task, so both branches reach `aggregate` at the same graph depth.
        if not state["facts"]:
            return [
                Send(
                    "check_fact",
                    {
                        "fact": None,
                        "police_report": state["police_report"],
                        "medical_records": state["medical_records"],
                        "witness_statement": state["witness_statement"],
                    },
                )
            ]
        return [
            Send(
                "check_fact",
                {
                    "fact": fact,
                    "police_report": state["police_report"],
                    "medical_records": state["medical_records"],
                    "witness_statement": state["witness_statement"],
                },
            )
            for fact in state["facts"]
        ]

    def check_fact(state: CheckFactState) -> dict:
        fact = state["fact"]
        if fact is None:
            return {}
        try:
            assessment = llm.call_structured(
                messages=[
                    {"role": "system", "content": FACT_CHECKER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": fact_checker_user_prompt(
                            fact.claim,
                            state["police_report"],
                            state["medical_records"],
                            state["witness_statement"],
                        ),
                    },
                ],
                schema=ConsistencyAssessment,
            )
            result = FactCheckResult(fact_id=fact.id, **assessment.model_dump())
            errors = []
            if result.flagged:
                confidence, confidence_reasoning, error = score_confidence(f"Fact: {fact.claim}", result.reasoning)
                result.confidence = confidence
                result.confidence_reasoning = confidence_reasoning
                if error:
                    errors.append(error)
            return {"fact_checks": [result], "errors": errors}
        except Exception as exc:
            return {"errors": [f"Consistency check failed for {fact.id}: {exc}"]}

    def aggregate(state: PipelineState) -> dict:
        verdicts = state["verdicts"]
        fact_checks = state["fact_checks"]
        report = AnalysisReport(
            citations=state["citations"],
            verdicts=verdicts,
            facts=state["facts"],
            fact_checks=fact_checks,
            citation_flagged_count=sum(1 for v in verdicts if v.flagged),
            fact_flagged_count=sum(1 for f in fact_checks if f.flagged),
            errors=state["errors"],
        )
        return {"report": report}

    def write_memo(state: PipelineState) -> dict:
        report = state["report"]
        citation_by_id = {c.id: c for c in report.citations}
        fact_by_id = {f.id: f for f in report.facts}

        findings = []
        for v in report.verdicts:
            if v.flagged:
                citation = citation_by_id.get(v.citation_id)
                label = citation.case_name if citation else v.citation_id
                confidence_note = f" (confidence {v.confidence:.0%})" if v.confidence is not None else ""
                findings.append(f"Citation '{label}': {v.reasoning}{confidence_note}")
        for fc in report.fact_checks:
            if fc.flagged:
                fact = fact_by_id.get(fc.fact_id)
                label = fact.claim if fact else fc.fact_id
                confidence_note = f" (confidence {fc.confidence:.0%})" if fc.confidence is not None else ""
                findings.append(f"Fact '{label}': {fc.reasoning}{confidence_note}")

        try:
            result = llm.call_structured(
                messages=[
                    {"role": "system", "content": JUDICIAL_MEMO_SYSTEM_PROMPT},
                    {"role": "user", "content": judicial_memo_user_prompt(findings)},
                ],
                schema=JudicialMemo,
            )
            return {"report": report.model_copy(update={"judicial_memo": result.memo})}
        except Exception as exc:
            error = f"Judicial memo generation failed: {exc}"
            return {"report": report.model_copy(update={"errors": report.errors + [error]}), "errors": [error]}

    graph = StateGraph(PipelineState)
    graph.add_node("extract_citations", extract_citations)
    graph.add_node("verify_citation", verify_citation)
    graph.add_node("extract_facts", extract_facts)
    graph.add_node("check_fact", check_fact)
    graph.add_node("aggregate", aggregate)
    graph.add_node("write_memo", write_memo)

    graph.add_edge(START, "extract_citations")
    graph.add_conditional_edges("extract_citations", route_to_verifiers, ["verify_citation"])
    graph.add_edge("verify_citation", "aggregate")

    graph.add_edge(START, "extract_facts")
    graph.add_conditional_edges("extract_facts", route_to_fact_checkers, ["check_fact"])
    graph.add_edge("check_fact", "aggregate")

    graph.add_edge("aggregate", "write_memo")
    graph.add_edge("write_memo", END)

    return graph.compile()


def run_pipeline(documents: dict[str, str], llm: StructuredLLM) -> AnalysisReport:
    app = build_pipeline(llm)
    result = app.invoke(
        {
            "msj_text": documents["motion_for_summary_judgment"],
            "police_report": documents.get("police_report", ""),
            "medical_records": documents.get("medical_records_excerpt", ""),
            "witness_statement": documents.get("witness_statement", ""),
            "citations": [],
            "verdicts": [],
            "facts": [],
            "fact_checks": [],
            "errors": [],
        }
    )
    return result["report"]
