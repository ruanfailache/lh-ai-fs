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
    ConsistencyAssessment,
    FactCheckResult,
    FactClaim,
    FactClaimList,
    VerdictAssessment,
)
from prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    FACT_CHECKER_SYSTEM_PROMPT,
    FACT_EXTRACTOR_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    fact_checker_user_prompt,
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
    citation: Citation


class CheckFactState(TypedDict):
    fact: FactClaim
    police_report: str
    medical_records: str
    witness_statement: str


def build_pipeline(llm: StructuredLLM):
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

    def route_to_verifiers(state: PipelineState) -> list[Send] | list[str]:
        # If extraction found (or could find) no citations, route straight to aggregate --
        # otherwise verify_citation never runs and aggregate (which waits on both fan-in
        # branches) would never fire either.
        if not state["citations"]:
            return ["aggregate"]
        return [Send("verify_citation", {"citation": citation}) for citation in state["citations"]]

    def verify_citation(state: VerifyCitationState) -> dict:
        citation = state["citation"]
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
            return {"verdicts": [verdict]}
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

    def route_to_fact_checkers(state: PipelineState) -> list[Send] | list[str]:
        # Same reasoning as route_to_verifiers: an empty list must still reach aggregate.
        if not state["facts"]:
            return ["aggregate"]
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
            return {"fact_checks": [result]}
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

    graph = StateGraph(PipelineState)
    graph.add_node("extract_citations", extract_citations)
    graph.add_node("verify_citation", verify_citation)
    graph.add_node("extract_facts", extract_facts)
    graph.add_node("check_fact", check_fact)
    graph.add_node("aggregate", aggregate)

    graph.add_edge(START, "extract_citations")
    graph.add_conditional_edges("extract_citations", route_to_verifiers, ["verify_citation", "aggregate"])
    graph.add_edge("verify_citation", "aggregate")

    graph.add_edge(START, "extract_facts")
    graph.add_conditional_edges("extract_facts", route_to_fact_checkers, ["check_fact", "aggregate"])
    graph.add_edge("check_fact", "aggregate")

    graph.add_edge("aggregate", END)

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
