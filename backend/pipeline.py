import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from llm import StructuredLLM
from models import AnalysisReport, Citation, CitationList, CitationVerdict, VerdictAssessment
from prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    verifier_user_prompt,
)


class PipelineState(TypedDict):
    msj_text: str
    citations: list[Citation]
    verdicts: Annotated[list[CitationVerdict], operator.add]
    errors: Annotated[list[str], operator.add]
    report: AnalysisReport


class VerifyCitationState(TypedDict):
    citation: Citation


def build_pipeline(llm: StructuredLLM):
    def extract_citations(state: PipelineState) -> dict:
        result = llm.call_structured(
            messages=[
                {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": state["msj_text"]},
            ],
            schema=CitationList,
        )
        return {"citations": result.citations}

    def route_to_verifiers(state: PipelineState) -> list[Send]:
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

    def aggregate(state: PipelineState) -> dict:
        verdicts = state["verdicts"]
        report = AnalysisReport(
            citations=state["citations"],
            verdicts=verdicts,
            flagged_count=sum(1 for v in verdicts if v.flagged),
        )
        return {"report": report}

    graph = StateGraph(PipelineState)
    graph.add_node("extract_citations", extract_citations)
    graph.add_node("verify_citation", verify_citation)
    graph.add_node("aggregate", aggregate)

    graph.add_edge(START, "extract_citations")
    graph.add_conditional_edges("extract_citations", route_to_verifiers, ["verify_citation"])
    graph.add_edge("verify_citation", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


def run_pipeline(msj_text: str, llm: StructuredLLM) -> AnalysisReport:
    app = build_pipeline(llm)
    result = app.invoke({"msj_text": msj_text, "citations": [], "verdicts": [], "errors": []})
    return result["report"]
