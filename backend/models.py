from typing import Literal

from pydantic import BaseModel, Field

SupportStatus = Literal["supports", "contradicts", "unsupported", "uncertain"]
QuoteAccuracy = Literal["accurate", "altered", "fabricated", "no_quote", "uncertain"]
ConsistencyStatus = Literal["consistent", "contradicted", "unverifiable"]


class Citation(BaseModel):
    id: str = Field(description="Stable identifier for this citation, e.g. 'cite-1'")
    raw_text: str = Field(description="The citation exactly as it appears in the document")
    case_name: str = Field(description="Case name, e.g. 'Privette v. Superior Court'")
    citation_string: str = Field(description="Formal reporter citation, e.g. '5 Cal.4th 689, 695 (1993)'")
    proposition: str = Field(description="The claim or proposition the brief uses this citation to support")
    quoted_text: str | None = Field(
        default=None, description="Exact text in quotation marks attributed to this authority, if any"
    )


class CitationList(BaseModel):
    citations: list[Citation]


class VerdictAssessment(BaseModel):
    """What the Verifier Agent produces for a single citation (no id — that's added by the pipeline)."""

    support_status: SupportStatus = Field(
        description="Whether the cited authority actually supports the proposition as stated"
    )
    quote_accuracy: QuoteAccuracy = Field(
        description="Whether any direct quote attributed to this authority is accurate"
    )
    reasoning: str = Field(description="Brief explanation for the verdict")
    flagged: bool = Field(description="True if this citation should be flagged as a likely problem")


class CitationVerdict(BaseModel):
    citation_id: str
    support_status: SupportStatus
    quote_accuracy: QuoteAccuracy
    reasoning: str
    flagged: bool
    confidence: float | None = Field(default=None, description="0-1 confidence in this flag, only set when flagged")
    confidence_reasoning: str | None = Field(default=None, description="Why the pipeline is (un)certain about this flag")


class FactClaim(BaseModel):
    id: str = Field(description="Stable identifier for this fact, e.g. 'fact-1'")
    raw_text: str = Field(description="The fact statement as it appears in the MSJ's Statement of Undisputed Material Facts")
    claim: str = Field(description="The atomic factual claim being made, paraphrased if needed for clarity")


class FactClaimList(BaseModel):
    facts: list[FactClaim]


class ConsistencyAssessment(BaseModel):
    """What the Consistency Checker Agent produces for a single fact (no id — added by the pipeline)."""

    consistency_status: ConsistencyStatus = Field(
        description="Whether the fact is consistent with, contradicted by, or unverifiable against the other documents"
    )
    reasoning: str = Field(description="Brief explanation, citing what the other documents say")
    flagged: bool = Field(description="True if this fact should be flagged as a likely problem")


class FactCheckResult(BaseModel):
    fact_id: str
    consistency_status: ConsistencyStatus
    reasoning: str
    flagged: bool
    confidence: float | None = Field(default=None, description="0-1 confidence in this flag, only set when flagged")
    confidence_reasoning: str | None = Field(default=None, description="Why the pipeline is (un)certain about this flag")


class ConfidenceAssessment(BaseModel):
    """What the Confidence Scoring Agent produces for a single already-flagged finding."""

    confidence: float = Field(ge=0, le=1, description="How confident the pipeline is in this flag, from 0 to 1")
    reasoning: str = Field(description="Brief explanation for the confidence level (what would increase or decrease it)")


class JudicialMemo(BaseModel):
    """What the Judicial Memo Agent produces: a single paragraph synthesizing the top findings."""

    memo: str = Field(description="One paragraph, addressed to a judge, synthesizing the flagged findings")


class AnalysisReport(BaseModel):
    citations: list[Citation]
    verdicts: list[CitationVerdict]
    facts: list[FactClaim]
    fact_checks: list[FactCheckResult]
    citation_flagged_count: int
    fact_flagged_count: int
    judicial_memo: str | None = Field(default=None, description="One-paragraph summary of the top flagged findings, written for a judge")
    errors: list[str] = Field(
        default_factory=list, description="Node-level failures (e.g. a single citation's verification call erroring out)"
    )
