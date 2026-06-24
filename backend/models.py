from typing import Literal

from pydantic import BaseModel, Field

SupportStatus = Literal["supports", "contradicts", "unsupported", "uncertain"]
QuoteAccuracy = Literal["accurate", "altered", "fabricated", "no_quote", "uncertain"]


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


class AnalysisReport(BaseModel):
    citations: list[Citation]
    verdicts: list[CitationVerdict]
    flagged_count: int
