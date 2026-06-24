from llm import FakeStructuredLLM
from models import (
    Citation,
    CitationList,
    ConfidenceAssessment,
    ConsistencyAssessment,
    FactClaim,
    FactClaimList,
    JudicialMemo,
    VerdictAssessment,
)
from pipeline import run_pipeline

DOCUMENTS = {
    "motion_for_summary_judgment": "dummy MSJ text",
    "police_report": "dummy police report text",
    "medical_records_excerpt": "dummy medical records text",
    "witness_statement": "dummy witness statement text",
}

DEFAULT_MEMO = JudicialMemo(memo="This is the synthesized memo for the judge.")


def make_fake_llm() -> FakeStructuredLLM:
    citations = CitationList(
        citations=[
            Citation(
                id="cite-1",
                raw_text="Privette v. Superior Court, 5 Cal.4th 689, 695 (1993)",
                case_name="Privette v. Superior Court",
                citation_string="5 Cal.4th 689, 695 (1993)",
                proposition="A hirer is never liable for injuries sustained by an independent contractor's employees.",
                quoted_text="A hirer is never liable for injuries sustained by an independent contractor's employees when the injuries arise from the contracted work.",
            ),
            Citation(
                id="cite-2",
                raw_text="Torres v. Granite Falls Dev. Corp., 198 Cal.App.4th 223 (2011)",
                case_name="Torres v. Granite Falls Dev. Corp.",
                citation_string="198 Cal.App.4th 223 (2011)",
                proposition="String cite supporting summary judgment for hirers.",
                quoted_text=None,
            ),
        ]
    )
    verdicts_by_case = {
        "Privette": VerdictAssessment(
            support_status="contradicts",
            quote_accuracy="altered",
            reasoning="Privette has well-known exceptions (e.g. Hooker, McKown); the quote drops the qualifying language.",
            flagged=True,
        ),
        "Torres": VerdictAssessment(
            support_status="uncertain",
            quote_accuracy="no_quote",
            reasoning="Could not verify this unpublished-sounding citation with confidence.",
            flagged=False,
        ),
    }

    def verify_matcher(messages: list[dict]) -> VerdictAssessment:
        # verify_citation's fan-out branches may run in any order, so pick the response by
        # looking at which case name is actually in this call's prompt rather than FIFO order.
        prompt = messages[-1]["content"]
        for case_name, verdict in verdicts_by_case.items():
            if case_name in prompt:
                return verdict
        raise AssertionError(f"No fake verdict registered for prompt: {prompt}")

    facts = FactClaimList(
        facts=[
            FactClaim(
                id="fact-1",
                raw_text="3. On or about March 14, 2021, Rivera was working on a scaffolding assembly...",
                claim="The incident occurred on March 14, 2021.",
            ),
            FactClaim(
                id="fact-2",
                raw_text="4. Rivera was not wearing required personal protective equipment...",
                claim="Rivera was not wearing required PPE at the time of the incident.",
            ),
        ]
    )
    fact_checks_by_claim = {
        "March 14, 2021": ConsistencyAssessment(
            consistency_status="contradicted",
            reasoning="The police report and medical records both state the incident occurred on March 12, 2021.",
            flagged=True,
        ),
        "PPE": ConsistencyAssessment(
            consistency_status="contradicted",
            reasoning="The police report and witness statement both say Rivera was wearing a hard hat, harness, and vest.",
            flagged=True,
        ),
    }

    def fact_check_matcher(messages: list[dict]) -> ConsistencyAssessment:
        prompt = messages[-1]["content"]
        for needle, result in fact_checks_by_claim.items():
            if needle in prompt:
                return result
        raise AssertionError(f"No fake fact check registered for prompt: {prompt}")

    confidence_by_needle = {
        "Privette": ConfidenceAssessment(confidence=0.9, reasoning="Well-known doctrine with a clearly altered quote."),
        "March 14": ConfidenceAssessment(confidence=0.95, reasoning="Exact date mismatch across three source documents."),
        "PPE": ConfidenceAssessment(confidence=0.85, reasoning="Both source documents explicitly describe PPE being worn."),
    }

    def confidence_matcher(messages: list[dict]) -> ConfidenceAssessment:
        prompt = messages[-1]["content"]
        for needle, result in confidence_by_needle.items():
            if needle in prompt:
                return result
        raise AssertionError(f"No fake confidence assessment registered for prompt: {prompt}")

    # extract_citations/extract_facts/write_memo only run once each, so plain FIFO lists work there.
    return FakeStructuredLLM(
        {
            "CitationList": [citations],
            "VerdictAssessment": verify_matcher,
            "FactClaimList": [facts],
            "ConsistencyAssessment": fact_check_matcher,
            "ConfidenceAssessment": confidence_matcher,
            "JudicialMemo": [DEFAULT_MEMO],
        }
    )


def test_pipeline_extracts_and_verifies_all_citations():
    report = run_pipeline(DOCUMENTS, llm=make_fake_llm())

    assert len(report.citations) == 2
    assert {c.id for c in report.citations} == {"cite-1", "cite-2"}
    assert len(report.verdicts) == 2


def test_pipeline_flags_altered_quote():
    report = run_pipeline(DOCUMENTS, llm=make_fake_llm())

    verdict_by_id = {v.citation_id: v for v in report.verdicts}
    assert verdict_by_id["cite-1"].flagged is True
    assert verdict_by_id["cite-1"].quote_accuracy == "altered"


def test_pipeline_does_not_flag_merely_uncertain_citation():
    report = run_pipeline(DOCUMENTS, llm=make_fake_llm())

    verdict_by_id = {v.citation_id: v for v in report.verdicts}
    assert verdict_by_id["cite-2"].support_status == "uncertain"
    assert verdict_by_id["cite-2"].flagged is False


def test_pipeline_extracts_and_checks_all_facts():
    report = run_pipeline(DOCUMENTS, llm=make_fake_llm())

    assert len(report.facts) == 2
    assert {f.id for f in report.facts} == {"fact-1", "fact-2"}
    assert len(report.fact_checks) == 2


def test_pipeline_flags_contradicted_fact():
    report = run_pipeline(DOCUMENTS, llm=make_fake_llm())

    check_by_id = {c.fact_id: c for c in report.fact_checks}
    assert check_by_id["fact-1"].consistency_status == "contradicted"
    assert check_by_id["fact-1"].flagged is True


def test_pipeline_computes_flagged_counts():
    report = run_pipeline(DOCUMENTS, llm=make_fake_llm())

    assert report.citation_flagged_count == 1
    assert report.fact_flagged_count == 2


def test_pipeline_only_scores_confidence_for_flagged_findings():
    report = run_pipeline(DOCUMENTS, llm=make_fake_llm())

    verdict_by_id = {v.citation_id: v for v in report.verdicts}
    check_by_id = {c.fact_id: c for c in report.fact_checks}

    assert verdict_by_id["cite-1"].confidence == 0.9
    assert verdict_by_id["cite-1"].confidence_reasoning is not None
    assert verdict_by_id["cite-2"].confidence is None  # not flagged -- never scored

    assert check_by_id["fact-1"].confidence == 0.95
    assert check_by_id["fact-2"].confidence == 0.85


def test_pipeline_writes_judicial_memo():
    report = run_pipeline(DOCUMENTS, llm=make_fake_llm())

    assert report.judicial_memo == DEFAULT_MEMO.memo


def test_pipeline_surfaces_node_failures_without_dropping_the_rest():
    citations = CitationList(
        citations=[
            Citation(
                id="cite-1",
                raw_text="Privette v. Superior Court",
                case_name="Privette v. Superior Court",
                citation_string="5 Cal.4th 689",
                proposition="p",
                quoted_text=None,
            )
        ]
    )

    def explode(_messages: list[dict]) -> VerdictAssessment:
        raise RuntimeError("simulated LLM failure")

    llm = FakeStructuredLLM(
        {
            "CitationList": [citations],
            "VerdictAssessment": explode,
            "FactClaimList": [FactClaimList(facts=[])],
            "ConsistencyAssessment": [],
            "JudicialMemo": [DEFAULT_MEMO],
        }
    )

    report = run_pipeline(DOCUMENTS, llm=llm)

    assert report.citations[0].id == "cite-1"
    assert report.verdicts == []
    assert len(report.errors) == 1
    assert "cite-1" in report.errors[0]
    assert "simulated LLM failure" in report.errors[0]


def test_pipeline_survives_extraction_failure():
    def explode_extraction(_messages: list[dict]) -> CitationList:
        raise RuntimeError("simulated extraction failure")

    llm = FakeStructuredLLM(
        {
            "CitationList": explode_extraction,
            "VerdictAssessment": [],
            "FactClaimList": [FactClaimList(facts=[])],
            "ConsistencyAssessment": [],
            "JudicialMemo": [DEFAULT_MEMO],
        }
    )

    report = run_pipeline(DOCUMENTS, llm=llm)

    assert report.citations == []
    assert report.verdicts == []
    assert any("Citation extraction failed" in err for err in report.errors)


def test_pipeline_survives_confidence_scoring_failure():
    citations = CitationList(
        citations=[
            Citation(
                id="cite-1",
                raw_text="Privette v. Superior Court",
                case_name="Privette v. Superior Court",
                citation_string="5 Cal.4th 689",
                proposition="p",
                quoted_text=None,
            )
        ]
    )
    verdict = VerdictAssessment(
        support_status="contradicts", quote_accuracy="altered", reasoning="overstated", flagged=True
    )

    def explode_confidence(_messages: list[dict]) -> ConfidenceAssessment:
        raise RuntimeError("simulated confidence scoring failure")

    llm = FakeStructuredLLM(
        {
            "CitationList": [citations],
            "VerdictAssessment": [verdict],
            "FactClaimList": [FactClaimList(facts=[])],
            "ConsistencyAssessment": [],
            "ConfidenceAssessment": explode_confidence,
            "JudicialMemo": [DEFAULT_MEMO],
        }
    )

    report = run_pipeline(DOCUMENTS, llm=llm)

    assert len(report.verdicts) == 1
    assert report.verdicts[0].flagged is True
    assert report.verdicts[0].confidence is None
    assert any("Confidence scoring failed" in err for err in report.errors)


def test_pipeline_survives_memo_failure():
    def explode_memo(_messages: list[dict]) -> JudicialMemo:
        raise RuntimeError("simulated memo failure")

    llm = FakeStructuredLLM(
        {
            "CitationList": [CitationList(citations=[])],
            "VerdictAssessment": [],
            "FactClaimList": [FactClaimList(facts=[])],
            "ConsistencyAssessment": [],
            "JudicialMemo": explode_memo,
        }
    )

    report = run_pipeline(DOCUMENTS, llm=llm)

    assert report.judicial_memo is None
    assert any("Judicial memo generation failed" in err for err in report.errors)
