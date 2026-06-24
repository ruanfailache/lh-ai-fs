from llm import FakeStructuredLLM
from models import Citation, CitationList, VerdictAssessment
from pipeline import run_pipeline


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

    # extract_citations only runs once, so a plain FIFO list works there.
    return FakeStructuredLLM(
        {
            "CitationList": [citations],
            "VerdictAssessment": verify_matcher,
        }
    )


def test_pipeline_extracts_and_verifies_all_citations():
    report = run_pipeline("dummy MSJ text", llm=make_fake_llm())

    assert len(report.citations) == 2
    assert {c.id for c in report.citations} == {"cite-1", "cite-2"}
    assert len(report.verdicts) == 2


def test_pipeline_flags_altered_quote():
    report = run_pipeline("dummy MSJ text", llm=make_fake_llm())

    verdict_by_id = {v.citation_id: v for v in report.verdicts}
    assert verdict_by_id["cite-1"].flagged is True
    assert verdict_by_id["cite-1"].quote_accuracy == "altered"


def test_pipeline_does_not_flag_merely_uncertain_citation():
    report = run_pipeline("dummy MSJ text", llm=make_fake_llm())

    verdict_by_id = {v.citation_id: v for v in report.verdicts}
    assert verdict_by_id["cite-2"].support_status == "uncertain"
    assert verdict_by_id["cite-2"].flagged is False


def test_pipeline_computes_flagged_count():
    report = run_pipeline("dummy MSJ text", llm=make_fake_llm())

    assert report.flagged_count == 1
