# Reflection

## Agent decomposition

The pipeline has six LLM-backed roles, but only four of them are separate LangGraph nodes:

1. **Citation Extractor** — pulls every citation out of the MSJ (body + footnotes) into structured `Citation` objects.
2. **Citation Verifier** — one call per citation (fanned out via `Send`), judges whether the cited authority actually supports the proposition and whether any direct quote is accurate.
3. **Fact Extractor** — pulls each numbered item out of the "Statement of Undisputed Material Facts" into a `FactClaim`.
4. **Consistency Checker** — one call per fact (fanned out), cross-references it against the police report, medical records, and witness statement.
5. **Confidence Scorer** — rates how sure the pipeline should be about an *already-flagged* finding.
6. **Judicial Memo writer** — synthesizes the flagged findings into one paragraph for a judge.

Citations and facts run as two independent parallel branches (both start right after the documents load), each doing its own extract → fan-out-verify, before converging on a single `aggregate` step and finally `write_memo`.

The Confidence Scorer is a deliberate compromise. It has its own prompt, schema, and reasoning step — a genuinely separate agent — but it's invoked as a second LLM call *inside* `verify_citation`/`check_fact` rather than as its own graph node. A "proper" separate node would need a barrier that waits for every citation/fact to finish before fanning out again over just the flagged ones, and after building and debugging the graph's actual fan-out/join behavior (see below), I judged that added topology risk wasn't worth it for what's functionally the same outcome. This is the kind of tradeoff I'd revisit with more time: it's slightly less "architecturally pure" than the spec's framing of a standalone scoring layer, in exchange for not adding another class of join bug to a graph that had already produced two.

## Prompts

Every prompt has an explicit, repeated instruction not to fabricate — "say uncertain/unverifiable rather than guessing" appears in the Verifier, the Consistency Checker, and implicitly governs the Confidence Scorer (low confidence is the honest response, not a flag of its own). The flagging rule is also explicit and narrow: uncertainty alone never triggers a flag, only an affirmative contradiction/overstatement does. This was a conscious choice to keep precision high even at the cost of recall — and the real eval run confirmed it (100% precision, recall well below 100%).

## What actually broke, and what that says about the design

The most honest part of this build is the debugging history, because the bugs only showed up once I stopped trusting the code and ran it against the real API and adversarial unit tests:

- **Errors were caught but silently dropped.** Early on, `verify_citation`/`check_fact` had `try/except` but the caught messages never made it into the response — a failed citation just vanished from the report with no trace. Fixed by adding `AnalysisReport.errors` and a test that asserts a simulated failure surfaces there.
- **A structurally empty branch could hang the graph.** If extraction returned zero citations (or failed and fell back to `[]`), the fan-out node never ran, and `aggregate` — which waited on a static edge from that node — never fired. This is a real bug independent of any LLM error; it would have hit a genuinely citation-free document too.
- **No retry on rate limits.** The OpenAI account used for testing has a very low requests-per-minute limit, and the graph's parallel fan-out hits it immediately. Without retry, a single 429 took down the entire request. Added exponential backoff (`max_retries=3` is now the project default for LLM calls).
- **A race between `aggregate` and `write_memo`.** This is the one I'm least proud of missing initially: my first fix for the empty-branch hang routed empty lists *directly* to `aggregate`, skipping the fan-out hop. That made the citations and facts branches reach `aggregate` at different graph depths depending on whether their list was empty — so `aggregate` could fire twice (once with partial data), and once `write_memo` existed to consume its output, the second late `aggregate` write collided with `write_memo`'s own write in the same step. The real fix was to always route through the fan-out node, even for empty lists (a single no-op sentinel task), so both branches reach `aggregate` at the same depth every time. This only surfaced because I added a node downstream of `aggregate` and re-ran the adversarial tests — it had been silently "working" before only because nothing was reading the duplicate output.

I think this is the most useful thing in this document: none of these were caught by reading the code. All four were caught by either a real API call or a test specifically designed to break a code path I hadn't exercised (an empty fan-out, a failing LLM call). That's the strongest argument I have for why the eval harness and the failure-path unit tests matter more than they might look like they do for a take-home.

## Eval design

`run_evals.py` calls the real API (this can't be meaningfully mocked — it's measuring actual model judgment, not pipeline wiring) and scores against a hand-curated `golden_set.json`. Each golden entry carries a `confidence` tag (`high`/`medium`/`low`) reflecting *my own* certainty as the person curating it:

- **High confidence**: things checkable from the documents alone (the March 12 vs. March 14 date contradiction) or extremely well-established law (the Privette doctrine has well-known exceptions).
- **Medium**: real cases I could identify, but where I'm reasoning about how they apply rather than quoting settled doctrine verbatim (Seabright, Kellerman).
- **Low**: the six footnote string-cite cases. I flagged all of them as "expected" because the proposition they support is itself an overstatement, but I have no way to verify whether the cases themselves are real or fabricated — I don't have case-law database access, and I didn't want to claim more certainty than I have. If the pipeline doesn't flag these, that's not necessarily a pipeline failure; it might be appropriately calibrated uncertainty.

**Hallucination rate** is defined structurally, not semantically: an extracted citation/fact "hallucinates" if no run of its words is findable in the source document at all. This catches invented citations, not incorrect legal reasoning about real ones — a narrower, more honestly-scoped definition than "did the model get it right."

Real runs against the documents: **100% precision, 0% hallucination, recall in the 25-58% range across runs** (see `backend/evals/last_run_results.json` for the latest). The recall variance itself is informative — some of it is genuine model non-determinism (gpt-4o at temperature 0 isn't perfectly deterministic), and some of it, before the rate-limit fixes above, was nodes silently failing and being scored as "not flagged" by default. I'd treat the precision and hallucination numbers as the trustworthy signal here, and the recall number as directionally correct but noisy until run several times.

## Prompt caching

`fact_checker_user_prompt` puts the ~2,700-token block of source documents *before* the per-call varying claim, specifically so OpenAI's automatic prefix-based caching can apply across the fact-checking fan-out (6 calls per run, same documents every time). This wasn't part of the original design — it came out of thinking about cost/latency once the fan-out pattern was in place — and is the kind of thing that's easy to miss if you don't think about prompt structure as a function of what's static vs. variable per call.

## What's not done, and why

- **UI.** The frontend is still the unmodified Vite scaffold. This was a deliberate scope cut: it's a different skillset/concern from the agent pipeline, and I'd rather ship a backend that's been genuinely debugged against the real API than split attention and ship a shallow version of both.
- **Orchestration resilience beyond what's here.** Retry + error surfacing + the join fix cover the failure modes I actually hit. I haven't load-tested concurrency limits or added a circuit breaker for sustained outages — diminishing returns for a fixed two-document-set demo.
- **A from-scratch case-law lookup.** The Verifier and Confidence Scorer rely entirely on the model's parametric legal knowledge. A production version of this would want real case-law retrieval (e.g. a citator API) rather than trusting the model's memory of what a 1993 case actually held.

If I had another pass, I'd spend it on the UI next, then on re-running the eval enough times to put a confidence interval on the recall number instead of reporting a single run.
