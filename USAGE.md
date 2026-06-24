# Usage

## Running the unit tests

The pipeline's unit tests use a `FakeStructuredLLM` (see [backend/llm.py](backend/llm.py)) instead
of calling OpenAI, so they're free and don't require an API key:

```bash
cd backend
pytest tests/ -v
```

## Running the eval suite

The eval harness measures the pipeline's actual quality against a hand-curated golden set of
known flaws in the fixture documents ([backend/evals/golden_set.json](backend/evals/golden_set.json)).
Unlike the unit tests above, this calls the real OpenAI API (a small number of calls per run) and
reports precision, recall, and a hallucination rate:

```bash
cd backend
cp .env.example .env   # add a real OPENAI_API_KEY
python run_evals.py
```

It prints a precision/recall/hallucination-rate summary plus a per-citation and per-fact
breakdown, and writes the full results to `backend/evals/last_run_results.json`.

## Observability

Setting `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` sends a trace of every
`/analyze` run to [LangSmith](https://smith.langchain.com) — the full LangGraph execution (citation
and fact-checking branches, fanned out per item) plus the prompt/response/token usage of every LLM
call inside it. It's a no-op if those variables aren't set.
