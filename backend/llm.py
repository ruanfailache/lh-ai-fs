import os
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def call_llm(
    messages: list[dict],
    model: str = "gpt-4o",
    temperature: float = 0,
) -> str:
    """Call the OpenAI API and return the response content."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


class StructuredLLM(ABC):
    """Returns a validated Pydantic instance for a given schema instead of raw text."""

    @abstractmethod
    def call_structured(self, messages: list[dict], schema: type[SchemaT]) -> SchemaT:
        raise NotImplementedError


class OpenAIStructuredLLM(StructuredLLM):
    def __init__(self, model: str = "gpt-4o", temperature: float = 0):
        # wrap_openai is a no-op unless LANGCHAIN_TRACING_V2=true is set, in which case it
        # reports each call (prompt, response, tokens, latency) as an LLM span in LangSmith.
        self.client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        self.model = model
        self.temperature = temperature

    def call_structured(self, messages: list[dict], schema: type[SchemaT]) -> SchemaT:
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format=schema,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError(f"Model refused or failed to produce a {schema.__name__}")
        return parsed


class FakeStructuredLLM(StructuredLLM):
    """Deterministic stand-in for tests: returns pre-built responses without calling the network.

    For each schema, register either:
    - a plain list of responses, consumed FIFO (fine when only one call happens for that schema), or
    - a callable(messages) -> BaseModel that inspects the prompt to decide which response to return.

    The callable form matters for fan-out nodes (e.g. one verifier call per citation), where
    branches may run in any order and a FIFO queue would attribute the wrong response to the
    wrong citation.
    """

    def __init__(self, responses_by_schema: dict[str, list[BaseModel] | Callable[[list[dict]], BaseModel]]):
        self._responses = dict(responses_by_schema)

    def call_structured(self, messages: list[dict], schema: type[SchemaT]) -> SchemaT:
        responses = self._responses.get(schema.__name__)
        if responses is None:
            raise AssertionError(f"FakeStructuredLLM has no registered response for {schema.__name__}")
        if callable(responses) and not isinstance(responses, list):
            return responses(messages)
        if not responses:
            raise AssertionError(f"FakeStructuredLLM ran out of queued responses for {schema.__name__}")
        return responses.pop(0)
