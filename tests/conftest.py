"""Shared pytest fixtures for the guitar_assistant test suite."""

from typing import Any, Final

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
import pytest

from guitar_assistant.retriever import build_vector_store

AVAILABLE_GUITAR_MODELS: Final = ("telecaster", "stratocaster", "sg")


class FakeChatModel(BaseChatModel):
    """Deterministic, network-free stand-in for ChatOpenAI.

    Routes every query to a fixed guitar model and answers with a fixed string,
    so `build_agent`'s route/generate chains work without any API calls.
    """

    guitar_model: str
    answer: str

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.answer))])

    def with_structured_output(
        self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        assert isinstance(schema, type)
        return RunnableLambda(lambda _input: schema(guitar_model=self.guitar_model))


class _FakeKeywordEmbeddings(Embeddings):
    """Deterministic, network-free stand-in for OpenAIEmbeddings.

    Encodes each text as a one-hot vector over `AVAILABLE_GUITAR_MODELS`, so similarity
    search behaves predictably without calling any embeddings API.
    """

    def _vectorize(self, text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if guitar_model in lowered else 0.0 for guitar_model in AVAILABLE_GUITAR_MODELS]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


@pytest.fixture(name="fake_embeddings")
def fixture_fake_embeddings() -> _FakeKeywordEmbeddings:
    return _FakeKeywordEmbeddings()


@pytest.fixture(name="fake_chat_model")
def fixture_fake_chat_model() -> FakeChatModel:
    return FakeChatModel(
        guitar_model="stratocaster", answer="The scale length is 25.5 in [stratocaster.md]."
    )


@pytest.fixture(name="available_guitar_models")
def fixture_available_guitar_models() -> tuple[str, ...]:
    return AVAILABLE_GUITAR_MODELS


@pytest.fixture(name="vector_store")
def fixture_vector_store(fake_embeddings, available_guitar_models):
    documents = [
        Document(
            page_content=f"Spec sheet content mentioning {guitar_model}.",
            metadata={"guitar_model": guitar_model, "source": f"{guitar_model}.md"},
        )
        for guitar_model in available_guitar_models
    ]
    return build_vector_store(documents, embeddings=fake_embeddings)
