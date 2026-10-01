"""Shared pytest fixtures for the guitar_assistant test suite."""

from typing import Any, Final

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import PrivateAttr
import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from guitar_assistant.retriever import build_vector_store

AVAILABLE_GUITAR_MODELS: Final = ("telecaster", "stratocaster", "sg")


@pytest.fixture(autouse=True)
def _isolate_from_repo_local_state(monkeypatch, tmp_path):
    """Run every test from an empty temp directory, isolated from this repo's local state.

    `retriever.DEFAULT_PERSIST_DIRECTORY` (`.chroma/`) and
    `manifest.DEFAULT_MANIFEST_PATH` (`ingestion_manifest.json`) are both relative
    paths, resolved against the process's current working directory. A test that
    calls `retriever.load_corpus()`/`GuitarAssistantModel.load_context` with no
    explicit `persist_directory` override (most unit tests pass one explicitly, but
    nothing enforces that) would otherwise silently read or write whatever a real
    `guitar-assistant-ingest` run has left in this repo's actual `.chroma/` —
    non-deterministic depending on local disk state, and not network-free. Every
    unit test's fakes are network-free regardless, so this only guards the
    filesystem side.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _clear_chromadb_system_cache():
    """Clear chromadb's cross-process `System` cache after every test.

    `Chroma(persist_directory=...)` keyed by settings is cached process-wide by
    `SharedSystemClient`. Many tests each open a persistent store in its own
    throwaway `tmp_path` (seeding/reopening/ingestion tests, plus
    `GuitarAssistantModel.load_context`'s own open) without ever clearing it, and
    the resulting pile-up of cached `System` objects pointing at directories pytest
    has since removed is what caused the intermittent
    `chromadb.errors.InvalidArgumentError`/`InternalError` ("Failed to pull logs
    from the log store") seen when running the full suite. Clearing the cache
    after each test keeps every persistent-store test starting from a clean slate.
    """
    yield
    SharedSystemClient.clear_system_cache()


class FakeChatModel(BaseChatModel):
    """Deterministic, network-free stand-in for ChatOpenAI.

    Routes every query to a fixed guitar model and answers with a fixed string,
    so `build_agent`'s route/generate chains work without any API calls.
    """

    guitar_model: str
    answer: str
    # Records each schema passed to with_structured_output, so routing tests can assert
    # whether the (fake) LLM was invoked at all, and which shortlist it was scoped to.
    _structured_output_calls: list[type] = PrivateAttr(default_factory=list)

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
        self._structured_output_calls.append(schema)
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
