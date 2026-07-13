"""Unit tests for the `guitar_assistant` CLI entrypoint (`guitar_assistant.__init__`)."""

from click.testing import CliRunner
from langchain_core.documents import Document

from guitar_assistant import main


def test_main_invokes_the_compiled_agent_with_the_cli_query(monkeypatch):
    # GIVEN a pipeline stubbed out to avoid network calls, whose compiled agent
    # records the state it was invoked with
    documents = [
        Document(
            page_content="content",
            metadata={"guitar_model": "stratocaster", "source": "stratocaster.md"},
        )
    ]
    monkeypatch.setattr("guitar_assistant.load_documents", lambda: documents)
    monkeypatch.setattr("guitar_assistant.build_vector_store", lambda docs: "fake-vector-store")
    invocations = []

    class _FakeAgent:
        def invoke(self, state: dict) -> dict:
            invocations.append(state)
            return {"answer": "answer"}

    monkeypatch.setattr("guitar_assistant.build_agent", lambda vector_store, models: _FakeAgent())

    # WHEN running the CLI with a query
    result = CliRunner().invoke(main, ["What is the scale length?"])

    # THEN the compiled agent was invoked with that query
    assert result.exit_code == 0
    assert invocations == [{"query": "What is the scale length?"}]
