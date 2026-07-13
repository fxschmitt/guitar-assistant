"""Unit tests for guitar_assistant.evaluation."""

from pathlib import Path
from typing import Any, Final

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from mlflow.entities import Feedback
import pytest

from guitar_assistant.evaluation import (
    GoldenQuestion,
    JudgeVerdict,
    correctness,
    grade_answer,
    load_golden_dataset,
)

_GOLDEN_DATASET_HEADER: Final = (
    "Question_ID,Category,Question,Expected_Answer,Evaluation_Criteria\n"
)


class _FakeJudge(BaseChatModel):
    """Deterministic, network-free stand-in for ChatOpenAI.

    Always returns a fixed `JudgeVerdict` from `with_structured_output`, so
    `grade_answer`'s prompt/parsing wiring can be tested without any API calls.
    """

    judge_verdict: JudgeVerdict

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
        raise NotImplementedError("Only with_structured_output is exercised by grade_answer.")

    def with_structured_output(
        self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        return RunnableLambda(lambda _input: self.judge_verdict)


@pytest.fixture(name="fake_judge")
def fixture_fake_judge() -> _FakeJudge:
    return _FakeJudge(
        judge_verdict=JudgeVerdict(passed=True, reasoning="Matches the reference answer.")
    )


def test_load_golden_dataset_parses_rows_into_dataclasses(tmp_path: Path):
    # GIVEN a golden dataset CSV with one data row
    csv_path = tmp_path / "test-questions.csv"
    csv_path.write_text(
        _GOLDEN_DATASET_HEADER + '1,Single Source,"What is X?","X is 5.","Accuracy of X"\n',
        encoding="utf-8",
    )
    # WHEN loading the golden dataset
    questions = load_golden_dataset(csv_path)
    # THEN each row maps onto a GoldenQuestion with matching fields
    assert questions == [
        GoldenQuestion(
            question_id="1",
            category="Single Source",
            question="What is X?",
            expected_answer="X is 5.",
            evaluation_criteria="Accuracy of X",
        )
    ]


def test_golden_question_to_scorer_inputs_shapes_inputs_and_expectations():
    # GIVEN a golden dataset question
    question = GoldenQuestion(
        question_id="1",
        category="Single Source",
        question="What is X?",
        expected_answer="X is 5.",
        evaluation_criteria="Accuracy of X",
    )
    # WHEN shaping it for mlflow.genai.evaluate()
    scorer_inputs = question.to_scorer_inputs()
    # THEN the question becomes the query input, and the reference answer/rubric
    # become the expectations correctness reads
    assert scorer_inputs == {
        "inputs": {"query": "What is X?"},
        "expectations": {
            "expected_answer": "X is 5.",
            "evaluation_criteria": "Accuracy of X",
        },
    }


def test_grade_answer_returns_the_judges_decision(fake_judge: _FakeJudge):
    # GIVEN a fake judge configured to return a fixed decision
    # WHEN grading an answer
    judge = grade_answer(
        question="What is X?",
        actual_answer="X is 5.",
        expected_answer="X is 5.",
        evaluation_criteria="Accuracy of X",
        judge=fake_judge,
    )
    # THEN grade_answer returns the judge's decision unchanged
    assert judge == fake_judge.judge_verdict


def test_correctness_scorer_wraps_grade_answer_as_feedback(
    fake_judge: _FakeJudge, monkeypatch: pytest.MonkeyPatch
):
    # GIVEN grade_answer using a fake judge configured to return a fixed decision
    monkeypatch.setattr(
        "guitar_assistant.evaluation.grade_answer",
        lambda **kwargs: grade_answer(judge=fake_judge, **kwargs),
    )
    # WHEN scoring one row's inputs/outputs/expectations
    feedback = correctness(
        inputs={"query": "What is X?"},
        outputs="X is 5.",
        expectations={"expected_answer": "X is 5.", "evaluation_criteria": "Accuracy of X"},
    )
    # THEN the feedback carries the judge's decision as value and reasoning as rationale
    assert isinstance(feedback, Feedback)
    assert feedback.value == fake_judge.judge_verdict.passed
    assert feedback.rationale == fake_judge.judge_verdict.reasoning
