"""Grade agent answers against the golden dataset (`task/test-questions.csv`).

Correctness here is judged relative to a known-good `expected_answer`, not
freshly re-derived from the source spec sheets — the golden dataset already
encodes the ground truth, so the judge only needs to compare the agent's
actual answer against it, independent of retrieval/corpus size. `grade_answer`
holds that grading logic; `correctness` adapts it into an `mlflow.genai.evaluate()`
scorer, so it backs both the end-to-end test's accuracy check and the logged
MLflow evaluation run.

A custom scorer rather than MLflow's builtin `Correctness`/`Guidelines` judges:
the golden dataset carries both a reference `expected_answer` *and* a per-row
`evaluation_criteria` rubric, which is a hybrid of what those two builtins
grade separately. Running both would cost two LLM calls per row for two
disjoint scores; `grade_answer`'s single prompt already weighs both signals
together in one call.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer
from pydantic import BaseModel

from guitar_assistant.agent import CHAT_MODEL

_JUDGE_PROMPT: Final = ChatPromptTemplate.from_template(
    "You are grading whether an AI assistant's answer to a guitar "
    "specification question is correct, using a known-correct reference answer and "
    "evaluation criteria.\n\n"
    "Question: {question}\n\n"
    "Reference (known-correct) answer: {expected_answer}\n\n"
    "Evaluation criteria: {evaluation_criteria}\n\n"
    "Assistant's actual answer: {actual_answer}\n\n"
    "Does the actual answer satisfy the evaluation criteria and match the "
    "reference answer's substance (wording may differ)?"
)


class JudgeVerdict(BaseModel):
    """An LLM judge's grading of one agent answer.

    Args:
        passed: Whether the actual answer satisfies the evaluation criteria.
        reasoning: A short explanation for the judge's decision.
    """

    passed: bool
    reasoning: str


@dataclass(frozen=True)
class GoldenQuestion:
    """One row of the golden dataset (`task/test-questions.csv`).

    Args:
        question_id: The dataset's question identifier.
        category: The query category (e.g. "Comparative - Cross Manufacturer").
        question: The question text to send to the agent.
        expected_answer: The known-correct reference answer.
        evaluation_criteria: What the answer needs to demonstrate to be graded correct.
    """

    question_id: str
    category: str
    question: str
    expected_answer: str
    evaluation_criteria: str

    def to_scorer_inputs(self) -> dict:
        """Shape this question for `mlflow.genai.evaluate()`'s `data` argument.

        Returns:
            A dict with an `inputs` key matching `correctness`'s (and
            `predict_fn`'s) `query` parameter, and an `expectations` key
            carrying the reference answer and grading rubric `correctness` reads.
        """
        return {
            "inputs": {"query": self.question},
            "expectations": {
                "expected_answer": self.expected_answer,
                "evaluation_criteria": self.evaluation_criteria,
            },
        }


def load_golden_dataset(path: Path) -> list[GoldenQuestion]:
    """Load the golden dataset from its CSV file.

    Args:
        path: Path to the golden dataset CSV, with `Question_ID`, `Category`,
            `Question`, `Expected_Answer`, and `Evaluation_Criteria` columns.

    Returns:
        One `GoldenQuestion` per data row, in file order.
    """
    with path.open(encoding="utf-8", newline="") as csv_file:
        return [
            GoldenQuestion(
                question_id=row["Question_ID"],
                category=row["Category"],
                question=row["Question"],
                expected_answer=row["Expected_Answer"],
                evaluation_criteria=row["Evaluation_Criteria"],
            )
            for row in csv.DictReader(csv_file)
        ]


def grade_answer(
    question: str,
    actual_answer: str,
    expected_answer: str,
    evaluation_criteria: str,
    judge: BaseChatModel | None = None,
) -> JudgeVerdict:
    """Grade an agent's answer against the golden dataset's reference answer.

    Args:
        question: The question that was asked.
        actual_answer: The agent's actual answer.
        expected_answer: The golden dataset's known-correct reference answer.
        evaluation_criteria: What the answer needs to demonstrate to be
            graded correct.
        judge: Chat model used to grade the answer. Defaults to `None`,
            which builds OpenAI's `CHAT_MODEL` lazily. Overridable for
            testing without network access.

    Returns:
        The judge's decision on whether the actual answer is correct.
    """
    chat_model = judge or ChatOpenAI(model=CHAT_MODEL)
    judge_chain = _JUDGE_PROMPT | chat_model.with_structured_output(JudgeVerdict)
    return cast(
        JudgeVerdict,
        judge_chain.invoke(
            {
                "question": question,
                "expected_answer": expected_answer,
                "evaluation_criteria": evaluation_criteria,
                "actual_answer": actual_answer,
            }
        ),
    )


@scorer
def correctness(inputs: dict, outputs: str, expectations: dict) -> Feedback:
    """Adapt `grade_answer` into an `mlflow.genai.evaluate()` scorer.

    Args:
        inputs: This row's `to_scorer_inputs()["inputs"]`, i.e. `{"query": ...}`.
        outputs: The agent's answer for `inputs["query"]`.
        expectations: This row's `to_scorer_inputs()["expectations"]`, carrying
            `expected_answer` and `evaluation_criteria`.

    Returns:
        `Feedback` carrying the judge's pass/fail decision as `value` and its
        explanation as `rationale`, so `mlflow.genai.evaluate()` can aggregate
        and log it as a run metric.
    """
    judge = grade_answer(
        question=inputs["query"],
        actual_answer=outputs,
        expected_answer=expectations["expected_answer"],
        evaluation_criteria=expectations["evaluation_criteria"],
    )
    return Feedback(value=judge.passed, rationale=judge.reasoning)
