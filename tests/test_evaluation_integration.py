"""Integration test for the evaluation: Checks the real judge against meta test-questions.

Requires a real `OPENAI_API_KEY` and hits the network, so it's marked
`integration` and excluded from the default `uv run pytest` run. Run
explicitly with `uv run pytest -m integration tests/test_evaluation_integration.py`.

Runs the real `grade_answer()` function against a small set of hand-crafted
calibration cases, to verify that the judge's grading is reasonable: it should
accept correct answers phrased differently from the reference (not overly
strict), and reject plausible-looking wrong answers (not overly lenient).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pytest

from guitar_assistant.evaluation import grade_answer


@dataclass(frozen=True)
class _CalibrationCase:
    """One hand-crafted question/answer pair for judge calibration.

    Args:
        label: Short identifier for this case, used as the test id.
        question: The question posed to the assistant.
        actual_answer: The assistant answer to grade.
        expected_answer: The known-correct reference answer.
        evaluation_criteria: What the answer needs to demonstrate to be graded correct.
        expected_passed: Whether a well-calibrated judge should mark this correct.
    """

    label: str
    question: str
    actual_answer: str
    expected_answer: str
    evaluation_criteria: str
    expected_passed: bool


_CALIBRATION_CASES: Final = (
    _CalibrationCase(
        label="exact_match",
        question="What is the scale length of the Telecaster?",
        actual_answer="The scale length of the Telecaster is 25.5 in (648 mm).",
        expected_answer=(
            "The scale length of the Telecaster is 25.5 in (648 mm), the standard "
            "Fender scale length."
        ),
        evaluation_criteria="Accuracy of scale length specification; source identification",
        expected_passed=True,
    ),
    _CalibrationCase(
        label="paraphrase",
        question="How many pickups does the Stratocaster have?",
        actual_answer=(
            "The Stratocaster comes with three single-coil pickups, positioned at the "
            "bridge, middle, and neck."
        ),
        expected_answer="The Stratocaster has 3 single-coil pickups (bridge, middle, neck).",
        evaluation_criteria="Correct pickup count and configuration",
        expected_passed=True,
    ),
    _CalibrationCase(
        label="verbose_correct",
        question="What type of pickups does the Gibson SG use?",
        actual_answer=(
            "The Gibson SG is equipped with two humbucker pickups, one at the bridge "
            "and one at the neck, giving it a thick, high-output tone. As background, "
            "humbuckers were developed to cancel the 60-cycle hum that plagued "
            "single-coil designs."
        ),
        expected_answer=(
            "The Gibson SG features 2 humbucker pickups (bridge and neck) for a thicker, "
            "higher-output tone than single-coil pickups."
        ),
        evaluation_criteria="Pickup type accuracy; understanding of tonal characteristics",
        expected_passed=True,
    ),
    _CalibrationCase(
        label="wrong_numeric_value",
        question="What is the scale length of the Gibson SG?",
        actual_answer="The scale length of the Gibson SG is 25.5 in.",
        expected_answer="The scale length of the Gibson SG is 24.75 in.",
        evaluation_criteria="Correct scale length specification",
        expected_passed=False,
    ),
    _CalibrationCase(
        label="off_by_small_amount",
        question="What is the typical weight of a Telecaster?",
        actual_answer="A Telecaster typically weighs around 8 lb.",
        expected_answer="A Telecaster typically weighs 6.5 - 7.5 lb.",
        evaluation_criteria="Accurate weight range",
        expected_passed=False,
    ),
    _CalibrationCase(
        label="swapped_facts",
        question="Compare the bridge type of the Telecaster and Stratocaster.",
        actual_answer=(
            "The Telecaster has a synchronized tremolo bridge, while the Stratocaster "
            "has a fixed bridge."
        ),
        expected_answer=(
            "The Telecaster has a fixed bridge with no tremolo, while the Stratocaster "
            "has a synchronized tremolo (vibrato) bridge."
        ),
        evaluation_criteria="Cross-document retrieval; accurate technical comparison",
        expected_passed=False,
    ),
    _CalibrationCase(
        label="partial_multipart",
        question="What are the differences between the Telecaster and the Stratocaster?",
        actual_answer="The Stratocaster has one more pickup than the Telecaster.",
        expected_answer=(
            "The Stratocaster differs from the Telecaster in three key ways: 3 pickups "
            "instead of 2, a synchronized tremolo bridge instead of a fixed bridge, and "
            "a 5-way switch instead of a 3-way switch."
        ),
        evaluation_criteria="Accurate comparison; synthesis of multiple specifications; "
        "clear differentiation",
        expected_passed=False,
    ),
    _CalibrationCase(
        label="wrong_negative_confirmation",
        question="Which guitar models support a tremolo/vibrato arm as standard equipment?",
        actual_answer=(
            "The Telecaster and the Stratocaster both support a tremolo arm as standard "
            "equipment. The Gibson SG does not."
        ),
        expected_answer=(
            "The Stratocaster supports a tremolo arm as standard equipment. The "
            "Telecaster does not have one in its standard configuration, and the Gibson "
            "SG typically ships with a fixed tune-o-matic/stopbar tailpiece instead."
        ),
        evaluation_criteria="Feature availability across models; negative confirmation "
        "for unsupported models",
        expected_passed=False,
    ),
    _CalibrationCase(
        label="hedging_non_answer",
        question="How does neck construction compare across all three guitar models?",
        actual_answer=(
            "I don't have enough information to compare neck construction across these "
            "models; please consult the manufacturer's documentation."
        ),
        expected_answer=(
            "Neck construction differs: the Telecaster and Stratocaster both use "
            "bolt-on maple necks, while the Gibson SG uses a set (glued) mahogany neck."
        ),
        evaluation_criteria="Complete construction comparison; understanding of design differences",
        expected_passed=False,
    ),
    _CalibrationCase(
        label="wrong_configuration_procedure",
        question="How do you select the bridge pickup only on a Stratocaster?",
        actual_answer=(
            "Set the 5-way selector switch to the neck position to select the bridge "
            "pickup on the Stratocaster."
        ),
        expected_answer=(
            "Set the 5-way selector switch to position 1 (all the way toward the "
            "bridge) to select the bridge pickup only on the Stratocaster."
        ),
        evaluation_criteria="Exact switch position knowledge; configuration procedure "
        "understanding",
        expected_passed=False,
    ),
    _CalibrationCase(
        label="contradicts_but_confident",
        question="Which guitar model has the shortest scale length?",
        actual_answer=(
            "The Stratocaster has the shortest scale length at 24.75 in, shorter than "
            "both the Telecaster and the Gibson SG."
        ),
        expected_answer=(
            "The Gibson SG has the shortest scale length at 24.75 in (629 mm). Both the "
            "Telecaster and Stratocaster use the longer 25.5 in (648 mm) Fender scale "
            "length."
        ),
        evaluation_criteria="Scale length comparison; identification of correct model; "
        "quantitative analysis",
        expected_passed=False,
    ),
)


@pytest.mark.integration
@pytest.mark.parametrize("case", _CALIBRATION_CASES, ids=lambda case: case.label)
def test_judge_matches_expected_verdict_for_calibration_case(case: _CalibrationCase) -> None:
    # GIVEN a hand-crafted question/answer pair with a known-correct pass/fail verdict
    # WHEN the real judge grades the actual answer
    verdict = grade_answer(
        question=case.question,
        actual_answer=case.actual_answer,
        expected_answer=case.expected_answer,
        evaluation_criteria=case.evaluation_criteria,
    )
    # THEN the judge's verdict matches the expected outcome
    assert verdict.passed == case.expected_passed, verdict.reasoning
