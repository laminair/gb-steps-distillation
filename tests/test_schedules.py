import ast
from pathlib import Path

import pytest

from gb_steps_post_training.distillation.schedules import schedule_range, scheduled_value

TRAINER = (
    Path(__file__).resolve().parent.parent
    / "src/gb_steps_post_training/distillation/custom_gold_trainer.py"
)


def test_constant_returns_final():
    assert scheduled_value("constant", None, 0.7, 50, 100) == 0.7
    assert scheduled_value("constant", 0.0, 0.7, 50, 100) == 0.7


@pytest.mark.parametrize(
    "step, expected", [(0, 0.0), (25, 0.25), (50, 0.5), (100, 1.0), (150, 1.0)]
)
def test_linear_ramps_and_holds(step, expected):
    assert scheduled_value("linear", 0.0, 1.0, step, 100) == pytest.approx(expected)


def test_linear_ramps_down():
    assert scheduled_value("linear", 1.0, 0.2, 50, 100) == pytest.approx(0.6)


@pytest.mark.parametrize("max_steps", [0, -1])
def test_linear_before_max_steps_known_returns_init(max_steps):
    assert scheduled_value("linear", 0.3, 1.0, 5, max_steps) == 0.3


def test_linear_always_returns_a_float():
    # The lmbda regression: "linear" fell through and returned None.
    for step in range(0, 101, 10):
        assert isinstance(scheduled_value("linear", 0.0, 1.0, step, 100), float)


def test_linear_without_init_raises():
    with pytest.raises(ValueError, match="init"):
        scheduled_value("linear", None, 1.0, 0, 100)


def test_unknown_schedule_raises_instead_of_returning_none():
    with pytest.raises(ValueError, match="cosine"):
        scheduled_value("cosine", 0.0, 1.0, 0, 100)


def test_schedule_range():
    assert schedule_range("constant", 0.0, 0.5) == (0.5, 0.5)
    assert schedule_range("linear", 0.0, 0.5) == (0.0, 0.5)
    assert schedule_range("linear", 0.9, 0.5) == (0.5, 0.9)


@pytest.mark.parametrize(
    "method", ["_get_current_lmbda", "_get_current_beta", "_get_current_alpha"]
)
def test_trainer_schedule_methods_delegate(method):
    # The trainer can't be imported without vllm, so check its source: each schedule getter must
    # go through scheduled_value rather than carry its own copy of the interpolation -- a
    # hand-copied branch is how lmbda's "linear" case ended up as dead code.
    tree = ast.parse(TRAINER.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CustomGOLDTrainer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    calls = {
        c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }
    assert "scheduled_value" in calls
