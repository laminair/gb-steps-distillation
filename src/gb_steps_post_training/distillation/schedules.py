"""Linear warm-up schedules for the GOLD trainer's lmbda, beta and alpha.

WHY THIS IS ITS OWN MODULE rather than three method bodies. The trainer had three hand-copied
versions of the same interpolation, and one of them (lmbda) lost its "linear" branch to dead code
below an early return: `_get_current_lmbda()` returned None, and every lmbda_schedule=linear run
crashed at step 0 on `None <= 0.0`. Nothing caught it because importing custom_gold_trainer pulls
in vllm, so the methods could not be exercised without the full GPU stack. Same reasoning as
on_policy_draw.py: a decision that cannot be tested cheaply ships untested.

Torch-free on purpose, so render_gold_config.py can share the same definition of "the schedule's
endpoints" when it validates a config.
"""

from __future__ import annotations

SCHEDULES = ("constant", "linear")


def scheduled_value(
    schedule: str, init: float | None, final: float, global_step: int, max_steps: int
) -> float:
    """Value of a scheduled hyperparameter at `global_step`.

    `constant` returns `final`. `linear` goes from `init` at step 0 to `final` at `max_steps`, and
    holds at `final` after that. Before the trainer knows `max_steps` (<= 0), it returns `init`.
    Any other schedule raises, rather than falling through to an implicit None -- that fall-through
    is exactly the bug this module replaces.
    """
    if schedule == "constant":
        return final
    if schedule == "linear":
        if init is None:
            raise ValueError("a linear schedule needs an init value; got None.")
        if max_steps <= 0:
            return init
        progress = min(1.0, global_step / max_steps)
        return init + progress * (final - init)
    raise ValueError(f"schedule must be one of {SCHEDULES}; got {schedule!r}.")


def schedule_range(schedule: str, init: float | None, final: float) -> tuple[float, float]:
    """(min, max) the schedule takes over the whole run. Linear is monotone, so its endpoints suffice."""
    if schedule == "linear" and init is not None:
        return (min(init, final), max(init, final))
    return (final, final)
