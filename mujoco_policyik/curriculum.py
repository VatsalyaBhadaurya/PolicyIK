"""Reverse curriculum for PolicyIK training (Section 4.5 / Appendix H.4).

Early in training the arm would rarely reach tolerance from random starts, so
episodes begin near the goal and the policy learns the final correction and
holding first; the start spread then grows. Here the start distribution
interpolates from a tight near-goal Gaussian (``std_start`` rad) to a wider one
(``std_end``), after which starts become fully random over the workspace.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class ReverseCurriculumCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        std_start: float = 0.1,
        std_end: float = 1.5,
        warm_frac: float = 0.8,
        update_every: int = 4096,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.std_start = float(std_start)
        self.std_end = float(std_end)
        self.warm_frac = float(warm_frac)
        self.update_every = int(update_every)
        self._next_update = 0

    def _current_std(self) -> float | None:
        frac = self.num_timesteps / max(self.total_timesteps, 1)
        if frac >= self.warm_frac:
            return None  # fully random starts
        p = frac / self.warm_frac
        return self.std_start + p * (self.std_end - self.std_start)

    def _on_step(self) -> bool:
        # Update the start spread on a coarse schedule, not every env step.
        if self.num_timesteps < self._next_update:
            return True
        self._next_update = self.num_timesteps + self.update_every
        std = self._current_std()
        self.training_env.env_method("set_start_std", std)
        self.logger.record("curriculum/start_std", -1.0 if std is None else std)
        if self.verbose:
            label = "random" if std is None else f"{std:.3f} rad"
            print(f"[curriculum] t={self.num_timesteps}: start spread -> {label}", flush=True)
        return True
