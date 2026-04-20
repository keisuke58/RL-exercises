from __future__ import annotations

import numpy as np
from itertools import cycle
from rl_exercises.environments import MarsRover


TRAIN_CONTEXTS = [
    (5.0, 0.0),
    (10.0, 0.2),
    (15.0, 0.4),
]

VAL_CONTEXTS = [
    (5.0, 0.0),
    (10.0, 0.2),
    (15.0, 0.4),
]

# Interpolation: within training range
TEST_INTERP_CONTEXTS = [
    (7.5, 0.1),
    (12.5, 0.3),
]

# Extrapolation: outside training range
TEST_EXTRAP_CONTEXTS = [
    (20.0, 0.6),
    (2.0, 0.5),
]


class ContextualMarsRover(MarsRover):
    """MarsRover with two context features: goal_reward and slip_prob.

    Context features:
      goal_reward: reward at the rightmost state (position 4)
      slip_prob:   probability that the chosen action is flipped (1 - P[s,a])

    On each reset(), the next context in the round-robin cycle is applied.
    """

    def __init__(
        self,
        contexts: list[tuple[float, float]],
        provide_context: bool = False,
        seed: int | None = None,
    ) -> None:
        self.contexts = list(contexts)
        self.provide_context = provide_context
        self._context_iter = cycle(self.contexts)
        self._current_context = self.contexts[0]

        goal_reward, slip_prob = self._current_context
        rewards = [1.0, 0.0, 0.0, 0.0, goal_reward]
        p = 1.0 - slip_prob
        transition_probabilities = np.full((5, 2), p)

        super().__init__(
            transition_probabilities=transition_probabilities,
            rewards=rewards,
            seed=seed,
        )

    def _apply_context(self, context: tuple[float, float]) -> None:
        goal_reward, slip_prob = context
        self._current_context = context
        self.rewards = [1.0, 0.0, 0.0, 0.0, goal_reward]
        p = 1.0 - slip_prob
        self.P = np.full((5, 2), p)
        self.transition_matrix = self.T = self.get_transition_matrix()

    def reset(self, *, seed=None, options=None):
        self._apply_context(next(self._context_iter))
        return super().reset(seed=seed, options=options)

    def get_context(self) -> tuple[float, float]:
        return self._current_context
