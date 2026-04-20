"""Level 3 experiment: contextual MarsRover with and without context."""
from __future__ import annotations

import numpy as np
from rich import print as printr
from rl_exercises.week_2.value_iteration import value_iteration
from rl_exercises.week_2.contextual_mars_rover import (
    ContextualMarsRover,
    TRAIN_CONTEXTS,
    VAL_CONTEXTS,
    TEST_INTERP_CONTEXTS,
    TEST_EXTRAP_CONTEXTS,
)


N_EVAL_EPISODES = 200
GAMMA = 0.9
SEED = 42


def make_env_for_context(context: tuple[float, float]) -> ContextualMarsRover:
    """Return an env locked to a single context (no rotation)."""
    return ContextualMarsRover(contexts=[context], seed=SEED)


def evaluate_policy(pi: np.ndarray, context: tuple[float, float]) -> float:
    """Evaluate policy pi on a fixed context for N_EVAL_EPISODES."""
    env = make_env_for_context(context)
    total = 0.0
    for _ in range(N_EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        while not done:
            action = int(pi[obs])
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    return total / N_EVAL_EPISODES


def optimal_policy_for_context(context: tuple[float, float]) -> np.ndarray:
    """Compute the VI-optimal policy for a given context."""
    env = make_env_for_context(context)
    _, pi = value_iteration(T=env.T, R_sa=env.get_reward_per_action(), gamma=GAMMA, seed=SEED)
    return pi


def averaged_policy_from_train() -> np.ndarray:
    """Compute a policy using MDP parameters averaged over training contexts."""
    n_states, n_actions = 5, 2
    T_avg = np.zeros((n_states, n_actions, n_states))
    R_avg = np.zeros((n_states, n_actions))
    for ctx in TRAIN_CONTEXTS:
        env = make_env_for_context(ctx)
        T_avg += env.T
        R_avg += env.get_reward_per_action()
    T_avg /= len(TRAIN_CONTEXTS)
    R_avg /= len(TRAIN_CONTEXTS)
    _, pi = value_iteration(T=T_avg, R_sa=R_avg, gamma=GAMMA, seed=SEED)
    return pi


def run_experiment(label: str, contexts: list[tuple[float, float]], pi_blind: np.ndarray) -> None:
    printr(f"\n[bold]--- {label} ---[/bold]")
    printr(f"{'Context':>25}  {'With context':>14}  {'Without context':>16}")
    for ctx in contexts:
        pi_ctx = optimal_policy_for_context(ctx)
        r_with = evaluate_policy(pi_ctx, ctx)
        r_without = evaluate_policy(pi_blind, ctx)
        printr(f"  goal={ctx[0]:4.1f} slip={ctx[1]:3.1f}   {r_with:14.2f}  {r_without:16.2f}")


if __name__ == "__main__":
    printr("[bold cyan]Computing context-blind policy from averaged training MDP...[/bold cyan]")
    pi_blind = averaged_policy_from_train()
    printr("Context-blind policy:", pi_blind)

    run_experiment("Validation Set", VAL_CONTEXTS, pi_blind)
    run_experiment("Test Set (Interpolation)", TEST_INTERP_CONTEXTS, pi_blind)
    run_experiment("Test Set (Extrapolation)", TEST_EXTRAP_CONTEXTS, pi_blind)
