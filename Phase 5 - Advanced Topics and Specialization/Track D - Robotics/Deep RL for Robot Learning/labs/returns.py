"""Inspectable CPU reference for GAE boundaries and PPO clipping; no dependencies."""

from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class Transition:
    reward: float
    value: float
    next_value: float  # TRUE post-action state, before reset; ignored on termination.
    terminated: bool = False
    truncated: bool = False


def gae(transitions: list[Transition], gamma: float, lam: float) -> tuple[list[float], list[float]]:
    """One environment's transitions, possibly containing episode boundaries.

    Reset-only events must be removed by the environment adapter before calling.
    At the end of the finite rollout, GAE recursion stops but TD still bootstraps
    from next_value when not terminated.
    """
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lambda must be in [0,1]")
    if not transitions:
        raise ValueError("a rollout must contain at least one transition")
    if not all(math.isfinite(x) for t in transitions for x in (t.reward, t.value, t.next_value)):
        raise ValueError("rollout values must be finite")
    advantages = [0.] * len(transitions)
    carry = 0.
    for index in reversed(range(len(transitions))):
        row = transitions[index]
        bootstrap = 0. if row.terminated else row.next_value
        delta = row.reward + gamma * bootstrap - row.value
        trace_continues = not (row.terminated or row.truncated)
        carry = delta + gamma * lam * trace_continues * carry
        advantages[index] = carry
    return advantages, [a + row.value for a, row in zip(advantages, transitions)]


def clipped_surrogate(ratio: float, advantage: float, epsilon: float = .2) -> float:
    """PPO objective contribution to MAXIMIZE, not the loss to minimize."""
    if not all(math.isfinite(x) for x in (ratio, advantage, epsilon)):
        raise ValueError("inputs must be finite")
    if ratio < 0 or not 0 <= epsilon < 1:
        raise ValueError("ratio must be nonnegative and clip epsilon in [0,1)")
    clipped = max(1 - epsilon, min(1 + epsilon, ratio))
    return min(ratio * advantage, clipped * advantage)


def discounted_chunk_target(rewards: list[float], next_value: float, gamma: float,
                            terminated: bool = False) -> float:
    """Semi-MDP target for the actions actually executed before this boundary.

    An external truncation keeps the final-state bootstrap; true termination
    removes it. A shorter final chunk uses its actual number of rewards.
    """
    if not rewards or type(terminated) is not bool or not 0 <= gamma <= 1:
        raise ValueError("nonempty rewards, boolean termination, and gamma in [0,1] required")
    if not all(math.isfinite(x) for x in [*rewards, next_value]):
        raise ValueError("rewards and next value must be finite")
    total = sum(gamma ** j * reward for j, reward in enumerate(rewards))
    result = total if terminated else total + gamma ** len(rewards) * next_value
    if not math.isfinite(result):
        raise ValueError("nonfinite chunk target")
    return result


def example() -> dict:
    rows = [Transition(1., .5, 1.), Transition(2., 1., 4., truncated=True),
            Transition(3., 9., 0., terminated=True)]
    advantages, targets = gae(rows, .9, .8)
    return {"execution_status": "synthetic arithmetic fixture; no training",
            "gamma": .9, "lambda": .8,
            "two_step_chunk_target": discounted_chunk_target([1., 2.], 10., .9),
            "two_step_terminal_target": discounted_chunk_target([1., 2.], 10., .9, True), "advantages": advantages, "value_targets": targets,
            "timeout_correct_target": targets[1],
            "timeout_wrong_terminal_target": 2.,
            "timeout_wrong_reset_bootstrap_target": 2. + .9 * 9.,
            "ppo_examples": [{"ratio": ratio, "advantage": advantage,
                              "objective": clipped_surrogate(ratio, advantage)}
                             for ratio in (.5, 1., 1.5) for advantage in (-2., 2.)]}


if __name__ == "__main__":
    print(json.dumps(example(), indent=2, allow_nan=False))
