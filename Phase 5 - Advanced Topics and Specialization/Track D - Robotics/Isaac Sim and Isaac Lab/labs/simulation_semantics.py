"""Scalar identification and paired-outcome experiments, independent of Isaac."""
import json
import math
import random
from robotics_contracts import finite, positive_int, wilson_interval


def actuator_response(commands, dt, tau, delay, initial=0):
    if finite(dt, "dt") <= 0 or finite(tau, "tau") <= 0:
        raise ValueError("positive dt and tau required")
    if type(delay) is not int or delay < 0:
        raise ValueError("delay must be a nonnegative integer")
    a = math.exp(-dt/tau)
    values = [finite(initial, "initial")]
    commands = [finite(x, "command") for x in commands]
    for index in range(len(commands)):
        delayed = commands[index-delay] if index >= delay else 0
        values.append(a*values[-1] + (1-a)*delayed)
    return values


def identify_actuator(commands, measured, dt, max_delay=6):
    """Fit each integer delay by least squares, constrained to 0<a<1.

    y[k+1]-u[k-d] = a*(y[k]-u[k-d]); a=exp(-dt/tau).
    This simple estimator is biased by noise in y[k]; use residuals and held-out data.
    """
    if len(measured) != len(commands)+1 or len(commands) < 3:
        raise ValueError("need K commands and K+1 measured states")
    if finite(dt, "dt") <= 0 or type(max_delay) is not int or max_delay < 0:
        raise ValueError("invalid dt or delay range")
    for x in [*commands, *measured]: finite(x, "signal")
    candidates = []
    for delay in range(max_delay+1):
        x, y = [], []
        for index, current in enumerate(measured[:-1]):
            u = commands[index-delay] if index >= delay else 0
            x.append(current-u); y.append(measured[index+1]-u)
        denominator = sum(v*v for v in x)
        if denominator < 1e-12: continue
        a = sum(v*w for v,w in zip(x,y))/denominator
        if not 0 < a < 1: continue
        residual = sum((w-a*v)**2 for v,w in zip(x,y))/len(x)
        candidates.append({"delay_steps": delay, "delay_s": delay*dt, "a": a,
                           "tau_s": -dt/math.log(a), "one_step_mse": residual})
    if not candidates:
        raise ValueError("input does not identify a stable first-order response")
    return min(candidates, key=lambda row: row["one_step_mse"]), candidates


def paired_regressions(baseline, candidate, max_regression=.01):
    """One-sided 95% Wilson bound on P(baseline succeeds AND candidate fails).

    This is a conservative declared outcome gate, not a proof of simulator equivalence.
    Dict keys must identify the exact same held-out scenarios, sampled independently.
    """
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError("paired scenario IDs must be nonempty and identical")
    if any(type(x) is not bool for x in [*baseline.values(), *candidate.values()]):
        raise ValueError("outcomes must be Boolean")
    if not 0 < finite(max_regression, "max_regression") < 1:
        raise ValueError("margin must lie between zero and one")
    losses = sum(baseline[key] and not candidate[key] for key in baseline)
    gains = sum(not baseline[key] and candidate[key] for key in baseline)
    upper = wilson_interval(losses, len(baseline), z=1.6448536269514722)[1]
    return {"episodes": len(baseline), "regressions": losses, "improvements": gains,
            "one_sided_95_upper_regression_probability": upper,
            "declared_margin": max_regression, "gate_passed": upper <= max_regression}


def report():
    rng = random.Random(42)
    commands = [rng.choice([-1., -.5, .5, 1.]) for _ in range(400)]
    truth = actuator_response(commands, .02, .08, 2)
    measured = [value+rng.gauss(0,.001) for value in truth]
    fitted, candidates = identify_actuator(commands, measured, .02)
    # No refitting on this held-out input distribution.
    validation = [.8*math.sin(i*.17) for i in range(200)]
    expected = actuator_response(validation, .02, .08, 2)
    predicted = actuator_response(validation, .02, fitted["tau_s"], fitted["delay_steps"])
    error = math.sqrt(sum((a-b)**2 for a,b in zip(expected,predicted))/len(expected))
    return {"scope": "synthetic scalar response and paired binary fixtures; no robot identification",
            "truth": {"tau_s": .08, "delay_steps": 2, "dt_s": .02}, "fitted": fitted,
            "delay_candidates": candidates, "held_out_rmse": error,
            "zero_regressions_100": paired_regressions(dict.fromkeys(range(100), True), dict.fromkeys(range(100), True)),
            "zero_regressions_1000": paired_regressions(dict.fromkeys(range(1000), True), dict.fromkeys(range(1000), True))}


if __name__ == "__main__": print(json.dumps(report(), indent=2, allow_nan=False))
