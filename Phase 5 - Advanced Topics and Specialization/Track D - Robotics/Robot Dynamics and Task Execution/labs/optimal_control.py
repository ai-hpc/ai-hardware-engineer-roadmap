"""Discrete finite-horizon LQR and a deliberately tiny finite-action MPC search."""
import itertools
import math

from dynamics import finite


def step(state, acceleration, dt):
    """Exact double-integrator transition for a held acceleration over dt."""
    position, velocity = state
    for value, name in ((position, "position"), (velocity, "velocity"),
                        (acceleration, "acceleration"), (dt, "dt")):
        finite(value, name)
    if dt <= 0:
        raise ValueError("dt must be positive")
    return position + velocity * dt + 0.5 * acceleration * dt * dt, velocity + acceleration * dt


def diagonal_cost(state, weights):
    return sum(weight * value * value for weight, value in zip(weights, state))


def finite_lqr(dt, horizon, q=(10.0, 1.0), r=0.1, terminal=(10.0, 1.0)):
    """Backward Riccati recursion, returning K[0:H] and the 2x2 initial P.

    State is (position error, velocity), control is acceleration. Q and terminal
    cost are diagonal, R is scalar. No input/state constraints are solved here.
    """
    if finite(dt, "dt") <= 0 or finite(r, "r") <= 0:
        raise ValueError("dt and r must be positive")
    if type(horizon) is not int or not 1 <= horizon <= 2000:
        raise ValueError("horizon must be an integer in 1..2000")
    if len(q) != 2 or len(terminal) != 2:
        raise ValueError("two diagonal state weights required")
    if any(finite(value, "cost weight") < 0 for value in (*q, *terminal)):
        raise ValueError("state cost weights must be nonnegative")
    a = ((1.0, dt), (0.0, 1.0))
    b = (0.5 * dt * dt, dt)
    p = [[terminal[0], 0.0], [0.0, terminal[1]]]
    backward_gains = []
    for _ in range(horizon):
        pb = [sum(p[i][j] * b[j] for j in range(2)) for i in range(2)]
        denominator = r + sum(b[i] * pb[i] for i in range(2))
        # cross = B^T P A; symmetry gives A^T P B = cross^T.
        cross = [sum(pb[i] * a[i][j] for i in range(2)) for j in range(2)]
        gain = [value / denominator for value in cross]
        atpa = [[sum(a[k][i] * p[k][ell] * a[ell][j]
                     for k in range(2) for ell in range(2))
                 for j in range(2)] for i in range(2)]
        p = [[(q[i] if i == j else 0.0) + atpa[i][j]
              - cross[i] * cross[j] / denominator for j in range(2)] for i in range(2)]
        if any(not math.isfinite(value) for row in p for value in row):
            raise ValueError("Riccati arithmetic overflow")
        backward_gains.append(gain)
    return list(reversed(backward_gains)), p


def lqr_rollout(initial, gains, dt, q=(10.0, 1.0), r=0.1,
                terminal=(10.0, 1.0), acceleration_limit=None):
    if acceleration_limit is not None and finite(acceleration_limit, "limit") <= 0:
        raise ValueError("acceleration limit must be positive")
    state, cost, trace = tuple(initial), 0.0, []
    for gain in gains:
        requested = -sum(k * x for k, x in zip(gain, state))
        applied = requested if acceleration_limit is None else max(
            -acceleration_limit, min(acceleration_limit, requested))
        cost += diagonal_cost(state, q) + r * applied * applied
        state = step(state, applied, dt)
        trace.append({"state": state, "requested_acceleration": requested,
                      "applied_acceleration": applied})
    return {"cost": cost + diagonal_cost(state, terminal), "trace": trace,
            "final_state": state}


def position_range(state, acceleration, dt):
    """Include an interior turning point; sample endpoints alone are insufficient."""
    end = step(state, acceleration, dt)
    positions = [state[0], end[0]]
    if acceleration != 0:
        turning_time = -state[1] / acceleration
        if 0 < turning_time < dt:
            positions.append(step(state, acceleration, turning_time)[0])
    return min(positions), max(positions)


def finite_action_mpc(initial, *, target=0.45, dt=0.2, horizon=5,
                      actions=(-1.0, 0.0, 1.0), position_bounds=(-1.0, 0.5),
                      speed_limit=1.0):
    """Enumerate all candidate sequences; return the best feasible plan or None.

    This is exact only for this finite action set, horizon, cost, and ideal model.
    It is not a general QP solver, a recursive-feasibility proof, or an emergency
    stop. Position constraints include the whole continuous segment per action.
    """
    if type(horizon) is not int or not 1 <= horizon <= 8:
        raise ValueError("horizon must be an integer in 1..8")
    if len(initial) != 2 or len(position_bounds) != 2:
        raise ValueError("two state components and two position bounds required")
    if not actions or len(actions) ** horizon > 50000:
        raise ValueError("candidate tree must contain 1..50000 sequences")
    for value in (*initial, *actions, *position_bounds, target, dt, speed_limit):
        finite(value, "MPC input")
    lower, upper = position_bounds
    if dt <= 0 or speed_limit <= 0 or lower >= upper:
        raise ValueError("invalid timing or state bounds")
    if not lower <= initial[0] <= upper or abs(initial[1]) > speed_limit:
        return None
    best = None
    for sequence in itertools.product(actions, repeat=horizon):
        state, cost, states = tuple(initial), 0.0, []
        for action in sequence:
            low, high = position_range(state, action, dt)
            next_state = step(state, action, dt)
            if low < lower - 1e-12 or high > upper + 1e-12 or abs(next_state[1]) > speed_limit + 1e-12:
                break
            cost += 40 * (state[0] - target) ** 2 + state[1] ** 2 + 0.1 * action ** 2
            state = next_state
            states.append(state)
        else:
            cost += 40 * (state[0] - target) ** 2 + 5 * state[1] ** 2
            if best is None or cost < best["cost"]:
                best = {"actions": sequence, "states": states, "cost": cost,
                        "candidate_sequences": len(actions) ** horizon}
    return best
