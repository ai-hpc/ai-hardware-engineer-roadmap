#!/usr/bin/env python3
"""CPU teaching fixtures for robotics contracts; no simulator or hardware claims."""
import json
import math


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def rates(num_envs, policy_steps, decimation, physics_dt_s, wall_s):
    for v, name in ((num_envs, "num_envs"), (policy_steps, "policy_steps"), (decimation, "decimation")):
        positive_int(v, name)
    for v, name in ((physics_dt_s, "physics_dt_s"), (wall_s, "wall_s")):
        if finite(v, name) <= 0:
            raise ValueError(name + " must be positive")
    return {
        "transitions_per_s": num_envs * policy_steps / wall_s,
        "vector_physics_steps_per_s": policy_steps * decimation / wall_s,
        "aggregate_environment_physics_steps_per_s": num_envs * policy_steps * decimation / wall_s,
        "per_environment_real_time_factor": policy_steps * decimation * physics_dt_s / wall_s,
        "policy_period_s": physics_dt_s * decimation,
    }


def rotate_xyzw(quaternion, vector):
    if len(quaternion) != 4 or len(vector) != 3:
        raise ValueError("expected xyzw quaternion and 3-vector")
    for x in (*quaternion, *vector):
        finite(x, "component")
    norm = math.sqrt(sum(x*x for x in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("quaternion must be unit length")
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return (vx + w*tx + y*tz-z*ty,
            vy + w*ty + z*tx-x*tz,
            vz + w*tz + x*ty-y*tx)


def partial_reset(rows, reset_ids, replacement):
    """Pure list fixture: copy every row, replace selected rows, never alias input."""
    ids = list(reset_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("reset IDs must be unique")
    if any(type(i) is not int or not 0 <= i < len(rows) for i in ids):
        raise ValueError("reset ID out of range")
    if any(len(row) != len(replacement) for row in rows):
        raise ValueError("replacement must have the state row width")
    selected = set(ids)
    return [list(replacement) if i in selected else list(row) for i, row in enumerate(rows)]


def wilson_interval(successes, episodes, z=1.959963984540054):
    positive_int(episodes, "episodes")
    if type(successes) is not int or not 0 <= successes <= episodes:
        raise ValueError("success count must lie between zero and episodes")
    if finite(z, "z") <= 0:
        raise ValueError("z must be positive")
    p = successes / episodes
    denominator = 1 + z*z/episodes
    center = (p + z*z/(2*episodes)) / denominator
    half = z*math.sqrt(p*(1-p)/episodes + z*z/(4*episodes*episodes)) / denominator
    return max(0.0, center-half), min(1.0, center+half)


def command_is_fresh(now_s, issued_s, ttl_s, current_episode, command_episode):
    for value, name in ((now_s, "now_s"), (issued_s, "issued_s"), (ttl_s, "ttl_s")):
        finite(value, name)
    if ttl_s <= 0:
        raise ValueError("ttl_s must be positive")
    if type(current_episode) is not int or type(command_episode) is not int:
        raise ValueError("episode IDs must be integers")
    # Same-clock fixture; cross-clock comparison needs an explicit synchronization contract.
    return current_episode == command_episode and 0 <= now_s-issued_s <= ttl_s


def main():
    rotated = rotate_xyzw((0, 0, math.sqrt(.5), math.sqrt(.5)), (1, 0, 0))
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(rotated, (0, 1, 0)))
    before = [[1, 10], [2, 20], [3, 30], [4, 40]]
    after = partial_reset(before, [1, 3], [0, 0])
    assert after == [[1, 10], [0, 0], [3, 30], [0, 0]]
    assert not command_is_fresh(1, .99, .1, 2, 1)
    print(json.dumps({
        "execution_status": "synthetic CPU fixtures; no Isaac execution",
        "rates": rates(16, 100, 2, 1/120, 2),
        "rotated_base_x": rotated,
        "partial_reset": after,
        "success_80_of_100_wilson_95": wilson_interval(80, 100),
        "expired_command_accepted": command_is_fresh(1, .5, .1, 2, 2),
        "previous_episode_command_accepted": command_is_fresh(1, .99, .1, 2, 1),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
