"""Analytical swept segments, rest-to-rest timing, and a yaw-only grasp fixture."""
import math

from dynamics import finite


def segment_clearance(start, end, center, obstacle_radius, robot_radius, margin=0.0):
    """Signed distance between a swept disk and a circle; <= 0 means contact."""
    if any(len(point) != 2 for point in (start, end, center)):
        raise ValueError("two-dimensional points required")
    for value in (*start, *end, *center, obstacle_radius, robot_radius, margin):
        finite(value, "geometry")
    if min(obstacle_radius, robot_radius, margin) < 0:
        raise ValueError("radii and margin must be nonnegative")
    delta = (end[0] - start[0], end[1] - start[1])
    length_squared = sum(value * value for value in delta)
    fraction = 0.0 if length_squared == 0 else max(0.0, min(1.0, sum(
        (center[i] - start[i]) * delta[i] for i in range(2)) / length_squared))
    closest = tuple(start[i] + fraction * delta[i] for i in range(2))
    return math.dist(closest, center) - obstacle_radius - robot_radius - margin


def cubic_duration(distance, speed_limit, acceleration_limit):
    """Shortest duration under these two bounds for this particular cubic shape."""
    for value in (distance, speed_limit, acceleration_limit):
        finite(value, "timing input")
    if distance < 0 or speed_limit <= 0 or acceleration_limit <= 0:
        raise ValueError("nonnegative distance and positive limits required")
    return max(1.5 * distance / speed_limit, math.sqrt(6 * distance / acceleration_limit))


def cubic_sample(start, end, duration, time):
    if len(start) != 2 or len(end) != 2:
        raise ValueError("two-dimensional endpoints required")
    for value in (*start, *end, duration, time):
        finite(value, "trajectory input")
    if duration <= 0 or not 0 <= time <= duration:
        raise ValueError("time must lie in a positive-duration segment")
    s = time / duration
    blend = 3 * s * s - 2 * s * s * s
    speed = (6 * s - 6 * s * s) / duration
    acceleration = (6 - 12 * s) / (duration * duration)
    delta = tuple(end[i] - start[i] for i in range(2))
    return (tuple(start[i] + blend * delta[i] for i in range(2)),
            tuple(speed * value for value in delta),
            tuple(acceleration * value for value in delta))


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def valid_pose(pose):
    if len(pose) != 4:
        raise ValueError("pose must be (x, y, z, yaw)")
    for value in pose:
        finite(value, "pose component")


def compose_pose(parent_child, child_object):
    """Yaw-only rigid transform; no roll/pitch and no general SE(3) claim."""
    valid_pose(parent_child)
    valid_pose(child_object)
    x, y, z, yaw = parent_child
    a, b, c, angle = child_object
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (x + cosine * a - sine * b, y + sine * a + cosine * b,
            z + c, wrap_angle(yaw + angle))


def inverse_pose(pose):
    valid_pose(pose)
    x, y, z, yaw = pose
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (-cosine * x - sine * y, sine * x - cosine * y, -z, -yaw)


def retention_error(gripper_before, object_before, gripper_after, object_after):
    before = compose_pose(inverse_pose(gripper_before), object_before)
    after = compose_pose(inverse_pose(gripper_after), object_after)
    return {"translation_m": math.dist(before[:3], after[:3]),
            "yaw_rad": abs(wrap_angle(after[3] - before[3]))}


def required_finger_force(mass, friction, load_factor=1.0, gravity=9.81):
    """Per-finger normal force for a symmetric two-finger vertical load model."""
    for value in (mass, friction, load_factor, gravity):
        if finite(value, "grasp input") <= 0:
            raise ValueError("grasp parameters must be positive")
    if load_factor < 1:
        raise ValueError("load factor must be at least one")
    return load_factor * mass * gravity / (2 * friction)


def evaluate_lift(gripper_before, object_before, gripper_after, object_after,
                  minimum_lift=0.15, position_tolerance=0.01, yaw_tolerance=0.05):
    for value in (minimum_lift, position_tolerance, yaw_tolerance):
        if finite(value, "task threshold") <= 0:
            raise ValueError("task thresholds must be positive")
    error = retention_error(gripper_before, object_before, gripper_after, object_after)
    lift = object_after[2] - object_before[2]
    retained = error["translation_m"] <= position_tolerance and error["yaw_rad"] <= yaw_tolerance
    return {"object_lift_m": lift, "retention_error": error, "retained": retained,
            "success": retained and lift >= minimum_lift}
