# 4. A commanded hand pose does not establish a successful pick

**Before:** [frames](../Robotics%20Foundations/Lecture-01.md) and
[trajectory execution](Lecture-03-Trajectory-Execution.md).
**Build:** transform a grasp into the world, estimate one simple contact-force
requirement, and distinguish hand motion from object retention.

## Name which frame each grasp belongs to

Suppose perception estimates an object's world pose `T_WO`. A grasp proposal
specifies a desired gripper pose relative to that object, `T_OG`. The desired
world gripper pose is:

```text
T_WG = T_WO × T_OG
```

The notation means that `T_AB` converts coordinates expressed in B into A. It
also describes B's pose relative to A. If a dataset instead stores the object's
pose relative to the gripper, that is `T_GO`; invert it before this composition.
Names such as `grasp_pose` do not establish the convention.

Our fixture supports translation in x/y/z and rotation about z only; it does not
implement arbitrary roll and pitch. Let the world object pose be:

```text
translation = (0.6, 0.2, 0.0) m
yaw = +90 degrees
object-relative gripper translation = (0.1, 0.0, 0.05) m
relative yaw = 0
```

Rotate the relative x offset into world y, then add the world translation. The
desired gripper pose is `(0.6, 0.3, 0.05) m` at `+90°`. Directly adding the
translations would incorrectly produce `(0.7,0.2,0.05)`.

**Guided variation:** change only the object yaw to zero. The world gripper
translation then becomes `(0.7,0.2,0.05)`. The object-frame grasp specification
has not changed; its world coordinates have.

## A grasp proposal still needs a feasible approach

A useful execution sequence separates object localization, pregrasp approach,
contact acquisition, closure, lift, transport, placement, and release. Each stage
has a distinct completion condition. Reaching the pregrasp pose does not mean
the fingers have made the intended contact.

Before execution, check inverse kinematics, joint limits, the gripper's opening,
approach clearance, and the geometry of the hand and object. Once the object is
held, its geometry moves with the robot and must be included during transport.
A target reachable by the tool center can still cause a wrist or carried-object
collision.

These checks connect a vision or VLA output to the motion and actuator contracts
from the previous lessons. A learned model's confidence is not a substitute for
an explicit geometric or execution condition.

## One contact calculation with visible assumptions

For an ideal symmetric two-finger grasp, let each finger apply normal force `N`
against opposing vertical faces. Assume Coulomb friction coefficient `mu`, a
stationary upright object, equal load sharing, and no relevant tipping moment.
Each contact can supply tangential force up to `mu × N`, so vertical holding
requires:

```text
2 × mu × N >= m × g
N_required = load_factor × m × g / (2 × mu)
```

For mass `0.5 kg`, `mu=0.4`, and illustrative load factor `1.5`, the estimate is
`9.196875 N` **per finger**. It is not the total of both normal forces. Halving
`mu` doubles the required normal force. Faster upward acceleration increases the
vertical load; a tipping moment requires a wrench analysis beyond this scalar
calculation.

The [Modern Robotics friction lesson](https://modernrobotics.northwestern.edu/nu-gm-book-resource/12-2-1-friction/)
develops contact/friction-cone models. Our assumed coefficient and load factor
are teaching values, not measured material properties or an actuator rating.
Increasing squeeze force can exceed a gripper's continuous capability or damage
an object; those are additional limits, not effects represented by this formula.

**Does this establish force closure?** No. Resisting this particular vertical
load is weaker than resisting arbitrary disturbance wrenches. Two ideal point
contacts in 3-D do not generally resist a torque about their connecting line.
Contact locations, normals, friction, and contact model matter. See the primary
[force-closure lesson](https://modernrobotics.northwestern.edu/nu-gm-book-resource/12-2-3-force-closure/).

## Measure the object relative to the hand

If an object remains rigidly held, its pose relative to the gripper stays
approximately constant:

```text
T_GO_before = inverse(T_WG_before) × T_WO_before
T_GO_after  = inverse(T_WG_after)  × T_WO_after
```

Compare relative translation and orientation using appropriate tolerances.
The supplied yaw-only fixture uses Euclidean translation error and the shortest
wrapped yaw difference. A full 3-D implementation needs a rotation metric on
SO(3); subtracting quaternion components is not a general angular error.

The paired cases use the **same commanded and final hand pose**:

| Observation after the hand rises 0.2 m | Object lift | Relative translation change | Result |
|---|---:|---:|---|
| Object rises with the hand | 0.2 m | approximately 0 m | Retained and lifted |
| Object stays on the table | 0 m | 0.2 m | Failed pick |

The demonstration defines lift success as at least `0.15 m` object lift,
relative translation change at most `0.01 m`, and yaw change at most `0.05 rad`.
Keeping everything still has zero retention error but fails the lift requirement.

These pose comparisons rely on valid object observations, frame calibration, and
timestamps. If the object becomes occluded, label retention unknown or use another
validated observation; do not reuse the last visible pose as proof of continued
contact. The CPU fixture has exact synthetic poses and no contact dynamics.

## Extend success to the actual task

A pick-and-place task needs additional criteria: the object reaches the allowed
placement region, is released, and remains there for a declared interval with
acceptable motion. The hand leaving the goal is not proof that the object was
released. A camera image alone may be ambiguous about whether the object is
supported, wedged, or still attached.

Use stage-specific outcomes such as `approach_failed`, `contact_not_confirmed`,
`retention_lost`, `placement_outside_region`, and `release_unconfirmed`. Keep an
overall task-success metric too. This helps explain why a model with excellent
action prediction or numerical parity can perform worse in closed-loop tasks.

**Independent transfer:** add three timestamped post-release object observations.
Require all three to lie inside a chosen placement region with speed below a
declared limit. Include a case that briefly enters the region and then falls out.
Use actual elapsed time, not just sample count, if observation rates differ.

Run `python3 labs/control_lab.py grasp` from this course directory. The
[implementation](labs/task_execution.py) supplies transform composition,
relative-pose error, the two-contact force estimate, and the lift predicate.
Complete the lesson with the hand calculation, paired outcomes, and a clear
account of which contact, perception, and release assumptions remain untested.
