# 2. Kinematics: what motion can the robot make?

**Purpose:** convert joint or wheel motion into tool or base motion, and recognize when the requested motion is impossible. **Prerequisite:** [frames](Lecture-01.md); the slope of a function. **Next:** [feedback](Lecture-03.md).

Kinematics describes motion without asking which forces produce it. **Forward kinematics** maps joint positions to tool pose. **Inverse kinematics** asks for joint positions that produce a requested pose. A learned action is not exempt from this geometry: a Cartesian target still needs a controller that can realize it.

## Build a two-link arm by hand

Our planar arm has two revolute joints and link lengths `l1` and `l2`. Joint angle `q1` is relative to the base; `q2` is relative to the first link. The second link's orientation in the base frame is therefore `q1 + q2`.

“Planar” means everything stays in one flat plane. “Revolute” means the joint
rotates, like a hinge. An arrow of length `l` at angle `q` contributes
`l*cos(q)` along x and `l*sin(q)` along y; the endpoint is the sum of the two
link arrows. Python trigonometric functions use radians.

\[
x=l_1\cos q_1+l_2\cos(q_1+q_2),\qquad
y=l_1\sin q_1+l_2\sin(q_1+q_2).
\]

For `l1 = l2 = 1 m`, `q1 = 0`, and `q2 = pi/2`, the tool is at `(1, 1)` m. Drawing the two arrows makes the equation easier to remember than memorizing it.

To find the joint angles for `(x, y)`, use the triangle's cosine rule:

\[
c_2=\frac{x^2+y^2-l_1^2-l_2^2}{2l_1l_2},\quad
q_2=\pm\arccos(c_2),\quad
q_1=\operatorname{atan2}(y,x)-\operatorname{atan2}(l_2\sin q_2,l_1+l_2\cos q_2).
\]

The plus and minus signs produce two elbow configurations. For `(1,1)`, the solutions are `(0, pi/2)` and `(pi/2, -pi/2)`. Both have the same position; their tool orientations differ. This arm has only two joints, so arbitrary x, y, **and** tool orientation cannot all be prescribed independently.

A target must satisfy `abs(l1-l2) <= sqrt(x*x+y*y) <= l1+l2`. A 3 m target is unreachable for two 1 m links. Check that condition before clipping floating-point values into the domain of `acos`; clipping a substantially unreachable target conceals the error. Joint limits and obstacles can invalidate geometrically reachable solutions too.

## The Jacobian translates small motions

The derivative tells us how output changes for a small input change. Collecting the derivatives into a matrix gives the **Jacobian**:

\[
\begin{bmatrix}\dot x\\\dot y\end{bmatrix}
=J(q)\begin{bmatrix}\dot q_1\\\dot q_2\end{bmatrix},\quad
J=\begin{bmatrix}
-l_1\sin q_1-l_2\sin(q_1+q_2)&-l_2\sin(q_1+q_2)\\
l_1\cos q_1+l_2\cos(q_1+q_2)&l_2\cos(q_1+q_2)
\end{bmatrix}.
\]

Each column says what tool velocity one joint produces while the other is still. Units are metres per radian; multiplying by radians per second yields metres per second.

The dots mean rates: `qdot` is joint angle change per second, and `xdot` is tool
position change per second. At the straight pose below, joint rates
`[0.1, 0.2] rad/s` give tool velocity `[0, 2*0.1 + 1*0.2] = [0, 0.4] m/s`.
This row-by-column calculation is what `J @ qdot` means; plain Python lists
need explicit sums instead of `@`.

With both 1 m links straight along x, `J = [[0,0],[2,1]]`. Both joints initially move the tool along y. There is no instantaneous x motion available: this is a **singularity**. Near singularities, solving for a desired tool velocity can demand very large joint velocities. Damping, limits, and changing the posture help, but do not make every desired motion achievable. [Modern Robotics: singularities](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-3-singularities/)

The supplied test checks each derivative independently using a central difference:

```text
J[:,j] approximately equals (FK(q + epsilon*unit_j) - FK(q - epsilon*unit_j)) / (2*epsilon)
```

Use `epsilon = 1e-6` rad for the supplied examples. Making epsilon arbitrarily tiny increases subtraction error; a finite difference is a numerical approximation with its own scale.

## A mobile base has different constraints

For an ideal differential-drive base with wheel separation `b`, and wheel **linear** speeds `v_left` and `v_right`:

\[
v=(v_{right}+v_{left})/2,\quad
\omega=(v_{right}-v_{left})/b,\quad
\dot x=v\cos\theta,\quad \dot y=v\sin\theta,\quad\dot\theta=\omega.
\]

If encoders report angular speeds, first multiply by wheel radius to obtain linear speeds. A differential-drive base cannot move sideways instantaneously. This constraint matters when translating a grid path into actual motion.

**Worked example:** `v_left=0.1 m/s`, `v_right=0.3 m/s`, `b=0.4 m` gives `v=0.2 m/s`, `omega=0.5 rad/s`. Starting at the origin facing x, one second of constant command produces a circular arc of radius `v/omega=0.4 m`: approximately `(0.19177, 0.04897, 0.5)` in metres/metres/radians. The lab integrates this constant arc exactly, rather than assuming the heading stays fixed during the step. Wheel slip is excluded.

## Practice and answer guidance

Run `python3 labs/foundations.py kinematics`.

**Guided variation:** use lengths `0.7` and `0.4` m. Predict the inner unreachable radius (0.3 m) and outer reach (1.1 m); test targets on both sides of those bounds. Set equal wheel speeds and predict straight motion; set equal-and-opposite speeds and predict rotation without translation.

**Independent task:** implement joint limits, select only valid IK branches, and reject a target when both branches violate the limits. Use FK to check the selected result, but also include a known hand-calculated solution. Record the two configurations and why one was rejected. Do not claim collision avoidance from an IK check.

**Self-check:** Does `J*qdot` predict a large finite displacement exactly? No; it is an instantaneous relation. Integrate while updating `J`, or compute FK at the final joint positions. Does a zero determinant mean the robot cannot move at all? No; it loses at least one task-space velocity direction. Can a VLA output `(dx,dy)` without a frame and timestep? It can output numbers, but the robot cannot unambiguously execute them.

**Handoff:** preserve reachability limits, action units, joint order, and a known-pose test. They become the checks on your learned policy's action adapter.
