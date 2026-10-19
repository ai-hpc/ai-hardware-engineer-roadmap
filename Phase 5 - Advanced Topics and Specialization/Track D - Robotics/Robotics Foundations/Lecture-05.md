# 5. Planning: choose a route the robot can actually use

**Purpose:** distinguish a geometric path, a timed trajectory, and a feedback controller. **Prerequisite:** [kinematics](Lecture-02.md) and [uncertain perception](Lecture-04.md). **Next:** [ROS integration](Lecture-06.md).

A **path** is a sequence or curve of configurations. A **trajectory** adds timing, velocity, and possibly acceleration. A **controller** responds to measured state to follow it. A collision-free path can still require an impossible turn or excessive acceleration; a feasible trajectory can still fail under delay, slip, or a changed obstacle.

## Solve a small search problem first

Represent a room as a grid. Free cells may be occupied by the robot's reference point; blocked cells may not. Our first robot is an ideal point that can move one cell up, down, left, or right at unit cost. This is deliberately simpler than a differential-drive robot.

**A\*** stores the best known cost `g` from the start and an estimate `h` of the remaining cost. It expands a candidate with smallest `f=g+h`. For four-neighbour unit-cost motion, Manhattan distance is:

```text
h(cell) = abs(cell.x - goal.x) + abs(cell.y - goal.y)
```

This ignores obstacles, so it never overestimates the shortest remaining route in this model. At each expansion, update a neighbour only when the new route is cheaper. Retain a parent pointer to reconstruct the route. Return “no path” if the frontier empties. The supplied implementation ignores stale queue entries after a cheaper route has superseded them.

### Worked example: a narrow doorway

The 7×7 map has a wall at x=3 except for y=3. The start is `(1,3)`, the goal `(5,3)`:

```text
...#...
...#...
...#...
.S...G.
...#...
...#...
...#...
```

The point robot takes four steps: `(1,3) -> (2,3) -> (3,3) -> (4,3) -> (5,3)`.

Now replace the point with a square footprint covering its own cell and all eight adjacent cells. The doorway is too small. **Obstacle inflation** marks a reference-point cell blocked whenever any part of that footprint would overlap an occupied cell. The supplied `inflate_cells(..., radius=1)` does exactly this discrete square-footprint check. It also treats outside the map as occupied, so the footprint cannot hang over the map boundary. Planning now correctly returns no path.

This discrete square model is not an exact circular footprint, an orientation-dependent arm collision model, or a proof about continuous physical geometry. A real planner needs the robot shape, map resolution, uncertainty margins, and a collision-checking method appropriate to the motion.

### Trace a detour rather than trusting the output

Use a 5×5 grid, start `(0,0)`, goal `(4,0)`, and a wall at `(2,0)` through `(2,3)`. At the start, `g=0`, `h=4`, so `f=4`. The free neighbour `(1,0)` has `g=1`, `h=3`, `f=4`; `(0,1)` has `g=1`, `h=5`, `f=6`. The search first explores toward the goal, but the wall prevents crossing at y=0. It must eventually reach y=4, cross beneath the wall, and return to y=0. A shortest route costs 12 steps: four down, four across, four up, in an order consistent with the wall. The heuristic's initial estimate of 4 was a lower bound, not a promise of a four-step route. Extend the wall through `(2,4)` and no route exists.

## Understand what the search guarantee assumes

The lab's A* returns a shortest path in its finite, four-neighbour, unit-cost graph. Breadth-first search is an independent oracle for that specific graph. Changing to diagonal moves, weighted terrain, or different motion primitives changes the edge costs and the required heuristic. For example, if diagonal moves cost `sqrt(2)`, Manhattan distance can overestimate the remaining cost; use a suitable octile or Euclidean lower bound instead.

Unknown space also needs a policy. Treating unknown cells as free helps exploration but can produce routes through unseen obstacles. Treating them as blocked can prevent progress. Record the policy and use sensing/replanning to reduce uncertainty. Do not claim that a map with no obstacle observation proves empty space.

## Connect to arms and mobile robots

For an arm, planning commonly searches **configuration space**: joint vectors such as `(q1,q2)`. An obstacle in workspace blocks all joint configurations whose links intersect it. Checking only the end-effector point misses collisions involving the elbow or another link. A straight Cartesian interpolation can cross an unreachable region or collide even when both endpoints are valid.

A differential-drive robot cannot instantaneously follow a sideways grid step. It can often rotate and then move, but turning clearance, speed, acceleration, and the current obstacle geometry still matter. Car-like robots add a turning-radius constraint and may need reverse manoeuvres. The route should be converted into feasible motion before the controller receives it.

When execution diverges from prediction, decide whether to replan, slow down, stop, or recover. The plan is conditional on a map and state estimate; it is not permission to keep moving through new contradictory evidence.

## Practice and answer guidance

Run `python3 labs/foundations.py planning`.

**Guided variation:** widen the doorway to three cells and use the radius-one square footprint. Predict whether a centered path exists, then rerun. Move the start to the map boundary and explain why footprint inflation makes that configuration invalid.

**Independent task:** create one map requiring a detour and one with no route. Compare A* path cost with breadth-first search, and validate every consecutive step and occupied-cell check. Save the map, footprint, path, total cost, and no-path result. “No route for this model” is a correct result, not a test failure.

**Self-check:** Why is a shortest route not necessarily the fastest executable motion? Turning, acceleration, speed limits, and tracking risk can dominate distance. Why can increasing inflation improve safety yet make navigation fail? It removes configurations, potentially closing all corridors. Why should an unreachable goal be separate from an invalid start? The first is a valid search result; the second may indicate a localization, footprint, or configuration fault.

**Handoff:** state the map representation, footprint, unknown-space policy, motion constraints, planning outcome, and the condition that triggers replanning. These are the assumptions behind a Nav2 or MoveIt result.
