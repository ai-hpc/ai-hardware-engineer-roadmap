# Coordinate tasks and paths without confusing them

Giving each robot a sensible task does not establish that their routes can coexist. Two individually shortest routes can collide; a fast reservation planner can fail even when a joint solution exists. This lab exposes each issue with inspectable CPU examples, then shows why a decentralized consensus rule also needs a stability argument.

Allow **90–120 minutes**. Use Python 3.10+, basic graphs, and the [single-robot planning lesson](../Robotics%20Foundations/Lecture-05.md). No ROS installation, simulator download or robot is needed. Run commands from this lesson directory:

```bash
python3 labs/coordination.py
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

The [reference output](labs/coordination-reference.json) comes from this code. **Sixteen tests** cover costs/capabilities, independent conflict checks, goal occupancy, edge swaps, reservation rollback, an incompleteness counterexample, consensus and malformed input.

## 1. Assign tasks using a stated objective

Suppose robot A and robot B each need one task. Estimated travel costs are:

| Robot | Task 0 | Task 1 |
| --- | ---: | ---: |
| A | 1 | 2 |
| B | 2 | 100 |

A greedy allocator processes A first and chooses its cheapest task 0. B receives task 1: total cost **101**. Assigning A→1 and B→0 costs **4**. Choosing the cheapest immediate action did not minimize the team's sum.

The implementation enumerates all permutations for at most eight robots. It treats `None` as a forbidden capability combination, not a huge artificial distance. This is an exact small assignment example; larger problems need appropriate solvers and models for capacity, precedence, charging, time windows and route interactions.

**Variation:** use `[[1,2],[1,None]]`. Greedy strands B even though A→1, B→0 is feasible. Explain why an infeasible greedy result is not proof that the assignment problem itself has no solution.

Travel-cost optimization alone ignores congestion. If both robots must cross a narrow shared passage, revisit the costs and schedules after planning; assignment and traffic can interact.

## 2. Define the motion model before claiming conflict freedom

The path experiment uses a 5×3 grid. Coordinates are `(x,y)` with x=0…4 and y=0…2. Each point robot moves one cardinal cell or waits per shared integer tick. A starts at `(0,1)` and wants `(4,1)`; B wants the reverse.

The independent shortest routes are the middle row. Both occupy `(2,1)` at tick 2. This is a **vertex conflict**. There is a second failure that endpoint occupancy alone misses: A can move from u to v while B moves v to u during the same tick. They exchange endpoints but collide between them. This is an **opposite-edge conflict** in our model.

The implementation forbids both. Once a path ends, its robot **holds the goal through the planning horizon**. It does not vanish from the workspace. These are explicit choices; different multi-agent pathfinding formulations make different choices about allowed conflicts and goal occupancy. [Definitions and variants](https://arxiv.org/abs/1906.08291).

## 3. Search in space and time

A search state is `(cell, tick)`. Reaching the same cell at a later tick can be useful because a robot or resource may have moved away. A reservation table records occupied `(tick,cell)` pairs and traversed `(tick,source,destination)` edges.

For each candidate successor:

1. Check that the destination is inside the grid and not blocked.
2. Reject a destination reserved at the next tick.
3. Reject the reverse of a reserved edge in that interval.
4. Accept a goal only if it remains available through the horizon.

Planning A first and then B produces this recorded schedule:

| Tick | A | B |
| --- | --- | --- |
| 0 | (0,1) | (4,1) |
| 1 | (1,1) | (3,1) |
| 2 | (2,1) | (3,0) |
| 3 | (3,1) | (2,0) |
| 4 | (4,1) | (1,0) |
| 5 | (4,1), holding | (0,0) |
| 6 | (4,1), holding | (0,1) |

An independent pairwise checker finds no vertex or opposite-edge conflict through horizon 12. The table reserves the entire returned path before moving to the next robot. A conflicting insertion is rejected without partially modifying the reservations. If any robot cannot be planned, `plan_fleet` raises instead of returning an apparently complete plan for only part of the team.

**Variation:** remove the two outer rows. The robots can no longer pass under this model. Waiting does not create a passing space. Record the failure rather than increasing the timeout until an unsafe route is returned.

## 4. Find a solution the priority planner misses

Priority planning is incomplete. Consider this T-shaped grid, where `#` cells are blocked:

```text
y=1:  #  alcove  #
y=0:  A  centre  B
       0    1    2   x
```

A's goal is the centre `(1,0)`; B's goal is A's start `(0,0)`. Planning A's shortest route first places it permanently in the centre at tick 1. B cannot pass. The executable raises `NoPlan`.

Yet this joint schedule is valid:

| Tick | A | B |
| --- | --- | --- |
| 0 | (0,0) | (2,0) |
| 1 | (1,0) | (2,0), waiting |
| 2 | (1,1), alcove | (1,0) |
| 3 | (1,0), goal | (0,0), goal |

The independent checker confirms the witness. A needed to visit its goal, leave it temporarily, and return. The greedy first route never considered that cooperation. A result of “no path under these priorities/reservations/horizon” is more precise than “no solution exists.” Joint search, conflict-based methods or replanning priorities are possible next experiments; each needs its own objective and completeness boundary.

**Check:** why does the search test future goal occupancy? A robot already at its goal may need to leave if another reserved route passes through that cell later. One regression test exercises exactly this case.

## 5. Test a local consensus rule

For two connected agents with scalar states x₀ and x₁, update both from the same old state:

\[
x'_0=x_0+\alpha(x_1-x_0),\qquad
x'_1=x_1+\alpha(x_0-x_1).
\]

Their average is unchanged. The disagreement obeys `e'=(1−2α)e`. Therefore this two-agent system converges for **0<α<1**. At α=1 the agents exchange values forever; larger gains increase disagreement magnitude. This statement assumes synchronous updates and the stated fixed, connected graph.

With α=0.25 and initial `[0,2]`, the trace is:

```text
[0, 2] → [0.5, 1.5] → [0.75, 1.25] → [0.875, 1.125] → …
```

Updating x₀ in place before computing x₁ would implement a different rule. The code reads the old vector for every edge and tests mean preservation. A disconnected third agent at 10 stays at 10; local agreement in one component is not whole-team consensus. Larger graphs have different allowable gain ranges, and delays/asynchronous updates require additional analysis.

Reynolds-style separation/alignment/cohesion rules are useful behavioral models, but they do not inherit this scalar consensus proof or provide physical collision guarantees. Read [Reynolds' original flocking paper](https://www.red3d.com/cwr/papers/1987/boids.html) for that separate model.

## 6. Carry the result into a fleet architecture

This fixture assumes a shared clock, exact positions and successful execution of every scheduled move. A real fleet needs finite footprints, swept-path checks, actuator limits, tracking-error margins and progress feedback. A valid discrete grid plan can still be physically invalid; continue with the [trajectory execution lab](../Robot%20Dynamics%20and%20Task%20Execution/Lecture-03-Trajectory-Execution.md).

A delayed robot can invalidate later reservations. Do not infer that a robot has stopped simply because its heartbeat timed out. Before another robot enters that space, establish the required position/stop/ownership evidence and a defined recovery process. Version plans and reject superseded commands; stale-plan rejection does not itself stop a robot already executing an earlier plan.

Namespaces separate ROS names but do not automatically rename TF frame IDs or align maps. Distinct ROS domain IDs separate discovery; robots that must communicate need a deliberate shared or bridged graph. [Open-RMF's traffic libraries](https://github.com/open-rmf/rmf_traffic) are a relevant next implementation to study, with actual fleet adapters and execution contracts. This lab does not run Open-RMF or simulate radio loss.

**Completion:** explain costs 101 versus 4, show a vertex and an opposite-edge conflict, reproduce the joint witness missed by priority planning, and derive the two-agent gain condition. Submit the raw paths, model assumptions, test output and one unresolved physical deployment requirement. The artifact is a verified discrete coordination experiment, not a collision-certified fleet.

Return to the [multi-robot survey](Lecture-01.md) or connect the result to the [industrial integration lecture](../Industrial%20and%20Embedded%20Robotics/Lecture-01.md).
