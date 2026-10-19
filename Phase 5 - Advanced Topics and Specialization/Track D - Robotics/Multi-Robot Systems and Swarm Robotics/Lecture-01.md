# Lecture 4: Multi-Robot Systems and Swarm Robotics

## Overview

Single-robot ROS 2 is already a distributed system. **Multi-robot** systems add **coordination**: who does which task, how maps are aligned, how robots avoid interference, and how one component's failure affects the fleet. **Swarm** robotics pushes toward **many** units with **local rules**; **HRI** layers add **humans** as peers in the loop.

**By the end of this lecture you should be able to:**

* Contrast **centralized**, **decentralized**, and **hierarchical** multi-robot architectures.
* Explain **task allocation** at a high level (auction vs optimization) and name failure modes.
* Describe **multi-robot SLAM / map merging** problems (loop closure across robots, relative pose).
* Summarize **Reynolds flocking** and why it scales.
* List **HRI** patterns: command, clarification, shared autonomy, and **safety-rated** speed reduction.

**Prerequisite:** Comfortable with Nav2, TF2, and ROS 2 graphs from [Lecture 1 — Advanced Robot Operating System](../Advanced%20Robot%20Operating%20System/Lecture-01.md).

**Worked practice:** [allocation, reservations, and consensus lab](Coordination-Lab.md).
Find the optimal assignment for two robots, expose a head-on edge conflict,
reserve space over time, and construct a case where priority planning fails
despite an existing joint solution. The exercises run with Python's standard
library and include the assumptions needed to interpret their results.

---

## 1. Multi-robot coordination

### 1.1 Why multi-robot is hard

* **Shared workspace:** Collisions are now **inter-robot**; your local planner must know (approximately) where others are.
* **Partial observability:** No robot sees the full world; **communication** is lossy and delayed.
* **Consistency:** Each robot may maintain its own **map frame**; merging requires **relative pose** estimates.

### 1.2 Task allocation

**Problem:** Assign tasks (visit locations, transport objects, patrol) to robots with different **capabilities** and **battery** states.

| Approach | Idea | Trade-off |
|----------|------|-----------|
| **Auction / market** | Robots bid on tasks from a central or distributed auctioneer | Simple; can be suboptimal |
| **Optimization** (MIP, OR-Tools) | Global cost minimization | Better quality; needs model and scale care |
| **Greedy heuristics** | Fast assignments | Good for online re-planning |

**Failure modes:** **Deadlock** when tasks block each other; **starvation** when one robot never gets work; **replanning storms** when communication flaps.

### 1.3 Multi-robot SLAM and map merging

Each robot may run **local SLAM** in its own frame. **Collaborative SLAM** merges maps when **relative transforms** between robots are known (shared observations, known landmarks, or communication of pose graphs). Systems like **Kimera-Multi** exemplify **distributed** mapping at research grade.

**Practical ROS 2 angle:** Namespaces (`/robot1`, `/robot2`), **multi-robot Nav2** setups, and **tf** trees that either share a common `map` or periodically **align** maps.

### 1.4 Formation control

**Leader–follower:** Follower tracks leader pose with offset; simple but single point of failure at leader.

**Virtual structure:** Team behaves as a rigid body; good for **inspection** or **coordinated transport**.

**Consensus-based:** Each agent adjusts state using **neighbor** errors; requires **connectivity** graphs (who talks to whom).

---

## 2. Swarm robotics

### 2.1 Reynolds rules (boids)

Classic **flocking** combines three forces:

1. **Separation:** Avoid crowding neighbors.
2. **Alignment:** Match average heading of neighbors.
3. **Cohesion:** Move toward average position of neighbors.

Add **obstacle avoidance** and you have a baseline for **many** agents. Parameters (weights, neighborhood radius) determine whether the swarm **clusters**, **disperses**, or **oscillates**.

These steering rules are heuristics; a separation term does not prove collision
avoidance under finite timesteps, limited actuation, or delayed observations.
Check the [swept motion](../Robot%20Dynamics%20and%20Task%20Execution/Lecture-03-Trajectory-Execution.md),
including motion between recorded positions and the other robot's footprint.

### 2.2 Decentralized vs centralized

| Style | Pros | Cons |
|-------|------|------|
| **Centralized** | Global optimum easier | Single point of failure, comms bottleneck |
| **Decentralized** | Can reduce central coordination bottlenecks | Connectivity, inconsistent state, and emergent failures still limit robustness and scale |

**Stigmergy** (indirect coordination via environment, e.g. trails) appears in large robot swarms and ant-inspired algorithms.

### 2.3 Scalability and fault tolerance

Ask: What happens at **10 → 100 → 1000** agents? **O(n²)** neighbor checks explode unless you use **spatial hashing** or **limited-range** sensing.

**Fault tolerance:** Robots drop out; the mission should **degrade** (fewer tasks/hour), not **collapse**—design **redundancy** and **timeout-based** reassignment.

---

## 3. Human–robot interaction (HRI)

### 3.1 Natural language

**LLMs + ASR** can turn speech into **task graphs**, but **grounding** remains hard: the robot must map words to **objects** and **places** it can actually reach. Dialogue for **clarification** (“Which bin?”) reduces costly mistakes.

### 3.2 Shared autonomy

The human provides **intent** (joystick direction, high-level “clean this room”); the robot handles **obstacles**, **grasp feasibility**, and **safety**. Tune **how much** autonomy to avoid both **boredom** (too much help) and **mistrust** (too little).

### 3.3 Safety near humans

For industrial collaborative applications, use the current robot and cell
integration standards and assessed protective functions described in
[the industrial lecture](../Industrial%20and%20Embedded%20Robotics/Lecture-01.md#13-collaborative-robots-cobots).
An ordinary fleet planner's reduced-speed request is not itself a safety-rated
protective function. Those industrial standards also do not automatically cover
every public-space or consumer swarm deployment.

---

## 4. ROS 2 practical notes

* **Namespaces and discovery:** Multiple robots on one network need clear **ROS_DOMAIN_ID** / DDS config and **namespacing** to avoid topic collisions.
  A namespace does not automatically rename TF frame identifiers, and conventional
  absolute `/tf` topics need deliberate remapping when isolating graphs. Use
  unique frame IDs or an explicitly shared frame tree; see the
  [local ROS integration exercise](../Robotics%20Foundations/ROS2-Lab.md).
  Separate DDS domains also separate ordinary discovery and communication; robots
  that must coordinate need a deliberately shared domain or an explicit bridge.
  Neither domain IDs nor namespacing provide authentication or prove isolation
  against a hostile participant.
* **Nav2 multi-robot:** Use official patterns for **multiple robots** in simulation; verify **global frame** alignment before **fleet** goals.
* **Open-RMF** supports fleet integration and traffic coordination. Study its
  [traffic implementation](https://github.com/open-rmf/rmf_traffic) after the
  reservation exercise. A negotiated schedule still needs adapters that report
  and execute the actual robot motion; a timed-out participant may still occupy
  the corridor.

---

## 5. Projects (from this roadmap)

* **Multi-robot exploration:** Three robots in Gazebo-class sim; **frontier exploration**; merge maps or align frames; log **comm** drops and replan.
* **Swarm formation:** ~20 agents with Reynolds + obstacles; plot **separation minima** over time.
* **Shared autonomy:** Joystick + obstacle avoidance blending; user study optional—at minimum, log **interventions per minute**.

---

## 6. Self-check

1. Why does **map merging** need **relative pose** between robots, not just two independent SLAM maps?
2. What is one **failure mode** of purely **greedy** task assignment?
3. How does **separation** in Reynolds differ from **obstacle avoidance**?

<details markdown="1"><summary>Answer guidance</summary>

1. Each local map can choose its own origin and axes. Corresponding geometry must
   be transformed into a consistent frame, with uncertainty and false matches
   handled, before a combined route has meaning.
2. Assigning the nearest robot to each new task can repeatedly favor one robot
   while starving another, or occupy a passage needed by a later task. Include
   task history, reachability, and shared-space constraints in the analysis.
3. Separation reacts to nearby agents; obstacle avoidance reacts to environmental
   geometry. Both need body shape, dynamics, and timing checks. Simply adding
   their requested steering vectors can still produce an invalid motion.

</details>

---

## Resources

* **"Multi-Robot Systems"** by Lynne Parker — coordination, task allocation, architectures.
* **"Swarm Intelligence: From Natural to Artificial Systems"** by Bonabeau, Dorigo, and Theraulaz.
* **ROS 2 / Nav2 multi-robot:** Search current Nav2 and ROS 2 docs for **multi-robot** examples.
* **Open-RMF:** [openrmf.org](https://openrmf.org/) — fleet management (optional deep dive).

---

## Related lectures

* [Advanced Robot Operating System](../Advanced%20Robot%20Operating%20System/Lecture-01.md) — Nav2, TF2, single-robot baseline.
* [Advanced Perception and AI for Robotics](../Advanced%20Perception%20and%20AI%20for%20Robotics/Lecture-01.md) — perception and learning for coordinated teams.
