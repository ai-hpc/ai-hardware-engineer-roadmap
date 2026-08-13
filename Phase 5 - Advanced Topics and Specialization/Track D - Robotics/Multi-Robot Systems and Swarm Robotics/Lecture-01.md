# Lecture 4: Multi-Robot Systems and Swarm Robotics

## Overview

Single-robot ROS 2 is already a distributed system. **Multi-robot** systems add **coordination**: who does which task, which map is ground truth, and how robots avoid **interference** without a single point of failure. **Swarm** robotics pushes toward **many** cheap units with **local rules**; **HRI** layers add **humans** as peers in the loop.

**By the end of this lecture you should be able to:**

* Contrast **centralized**, **decentralized**, and **hierarchical** multi-robot architectures.
* Explain **task allocation** at a high level (auction vs optimization) and name failure modes.
* Describe **multi-robot SLAM / map merging** problems (loop closure across robots, relative pose).
* Summarize **Reynolds flocking** and why it scales.
* List **HRI** patterns: command, clarification, shared autonomy, and **safety-rated** speed reduction.

**Prerequisite:** Comfortable with Nav2, TF2, and ROS 2 graphs from [Lecture 1 — Advanced Robot Operating System](../Advanced%20Robot%20Operating%20System/Lecture-01.md).

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

### 2.2 Decentralized vs centralized

| Style | Pros | Cons |
|-------|------|------|
| **Centralized** | Global optimum easier | Single point of failure, comms bottleneck |
| **Decentralized** | Robust, scalable | Harder analysis, emergent failures |

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

**ISO/TS 15066** informs **speed** and **force** when humans enter collaborative spaces. Software implements **reduced speed** in proximity; **hardware** still owns **E-stop** and **protective stop** chains.

---

## 4. ROS 2 practical notes

* **Namespaces and discovery:** Multiple robots on one network need clear **ROS_DOMAIN_ID** / DDS config and **namespacing** to avoid topic collisions.
* **Nav2 multi-robot:** Use official patterns for **multiple robots** in simulation; verify **global frame** alignment before **fleet** goals.
* **Open-RMF** (outside this lecture’s depth) targets **fleet** management in facilities; worth a follow-on if you deploy **many** robots in buildings.

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
