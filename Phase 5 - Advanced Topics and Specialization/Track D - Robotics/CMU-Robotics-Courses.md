# Carnegie Mellon University: Robotics Courses Reference

A reference of CMU's Robotics Institute undergraduate programs and courses relevant to the **AI Hardware Engineer Roadmap** — compiled from official course pages and catalogs.

**Sources:**
- [Bachelor of Science in Robotics](https://www.ri.cmu.edu/education/academic-programs/bachelor-of-science-in-robotics/)
- [Robotics Program Catalog](http://coursecatalog.web.cmu.edu/schools-colleges/schoolofcomputerscience/robotics/#roboticscurriculumtextcontainer)

---


## 1. Bachelor of Science in Robotics (BSR)

**Housed in SCS.** Announced Fall 2023. Emphasizes problem-solving, systems thinking, and interdisciplinary teams.

### Core Concepts

- **RoboMath** — Mathematical foundations for robotics
- **Computer Vision and Sensing**
- **Mechanisms and Manipulation**
- **General Robotic Systems**
- **Planning and Control**
- **Robotic Building Practices**
- **Capstone Project**

### Degree Requirements Summary

| Area | Courses | Units |
|------|---------|-------|
| Robotics (core + electives) | 11 | 125 |
| Computer Science (core + SCS elective) | 6 | 51 |
| Mathematics | 6 | 67 |
| Ethics | 1 | 12 |
| Science / Engineering | 4 | 36 |
| Humanities / Arts | 7 | 63 |
| Computing @ CMU | 1 | 3 |
| First Year Seminar | 1 | 3 |
| **Total** | **26** | **360** |

### Computer Science Core

| Course | Name |
|--------|------|
| 07-128 | First Year Immigration Course |
| 15-122 | Principles of Imperative Computation |
| 15-213 | Introduction to Computer Systems |
| 15-251 | Great Ideas in Theoretical Computer Science |

### Robotics Core

| Course | Name |
|--------|------|
| 16-280 or 16-311 | Intelligent Robot Systems (or Intro to Robotics; 16-281 General Robotics starting Spring 2026) |
| 16-220 | Robot Building Practices |
| 16-299 | Introduction to Feedback Control Systems |
| 16-384 | Robot Kinematics and Dynamics |
| 16-385 | Computer Vision |
| 16-450 | Robotics Systems Engineering (senior fall) |
| 16-474 | Robotics Capstone (senior spring) |

### Ethics (one of)

- 16-161 Artificial Intelligence and Humanity
- 16-735 Ethics and Robotics

### Mathematics

| Course | Name |
|--------|------|
| 15-151 | Mathematical Foundations for Computer Science |
| 16-211 | Foundational Mathematics of Robotics |
| 21-120 | Differential and Integral Calculus |
| 21-122 | Integration and Approximation |
| 21-241 | Matrices and Linear Transformations |
| + Probability (15-259, 21-235, 36-218, or 36-225) | |

### Robotics Electives

3 general robotics electives (16-3xx, 16-4xx). Up to 12 units of 16-597 (Undergraduate Reading and Research) or 99-270 (Summer Undergraduate Research) may count.

### Sample Course Sequence (4 years)

| Year | Fall | Spring |
|------|------|--------|
| **1** | 07-128, 15-122, 15-151, 21-122, 76-101, 99-101 | 15-213, 16-180, 21-241, Science/Eng, H&A |
| **2** | Probability, 16-220, 16-211, H&A | 16-299, 15-251, 16-385, H&A |
| **3** | 16-384, H&A, 16-280, H&A, Free | Robotics Elective x2, H&A, Free |
| **4** | 16-450, Ethics, Science/Eng, Robotics Elective | 16-474, SCS Elective, Science/Eng, Free |

### Notable Courses

- **16-281 General Robotics** — Rube Goldberg machines, mini Urban Search and Rescue (replaces 16-311 from Spring 2026)
- **16-384 Robot Kinematics and Dynamics** — Hebi robotic arm control
- **16-385 Computer Vision** — Image processing, detection, recognition, geometry-based vision

---

## 2. Robotics Minor & Additional Major

### Robotics Minor

- **Overview:** 16-280 or 16-311 (16-281 from Spring 2026)
- **Controls:** One of 06-464, 16-299, 18-370, 24-451, 24-773
- **Mechanisms:** 16-384 or (15-362 + 33-141)
- **Robot Building:** One of 16-220, 16-362, 16-423, 18-349, 18-500, 18-578, 24-671
- **Electives:** 2 general robotics electives
- **Double-count:** Max 2 courses from major
- **QPA:** 2.5 in minor curriculum

### Additional Major in Robotics

- **Application:** Jan 7-Feb 3
- **Prereq:** C language, basic programming, algorithms (15-122)
- **Double-count:** Max 6 courses from primary major (5 for CS majors)
- **QPA:** 3.0 in additional major

---

## 3. Key Robotics Courses (Catalog)

| Code | Name | Notes |
|------|------|-------|
| 16-180 | Concepts of Robotics | Intro; perception, cognition, action |
| 16-211 | Foundational Mathematics of Robotics | Coordinate transforms, Jacobians, optimization, neural nets |
| 16-220 | Robot Building Practices | CAD, 3D printing, electronics, PCB, motor controllers |
| 16-280 | Intelligent Robot Systems | Search, planning, perception, control, ROS, embedded |
| 16-299 | Introduction to Feedback Control Systems | Classical control, state-space, LQR, nonlinear |
| 16-311 | Introduction to Robotics | LEGO robots, vision, motion planning, kinematics (-> 16-281) |
| 16-350 | Planning Techniques for Robotics | Path/motion planning, ground/aerial/humanoids |
| 16-362 | Mobile Robot Algorithms Laboratory | Multirotor autonomy, Python/C++ simulator |
| 16-384 | Robot Kinematics and Dynamics | Forward/inverse kinematics, Jacobians |
| 16-385 | Computer Vision | Image processing, detection, recognition |
| 16-450 | Robotics Systems Engineering | Systems engineering, design, prototyping |
| 16-474 | Robotics Capstone | Build, integrate, test robot from 16-450 |
| 16-663 | F1Tenth Autonomous Racing | 1/10 scale autonomous race car |
| 16-664 | Self-Driving Cars: Perception & Control | Deep learning, sensor fusion, localization |
| 16-735 | Ethics and Robotics | AI/ML in society, power, ethics |
| 16-831 | Introduction to Robot Learning | RL, imitation learning, visual learning |
| 16-884 | Deep Learning for Robotics | Robot learning, deep RL, control |

---

## 4. Self-Study Mapping to Roadmap

For learners following the **AI Hardware Engineer Roadmap**, here is how CMU's robotics curriculum aligns:

| Roadmap Phase | CMU Course(s) | Overlap |
|---------------|---------------|---------|
| **Phase 3: Sensor Fusion** | 16-299 (Control), 16-385 (Vision) | Kalman filtering, multi-sensor perception before Phase 4 Jetson integration |
| **Phase 4 Track B (Jetson): ROS2** | 16-280 (ROS), 16-220 (Robot Building) | ROS 2, embedded robot software |
| **Phase 5: Robotics** | 16-220, 16-280, 16-384, 16-450, 16-474 | Mechanics, control, planning |
| **Phase 5: Autonomous Driving** | 16-663 (F1Tenth), 16-664 (Self-Driving Cars) | Perception, planning, control for autonomy |

### Suggested Self-Study Order (CMU-Inspired, Robotics focus)

1. **15-122 equivalent** — Imperative programming (C/Python)
2. **21-120, 21-122, 21-241** — Calculus, linear algebra
3. **16-220** — Robot building (CAD, electronics, prototyping)
4. **16-299** — Feedback control
5. **16-384** — Kinematics and dynamics
6. **16-385** — Computer vision
7. **16-280** — Intelligent robot systems (ROS, planning, perception)

---

## Links

| Resource | URL |
|----------|-----|
| B.S. Robotics | https://www.ri.cmu.edu/education/academic-programs/bachelor-of-science-in-robotics/ |
| Robotics Catalog | http://coursecatalog.web.cmu.edu/schools-colleges/schoolofcomputerscience/robotics/ |
| SCS Undergraduate | https://www.cs.cmu.edu/academics/undergraduate/ |
| CMU Admission | https://admission.cmu.edu/ |

---

*Last updated: February 2025. Course offerings and requirements may change; verify with CMU official sources.*
