# Module 5 — Safety Standards and Deployment

<div class="course-identity auto-course" style="--course-accent: #0d9488; --course-accent-rgb: 13, 148, 136;" markdown="1">
<div class="course-identity__icon">MSSA</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Specialization</p>
<p class="course-identity__title">Specialized course identity for Module 5 — Safety Standards and Deployment.</p>
<p class="course-identity__meta">Artifact: specialization case study · Measure: performance, reliability, role fit</p>
</div>
</div>


**Parent:** [Phase 5 — Autonomous Driving](../Guide.md)

**Time:** 3–6 months

**Prerequisites:** Modules 1–2 (fundamentals + openpilot system understanding). Module 4 (Advanced Perception) is recommended but not required.

---

## Why safety and deployment

A perception model that works 99% of the time is not safe enough for a 2-ton vehicle at highway speed. This module covers the standards, testing methodologies, and deployment patterns that bridge the gap between a working prototype and a production ADAS.

---

## 1. Functional Safety Standards

* **ISO 26262 (Road Vehicles — Functional Safety):**
    * ASIL classification: A (lowest) through D (highest) safety integrity levels.
    * Hazard Analysis and Risk Assessment (HARA): severity, exposure, controllability.
    * Safety goals → functional safety requirements → technical safety requirements.
    * Safety lifecycle: concept → development → production → operation.
    * Hardware metrics: SPFM (Single Point Fault Metric), LFM (Latent Fault Metric).

* **SOTIF (ISO 21448 — Safety of the Intended Functionality):**
    * Addresses risks from sensor limitations, algorithm uncertainty, and unpredictable environments — not just hardware faults (which ISO 26262 covers).
    * Known/unknown unsafe scenarios, triggering conditions.
    * Validation strategy: reduce residual risk from intended functionality.

* **Safety architecture patterns:**
    * **Redundancy:** Dual-channel monitoring (e.g., two independent perception paths).
    * **Diverse redundancy:** Different algorithms or sensors for the same safety function.
    * **Plausibility monitoring:** Cross-check perception outputs against expected physical constraints.
    * **Graceful degradation:** Fallback to simpler, safer behavior when primary system degrades.

**Projects:**
* Perform a HARA for a lane-keeping assist function. Assign ASIL levels to identified safety goals and propose mitigations.
* Design a safety architecture for an ACC system: identify single-point failures and propose redundancy/monitoring.

---

## 2. V2X (Vehicle-to-Everything) Communication

* **Communication standards:**
    * **DSRC (Dedicated Short-Range Communications):** 802.11p-based, mature but limited bandwidth.
    * **C-V2X (Cellular V2X):** LTE-V2X (PC5 sidelink), 5G NR-V2X — higher bandwidth, lower latency.
    * V2V (vehicle-to-vehicle), V2I (vehicle-to-infrastructure), V2P (vehicle-to-pedestrian).

* **Cooperative perception:**
    * Vehicles share sensor data or object detections via V2X.
    * Extends effective sensing range beyond individual vehicle FoV.
    * Challenges: latency, bandwidth, data format standardization.

* **V2X security:**
    * IEEE 1609.2, ETSI ITS Security.
    * Certificate authorities, pseudonymous authentication.
    * Privacy-preserving communication (location privacy vs. safety).

---

## 3. ADAS Validation and Testing

* **Scenario-based testing:**
    * Scenario databases: OpenSCENARIO, ASAM OSI.
    * Systematic coverage: edge cases, SOTIF-relevant scenarios, ODD (Operational Design Domain) boundaries.
    * Concrete vs. logical vs. functional scenarios.

* **Hardware-in-the-Loop (HIL) testing:**
    * Inject synthetic sensor data into production ECUs.
    * Validate ADAS software under controlled, repeatable conditions.
    * Closed-loop HIL: ECU outputs feed back into simulation.

* **Shadow mode deployment:**
    * Run experimental perception in parallel with production system — no actuation.
    * Log disagreements between experimental and production outputs.
    * Offline evaluation: curate targeted test sets from disagreements.
    * Metric: disagreement rate, false positive/negative analysis.

* **Field operational tests (FOT):**
    * Controlled real-world testing with safety drivers.
    * Data collection: metrics, edge cases, system performance under ODD.
    * Regulatory requirements by region (US, EU, China).

**Projects:**
* Build a simple HIL test rig that injects synthetic camera frames into an ADAS perception node. Validate detection accuracy across day/night/fog scenarios.
* Deploy an experimental perception algorithm in shadow mode alongside a baseline. Collect and analyze disagreements to identify algorithmic weaknesses.
* Create an OpenSCENARIO scenario for a pedestrian crossing at an intersection. Run it in CARLA with your Module 1 controller.

---

## Resources

| Resource | Why |
|----------|-----|
| ISO 26262 Standard | Foundational safety standard for automotive electronics |
| ISO 21448 (SOTIF) | Safety of intended functionality for ADAS/AD |
| *Autonomous Vehicles and Functional Safety* (Tier 1 guides) | Practical guides to applying ISO 26262 + SOTIF |
| [5GAA](https://5gaa.org/) | C-V2X standards, use cases, deployment guidance |
| [OpenSCENARIO](https://www.asam.net/standards/detail/openscenario/) | Scenario description standard for ADAS testing |
| [CARLA](https://carla.org/) | Simulation for HIL and scenario-based testing |

---

## Next

→ **[Module 6 — Lauterbach TRACE32 Debug](../6.%20Lauterbach%20TRACE32%20Debug/Guide.md)** (optional) — In-circuit debug and trace for automotive ECUs.
