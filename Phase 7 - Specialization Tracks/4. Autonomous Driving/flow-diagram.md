# Openpilot Pipeline — Flow Diagram

End-to-end data flow in **openpilot** (comma.ai), from camera input to CAN actuation. Based on the actual process layout in [`openpilot/`](openpilot/).

---

## High-Level Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  camerad    │───▶│   modeld   │───▶│  plannerd   │───▶│ controlsd  │───▶│   pandad    │
│  (camera)   │    │  (tinygrad) │    │  radard     │    │  (lat+long) │    │   (CAN)     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                    │
                          │                    └── radarState (lead vehicle)
                          └── modelV2 (lanes, plan, pose)
```

---

## Process-Level Flow (Openpilot)

```mermaid
flowchart TB
    subgraph SENSORS["📷 Sensors"]
        CAM[camerad<br/>Road + driver cameras]
        RADAR[Radar CAN<br/>optional, per car]
        IMU[IMU / GPS<br/>locationd]
    end

    subgraph PERCEPTION["🔍 Perception (modeld)"]
        VISION[driving_vision_tinygrad.pkl]
        POLICY[driving_policy_tinygrad.pkl]
        VISION --> POLICY
    end

    subgraph DM["👁️ Driver Monitoring"]
        DMODEL[dmonitoringmodeld]
        DMON[dmonitoringd]
        DMODEL --> DMON
    end

    subgraph LOCALIZATION["📍 Localization"]
        CALIB[calibrationd]
        LOC[locationd]
        PARAMS[paramsd]
        TORQUE[torqued]
    end

    subgraph PLANNING["📐 Planning"]
        RADARD[radard<br/>lead vehicle tracking]
        PLANNERD[plannerd<br/>LongitudinalPlanner, LDW]
    end

    subgraph CONTROL["🎛️ Control"]
        CONTROLS[controlsd<br/>LatControl + LongControl]
    end

    subgraph ACTUATION["⚙️ Actuation"]
        CARD[card<br/>Car interface]
        PANDAD[pandad<br/>CAN bus]
    end

    CAM --> PERCEPTION
    CAM --> DM
    RADAR --> RADARD
    IMU --> LOCALIZATION

    PERCEPTION --> |modelV2| PLANNERD
    PERCEPTION --> |modelV2| CONTROLS
    LOCALIZATION --> |livePose, liveCalibration| CONTROLS
    RADARD --> |radarState| PLANNERD
    PLANNERD --> |longitudinalPlan| CONTROLS
    DM --> |driverMonitoringState| CONTROLS

    CONTROLS --> |carControl| CARD
    CARD --> PANDAD
```

---

## Message Flow (cereal)

```mermaid
flowchart LR
    subgraph modeld
        FRAME[VisionIpc<br/>camera frames]
        MDL[modelV2]
        FRAME --> MDL
    end

    subgraph plannerd
        RADAR[radarState]
        PLAN[longitudinalPlan]
        MDL --> PLAN
        RADAR --> PLAN
    end

    subgraph controlsd
        CC[carControl]
        PLAN --> CC
        MDL --> CC
        POSE[livePose] --> CC
        DMS[driverMonitoringState] --> CC
    end

    subgraph pandad
        CAN[CAN messages]
        CC --> CAN
    end
```

---

## modeld: Perception Detail

```mermaid
flowchart TB
    subgraph INPUT
        VIPC[VisionIpcClient<br/>road camera frames]
        WARP[get_warp_matrix<br/>calibration]
    end

    subgraph TINYGRAD["tinygrad models"]
        VISION[driving_vision<br/>lane lines, pose, road edges]
        POLICY[driving_policy<br/>plan: velocity, curvature]
        VISION --> POLICY
    end

    subgraph OUTPUT["modelV2"]
        LANES[laneLines]
        EDGES[roadEdges]
        POSE[pose]
        PLAN[position / velocity / acceleration]
        ACTION[desiredCurvature, desiredAcceleration]
    end

    VIPC --> INPUT
    WARP --> INPUT
    INPUT --> TINYGRAD
    TINYGRAD --> OUTPUT
```

**modeld outputs** (from `fill_model_msg`, `parse_model_outputs`):
- Lane lines, road edges, lane line probabilities
- Pose (road transform, device pose)
- Plan (position, velocity, acceleration over time)
- Action (desired curvature, desired acceleration, shouldStop)
- FCW (forward collision) probabilities

---

## plannerd → controlsd

```mermaid
flowchart LR
    subgraph plannerd
        LONG[LongitudinalPlanner]
        LDW[LaneDepartureWarning]
        LONG --> longitudinalPlan
        LDW --> driverAssistance
    end

    subgraph controlsd
        LAT[LatControl<br/>Angle / PID / Torque]
        LONG_CTRL[LongControl]
        LAT --> actuators
        LONG_CTRL --> actuators
    end

    modelV2 --> plannerd
    radarState --> plannerd
    longitudinalPlan --> controlsd
    modelV2 --> controlsd
```

**plannerd** subscribes: `modelV2`, `carState`, `radarState`, `controlsState`, `liveParameters`  
**plannerd** publishes: `longitudinalPlan`, `driverAssistance`

**controlsd** subscribes: `modelV2`, `longitudinalPlan`, `livePose`, `liveCalibration`, `carState`, `driverMonitoringState`  
**controlsd** publishes: `carControl` (actuators)

---

## controlsd → CAN

```mermaid
flowchart LR
    subgraph controlsd
        CC[carControl]
        STEER[steer]
        GAS[gas]
        BRAKE[brake]
    end

    subgraph card
        CI[CarInterface]
    end

    subgraph pandad
        CAN[CAN]
    end

    CC --> CI
    CI --> CAN
```

---

## Openpilot-Specific Notes

| Aspect | Openpilot |
|--------|-----------|
| **Sensors** | Camera(s) primary; radar optional (car-dependent) |
| **Perception** | End-to-end NN (vision + policy) in modeld; no explicit 2D/3D detection |
| **Planning** | Plan from model; plannerd adds longitudinal (ACC, lead follow) and LDW |
| **Control** | LatControl (angle/PID/torque), LongControl; vehicle-specific via opendbc |
| **Inference** | tinygrad (driving_vision, driving_policy, dmonitoring) |
| **Messaging** | cereal (capnp) over IPC |

---

## Process → Source Map

| Process | Path |
|---------|------|
| camerad | `system/camerad/` — [camerad Guide](camerad/Guide.md) |
| modeld | `selfdrive/modeld/modeld.py` |
| dmonitoringmodeld | `selfdrive/modeld/dmonitoringmodeld.py` |
| dmonitoringd | `selfdrive/monitoring/dmonitoringd.py` |
| locationd | `selfdrive/locationd/locationd.py` |
| calibrationd | `selfdrive/locationd/calibrationd.py` |
| radard | `selfdrive/controls/radard.py` |
| plannerd | `selfdrive/controls/plannerd.py` |
| controlsd | `selfdrive/controls/controlsd.py` |
| card | `selfdrive/car/card.py` |
| pandad | `selfdrive/pandad/` |
