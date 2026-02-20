# 6D Kalman Filter Explained Simply

## What is a 6D Kalman Filter?

A 6D Kalman filter tracks **6 things at once**:
- 3D Position: Where something is (x, y, z)
- 3D Velocity: How fast it's moving (vx, vy, vz)

Think of it like tracking a drone flying through the air - you want to know both where it is AND how fast it's going in all three directions.

## The Big Picture

Imagine you're tracking a drone with GPS:
- GPS tells you **where** the drone is (but with some error)
- GPS does NOT tell you **how fast** it's moving
- The Kalman filter figures out the velocity automatically!

### How Does It Work?

The filter uses a simple idea:
> "If I know where something was a moment ago, and I know where it is now, I can figure out how fast it's moving!"

## The State Vector

The **state vector** is like a report card that contains everything we know:

```
State = [x ]  ← position in x direction (meters)
        [y ]  ← position in y direction (meters)
        [z ]  ← position in z direction (meters)
        [vx]  ← velocity in x direction (meters/second)
        [vy]  ← velocity in y direction (meters/second)
        [vz]  ← velocity in z direction (meters/second)
```

### Example State

```
State = [10.5]  ← 10.5 meters east
        [8.2 ]  ← 8.2 meters north
        [3.0 ]  ← 3.0 meters up
        [2.0 ]  ← moving 2 m/s east
        [1.5 ]  ← moving 1.5 m/s north
        [0.5 ]  ← moving 0.5 m/s up
```

This tells us: The drone is at position (10.5, 8.2, 3.0) and moving at velocity (2.0, 1.5, 0.5).

## The Two-Step Dance

The Kalman filter does two things over and over:

### Step 1: PREDICT (Where will it be?)

**The Question**: "Based on where it is now and how fast it's moving, where will it be in 0.1 seconds?"

**The Math** (simple version):
```
new_position = old_position + velocity × time
new_velocity = old_velocity (stays the same)
```

**Example**:
```
Current: x = 10.5m, vx = 2.0 m/s
Time: 0.1 seconds

Predicted: x = 10.5 + 2.0 × 0.1 = 10.7m
           vx = 2.0 m/s (unchanged)
```

### Step 2: UPDATE (What did we actually measure?)

**The Question**: "GPS says it's at 10.6m. My prediction said 10.7m. What's the truth?"

**The Answer**: Use a weighted average!
- If prediction is more certain → trust it more
- If GPS is more certain → trust it more

**Example**:
```
Prediction: 10.7m (uncertainty ±0.5m)
GPS: 10.6m (uncertainty ±2.0m)

Prediction is more certain, so trust it more!
Final estimate: 10.68m (closer to prediction)
```

## The Magic: Estimating Velocity Without Measuring It!

This is the coolest part! GPS only tells us position, but the filter figures out velocity.

### How?

**Observation 1**: At time 0s, position = 10.0m
**Observation 2**: At time 1s, position = 12.0m

**The Filter Thinks**:
> "It moved 2 meters in 1 second, so velocity must be about 2 m/s!"

But it's smarter than that - it uses ALL the measurements over time to get a really good estimate.

## Understanding Uncertainty

The filter doesn't just track position and velocity - it also tracks **how sure it is**.

### Uncertainty Visualization

```
Very Uncertain:  [====±5m====]
Somewhat Certain: [==±2m==]
Very Certain:     [±0.5m]
```

### How Uncertainty Changes

**During PREDICT**: Uncertainty GROWS
```
Before: Position ±0.5m
After:  Position ±0.8m (less certain because things can change)
```

**During UPDATE**: Uncertainty SHRINKS
```
Before: Position ±0.8m
After:  Position ±0.4m (more certain because we got new info!)
```

Over time, uncertainty oscillates but generally decreases:
```
Time:        0s    1s    2s    3s    4s    5s
Uncertainty: ±5m → ±3m → ±2m → ±1.5m → ±1m → ±0.8m
```

## The Matrices Explained Simply

Don't be scared of matrices! They're just organized ways to do math.

### F Matrix (State Transition)

**What it does**: Predicts the next state

```
F = [1  0  0  dt  0   0 ]
    [0  1  0  0   dt  0 ]
    [0  0  1  0   0   dt]
    [0  0  0  1   0   0 ]
    [0  0  0  0   1   0 ]
    [0  0  0  0   0   1 ]
```

**What it means**:
- Row 1: `new_x = old_x + vx × dt` (position changes by velocity × time)
- Row 4: `new_vx = old_vx` (velocity stays the same)

### H Matrix (Measurement)

**What it does**: Extracts what we can measure from the state

```
H = [1  0  0  0  0  0]  ← measure x position
    [0  1  0  0  0  0]  ← measure y position
    [0  0  1  0  0  0]  ← measure z position
```

**What it means**: GPS gives us position (x, y, z) but NOT velocity!

### Q Matrix (Process Noise)

**What it does**: Says "the world is unpredictable"

**Why we need it**:
- Wind might push the drone
- Motors might vary slightly
- Physics isn't perfect

**Effect**: Adds uncertainty during prediction

### R Matrix (Measurement Noise)

**What it does**: Says "GPS isn't perfect"

```
R = [2.0  0    0  ]  ← x has ±√2 = ±1.4m error
    [0    2.0  0  ]  ← y has ±√2 = ±1.4m error
    [0    0    3.0]  ← z has ±√3 = ±1.7m error
```

**What it means**: GPS is noisier in altitude (z) than horizontal (x, y)

## Step-by-Step Example

Let's track a drone for 3 time steps!

### Initial State (t=0s)

```
Position: [0, 0, 0] meters
Velocity: [1, 0.5, 0.2] m/s
Uncertainty: ±10m for position, ±5 m/s for velocity
```

### Time t=0.1s

**PREDICT**:
```
New position = [0, 0, 0] + [1, 0.5, 0.2] × 0.1
             = [0.1, 0.05, 0.02] meters

New velocity = [1, 0.5, 0.2] m/s (unchanged)

Uncertainty: ±10.1m (grew slightly)
```

**GPS MEASUREMENT**: [0.15, 0.08, 0.01] meters (noisy!)

**UPDATE**:
```
Prediction: [0.1, 0.05, 0.02] ± 10.1m
GPS:        [0.15, 0.08, 0.01] ± 1.4m

GPS is more certain, so trust it more!

Final estimate: [0.14, 0.07, 0.015] meters
Uncertainty: ±1.2m (much better!)
```

### Time t=0.2s

**PREDICT**:
```
Position = [0.14, 0.07, 0.015] + [1, 0.5, 0.2] × 0.1
         = [0.24, 0.12, 0.035] meters

Uncertainty: ±1.3m (grew slightly)
```

**GPS MEASUREMENT**: [0.22, 0.13, 0.04] meters

**UPDATE**:
```
Prediction: [0.24, 0.12, 0.035] ± 1.3m
GPS:        [0.22, 0.13, 0.04] ± 1.4m

About equal certainty, so average them!

Final estimate: [0.23, 0.125, 0.0375] meters
Uncertainty: ±0.9m (even better!)
```

### Time t=0.3s

And so on... The filter keeps getting better!

## Why 6D Instead of 3D?

You might ask: "Why not just track position (3D) and calculate velocity ourselves?"

### Problems with Simple Approach

**Approach 1**: Just use GPS position
- ❌ Very noisy
- ❌ No velocity information
- ❌ Can't predict future position

**Approach 2**: Calculate velocity = (new_pos - old_pos) / time
- ❌ Very noisy (noise in position gets amplified!)
- ❌ No smoothing
- ❌ Jumps around a lot

### Benefits of 6D Kalman Filter

✅ Smooth position estimates
✅ Smooth velocity estimates
✅ Can predict future positions
✅ Handles missing measurements
✅ Optimal combination of all information
✅ Tracks uncertainty

## Real-World Applications

### 1. Drone Navigation

```
GPS → 6D Kalman Filter → Smooth position + velocity
                       → Control system
                       → Stable flight!
```

### 2. Self-Driving Cars

```
GPS + Wheel sensors → 6D Kalman Filter → Car position + speed
                                       → Navigation system
                                       → Safe driving!
```

### 3. Smartphone Location

```
GPS + Accelerometer → 6D Kalman Filter → Your location + speed
                                       → Maps app
                                       → Accurate directions!
```

### 4. Rocket Tracking

```
Radar → 6D Kalman Filter → Rocket position + velocity
                         → Trajectory prediction
                         → Mission control!
```

## Common Questions

### Q1: Why does uncertainty grow during prediction?

**A**: Because the future is uncertain! Even if you know exactly where something is now, you can't be 100% sure where it will be in the future. Things can change!

### Q2: How does the filter know how much to trust each source?

**A**: It uses the uncertainty values! Lower uncertainty = more trust.

```
If prediction uncertainty = ±0.5m and GPS uncertainty = ±2m
→ Trust prediction 4× more than GPS!
```

### Q3: What if GPS is completely wrong?

**A**: The filter will notice! If GPS says something crazy (like the drone teleported 100 meters), the innovation (difference between prediction and measurement) will be huge. The filter can detect and reject outliers.

### Q4: Can it track acceleration too?

**A**: Yes! You'd make it a 9D filter:
```
State = [x, y, z, vx, vy, vz, ax, ay, az]
```

But that's more complex and often not needed.

## Visualizing the Filter

### The Spiral Trajectory Example

Imagine a drone flying in a spiral while climbing:

```
Top View (X-Y):          Side View (X-Z):

    ╱─╲                      ╱
   ╱   ╲                    ╱
  │  •  │                  ╱
   ╲   ╱                  ╱
    ╲─╱                  •────────

  Circular              Climbing
  motion                upward
```

The 6D Kalman filter tracks:
- X, Y positions (circular motion)
- Z position (climbing)
- VX, VY velocities (changing for circular motion)
- VZ velocity (constant upward)

### What the Filter Sees

```
Time 0s:  Position (0, 0, 0),    Velocity (1.0, 0.0, 0.2)
Time 1s:  Position (0.9, 0.1, 0.2), Velocity (0.9, 0.4, 0.2)
Time 2s:  Position (1.6, 0.5, 0.4), Velocity (0.7, 0.7, 0.2)
Time 3s:  Position (2.0, 1.0, 0.6), Velocity (0.4, 0.9, 0.2)
...
```

The filter smoothly tracks this complex motion!

## Key Takeaways

1. **6D = Position + Velocity**: Track where something is AND how fast it's moving

2. **Two Steps**:
   - PREDICT: Use physics to guess next state
   - UPDATE: Use measurements to correct the guess

3. **Estimates Velocity**: Even though GPS only measures position!

4. **Tracks Uncertainty**: Knows how confident it is

5. **Optimal**: Mathematically proven to be the best linear estimator

6. **Real-Time**: Works as measurements arrive, no need to wait

## Try It Yourself!

Run the example code:
```bash
python kalman_6d.py
```

You'll see:
- A 3D spiral trajectory
- Noisy GPS measurements (red dots)
- Smooth Kalman estimates (blue line)
- How uncertainty decreases over time
- Velocity estimation without direct measurement!

## Further Reading

- [Chapter 10: Matrix Form](10-matrix-form.md) - Full mathematical treatment
- [Chapter 11: Multidimensional KF](11-multidimensional-kf.md) - More examples
- [Cheat Sheet](cheat-sheet.md) - Quick reference for equations

---

**Remember**: The Kalman filter is just a smart way to combine predictions and measurements. It's like having a really good assistant who:
1. Remembers where things were
2. Predicts where they'll be
3. Checks measurements
4. Figures out the truth
5. Never stops learning!
