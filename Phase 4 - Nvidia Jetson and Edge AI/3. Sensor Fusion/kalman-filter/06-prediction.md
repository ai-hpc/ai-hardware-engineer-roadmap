# Prediction

**Level: Middle School (Ages 13-15)**

## Introduction

So far we've learned how to combine measurements. But what if you need to know where something is BEFORE you measure it? That's where **prediction** comes in!

## What is Prediction?

**Prediction** is using what you know now to estimate what will happen in the future.

### Example 1: Catching a Ball

When someone throws you a ball:
1. You see where it is NOW
2. You see how fast it's moving
3. Your brain PREDICTS where it will be in 1 second
4. You move your hands to that spot!

You don't wait for the ball to arrive - you predict and act!

### Example 2: Walking to School

You leave home at 8:00 AM. School is 10 minutes away.

**Prediction**: "I'll arrive at 8:10 AM"

This prediction is based on:
- Where you are now (home)
- How fast you walk (1 minute per block)
- How far school is (10 blocks)

## Motion Models

A **motion model** describes how things move. Let's start simple!

### Constant Position Model

**Rule**: Things don't move.

```
Future Position = Current Position
```

**Example**: A book on a table
- Now: position = 5 feet from wall
- In 10 seconds: position = 5 feet from wall

This works for stationary objects!

### Constant Velocity Model

**Rule**: Things move at a steady speed.

```
Future Position = Current Position + Velocity × Time
```

**Example**: A toy car rolling at 2 feet per second
- Now: position = 10 feet
- Velocity: 2 feet/second
- After 5 seconds: position = 10 + (2 × 5) = 20 feet

### Constant Acceleration Model

**Rule**: Things speed up or slow down steadily.

```
Future Position = Current Position + Velocity × Time + ½ × Acceleration × Time²
```

**Example**: A ball rolling down a ramp
- Now: position = 0 feet, velocity = 0 ft/s
- Acceleration: 2 ft/s²
- After 3 seconds: position = 0 + 0×3 + ½×2×3² = 9 feet

## State and State Transition

### What is State?

**State** is everything you need to know about a system at one moment.

**Example: Tracking a car**
- State = [position, velocity]
- State = [100 meters, 20 m/s]

This tells you where the car is AND how fast it's going.

### State Transition

**State transition** is how the state changes over time.

**Example**: Car moving at constant velocity
```
Initial state: [position=100m, velocity=20m/s]
Time passes: 2 seconds
New state: [position=140m, velocity=20m/s]
```

The position changed, but velocity stayed the same!

## Prediction Equations

### For Constant Velocity

**State**: [position, velocity]

**Prediction after time Δt**:
```
new_position = old_position + velocity × Δt
new_velocity = old_velocity
```

**Example**:
```
Current state: [100m, 20m/s]
Time step: 0.5 seconds

Predicted position = 100 + 20×0.5 = 110m
Predicted velocity = 20m/s

Predicted state: [110m, 20m/s]
```

### For Constant Acceleration

**State**: [position, velocity, acceleration]

**Prediction after time Δt**:
```
new_position = old_position + velocity×Δt + ½×acceleration×Δt²
new_velocity = old_velocity + acceleration×Δt
new_acceleration = old_acceleration
```

## Prediction Uncertainty

Predictions aren't perfect! The further into the future you predict, the less certain you are.

### Example: Weather Forecast

- Tomorrow's forecast: ±2°F uncertainty
- Next week's forecast: ±5°F uncertainty
- Next month's forecast: ±10°F uncertainty

**Rule**: Uncertainty grows with time!

### Why Uncertainty Grows

**Example**: Toy car rolling

You know:
- Position: 10 ± 0.1 feet
- Velocity: 2 ± 0.2 feet/second

After 5 seconds:
- Position = 10 + 2×5 = 20 feet
- But velocity uncertainty affects position!
- Position uncertainty = 0.1 + 0.2×5 = 1.1 feet

The prediction is 20 ± 1.1 feet (much more uncertain!)

## Process Noise

Real systems don't follow perfect models. There's always some randomness called **process noise**.

### Example: Walking to School

Your model says: "I walk at exactly 1 block per minute"

Reality:
- Sometimes you walk faster
- Sometimes you stop to tie your shoe
- Sometimes you wait at a crosswalk

This randomness is process noise!

### Adding Process Noise to Predictions

```
Prediction Uncertainty = Model Uncertainty + Process Noise
```

**Example**: Car tracking
- Model uncertainty: ±2 meters
- Process noise (wind, road bumps): ±1 meter per second
- After 3 seconds: ±2 + ±1×3 = ±5 meters

## The Predict-Update Cycle

The Kalman filter alternates between two steps:

### Step 1: PREDICT
"Based on physics, where should things be?"

```
Predicted State = f(Previous State, Time)
Predicted Uncertainty = Previous Uncertainty + Process Noise
```

### Step 2: UPDATE (Measure)
"What do my sensors actually see?"

```
New State = Weighted Average(Prediction, Measurement)
New Uncertainty = Reduced (because we combined information!)
```

### Example: Tracking a Drone

**Time 0.0s**:
- State: [altitude=100m, velocity=5m/s up]
- Uncertainty: ±1m

**Time 0.5s - PREDICT**:
- Predicted altitude: 100 + 5×0.5 = 102.5m
- Predicted velocity: 5m/s
- Predicted uncertainty: ±1.5m (grew!)

**Time 0.5s - UPDATE**:
- Measurement: 102m ± 2m
- Combined estimate: 102.3m ± 1.2m
- (Uncertainty decreased!)

**Time 1.0s - PREDICT**:
- Predicted altitude: 102.3 + 5×0.5 = 104.8m
- Predicted uncertainty: ±1.7m

And so on...

## Practice Problems

### Problem 1: Simple Prediction

A robot is at position 50cm, moving at 10cm/s.
Where will it be after 3 seconds?

### Problem 2: With Acceleration

A car starts at position 0m with velocity 0m/s.
It accelerates at 2m/s².
a) Where is it after 5 seconds?
b) What's its velocity after 5 seconds?

### Problem 3: Uncertainty Growth

Initial: position = 100 ± 2 meters, velocity = 10 ± 1 m/s
After 4 seconds, what's the position uncertainty?
(Assume velocity uncertainty contributes to position)

### Problem 4: Predict-Update Cycle

**Initial state**: [position=0m, velocity=5m/s], uncertainty=±1m

**Step 1**: Predict after 2 seconds
**Step 2**: Measure position=11m ± 2m
**Step 3**: Update estimate (use Kalman gain)

What's the final estimate?

### Problem 5: Process Noise

You're tracking a ball rolling on a bumpy surface.
- Initial uncertainty: ±0.5m
- Process noise: ±0.3m per second
- After 10 seconds of prediction (no measurements), what's the uncertainty?

## Real World Applications

### Self-Driving Cars

The car predicts:
- "Based on my speed and steering, I'll be here in 0.1 seconds"
- Then measures with cameras and GPS
- Updates the estimate
- Repeats 10 times per second!

### Missile Tracking

Radar tracks a missile:
- Predict where it will be based on trajectory
- Measure with radar (noisy!)
- Update estimate
- Predict next position
- Repeat

### Robot Navigation

A robot vacuum:
- Predicts position based on wheel rotations
- Measures position with wall sensors
- Updates estimate
- Predicts next position
- Repeat

### Weather Forecasting

Meteorologists:
- Run physics models to predict weather
- Get measurements from weather stations
- Update the model
- Predict further into future
- Repeat

## Try It Yourself!

### Experiment 1: Ball Rolling

1. Roll a ball across the floor
2. Measure its position at 0s, 1s, 2s
3. Calculate velocity from first two measurements
4. PREDICT where it will be at 3s
5. MEASURE where it actually is at 3s
6. How close was your prediction?

### Experiment 2: Walking Speed

1. Walk at a steady pace
2. Measure distance after 10 seconds
3. Calculate your velocity
4. PREDICT how far you'll walk in 30 seconds
5. Actually walk for 30 seconds and measure
6. Was your prediction close? Why or why not?

### Experiment 3: Prediction Uncertainty

1. Drop a ball from different heights
2. Predict how long it will take to hit ground (use t = √(2h/g))
3. Measure actual time
4. Calculate prediction error for each height
5. Does error grow with height?

### Experiment 4: Process Noise

1. Set up a toy car to roll down a ramp
2. Time it 5 times
3. Calculate average time and standard deviation
4. The standard deviation is your process noise!
5. Use this to predict uncertainty for future runs

## Key Concepts

1. **Prediction** uses current state to estimate future state
2. **Motion models** describe how things move (constant velocity, acceleration, etc.)
3. **State** contains all information needed (position, velocity, etc.)
4. **Uncertainty grows** during prediction
5. **Process noise** accounts for model imperfections
6. **Predict-Update cycle** is the heart of Kalman filtering

## What's Next?

Now we have all the pieces! The next chapter will put everything together and show you the complete **Kalman Filter** algorithm. Get ready - this is where it all comes together!

---

**Key Vocabulary**
- **Prediction**: Estimating future state from current state
- **Motion Model**: Mathematical description of how things move
- **State**: All variables needed to describe a system
- **State Transition**: How state changes over time
- **Process Noise**: Random variations in the system
- **Predict-Update Cycle**: Alternating between prediction and measurement
