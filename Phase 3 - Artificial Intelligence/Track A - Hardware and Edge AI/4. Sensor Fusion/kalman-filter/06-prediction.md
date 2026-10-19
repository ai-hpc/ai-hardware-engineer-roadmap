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

Predictions are uncertain. In common motion models, unknown velocity and acceleration make a distant prediction less certain. The amount follows the model and covariance, not a rule that every variance must always increase.

### Example: Weather Forecast

- Tomorrow's forecast: ±2°F uncertainty
- Next week's forecast: ±5°F uncertainty
- Next month's forecast: ±10°F uncertainty

These weather intervals are illustrative; they do not define a Kalman noise model.

### Why Uncertainty Grows

**Example**: Toy car rolling

You know:
- Position: 10 ± 0.1 feet
- Velocity: 2 ± 0.2 feet/second

After 5 seconds:
- Position = 10 + 2×5 = 20 feet
- But velocity uncertainty affects position!
- Interpret both ± values as standard deviations of independent errors. Position variance becomes `0.1² + 5²*0.2² = 1.01 ft²`.
- Position standard deviation is `sqrt(1.01)=1.005 feet`.

The prediction is approximately `20 ± 1.005 feet`. Adding standard deviations gives a conservative bound for bounded errors, not this independent-error variance calculation. If the errors are correlated, include `2*dt*Cov(position,velocity)` as well.

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
Predicted covariance = Propagated covariance + Process-noise covariance
```

**Example**: Car tracking
- Simplify to a position random walk with initial standard deviation 2 m.
- Each one-second step adds an independent zero-mean displacement with standard deviation 1 m.
- After three steps, variance is `2²+3*1²=7 m²`, so standard deviation is `sqrt(7)=2.646 m`.

A fixed unknown velocity bias of standard deviation 1 m/s produces a different result: after three seconds its position-error variance is `3²*1²=9 m²`, giving total variance 13 m². Describe the stochastic process before assigning a number to `Q`.

## The Predict-Update Cycle

The Kalman filter alternates between two steps:

### Step 1: PREDICT
"Based on physics, where should things be?"

```
Predicted State = f(Previous State, Time)
Predicted Covariance = F * Previous Covariance * Fᵀ + Q
```

### Step 2: UPDATE (Measure)
"What do my sensors actually see?"

```
New State = Weighted Average(Prediction, Measurement)
New Uncertainty = Reduced (because we combined information!)
```

### Example: Tracking a Drone

For this scalar illustration, velocity is treated as known exactly, `±` means one standard deviation, and each 0.5-second step adds independent position variance `Q=1.25 m²`. If velocity is estimated, use the full two-state covariance in lesson 08.

**Time 0.0s**:
- State: [altitude=100m, velocity=5m/s up]
- Uncertainty: ±1m

**Time 0.5s - PREDICT**:
- Predicted altitude: 100 + 5×0.5 = 102.5m
- Predicted velocity: 5m/s
- Predicted uncertainty: ±1.5m (grew!)

**Time 0.5s - UPDATE**:
- Measurement: 102m ± 2m
- Combined estimate: 102.32m ± 1.2m (gain=2.25/(2.25+4)=0.36)
- (Uncertainty decreased!)

**Time 1.0s - PREDICT**:
- Predicted altitude: 102.32 + 5×0.5 = 104.82m
- Predicted uncertainty: sqrt(1.2²+1.25)=±1.640m

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
(Assume these are independent one-standard-deviation errors and no process noise.)

### Problem 4: Predict-Update Cycle

**Initial state**: position 0 m with standard deviation 1 m; velocity known exactly as 5 m/s; no process noise.

**Step 1**: Predict after 2 seconds
**Step 2**: Measure position=11m ± 2m
**Step 3**: Update estimate (use Kalman gain)

What's the final estimate?

### Problem 5: Process Noise

You're tracking a ball rolling on a bumpy surface.
- Initial uncertainty: ±0.5m
- Each one-second step adds independent displacement noise with standard deviation 0.3 m
- After 10 seconds of prediction (no measurements), what's the uncertainty?

### Check your answers

1. `50+10*3=80 cm`.
2. Position `0.5*2*5²=25 m`; velocity `2*5=10 m/s`.
3. Position variance `2²+4²*1²=20 m²`; standard deviation `sqrt(20)=4.472 m`.
4. Predicted position 10 m, variance 1; gain `1/(1+4)=.2`; corrected position 10.2 m, variance .8 m², standard deviation .8944 m.
5. Variance `.5²+10*.3²=1.15 m²`; standard deviation `sqrt(1.15)=1.0724 m`. Repeated independent errors and one persistent bias accumulate differently.

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
4. This spread contains timing/measurement noise and variation in the motion. Separate these effects before using it to derive a process covariance.
5. Use this to predict uncertainty for future runs

## Key Concepts

1. **Prediction** uses current state to estimate future state
2. **Motion models** describe how things move (constant velocity, acceleration, etc.)
3. **State** contains all information needed (position, velocity, etc.)
4. **Covariance propagates** through the motion model and receives process noise
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
