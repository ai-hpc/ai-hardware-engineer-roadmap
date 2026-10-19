# The Kalman Filter Idea

**Level: High School (Ages 16-18)**

## Putting It All Together

You've learned all the pieces. Now let's see how they fit together into the **Kalman Filter**!

## The Big Picture

The Kalman filter is an algorithm that:
1. Keeps track of what you know (state + uncertainty)
2. Predicts what will happen next
3. Measures what actually happened
4. Combines prediction and measurement under explicit model assumptions
5. Repeats forever

It's like having a really smart assistant that:
- Remembers everything
- Makes educated guesses
- Listens to sensors
- Figures out the truth
- Never stops learning

## The Two-Step Dance

### Step 1: PREDICT (Time Update)

"What do I expect to happen?"

**Inputs**:
- Previous state estimate
- Previous uncertainty
- Motion model
- Time elapsed

**Outputs**:
- Predicted state
- Predicted uncertainty, propagated through the motion model

**Example**: Tracking a car
```
Previous: position=100m ± 2m, velocity=20m/s ± 1m/s
Time: 1 second passes
Predicted: position=120m ± 3m, velocity=20m/s ± 1m/s
Assumptions: ± is one standard deviation, prior covariance is zero,
and this step adds 4 m² of independent position process-noise variance.
```

### Step 2: UPDATE (Measurement Update)

"What did I actually observe?"

**Inputs**:
- Predicted state
- Predicted uncertainty
- Measurement
- Measurement uncertainty

**Outputs**:
- Updated state (weighted average!)
- Updated uncertainty (smaller!)

**Example**: Continuing from above
```
Predicted: position=120m ± 3m
Measured: position=118m ± 4m
Updated: position=119.3m ± 2.4m
```

Notice: Uncertainty decreased from ±3m to ±2.4m!

## Why It's Optimal

With a correctly specified linear model, Gaussian initial state, and independent zero-mean Gaussian process/measurement noise of known covariance, the Kalman filter computes the posterior mean and covariance. The mean minimizes expected squared error under that model. This does not guarantee the smallest error on every individual sample or when the model is wrong.

Without Gaussian noise, the linear minimum-variance interpretation can still apply under the corresponding second-moment assumptions; a particle filter is an option for complex non-Gaussian posteriors, not an automatic requirement for every non-Gaussian sensor.

## The Kalman Filter Loop

```
Initialize:
  - Set initial state
  - Set initial uncertainty

Loop forever:
  1. PREDICT:
     - Use motion model to predict next state
     - Propagate covariance and add process-noise covariance

  2. UPDATE:
     - Get measurement from sensor
     - Calculate Kalman Gain
     - Update state (weighted average)
     - Decrease uncertainty

  3. Repeat
```

## Detailed Example: Tracking a Train

Let's track a train moving at constant velocity.

### Setup

**State**: [position, velocity]
**Motion model**: Constant velocity
**Sensor**: GPS (measures position only)

### Initial Conditions (t=0)

```
State: [0m, 25m/s]
Uncertainty: P = [10² 0  ] = [100  0]
                 [0  5² ]   [0   25]
```

Position uncertainty: ±10m
Velocity uncertainty: ±5m/s

### Time t=1s: PREDICT

**Motion model**:
```
new_position = old_position + velocity × Δt
new_velocity = old_velocity
```

**Predicted state**:
```
position = 0 + 25×1 = 25m
velocity = 25m/s
State: [25m, 25m/s]
```

**Predicted covariance:** use `F=[[1,1],[0,1]]` and, for this ideal one-second example, `Q=0`. Then:

```text
P_minus = F P Fᵀ = [[125,25],[25,25]]
```

Position standard deviation is `sqrt(125)=11.18 m`; velocity standard deviation remains `5 m/s`. The off-diagonal 25 is essential: an error in velocity has now produced an error in position.

### Time t=1s: UPDATE

GPS measures position `23 m` with standard deviation `5 m`, so `H=[1,0]`, `R=25 m²`. Calculate from the unchanged predicted covariance:

```text
residual = 23 - 25 = -2 m
S = 125 + 25 = 150 m²
K = [125/150, 25/150]ᵀ = [5/6, 1/6]ᵀ
position_plus = 25 + (5/6)*(-2) = 23.3333 m
velocity_plus = 25 + (1/6)*(-2) = 24.6667 m/s
P_plus = [[125/6,25/6],[25/6,125/6]]
```

A position observation changes velocity because of the cross-covariance. Both posterior standard deviations are numerically `sqrt(125/6)=4.5644`, with different units: meters and meters per second. The gain's velocity component also has units `1/s`; a matrix gain is not a set of dimensionless percentages.

### Time t=2s: PREDICT

The next mean is `[23.3333 + 24.6667, 24.6667] = [48,24.6667]`. Propagate the **whole** posterior covariance again; do not discard the off-diagonal term. The executable series tests reproduce these numbers and catch the earlier mistake of holding velocity unchanged.

## Key Insights

### Insight 1: Uncertainty Oscillates

```
PREDICT → Uncertainty increases
UPDATE → Uncertainty decreases
PREDICT → Uncertainty increases
UPDATE → Uncertainty decreases
...
```

This pattern is common for a random walk or constant-velocity example. Prediction need not increase every variance: with scalar `F=.5`, `P=4`, and `Q=0`, the next variance is 1. Steady-state behavior needs suitable model/observability assumptions; it is not guaranteed for arbitrary changing models.

### Insight 2: Kalman Gain Adapts

- When prediction is certain: K is small (trust prediction)
- When measurement is certain: K is large (trust measurement)
- It automatically adjusts!

### Insight 3: Information Never Lost

Under the correct independent-noise model, correction cannot increase the covariance in exact arithmetic. One realized state error can still grow. A biased, stale or duplicated measurement may hurt and must not be treated as fresh independent evidence.

### Insight 4: Works in Real-Time

The standard filter processes time-ordered measurements recursively. Delayed or out-of-order observations need a declared discard, buffering, or replay policy; see the [timing lab](../Camera-Time-Fusion-Lab/Guide.md).

## What Makes It "Kalman"?

Kalman published the recursive estimation formulation in 1960. Its useful properties include:

1. **Recursive**: Only needs current state, not all history
2. **Optimal under assumptions**: Minimizes expected squared error for the stated model
3. **Incremental**: Runtime depends on state size and implementation; measure your deadline
4. **General**: Works for many different problems

## Assumptions and Limitations

The Kalman filter works best when:

### ✓ Good Assumptions

1. **Linear system**: State changes linearly
2. **Gaussian noise**: Errors follow bell curve
3. **Known models**: You know how things move
4. **White noise**: Errors are uncorrelated across time (independence follows for jointly Gaussian noise)

### ✗ Limitations

1. **Nonlinear systems**: May need EKF, UKF or another suitable estimator
2. **Non-Gaussian noise**: Consider robustness or richer posterior approximations when needed
3. **Unknown models**: Need adaptive filters
4. **Outliers**: Need robust methods

## Comparison to Other Methods

### Simple Average

```
Estimate = Average of all measurements
```

**Problems**:
- Treats all measurements equally
- Doesn't use motion model
- Doesn't adapt to changing uncertainty

### Moving Average

```
Estimate = Average of last N measurements
```

**Problems**:
- Arbitrary window size
- Doesn't use motion model
- Doesn't weight by uncertainty

### Kalman Filter

```
Estimate = Optimal weighted average of prediction and measurement
```

**Advantages**:
- Uses motion model
- Weights by uncertainty
- Adapts automatically
- Mathematically optimal

## Real-World Success Stories

### Apollo Moon Landing (1960s)

The Kalman filter guided Apollo spacecraft to the moon! It combined:
- Inertial measurements (accelerometers, gyros)
- Star tracker observations
- Ground radar

### GPS Navigation (1990s-present)

Your phone uses Kalman filtering to:
- Combine GPS satellites
- Use motion sensors
- Smooth out noise
- Predict position between updates

### Self-Driving Cars (2010s-present)

Autonomous vehicles use Kalman filters to:
- Fuse camera, radar, lidar
- Track other vehicles
- Estimate own position
- Predict future states

### Weather Forecasting

Meteorologists use Kalman filtering to:
- Combine weather models
- Incorporate measurements
- Update forecasts
- Reduce uncertainty

## Practice Problems

### Problem 1: Conceptual Understanding

A robot is tracking its position. Answer true or false:

a) Uncertainty always decreases over time
b) Kalman Gain is always between 0 and 1
c) Predictions are always more accurate than measurements
d) The filter needs to store all past measurements

### Problem 2: Predict-Update Sequence

Given:
- State: [position=50m, velocity=10m/s]
- Position uncertainty: ±5m
- Velocity uncertainty: ±2m/s

After 2 seconds:
a) What's the predicted position?
b) If measurement is 68m ± 8m, calculate Kalman Gain
c) What's the updated position?

### Problem 3: Uncertainty Analysis

Initial uncertainty: ±10m
After PREDICT: ±12m
After UPDATE with measurement (±5m): ?

Calculate the updated uncertainty.

### Problem 4: Kalman Gain Interpretation

For each scenario, predict if K will be close to 0, 0.5, or 1:

a) Prediction: ±2m, Measurement: ±20m
b) Prediction: ±20m, Measurement: ±2m
c) Prediction: ±5m, Measurement: ±5m

### Problem 5: Real-World Application

You're tracking a drone with:
- GPS: Updates every 1 second, ±5m accuracy
- IMU: Updates every 0.01 seconds, drifts ±0.1m per second

Design a Kalman filter strategy:
a) What's your state vector?
b) When do you PREDICT?
c) When do you UPDATE?
d) Which sensor is more reliable short-term? Long-term?

### Answers and assumptions

1. (a) False. (b) Only for the scalar `H=1` case with nonnegative variances and positive innovation variance; a matrix gain can be negative, exceed one, or carry units. (c) False. (d) The ordinary forward filter stores its sufficient state; delayed-data replay and smoothing retain additional history.
2. Assume initial position/velocity errors are uncorrelated and `Q=0`: predicted position `70 m`, position variance `25+2²*4=41 m²`, position gain `41/(41+64)=41/105`, corrected position `69.21905 m`. Predicted cross-covariance is 8, so velocity also becomes `10−16/105=9.84762 m/s`.
3. Posterior variance is `144*25/(144+25)=21.30178 m²`; standard deviation is `4.61538 m`.
4. Gains are `4/404≈.00990`, `400/404≈.99010`, and `.5`.
5. Position, velocity, orientation and IMU biases are possible states for a full inertial estimator. Predict using timestamped IMU data and update using timestamped position observations. The two quoted “accuracy/drift” numbers do not fully specify covariance, bias, gravity/frame conventions or independence, so they do not determine a complete filter.

## Try It Yourself!

### Experiment 1: Manual Kalman Filter

Track a rolling ball:
1. Measure position at t=0, 1, 2 seconds
2. Calculate velocity from first two points
3. PREDICT position at t=3
4. MEASURE actual position at t=3
5. UPDATE estimate using weighted average
6. Repeat for t=4, 5, 6...

### Experiment 2: Uncertainty Tracking

1. Start with position estimate: 0 ± 10m
2. PREDICT a stationary random walk for one second: add variance `2²=4 m²`, giving `P=104 m²`, rather than adding standard deviations
3. UPDATE with measurement (±5m)
4. Track how uncertainty changes over 10 cycles
5. Plot uncertainty vs time

### Experiment 3: Kalman Gain Behavior

For different uncertainty ratios:
1. Prediction: ±1m, Measurement: ±10m → Calculate K
2. Prediction: ±5m, Measurement: ±5m → Calculate K
3. Prediction: ±10m, Measurement: ±1m → Calculate K

Plot K vs uncertainty ratio.

### Experiment 4: Comparison

Track the same object with:
1. Simple average of all measurements
2. Moving average (last 3 measurements)
3. Kalman filter

Which is most accurate? Most responsive?

## Key Concepts

1. **Two-step process**: Predict, then Update
2. **Covariance evolves**: Propagate through the model; valid measurements reduce modeled uncertainty
3. **Kalman Gain**: Automatically balances prediction vs measurement
4. **Optimal under assumptions**: The model and noise assumptions matter
5. **Recursive**: Only needs current state
6. **Real-time**: Processes data as it arrives

## What's Next?

The next chapter will implement a complete 1D Kalman filter in Python. You'll see the actual code and run it yourself!

---

**Key Vocabulary**
- **Kalman Filter**: Optimal recursive state estimator
- **Time Update**: Prediction step
- **Measurement Update**: Correction step
- **Kalman Gain**: Optimal weighting factor
- **Recursive**: Uses only current state, not full history
- **Optimal**: Minimizes mean squared error
