# Weighted Averages

**Level: Middle School (Ages 13-15)**

## The Problem with Regular Averages

Imagine you're trying to find out the temperature outside. You have:
- Your guess: 80°F (you're just guessing)
- A cheap thermometer: 72°F (somewhat reliable)
- A weather station: 70°F (very reliable)

Regular average: (80 + 72 + 70) ÷ 3 = 74°F

But wait! Should your wild guess count as much as the weather station? Probably not!

## What is a Weighted Average?

A **weighted average** gives more importance to some numbers than others.

### Formula
```
Weighted Average = (value₁ × weight₁ + value₂ × weight₂ + ...) ÷ (sum of weights)
```

### Example: Temperature with Weights

Let's give weights based on reliability:
- Your guess: 80°F, weight = 1 (not very reliable)
- Cheap thermometer: 72°F, weight = 3 (somewhat reliable)
- Weather station: 70°F, weight = 6 (very reliable)

```
Weighted Average = (80×1 + 72×3 + 70×6) ÷ (1+3+6)
                 = (80 + 216 + 420) ÷ 10
                 = 716 ÷ 10
                 = 71.6°F
```

Notice: 71.6°F is much closer to the weather station (70°F) than to your guess (80°F)!

## Why Weights Matter

### Example 1: School Grades

Your teacher calculates your grade:
- Homework: 85% (weight = 20%)
- Quizzes: 90% (weight = 30%)
- Final exam: 78% (weight = 50%)

**Regular average**: (85 + 90 + 78) ÷ 3 = 84.3%

**Weighted average**:
```
(85×0.20 + 90×0.30 + 78×0.50)
= 17 + 27 + 39
= 83%
```

The final exam counts more, so it pulls your grade down!

### Example 2: Combining Measurements

You measure the length of a table:
- Measurement 1: 48.2 inches (uncertainty ± 0.5 inches)
- Measurement 2: 48.0 inches (uncertainty ± 0.1 inches)

Which measurement should you trust more? Measurement 2! It has lower uncertainty.

## Weights from Uncertainty

Here's the key insight: **Lower uncertainty = Higher weight**

### The Rule
```
Weight = 1 ÷ (Uncertainty²)
```

Or in math terms:
```
Weight = 1 ÷ Variance
```

### Example: Table Measurements

**Measurement 1**: 48.2 inches, uncertainty = 0.5
- Variance = 0.5² = 0.25
- Weight = 1 ÷ 0.25 = 4

**Measurement 2**: 48.0 inches, uncertainty = 0.1
- Variance = 0.1² = 0.01
- Weight = 1 ÷ 0.01 = 100

Measurement 2 gets 25 times more weight! (100 vs 4)

**Weighted average**:
```
(48.2×4 + 48.0×100) ÷ (4+100)
= (192.8 + 4800) ÷ 104
= 4992.8 ÷ 104
= 48.01 inches
```

The result is very close to the more accurate measurement!

## The Kalman Filter Way

The Kalman filter uses weighted averages to combine:
1. **Predictions** (what you expect)
2. **Measurements** (what you observe)

Each has its own uncertainty, so each gets its own weight!

### Example: Tracking a Car

**Prediction**: The car is at position 100 meters (uncertainty ± 5 meters)
**Measurement**: GPS says 95 meters (uncertainty ± 10 meters)

Which should you trust more? The prediction! It has lower uncertainty.

**Prediction weight**: 1 ÷ 5² = 1 ÷ 25 = 0.04
**Measurement weight**: 1 ÷ 10² = 1 ÷ 100 = 0.01

**Weighted estimate**:
```
(100×0.04 + 95×0.01) ÷ (0.04+0.01)
= (4 + 0.95) ÷ 0.05
= 4.95 ÷ 0.05
= 99 meters
```

The estimate is closer to the prediction (100m) than the measurement (95m) because the prediction is more certain!

## Kalman Gain: The Magic Number

The Kalman filter uses something called **Kalman Gain** (K) to decide how much to trust the measurement vs. the prediction.

### Formula
```
K = Prediction Uncertainty ÷ (Prediction Uncertainty + Measurement Uncertainty)
```

### What K Means

- **K = 0**: Don't trust the measurement at all (use prediction)
- **K = 1**: Don't trust the prediction at all (use measurement)
- **K = 0.5**: Trust both equally

### Example: Car Tracking

Prediction uncertainty: 5 meters
Measurement uncertainty: 10 meters

```
K = 5² ÷ (5² + 10²)
  = 25 ÷ (25 + 100)
  = 25 ÷ 125
  = 0.2
```

**Update formula**:
```
New Estimate = Prediction + K × (Measurement - Prediction)
             = 100 + 0.2 × (95 - 100)
             = 100 + 0.2 × (-5)
             = 100 - 1
             = 99 meters
```

Same answer as the weighted average!

## Understanding Kalman Gain

### Case 1: Very Certain Prediction

Prediction: 100 ± 1 meter
Measurement: 95 ± 10 meters

```
K = 1² ÷ (1² + 10²) = 1 ÷ 101 ≈ 0.01
```

K is very small! Trust the prediction more.

```
New Estimate = 100 + 0.01 × (95 - 100)
             = 100 + 0.01 × (-5)
             = 100 - 0.05
             = 99.95 meters
```

Very close to the prediction (100m)!

### Case 2: Very Certain Measurement

Prediction: 100 ± 10 meters
Measurement: 95 ± 1 meter

```
K = 10² ÷ (10² + 1²) = 100 ÷ 101 ≈ 0.99
```

K is almost 1! Trust the measurement more.

```
New Estimate = 100 + 0.99 × (95 - 100)
             = 100 + 0.99 × (-5)
             = 100 - 4.95
             = 95.05 meters
```

Very close to the measurement (95m)!

### Case 3: Equal Uncertainty

Prediction: 100 ± 5 meters
Measurement: 95 ± 5 meters

```
K = 5² ÷ (5² + 5²) = 25 ÷ 50 = 0.5
```

K is 0.5! Trust both equally.

```
New Estimate = 100 + 0.5 × (95 - 100)
             = 100 + 0.5 × (-5)
             = 100 - 2.5
             = 97.5 meters
```

Right in the middle!

## Updating Uncertainty

After combining prediction and measurement, the uncertainty also updates!

### Formula
```
New Uncertainty = (1 - K) × Prediction Uncertainty
```

### Example

Prediction uncertainty: 5 meters
K = 0.2

```
New Uncertainty = (1 - 0.2) × 5
                = 0.8 × 5
                = 4 meters
```

The uncertainty decreased! We're more certain after combining information.

### Key Insight

**Combining information always reduces uncertainty!**

Even if both sources are uncertain, combining them makes you more certain.

## Practice Problems

### Problem 1: Simple Weighted Average

You have three measurements:
- A: 50 (weight = 1)
- B: 60 (weight = 2)
- C: 55 (weight = 3)

Calculate the weighted average.

### Problem 2: From Uncertainty to Weights

Two measurements:
- Measurement 1: 100 ± 5
- Measurement 2: 110 ± 10

a) Calculate the weight for each measurement
b) Calculate the weighted average
c) Which measurement had more influence? Why?

### Problem 3: Kalman Gain

Prediction: 75 ± 3
Measurement: 80 ± 6

a) Calculate the Kalman Gain (K)
b) Calculate the new estimate
c) Calculate the new uncertainty
d) Did the uncertainty increase or decrease?

### Problem 4: Extreme Cases

For each case, predict whether K will be close to 0, 0.5, or 1:

a) Prediction: 50 ± 1, Measurement: 60 ± 20
b) Prediction: 50 ± 20, Measurement: 60 ± 1
c) Prediction: 50 ± 5, Measurement: 60 ± 5

### Problem 5: Real World

You're tracking a drone:
- Your physics model predicts: altitude = 100 feet ± 2 feet
- Barometer measures: altitude = 95 feet ± 8 feet

a) Calculate K
b) What's your best estimate of the altitude?
c) What's the new uncertainty?

## Real World Applications

### GPS + Inertial Navigation

Your phone combines:
- **GPS**: Accurate but slow updates (1 Hz)
- **Accelerometer**: Fast but drifts over time (100 Hz)

The Kalman filter uses weighted averages to combine both!

### Robot Localization

A robot uses:
- **Wheel encoders**: Measure how far wheels turned
- **Laser scanner**: Measures distance to walls

Both have uncertainty. Weighted average gives best position estimate!

### Weather Forecasting

Meteorologists combine:
- **Computer models**: Predict weather
- **Actual measurements**: From weather stations

Weighted average gives the forecast you see!

## Try It Yourself!

### Experiment 1: Weighted Coin Flip

1. Flip a coin 10 times, count heads
2. Flip a different coin 100 times, count heads
3. Which result do you trust more for estimating probability of heads?
4. Calculate a weighted average (weight by number of flips)

### Experiment 2: Measuring with Different Tools

1. Measure something with a ruler (5 times)
2. Measure the same thing with a tape measure (5 times)
3. Calculate average and standard deviation for each
4. Calculate weights based on standard deviations
5. Calculate weighted average of the two averages

### Experiment 3: Prediction vs Measurement

1. Drop a ball from a height
2. Predict how long it will take to hit the ground (use physics: t = √(2h/g))
3. Measure the actual time with a stopwatch
4. Estimate uncertainty for each
5. Calculate weighted average

## Key Concepts

1. **Weighted averages** give more importance to more reliable information
2. **Weight = 1 ÷ Variance** (lower uncertainty = higher weight)
3. **Kalman Gain** determines how much to trust measurement vs prediction
4. **Combining information reduces uncertainty**
5. **The Kalman filter is essentially a smart weighted average**

## What's Next?

Now that we understand weighted averages, the next chapter will teach us about **prediction** - how to estimate where things will be in the future based on where they are now!

---

**Key Vocabulary**
- **Weighted Average**: Average where some values count more than others
- **Weight**: How much importance to give a value
- **Kalman Gain**: The weight given to the measurement in Kalman filter
- **Variance**: Uncertainty squared
- **Update**: Combining prediction and measurement to get new estimate
