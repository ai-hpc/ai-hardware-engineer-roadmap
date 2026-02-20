# Averages and Uncertainty

**Level: Middle School (Ages 13-15)**

## Introduction

You've learned about estimation and combining information. Now let's add some math to make our estimates even better! Don't worry - we'll start simple.

## What is an Average?

An **average** (also called the **mean**) is a way to find the "middle" of a group of numbers.

### Formula
```
Average = (Sum of all numbers) ÷ (How many numbers)
```

### Example 1: Test Scores
Your test scores: 85, 90, 88, 92, 85

Average = (85 + 90 + 88 + 92 + 85) ÷ 5 = 440 ÷ 5 = 88

Your average score is 88.

### Example 2: Measuring Height
You measure your height 5 times:
- 65.2 inches
- 65.5 inches
- 65.1 inches
- 65.4 inches
- 65.3 inches

Average = (65.2 + 65.5 + 65.1 + 65.4 + 65.3) ÷ 5 = 327.5 ÷ 5 = 65.5 inches

The average is probably closer to your true height than any single measurement!

## Why Averages Matter for Estimation

When you have noisy measurements, the average is usually a better estimate than any single measurement.

### The Dart Board Analogy

Imagine throwing 10 darts at a target:
- Some land left of center
- Some land right of center
- Some land high
- Some land low

The **average position** of all darts is probably very close to where you were aiming!

## What is Uncertainty?

**Uncertainty** is how much you're not sure about something. It's like saying "I think it's 50, but I could be wrong by about 5."

### Example: Guessing Jellybeans

You guess there are 100 jellybeans in a jar.

**Low uncertainty**: "I'm pretty sure it's between 95 and 105"
**High uncertainty**: "It could be anywhere from 50 to 150"

## Measuring Uncertainty: Variance

**Variance** measures how spread out numbers are from the average.

### Example: Two Students

**Student A's test scores**: 88, 89, 90, 89, 89
- Average: 89
- Scores are very close together (low variance)
- Consistent performance!

**Student B's test scores**: 70, 95, 85, 100, 95
- Average: 89 (same average!)
- Scores are spread out (high variance)
- Inconsistent performance!

### Calculating Variance

**Step 1**: Find the average
**Step 2**: Find how far each number is from the average
**Step 3**: Square those differences
**Step 4**: Average the squared differences

#### Example: Student A
Scores: 88, 89, 90, 89, 89
Average: 89

Differences from average:
- 88 - 89 = -1
- 89 - 89 = 0
- 90 - 89 = 1
- 89 - 89 = 0
- 89 - 89 = 0

Squared differences:
- (-1)² = 1
- (0)² = 0
- (1)² = 1
- (0)² = 0
- (0)² = 0

Variance = (1 + 0 + 1 + 0 + 0) ÷ 5 = 0.4

#### Example: Student B
Scores: 70, 95, 85, 100, 95
Average: 89

Differences: -19, 6, -4, 11, 6
Squared: 361, 36, 16, 121, 36

Variance = (361 + 36 + 16 + 121 + 36) ÷ 5 = 114

Student B has much higher variance (114 vs 0.4)!

## Standard Deviation

**Standard deviation** is the square root of variance. It's easier to understand because it's in the same units as your measurements.

```
Standard Deviation = √Variance
```

**Student A**: √0.4 = 0.63 points
**Student B**: √114 = 10.68 points

Student B's scores vary by about 11 points, while Student A's vary by less than 1 point!

## Uncertainty in Measurements

### Example: Temperature Sensor

You measure temperature 10 times:
72, 73, 71, 72, 74, 72, 71, 73, 72, 71

**Average**: 72.1°F
**Variance**: 1.09
**Standard Deviation**: 1.04°F

**What this means**: The temperature is probably 72.1°F, give or take about 1°F.

We can write this as: **72.1 ± 1.0°F**

## Combining Averages

When you have multiple measurements, you can combine them!

### Example: Two Thermometers

**Thermometer A** (less accurate):
- Measurements: 70, 72, 71, 73, 70
- Average: 71.2°F
- Standard deviation: 1.3°F

**Thermometer B** (more accurate):
- Measurements: 72.0, 72.1, 71.9, 72.0, 72.1
- Standard deviation: 0.08°F
- Average: 72.02°F

Which should you trust more? Thermometer B! It has much lower uncertainty.

## The 68-95-99.7 Rule

For normally distributed data (bell curve):
- **68%** of measurements fall within 1 standard deviation
- **95%** fall within 2 standard deviations
- **99.7%** fall within 3 standard deviations

### Example: Height Measurements

Your average height: 65.5 inches
Standard deviation: 0.2 inches

- 68% chance your true height is between 65.3 and 65.7 inches
- 95% chance it's between 65.1 and 65.9 inches
- 99.7% chance it's between 64.9 and 66.1 inches

## Uncertainty Decreases with More Measurements

The more measurements you take, the more certain you become!

### Formula
```
Uncertainty of average = Standard Deviation ÷ √(Number of measurements)
```

### Example: Measuring a Table

One measurement: 48 ± 0.5 inches (uncertainty = 0.5)
Four measurements: 48 ± 0.25 inches (uncertainty = 0.5 ÷ √4 = 0.25)
Nine measurements: 48 ± 0.17 inches (uncertainty = 0.5 ÷ √9 = 0.17)

More measurements = more certainty!

## Kalman Filter Connection

The Kalman filter keeps track of:
1. **The estimate** (like an average)
2. **The uncertainty** (like variance)

It updates both as it gets new information!

## Practice Problems

### Problem 1: Calculate Average and Variance

You measure the length of a pencil 5 times (in cm):
19.2, 19.5, 19.3, 19.4, 19.1

a) What's the average length?
b) What's the variance?
c) What's the standard deviation?
d) Write the length as: average ± standard deviation

### Problem 2: Comparing Uncertainty

**Sensor A**: Average = 50, Standard Deviation = 5
**Sensor B**: Average = 50, Standard Deviation = 2

Which sensor is more reliable? Why?

### Problem 3: More Measurements

You measure something once and get 100 ± 10.
If you take 4 measurements and average them, what's the new uncertainty?

### Problem 4: Temperature Readings

Morning temperatures for a week: 65, 67, 66, 68, 65, 66, 67

a) What's the average temperature?
b) What's the standard deviation?
c) What temperature range contains 95% of the measurements?

## Real World Applications

### GPS Accuracy

Your phone's GPS might say:
- "You are at this location ± 5 meters"

The ±5 meters is the uncertainty! Sometimes it's ±3 meters (more certain), sometimes ±20 meters (less certain).

### Scientific Measurements

Scientists always report measurements with uncertainty:
- "The speed of light is 299,792,458 ± 1 m/s"
- "The patient's temperature is 98.6 ± 0.2°F"

### Weather Forecasts

"High of 75°F" really means "probably between 73°F and 77°F"

The forecast has uncertainty!

## Try It Yourself!

### Experiment 1: Reaction Time

1. Use an online reaction time test
2. Take the test 20 times
3. Calculate the average
4. Calculate the standard deviation
5. What's your reaction time ± uncertainty?

### Experiment 2: Measuring Objects

1. Measure the length of your desk 10 times
2. Calculate average and standard deviation
3. Measure it 10 more times
4. Calculate the new average and standard deviation
5. Did the uncertainty decrease?

### Experiment 3: Comparing Sensors

1. Use two different rulers to measure the same object
2. Take 5 measurements with each
3. Calculate average and standard deviation for each
4. Which ruler is more consistent (lower standard deviation)?

## Key Concepts

1. **Average** = Best single estimate from multiple measurements
2. **Variance** = How spread out the measurements are
3. **Standard Deviation** = Square root of variance (easier to interpret)
4. **Uncertainty** = How much you might be wrong
5. **More measurements** = Less uncertainty

## What's Next?

Now that we understand averages and uncertainty, the next chapter will teach us about **weighted averages** - how to combine measurements when some are more reliable than others!

---

**Key Vocabulary**
- **Mean/Average**: Sum of values divided by count
- **Variance**: Average of squared differences from mean
- **Standard Deviation**: Square root of variance
- **Uncertainty**: How much error might be in your estimate
- **Normal Distribution**: Bell curve shape of many natural measurements
