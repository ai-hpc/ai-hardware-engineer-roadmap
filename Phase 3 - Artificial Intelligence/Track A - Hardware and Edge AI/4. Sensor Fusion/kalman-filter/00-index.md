# Kalman Filter Learning Path

A path from estimation intuition to tested numerical kernels and a timestamped camera-fusion experiment. Advanced lessons identify the additional work needed for full inertial or visual odometry systems.

## Learning Path

### Level 1: First intuition
- [01 - What is Estimation?](01-what-is-estimation.md) - Understanding guessing and improving guesses
- [02 - Noisy Measurements](02-noisy-measurements.md) - Why sensors aren't perfect
- [03 - Combining Information](03-combining-information.md) - Using multiple clues

### Level 2: Arithmetic and uncertainty
- [04 - Averages and Uncertainty](04-averages-and-uncertainty.md) - Mean and variance
- [05 - Weighted Averages](05-weighted-averages.md) - Trusting some measurements more
- [06 - Prediction](06-prediction.md) - Guessing the future from the past

### Level 3: Algebra and code
- [07 - The Kalman Filter Idea](07-kalman-filter-idea.md) - Putting it all together
- [08 - One Dimensional Kalman Filter](08-1d-kalman-filter.md) - First real implementation
- [09 - Tracking a Moving Object](09-tracking-moving-object.md) - Position and velocity

### Level 4: Undergraduate (College)
- [10 - Matrix Form](10-matrix-form.md) - Mathematical notation
- [11 - Multidimensional Kalman Filter](11-multidimensional-kf.md) - Multiple states
- [12 — Process noise and measurement noise](12-process-measurement-noise.md)
- [13 — A complete Kalman update, with numbers](13-complete-algorithm.md)

### Level 5: Graduate (Advanced)
- [14 — Extended Kalman filter: linearize the right function](14-extended-kalman-filter.md)
- [15 — Unscented Kalman filter: propagate a set of representative points](15-unscented-kalman-filter.md)
- [16 — Error-state filtering: small corrections around a nominal pose](16-error-state-kalman-filter.md)
- [17 — MSCKF: use a feature track without keeping every landmark](17-multi-state-constraint-filter.md)

### Level 6: Numerical methods and integration
- [18 — Kalman smoothing: use later information to revise an earlier estimate](18-kalman-smoothing.md)
- [19 — Outlier rejection: test a residual against its uncertainty](19-outlier-rejection.md)
- [20 — Numerical implementation: preserve the meaning of covariance](20-numerical-implementation.md)
- [21 — Integrate an estimator into a robot observation pipeline](21-real-world-integration.md)

## How to Use This Guide

1. **Start at your level** - Don't skip ahead too fast
2. **Do the exercises** - Each chapter has hands-on problems
3. **Code along** - Implementation examples in Python
4. **Build intuition first** - Understand "why" before "how"

## Prerequisites by Level

- **Level 1-2**: Basic arithmetic
- **Level 3**: Algebra, basic probability
- **Level 4**: Linear algebra, probability theory
- **Level 5**: Calculus, statistics
- **Level 6**: Advanced mathematics, programming

## Quick Reference

Once you've learned the basics, use these for quick lookup:
- [Kalman Filter Equations Cheat Sheet](cheat-sheet.md)
- [Common mistakes and diagnostic checks](common-mistakes.md)
- [Executable implementation templates](implementation-templates.md)

Let's begin the journey!

**Primary theory reference:** [Särkkä, Bayesian Filtering and Smoothing (author-provided 2013 edition)](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf), chapters 4–5 and 8–9. New examples in this course are original teaching fixtures. Source/API checks and executable tests: 2026-10-18.
