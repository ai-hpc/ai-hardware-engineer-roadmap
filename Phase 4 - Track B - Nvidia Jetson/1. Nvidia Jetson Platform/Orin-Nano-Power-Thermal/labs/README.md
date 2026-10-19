# Energy and thermal modeling on a CPU

**Time:** 35–50 minutes. **Prerequisites:** multiplication, averages, and basic Python. **Hardware:** any computer with Python 3.10+. Every input here is synthetic; this lab does not measure a Jetson.

You will integrate a power trace, compare energy per useful result, and predict heating with a deliberately simple model. These are the calculations to get right before connecting a power meter.

## 1. Run the supplied experiment

Download [energy_thermal.py](energy_thermal.py) and [test_energy_thermal.py](test_energy_thermal.py) into one folder, or enter this `labs` directory in the repository:

```bash
python3 energy_thermal.py
python3 -m unittest discover -s . -p 'test_*.py' -v
```

The eleven tests cover irregular sampling, invalid traces, success counts, baseline subtraction, and the thermal solution. The [saved example](example-result.json) and [validation record](validation.json) identify the synthetic inputs and executed checks.

## 2. Calculate energy with the time intervals

The fictional meter reports complete-system DC input:

| Timestamp (s) | Power (W) |
| --- | --- |
| 0 | 6 |
| 1 | 10 |
| 3 | 10 |
| 4 | 6 |

One watt is one joule per second. Approximate each interval by a trapezoid: average its two endpoint powers, then multiply by its duration.

- 0–1 s: `(6 + 10) / 2 × 1 = 8 J`.
- 1–3 s: `(10 + 10) / 2 × 2 = 20 J`.
- 3–4 s: `(10 + 6) / 2 × 1 = 8 J`.

Total energy is **36 J**. Average power is `36 / 4 = 9 W`. Averaging the four readings produces 8 W and is incorrect because the time intervals differ. Integration assumes the trace between samples is reasonably approximated by straight lines; it cannot reconstruct an unobserved short power spike.

The function requires increasing timestamps and a caller-selected maximum gap. Here a gap over two seconds is invalid. The implementation permits only the floating-point rounding error associated with the timestamps and gap limit, so a nominal 0.3-second interval does not fail because binary floating point represents it slightly above 0.3. It rejects timestamps too coarse to resolve the allowed gap. A missing reading is unknown, not zero watts. A gap policy should follow the meter's sample interval and the workload's dynamics; two seconds is a teaching choice, not a suitable universal sampling rate.

**Try:** change a timestamp from 3 to 1. The repeated time must fail. Then delete the middle reading while keeping the two-second gap rule. The trace must fail instead of quietly bridging the missing interval.

## 3. Count useful completions over the same interval

Configuration A attempts 200 tasks in this interval; 180 results are valid and meet the application deadline. Its energy is `36 / 180 = 0.20 J` per useful result.

Configuration B draws a constant 7 W for four seconds, using 28 J, but produces only 100 valid results on time. It costs `28 / 100 = 0.28 J` per useful result. B uses less total energy while delivering less useful work; A is more efficient by this metric. Neither automatically meets an application requirement such as 99.9% successful deadlines.

This denominator is deliberately strict. Record attempted, completed, correct, and on-time counts separately. Define which combination is the intended outcome before comparing configurations. Measure energy and result counts over compatible boundaries: otherwise an asynchronous queue can move work or energy outside the accounting window. For a finite job, include drain time; for steady state, explain the window and boundary handling.

If A's matched idle baseline is 4 W, its estimated incremental energy is `36 − 4 × 4 = 20 J`. This is a separate diagnostic. The battery still supplied 36 J. A negative subtraction usually calls for investigating the baseline, measurement noise, or mismatched system states; the code leaves it visible.

**Check:** why can adding a module's total input power to one of its internal rail readings double-count energy? The total already includes that internal contribution. Only sum distinct, nonoverlapping boundaries.

## 4. Predict heating with a one-node model

Treat the device as one temperature with thermal resistance `R` in °C/W and heat capacity `C` in J/°C. This simplification gives:

\[
C\frac{dT}{dt}=P-\frac{T-T_a}{R},\qquad
T_\infty=T_a+PR,\qquad \tau=RC.
\]

For constant power and ambient temperature, the exact solution is:

\[
T(t)=T_\infty+(T_0-T_\infty)e^{-t/\tau}.
\]

Use our hypothetical values: ambient and starting temperature 35°C, power 15 W, resistance 2°C/W, and capacity 40 J/°C. Equilibrium is **65°C** and the time constant is **80 s**. At 80 seconds the model gives **53.9636°C**, about 63.2% of the way from 35°C to 65°C. At 240 seconds it reaches **63.5064°C**, about 95% of that rise. A short cold benchmark will not reveal the eventual temperature.

The implementation uses this exact constant-input solution, so subdividing a 240-second interval into 24 ten-second steps gives the same result to floating-point accuracy. If you implement explicit Euler integration instead, the scalar model requires `0 < dt / (RC) < 2` for numerical stability, and substantially smaller steps for accuracy. Numerical instability is not thermal runaway.

**Try:** change ambient to 45°C while keeping power and resistance fixed. Equilibrium becomes 75°C. Then lower resistance to 1°C/W: equilibrium becomes 60°C and the time constant becomes 40 s. These are predictions of the model, not a fan or heatsink specification.

## 5. Explain what is missing

A real enclosure has several thermal masses, contact resistances, airflow paths, temperature-dependent leakage, fans, power limits, and spatial hot spots. A measured module input watt is also not necessarily a watt dissipated in the one thermal node you modeled. Estimate parameters from a stated experiment and test the model on a different heating or cooling run. A fit to one trace does not validate all operating conditions.

**Completion:** submit the 36 J hand calculation, the two efficiency results, one rejected bad trace, and the two thermal predictions. Explain which power boundary, sampling uncertainty, and successful-result definition a hardware experiment would need. Continue with the [power and thermal guide](../Guide.md).
