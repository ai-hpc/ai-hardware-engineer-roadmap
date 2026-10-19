# Common estimation mistakes and how to diagnose them

| Symptom | Likely mistake | Discriminating check |
|---|---|---|
| Confident drift | Bias omitted or correlated evidence counted twice | Compare known bias injection with zero-bias run; inspect residual autocorrelation |
| Good pixel fit, wrong distance | Wrong target size or transform direction | Recover a known metric fixture; invert pose explicitly |
| Every delayed image rejected | Arrival time substituted for capture time | Replay the same values with correct capture timestamps |
| Quality changes when input frequency changes | Wrong Q discretization or fixed dt | Compare continuous-noise predictions over subdivided intervals |
| Position right, velocity nonsense | State ordering/units wrong | Track a noiseless constant-velocity sequence with hand-computed prediction |
| Covariance becomes negative | Numerical cancellation, bad matrices, or invalid linearization | Joseph update; inspect eigenvalues and units; compare an analytic scalar case |
| Impossible 358° turn | Unwrapped angle residual | Test 179° versus −179° |
| Online performance looks too good | Future samples/smoothed labels used as current observations | Audit arrival-time availability at every output |
| Filter trusts a repeated sample more each time | Duplicate observation not rejected | Preserve source/sequence identity; test replay idempotence in deployment |

**Worked check:** two identical copies of one measurement with variance 1 do not provide two independent pieces of evidence. Treating them as independent yields an apparent measurement variance 0.5 and unjustified confidence.

Use [lessons 12–21](00-index.md) for derivations and the [CPU lab](../Camera-Time-Fusion-Lab/Guide.md) for executable counterexamples. Fix one cause at a time; a lower fitted-sequence error is not proof of consistency on new data.
