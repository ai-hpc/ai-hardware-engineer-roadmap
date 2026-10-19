# Lecture 10 — Action Chunking, ACT, and Temporal Ensembling

Design the interface between a policy that predicts a sequence and a controller that must choose one action now.

**Prerequisites:** Lecture 09; sequence models and basic variational inference. **Effort:** setup 30 minutes, lab 3 hours, analysis 90 minutes. **Hardware:** CPU for scheduling experiments; optional GPU for an ACT implementation. **Stack/roles:** model architecture, inference scheduling, and control; robot learning and inference optimization engineers. **Artifact:** a chunk scheduler with timing traces, validity masks, and controlled comparisons.

## Three horizons, three different decisions

A chunk policy predicts \(A_t=(a_t,\ldots,a_{t+H-1})\) from observations at time \(t\). The prediction horizon \(H\), execution horizon \(K\), and observation-history length \(O\) are separate variables. Predicting 32 actions does not require executing all 32 before observing again. A receding-horizon controller might execute four, acquire fresh observations, and replace the remaining plan.

Chunking amortizes expensive visual and language processing across several actions. It also represents correlations across a short movement. The cost is increasing commitment to a plan based on older information. Whether this is useful depends on camera rate, inference latency, contact sensitivity, actuator control rate, and how much uncertainty evolves while the chunk executes.

ACT combines a transformer action-chunk predictor with a conditional variational autoencoder and a temporal ensembling mechanism. The original ALOHA work supplies the experimental context and exact architectural choices; use it as an instance of this design space rather than assuming every chunk policy is ACT. [ACT / ALOHA paper, RSS 2023](https://roboticsproceedings.org/rss19/p016.html).

## What the latent variable learns

During training, an encoder observes the demonstrated action sequence and robot state and forms \(q_\phi(z\mid A,q)\). A decoder predicts the chunk from observation and latent \(z\). A simplified teaching objective is

$$
\mathcal L=\mathbb E_{z\sim q_\phi}[\|A-f_\theta(o,z)\|_1]
+\beta D_{KL}(q_\phi(z\mid A,q)\|\mathcal N(0,I)).
$$

For a diagonal Gaussian, the KL term is \(\frac12\sum_i(\mu_i^2+\sigma_i^2-\log\sigma_i^2-1)\). The reconstruction term rewards demonstrated motion; the KL term discourages an arbitrary latent code that works only when an entire target sequence is already known. At deployment the future demonstration is unavailable. The chosen prior or fixed-latent inference rule must match the implementation being reproduced.

This distinction is operationally useful. If reconstruction is excellent only when using the training encoder, your deployment path has not been validated. If the decoder ignores the observation because the latent encodes everything, the policy can replay familiar trajectories without responding to a moved object. Test both failures deliberately.

## Temporal ensembling without indexing ambiguity

Several chunks can contain a prediction for the same absolute control timestep \(t\). Let \(\hat a_t^{(s)}\) mean the action for timestep \(t\) predicted from a chunk generated at timestep \(s\). An ensemble computes

$$
\bar a_t=\frac{\sum_{s\in V_t}w_{t,s}\hat a_t^{(s)}}{\sum_{s\in V_t}w_{t,s}},
$$

where \(V_t\) contains only valid, already available predictions. Define weight indexing explicitly. With age \(t-s\), \(w_{t,s}=e^{-\lambda(t-s)}\) favors recent predictions. This is a teaching variant; reproducing ACT requires checking the original implementation's ordering and decay convention. A formula such as \(e^{-\lambda i}\) is incomplete until the meaning of \(i\) is stated.

Never use nonzero action values as the validity mask: zero is a valid motor command. Preserve generation time, observation timestamp, target control time, and chunk identity. Discard chunks from an old episode after reset. Averages are meaningful only when all predictions share the same coordinate frame, units, and action semantics. Averaging discrete gripper modes or rotations requires a separate rule.

## Check the actual ACT source convention

The source inspection on October 18 pins ACT to commit `742c753c0d4a5d87076c8f69e5628c79a8cc5488`; the [source manifest](labs/act-source-evidence.json) records three file hashes. No upstream ACT training or robot rollout was executed.

The pinned evaluation code stores chunks by generation timestep, selects overlapping predictions in that row order, then applies exponentially decreasing weights from index zero. Thus **older retained predictions receive greater weight** in this implementation. The recent-favoring age formula above is a deliberately different scheduler, not a reproduction of that default. The same source infers population from every action coordinate being nonzero: an otherwise valid vector containing one zero is excluded. Preserve that behavior when explaining the historical implementation, but use explicit Boolean validity in a new controller. [Pinned evaluator](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py#L249-L259)

The deployed decoder supplies a **zero latent**, whereas the training encoder samples a latent using the demonstrated future. Sampling a random prior at inference is an experiment, not the pinned default. This distinction also means the scalar multimodal generators in the next lab are not implementations of ACT. [Pinned latent path](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/detr/models/detr_vae.py#L98-L115)

The policy multiplies reconstruction errors by the nonpadding mask, then averages over the full padded tensor. It does not divide by the number of valid coordinates. With one valid scalar error of 1 and three padded entries, that reduction is 0.25; a valid-count-normalized reduction is 1. This changes the reconstruction scale relative to the KL term when padding fractions differ. State which reduction you reproduce before comparing losses or changing the KL coefficient. [Pinned loss](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/policy.py#L24-L34)

## Worked example: smoothness can create delay

Suppose three available predictions for one joint target are 0.20, 0.24, and 0.32 radians, ordered oldest to newest. A uniform average is approximately 0.253 radians. Recent-favoring weights \((1,2,4)\) produce approximately 0.280 radians. Both reduce the jump to the newest prediction. If the newest prediction reflects a newly visible obstacle, smoothing also delays the response.

For an illustrative 50 Hz actuator stream and \(K=5\), replanning occurs nominally every 100 ms. A measured inference latency of 70 ms could fit that interval, but a 140 ms tail event cannot. A scheduler needs a declared fallback when a replacement chunk misses its deadline. The policy invocation rate, action streaming rate, and physical servo rate should appear as separate columns in your report.

Start with the already executable [dataset and execution lab](Dataset-and-Execution-Lab.md) for delayed arrivals, zeros, and reset epochs, then the [trained policy in a timed loop](../../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-02-Timed-Robot-Loop.md) for the effect of queue policy on a learned controller. These validate scheduling mechanisms; neither implements ACT.

## Build a chunk scheduler before training a transformer

**Exercise / pseudocode.** Implement a synthetic chunk producer that predicts a bounded trajectory, adds seeded noise, and has configurable completion latency. The scheduler is real code written by the learner; the synthetic producer is not an ACT quality benchmark.

1. Represent each chunk as generation ID, episode ID, source timestamp, target timestep range, predictions, and a Boolean validity mask.
2. Implement execute-\(K\), latest-available prediction, uniform ensemble, and recent-weighted ensemble. Keep action clipping and rate limits identical.
3. Run a smooth target and a target that changes suddenly. Sweep \(H\in\{8,32\}\), \(K\in\{1,4,8\}\), and fixed versus jittered completion latency.
4. Measure target tracking error, action total variation, deadline misses, maximum observation age at execution, and scheduler CPU cost. State that total variation is a numerical smoothness proxy, not a mechanical wear measurement.
5. Inject a zero-valued chunk, a delayed old-episode chunk after reset, and a chunk with the wrong joint order. Require explicit handling for each.
6. Optionally train a small ACT-style model on the state trajectories from Lecture 09. Compare deployment inference with the training encoder disabled. Record the code revision if using upstream software.

Stop the scheduler sweep after a fixed 24 configurations and three seeds; select further settings using a written hypothesis. **Acceptance criteria:** target timesteps align correctly; stale episode data never executes; zeros remain valid; reported smoothness gains include their response-delay cost; no configuration is declared superior from offline loss alone.

## Failure clinic

- **Smooth open-loop motion, repeated contact failure:** shorten execution horizon and inspect whether new sensory information arrives before replanning.
- **First action jumps at every refresh:** inspect overlap alignment and delta-action reference state.
- **GPU gets slower as chunks grow:** measure the action decoder separately; fewer invocations do not imply smaller individual forward passes.
- **Reset moves the robot toward the previous task:** clear queued chunks and pending asynchronous results using an episode token.
- **Ensembling creates a physically invalid pose:** reconsider the representation and the space in which averaging occurs.

## Self-check and answer guidance

1. **Does chunking make inference cheaper per invocation?** Usually that is not its guarantee; it can reduce cost per executed action through amortization.
2. **Why include a latent regularizer?** The action encoder sees privileged future targets during training; regularization supports a usable deployment latent distribution.
3. **Can temporal ensembling average arbitrary action representations?** No. Joint targets, relative poses, and categorical gripper decisions need representation-aware handling.
4. **What is the essential optimization constraint?** Improve end-to-end task behavior while satisfying observation-age and action-deadline limits.

[Previous: behavioral cloning](Lecture-09.md) · [Course index](README.md) · [Next: diffusion policies](Lecture-11.md)
