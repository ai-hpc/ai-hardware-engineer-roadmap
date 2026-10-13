# Lecture 02 — Historical foundations: control, imitation, and deep visuomotor learning

Reconstruct the problems inherited by modern VLAs and demonstrate why successful supervised prediction can fail in a feedback loop.

**Prerequisites:** [Lecture 01](Lecture-01.md), linear regression, gradients. **Effort:** 30 minutes setup, two hours lab, one hour analysis. **Hardware:** CPU laptop; optional plotting package. **Stack:** control algorithms, training systems, embedded execution. **Roles:** robot learning engineer, control engineer, research engineer.

[Course](README.md) · [Previous](Lecture-01.md) · [Next](Lecture-03.md)

## Read history as changing assumptions

The history of VLA systems is not a sequence in which a larger model makes earlier engineering obsolete. Each generation changes where knowledge comes from, what observations a policy consumes, and which part of control is learned. A calibrated proportional controller remains useful beneath a foundation model. A large dataset remains vulnerable to misaligned timestamps.

Before internet-scale pretraining, robot builders commonly separated state estimation, planning, and control. A model of dynamics supported trajectory optimization; a feedback controller corrected deviations. This decomposition exposes interfaces and permits reasoning about stability, but perception errors and incomplete models create difficult boundaries between modules.

Brooks's layered robot architecture emphasized directly connected sensing and behavior. His 1985 technical report, followed by the 1986 journal paper, is a historical reference for reactive layered control rather than a VLA ancestor with language inputs. [Author-hosted report](https://people.csail.mit.edu/brooks/papers/AIM-864.pdf)

ALVINN demonstrated neural mapping from sensory inputs to vehicle steering decades before modern VLAs. The NeurIPS proceedings place the work in the 1988 conference volume; CMU's publication catalog lists the proceedings publication in 1989. Record both conventions when building a timeline instead of inventing an exact single release date. [Proceedings](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html), [CMU catalog](https://www.ri.cmu.edu/pubs/page/2/?authors-first-last-and-separated=Dean+Pomerleau&publication-year=&wpv-publication-taxonomies=0&wpv_filter_submit=Search&wpv_post_search=&wpv_view_count=4514-TCPID4515)

The important inherited idea is learning a sensor-to-action mapping. The missing pieces for a modern VLA include broad language conditioning, extensive embodiment diversity, and large pretrained representations. Calling every historical visuomotor policy a VLA would hide those differences.

## Imitation learning changes the source of supervision

Instead of programming every behavior, record an expert's observations and actions. For a deterministic policy \(f_\theta\), ordinary behavior cloning minimizes

$$
\mathcal L(\theta)=\frac1N\sum_{i=1}^{N}
\|f_\theta(o_i)-a_i^*\|_W^2,
$$

where \(W\) controls relative action scales. This looks like standard supervised learning, but the input distribution during deployment depends on earlier predictions. A slightly wrong action can produce an unfamiliar image, causing a larger mistake at the next step.

Imagine demonstrations of a car exactly centered in a lane. A learner can fit steering commands near zero perfectly and still have no information about recovery from a 20 cm lateral offset. Randomly splitting adjacent frames into train and test sets will not reveal the missing recovery states. An episode-level split is better, but a deliberate perturbed-start evaluation is still needed.

DAgger addresses the mismatch by collecting states visited by the learner and querying an expert for appropriate actions, then aggregating those labels with existing data. The 2011 paper formalizes the connection to no-regret online learning. Its guarantees depend on assumptions about the online learner and expert access; the method does not make physical exploration costless. [Original paper](https://proceedings.mlr.press/v15/ross11a.html)

## Worked example: the horizon amplifies small error

For a deliberately simplified calculation, assume each step has an independent 1% chance of an unrecoverable mistake. The probability of completing 100 steps without such a mistake is

$$
P(\text{completion})=(1-0.01)^{100}\approx0.366.
$$

This 36.6% result is not DAgger's theorem and not a robotics benchmark. It is an original toy model that explains why 99% per-step accuracy can be inadequate. Correlated errors, recovery, and task redundancy change the number, often substantially.

A different example isolates feedback. Let lateral error evolve as \(x_{t+1}=x_t+u_t+w_t\). Expert control is \(u_t=-0.5x_t\), giving \(x_{t+1}=0.5x_t+w_t\). A learner trained only at \(x=0\) may choose \(u=0\), yielding a random walk under disturbances. Both controllers fit the centered demonstrations, but their closed-loop behavior is completely different.

With no disturbance and initial error \(x_0=0.4\), the expert produces \(0.4,0.2,0.1,0.05,0.025\). The zero-action learner remains at 0.4. The missing training information is the action derivative with respect to error, not merely the mean action.

## Deep visuomotor policies before large language models

Levine and colleagues' 2016 work learned image-to-torque policies using guided policy search. A trajectory-centric method supplies supervision for a neural policy, allowing perception and control to be optimized together. The paper is a concrete milestone for learned visuomotor manipulation, with substantially different model scale and training assumptions from later language-conditioned generalists. [JMLR article](https://www.jmlr.org/papers/v17/15-522.html)

The reusable engineering lesson is to distinguish the information available to a teacher during training from that available to a policy at deployment. A teacher may use privileged state from motion capture or simulation. If the deployed observation omits a crucial variable, imitation can become irreducibly ambiguous. Record this asymmetry in the dataset specification.

End-to-end training also does not mean every physical subsystem is differentiable or learned. Motor drivers, sensing, timing, robot mechanics, and constraints still exist. When interpreting a paper, draw the learned boundary and label each frozen or external component.

## What pre-2019 history contributes to a 2026 engineer

Four durable questions connect these milestones. Does the policy see enough state? Does its training distribution cover recovery? Is supervision physically achievable by the deployed embodiment? Does the execution loop preserve the timing under which the behavior was learned?

The hardware consequences follow directly. A feedback loop may prefer a small predictable model over a larger model with long latency spikes. Teacher data collection can consume more engineering effort than training. Image processing can dominate a lightweight controller's runtime. A model trained on clean simulator state can fail when an embedded estimator introduces delay or bias.

Keep task success, control error, and compute cost separate in experiments. A lower action MSE is one diagnostic. It is not evidence of better long-horizon behavior unless evaluated in a loop.

## Lab: reproduce distribution shift without a robot

**Exercise / pseudocode; the learner implements the experiment.** Use the scalar system above for 50-step episodes. Generate 200 expert examples with states in \([-0.02,0.02]\). Compare a constant-action baseline with a linear controller fit to those examples. Test on 20 initial states spanning \([-0.5,0.5]\), first without noise, then with fixed-seed Gaussian disturbances of standard deviation 0.01.

Log every \(x_t,u_t\), clipping event, and episode identifier. Cap actions at magnitude 0.1 and declare failure when \(|x_t|>1\). Measure training MSE, mean absolute state error, failures, and recovery time to \(|x|<0.03\). Explain why the cap changes the numerical trajectory from the worked example.

For the guided variation, add a nonlinear expert term or biased observations so the extrapolation problem is visible even for a fitted model. Collect 100 learner-visited states, label them with the expert, retrain, and compare using identical starting states and disturbances. Call this a DAgger-inspired educational experiment, not a reproduction of the original paper.

For independent transfer, replace current state input with a one-step-delayed measurement and add velocity to the observation. Decide whether more demonstrations, memory, or a different controller best addresses the failure. Stop after the baseline, one aggregation round, and the delay experiment.

Submit the data-generation seed, fitted parameters, rollout CSV, two plots, and a short timeline that separates publication dates from later code availability. This CPU lab cannot reproduce real visuomotor representation learning or validate a physical controller.

## Self-check and exit criteria

1. **What did language pretraining add to early visuomotor learning?** A source of semantic supervision and instruction representations; it did not introduce the general idea of learning actions from images.
2. **Why is a random frame split weak evidence?** Nearly identical neighboring frames can leak across splits, while off-trajectory states remain absent.
3. **What distinguishes DAgger from simply collecting more expert episodes?** The query states come from the learner's induced distribution; explain the expert-labeling and execution assumptions.

Exit with a demonstrated case where offline error and closed-loop quality disagree, and a proposed data or interface change that addresses the actual cause.

[Previous](Lecture-01.md) · [Next: language grounding](Lecture-03.md)
