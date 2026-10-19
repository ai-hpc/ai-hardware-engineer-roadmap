# Lecture 27 — Gemini Robotics: reasoning, action, and access boundaries

Design and evaluate a hierarchical robotics system while separating an embodied reasoning API from an executable action policy.

**Stack:** application orchestration, model workload, distributed inference, local control. **Roles:** robotics software architect, evaluation engineer. **Prerequisites:** [Lecture 26](Lecture-26.md), task planning, action contracts, and timestamped evaluation. **Next:** [Open VLA ecosystem](Lecture-28.md). **Cutoff:** 2026-10-13.

## Follow the three product roles

The initial Gemini Robotics announcement on 2025-03-12 distinguished a VLA from an embodied reasoning model. That distinction remains essential: a model that describes where an object is or calls a tool is not automatically a model that emits the control signal your robot expects. [Initial announcement](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/).

On-Device followed on 2025-06-24, with local execution and an SDK offered through trusted testing. Version 1.5 on 2025-09-25 emphasized a reasoning/action pairing and made ER 1.5 available through the Gemini API while the VLA remained partner-access. [On-Device announcement](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/), [1.5 announcement](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/).

ER 1.6, announced 2026-04-14, added reported improvements in spatial and multi-view reasoning, including instrument reading. A good gauge-reading result is evidence for that perception/reasoning task, not evidence of a successful manipulation controller. [ER 1.6 announcement](https://deepmind.google/blog/gemini-robotics-er-1-6/).

The newest family verified in this review is **Gemini Robotics 2**, announced 2026-07-30: Robotics 2 for action, ER 2 for embodied reasoning, and On-Device 2 for local action inference. The announcement describes whole-body manipulation, dexterity, and multi-robot demonstrations. It offers the VLA and On-Device models to early-access partners. [Robotics 2 announcement](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/).

ER 2's card describes text output and API distribution, including AI Studio. On-Device 2's card describes numerical robot actions and trusted-tester availability. No public downloadable Robotics 2 action weights were verified in this review. These are distinct access categories; API availability for ER does not grant access to the VLA. [ER 2 card](https://deepmind.google/models/model-cards/gemini-robotics-er-2/), [On-Device 2 card](https://deepmind.google/models/model-cards/gemini-robotics-on-device-2/).

## Mental model: hierarchy introduces contracts

A useful explanatory decomposition is

\[
g_k\sim p_\phi(g\mid o_{0:t},\ell,m_k),\qquad
A_t\sim\pi_\theta(A\mid o_t,g_k,e).
\]

Here \(g_k\) is a subgoal selected on a slower planning clock; \(m_k\) is task memory; \(A_t\) is a motor-action chunk selected on a faster clock. This equation is a teaching abstraction, not a published reconstruction of Google's proprietary implementation.

Every subgoal needs a completion predicate, timeout, allowed resources, and a recovery transition. “Put the mug away” is ambiguous unless the destination and success criteria are grounded. A well-typed instruction could reference an object ID and target region, but the system must know whether those IDs remain valid after the scene changes.

Avoid letting free-form prose directly select arbitrary robot functions. Define a small tool surface such as `inspect_region`, `execute_named_skill`, `check_completion`, and `request_operator`. Validate tool names and arguments; map accepted calls onto independently tested controllers. A text model may propose a useful plan while misunderstanding the physical preconditions of one step.

A semantic planner also needs an uncertainty channel. “Object not visible,” “object inaccessible,” and “object absent” should be different states. The first can motivate another camera view; the second can require a different approach; the third may require a revised task. Collapsing all three into a repeated grasp command creates endless retries.

## Worked example: why long-horizon success is difficult

Consider an **illustrative** tidying task with eight required subtasks. If each succeeds independently with probability 0.95 and no recovery exists,

\[
P(\text{complete})=0.95^8\approx0.663.
\]

The independence assumption is often false: a misplaced object can make later steps harder. This estimate is therefore a teaching calculation, not a predicted benchmark result.

Suppose a valid retry can recover 60% of failures with no side effects. The toy effective per-step success is \(0.95+0.05\times0.60=0.98\), yielding \(0.98^8\approx0.851\). But that improvement requires a detector that recognizes failure. A system that falsely marks failure as success never invokes recovery.

Measure the detector separately with a confusion matrix. False completion may be more damaging than delayed completion because it corrupts the planner's state. Report precision of “done,” recall of actual completion, detection delay, and calibration by task. Do not multiply vendor scores across different datasets to estimate full-system reliability.

The control path has its own budget. If planning takes 900 ms while a grasp controller updates every 20 ms, the controller must not block waiting for another paragraph of reasoning. Design a valid subgoal lifetime and cancellation path. When the observed state invalidates the plan, cancel future actions using the controller's defined transition; an eventual planner correction cannot undo already executed motion.

## Build it: a planner/action contract without privileged access

**Effort:** 45 minutes setup, 2–3 hours execution, 1 hour analysis. **Hardware:** a laptop for a state-machine fixture; a supported simulation machine for the optional closed-loop extension. API use is optional and requires the learner's own available access. No paid requests or model executions were performed while authoring this lesson.

Build a tabletop sorting fixture with three objects, two bins, one occlusion, and one intentionally failed grasp. Use a deterministic planner or recorded planner proposals first. An accessible ER model can later replace that component without changing the controller contract. The fixture teaches orchestration and evaluation; it is not a reproduction of Robotics 2.

1. Define scene state independently of the planner: object poses, visibility, bin membership, grasp attachment, and time. Define success before implementing prompts.
2. Define a structured subgoal record containing `episode_id`, `subgoal_id`, `object_id`, `destination_id`, `observation_time`, `expires_at`, and `completion_test`. Reject unknown IDs and impossible transitions.
3. Create an executable skill with observable preconditions and postconditions. The initial fixture may update a synthetic state; label it as such. The optional simulator version must derive completion from simulator state or a separately evaluated perception function.
4. Evaluate the planner against recorded observations. Score valid tool selection, argument validity, task completion, unnecessary retries, and hallucinated objects. Record input/output data and model version when an API is used.
5. Inject stale observations, an occluded object, a failed grasp, and an episode reset. Confirm that stale subgoals do not survive reset and that a failed skill does not mark its postcondition true.
6. Stop after 24 predeclared fixture scenarios. Use those scenarios to improve your contract, then evaluate a separate held-out set; avoid repeatedly tuning prompts on the test set.

**Exercise / pseudocode:**

```python
proposal = planner.propose(scene_snapshot, task_memory)
if not schema.valid(proposal):
    outcome = "invalid_proposal"
elif proposal.episode_id != current_episode:
    outcome = "stale_episode"
elif not proposal.observation_time <= clock.now() < proposal.expires_at:
    outcome = "future_or_expired_proposal"
elif not preconditions_hold(proposal, current_scene):
    outcome = "reobserve_or_replan"
else:
    # The executor checks epoch, deadline and preconditions again at dispatch.
    # A queue wait or scene reset after this check must invalidate the proposal.
    result = skills.execute_with_dispatch_validation(proposal)
    outcome = independent_completion_check(result, current_scene)
task_memory.record(proposal, outcome)
```

All timestamp comparisons above assume a common monotonic clock mapping. Define exact-boundary behavior (`now == expires_at` is expired here) and invalidate an in-flight skill at reset through the controller's supported cancellation path. The [existing offline dispatcher](Dataset-and-Execution-Lab.md) provides runnable reset, late-result, and duplicate-result cases; it is not an implementation of this planner.

For a real API experiment, save the exact public model identifier and dated response metadata. Keep request latency separate from local controller latency. Avoid claiming deterministic reproducibility when a provider can update a service behind a moving identifier.

## Guided variation: collaboration needs resource ownership

Give two simulated robots one shared tray. Add a resource lease with an owner, expiration, and release condition. Run conflicting proposals: both robots attempt to place into the same region simultaneously. Your scheduler should serialize or reject the conflict before execution. This is a coordination exercise; it does not reproduce Google's multi-robot model.

For independent transfer, redesign the fixture as a mobile manipulator that must navigate before grasping. Keep navigation success, object localization, grasp success, and final placement separate. Explain which new observations and completion tests are required.

## Failure modes, answers, and completion

The common misconception is that better verbal reasoning guarantees better physical control. Other failures include treating tool-call validity as task success, accepting self-reported completion without evidence, hiding network delay inside aggregate inference time, and assuming an on-device announcement supplies public weights.

1. **What can you reproduce without VLA access?** The planner/tool interface, validation, fixture scenarios, and perhaps public ER tasks. You cannot reproduce the inaccessible action model's reported robot performance.
2. **Why separate completion detection from policy output?** The policy can be confidently wrong; an independently defined measurement prevents its narrative from becoming its own score.
3. **Does an explanation reveal the actual cause of an action?** Not necessarily. Test interventions on observations and plans instead of assuming the explanation is faithful.
4. **What should a timed-out subgoal do?** Follow a defined controller transition, invalidate stale commands, record the cause, and reobserve before replanning.

**Ship:** a typed planner/action contract, executable scenario matrix, completion-detector confusion matrix, latency decomposition, and a statement of which proprietary capabilities you did and did not test. A reviewer should be able to replace your planner while preserving the same evaluation.
