# Lecture 03 — Language grounding: CLIP, CLIPort, and SayCan

Separate recognizing a requested object, locating a feasible interaction, and selecting an executable skill.

**Prerequisites:** [Lecture 02](Lecture-02.md), cosine similarity, softmax, basic image coordinates. **Effort:** 30 minutes setup, two hours lab, one hour analysis. **Hardware:** CPU for the required synthetic lab; an optional image encoder requires its own memory assessment. **Stack:** representation learning, policy architecture, runtime integration. **Roles:** multimodal engineer, robot learning engineer, application engineer.

[Course](README.md) · [Previous](Lecture-02.md) · [Next](Lecture-04.md)

## Three meanings of grounding

Suppose the instruction is “put the ripe fruit into the bowl.” Semantic grounding identifies which visible object matches “ripe fruit.” Spatial grounding connects that object to a grasp pose and the bowl to a placement region. Affordance grounding asks whether this robot can execute that grasp now. A system can succeed at any one of these and fail at the others.

The misconception is that image-text similarity measures action feasibility. A strong match between an image and “red apple” says little about whether the apple is occluded, attached to a branch, too heavy for the gripper, or outside the reachable workspace.

CLIP's 2021 paper trained visual and language representations with paired image-text supervision, demonstrating transfer through natural-language descriptions. It did not train a robot controller or provide metric grasp geometry. [CLIP paper](https://arxiv.org/abs/2103.00020)

CLIPort, also introduced in 2021, combined semantic features with a spatial manipulation pathway for language-conditioned pick-and-place behavior. It illustrates why recognizing concepts and preserving spatial precision can benefit from different architectural biases. Its evidence concerns the reported tabletop settings, not arbitrary robot control. [CLIPort paper](https://arxiv.org/abs/2109.12098)

SayCan's 2022 work combined language-based skill selection with skill feasibility estimates. Its high-level output selects among available behaviors; a separate low-level skill executes the choice. This is a useful historical boundary between language-conditioned planning and a policy directly predicting a continuous action. [SayCan paper](https://arxiv.org/abs/2204.01691)

## What a representation learns to ignore

For normalized image embedding \(v_i\) and text embedding \(t_j\), a similarity score is \(s_{ij}=v_i^\top t_j/\tau\), with temperature \(\tau>0\). A contrastive objective encourages matched examples to score above unmatched examples. This supplies a semantic organization of features, not a guarantee that every spatial detail survives.

A global feature can remain nearly constant as an apple moves a few centimeters across a table. That invariance helps recognition while harming precise control if the global feature is the only visual input. Patch features, depth, crop coordinates, keypoints, or a spatial prediction head can retain location information.

Ask which transformations should leave an output unchanged. Object identity should usually survive a camera translation. An end-effector displacement in camera coordinates generally should not. A design that applies the same invariance to both tasks creates a representational conflict.

Likewise, a language encoder may map “place left of the cup” and “place right of the cup” to nearby embeddings because most words overlap. The action head must still treat that contrast as decisive. Average semantic retrieval accuracy can hide a catastrophic failure on relational words.

## From pixels to a manipulation primitive

A pick-and-place interface might represent an action as two image locations and orientations: \(a=(u_p,v_p,\theta_p,u_q,v_q,\theta_q)\). A calibration process maps those pixels onto a workspace surface, and a motion primitive carries out the approach, grasp, lift, transfer, and release.

This representation is compact because the primitive contains substantial prior knowledge. It assumes suitable geometry and a valid route between the two poses. It does not directly express arbitrary compliant insertion, manipulation while moving a base, or a changing contact force. Architecture comparisons must include the capability of the executor underneath the learned head.

As an original scale calculation, a 512-pixel image covering 0.8 m of a planar workspace corresponds to 1.5625 mm per pixel along that axis under uniform orthographic mapping. A four-pixel localization error is 6.25 mm. Perspective, depth, lens distortion, and calibration uncertainty invalidate the uniform approximation unless handled explicitly.

Increasing resolution can improve the representable spatial grid while increasing image processing and encoder cost. It will not repair an incorrect camera-to-base transform. Measure localization error and transform error independently.

## Worked example: semantic preference times feasibility

Consider three candidate skills for “bring me something to drink.” Use synthetic scores:

| Skill | Language score | Feasibility estimate | Product |
|---|---:|---:|---:|
| Pick the sealed bottle | 0.65 | 0.20 | 0.130 |
| Pick the mug | 0.25 | 0.90 | 0.225 |
| Pick the sponge | 0.10 | 0.95 | 0.095 |

The bottle is semantically preferred but the mug wins the product. These are invented teaching values, not SayCan results. They demonstrate a selection rule, not a calibrated probability of finishing a multistep task.

For many small factors, calculate scores in log space:

$$
\operatorname{score}(k)=\log p_{\mathrm{lang}}(k\mid l,h)
+\lambda\log p_{\mathrm{feas}}(k\mid o).
$$

Here \(\lambda\) is a design choice in our toy system. With \(\lambda=0\), the language preference dominates; with larger \(\lambda\), feasibility matters more. Handle exact zero explicitly: an impossible skill should be excluded rather than accidentally revived through an arbitrary numerical floor.

Scores from unrelated models are not automatically calibrated on a compatible scale. A value function's output may estimate expected return rather than binary success. Before interpreting it as feasibility, inspect its training target, reward discount, termination definition, and validation data.

## Lab: build a grounded skill selector

**Exercise / pseudocode; no external model or robot is required.** Create a small dataset of 12 scenes. Each scene includes named objects, a language instruction, reachability flags, graspability estimates, and whether the gripper is already occupied. Define five skills: pick object, place object, open gripper, move to inspection pose, and finish.

Implement the numerical selector above using supplied semantic scores. Preserve a trace containing every candidate, each component score, rejection reason, and chosen skill. Add preconditions: a pick requires an empty gripper; a place requires a held object; finish requires the measured goal condition.

The positive fixture should select the mug from the worked example. Negative fixtures must include an unreachable bottle, an occupied gripper, a zero-feasibility candidate, and a high-scoring finish action before task completion. Specify the expected rejection for each before running the selector.

For guided variation, reduce all feasibility estimates by a common multiplicative factor. With fixed nonzero values and a product ranking, the ordering should not change, although a separate absolute acceptance threshold might. Explain why confusing ranking with calibration produces misleading claims.

For independent transfer, add “the object left of the blue bowl.” Construct two scenes that swap object positions while preserving object names. A correct selector must change its target. Then remove spatial information and document why the task becomes underdetermined.

Record selection accuracy against your declared fixtures, impossible-skill selections, false completion count, and decision latency. Stop after all 12 scenes and the two variations. Deliver JSON inputs, code, raw traces, and a failure taxonomy separating semantic, spatial, affordance, and execution errors.

This lab tests interface logic and scoring. Synthetic semantic values cannot establish language understanding; symbolic reachability flags cannot validate a real collision checker; selecting the right primitive cannot establish grasp success.

## Failure analysis and self-check

A common failure is reporting planning success when only the skill sequence is plausible. Define execution success using environmental evidence: the target object is in the destination, the gripper released it, and the goal remains true after settling. Another failure is treating the list of available skills as open-ended. A planner cannot create a missing low-level behavior by naming it.

1. **Why retain spatial features alongside semantic embeddings?** Recognition can benefit from invariance that destroys information needed for precise actuation.
2. **Can feasibility scoring recover a missing skill?** No. It can select or reject available skills; an absent capability needs a new executor or training procedure.
3. **What evidence would show relational language grounding?** Counterfactual scenes and instructions where changing “left” to “right” changes the selected physical target while irrelevant changes do not.

Exit when your trace exposes why every skill was chosen and your tests distinguish correct words from achievable actions.

[Previous](Lecture-02.md) · [Next: RT-1, RT-2, and PaLM-E](Lecture-04.md)
