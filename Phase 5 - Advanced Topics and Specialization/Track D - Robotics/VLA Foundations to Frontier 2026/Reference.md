# VLA notation, glossary, and study prompts

[Course overview](README.md) · [Lab guide](Lab-Guide.md)

This reference supports the lectures. Symbols can differ across papers: translate each paper's notation into an explicit observation/action/time contract before comparing equations.

## Shared notation

| Symbol | Meaning | Common mistake |
|---|---|---|
| \(o_t\) | Observable sensor input at robot time \(t\) | Treating it as complete physical state |
| \(s_t\) | State; a paper may mean physical, estimated, or proprioceptive state | Failing to identify privileged simulator information |
| \(\ell\) | Language instruction or token sequence | Assuming it encodes an executable, unambiguous task |
| \(h_t\) | Observation/action history or learned memory | Reusing it after an episode reset |
| \(a_t\) | One action in a specified representation, frame, and unit | Confusing torque, velocity, displacement, and pose |
| \(A_t\in\mathbb R^{H\times D}\) | Predicted chunk, horizon \(H\), action dimension \(D\) | Assuming all \(H\) actions are executed before replanning |
| \(K\) | Executed prefix length before replanning | Confusing it with prediction horizon |
| \(\Delta t_a\) | Interval between predicted action targets | Confusing it with physics or inference duration |
| \(\pi_\theta\) | Policy parameterized by \(\theta\) | Assuming it exposes a tractable normalized likelihood |
| \(m_{t,d}\) | Validity mask | Treating padded zeros as measured neutral actions |
| \(\mu,\sigma\) | Normalization center and positive scale | Fitting them on held-out data or swapping embodiments |
| \(\tau\in[0,1]\) | Flow/diffusion interpolation time in a chosen convention | Confusing it with physical robot time \(t\) |
| \(v_\theta(x,\tau,c)\) | Conditional vector field for flow sampling | Assuming every paper uses noise-to-data direction |
| \(T_{AB}\) | Transform taking coordinates in B into A in this course | Reversing multiplication order or transform direction |

For dimensions, write batch \(B\), camera count \(C\), image height/width \(H_I,W_I\), visual tokens \(N_V\), language tokens \(N_L\), and embedding dimension \(d\). Avoid using one unlabeled \(H\) for both image height and action horizon in code or reports.

## Equations worth reconstructing from memory

**Masked behavior cloning:**

\[
L_{BC}=\frac{\sum_{b,t,d}m_{btd}(\hat a_{btd}-a_{btd})^2}
{\sum_{b,t,d}m_{btd}}.
\]

This is one regression objective, not a universal action-model likelihood. Handle a zero-valid-entry batch explicitly. Other representations use token cross-entropy, mixture likelihoods, or generative objectives. See [Lecture 09](Lecture-09.md).

**Noise-to-data conditional flow example:**

\[
x_\tau=(1-\tau)\epsilon+\tau A,\qquad
L_{FM}=\mathbb E\|v_\theta(x_\tau,\tau,c)-(A-\epsilon)\|^2.
\]

At inference integrate from a noise sample toward data under the learned field. Match interpolation direction, time sampling, parameterization, and solver to the chosen model. This squared loss is not itself a normalized log probability for PPO. See [Lectures 12](Lecture-12.md) and [17](Lecture-17.md).

**Action reconstruction:** \(a=\sigma\odot\tilde a+\mu\). This affine formula does not describe every dataset's quantile, clipped, or binary-gripper transform. Save and apply the actual training transform. See [Lecture 16](Lecture-16.md).

**Chunk timing:** nominal replanning interval is \(K\Delta t_a\); under the Lecture 22 convention, the first non-expired index after delay \(L\) is \(\lceil L/\Delta t_a\rceil\). Inference jitter, controller commit windows, and sensor age still matter. See [Lecture 22](Lecture-22.md).

**Task-macro success:** \(\hat p=J^{-1}\sum_j n_j^{-1}\sum_i s_{ji}\). Compare policies with paired initial states and uncertainty appropriate to tasks and episodes. See [Lecture 19](Lecture-19.md).

## Vocabulary by engineering decision

| Term | Working meaning | Study prompt |
|---|---|---|
| VLM | A model combining visual and language representations | Which output or supervision turns this into an action policy? |
| VLA | A vision/language-conditioned action model | What exactly is its action representation? |
| Embodiment | Robot morphology, sensing, actuation, and related interface | Which differences can an adapter express, and which require learning? |
| Proprioception | Internal robot measurements such as joint state | Are values measured, estimated, normalized, and timestamped? |
| POMDP | Decision process with partial observations of underlying state | What history helps resolve hidden state? |
| Imitation learning | Learning behavior from demonstrations or expert interaction | Whose state distribution generated the labels? |
| Behavioral cloning | Supervised prediction of demonstrated behavior | What happens after the learner leaves demonstration states? |
| DAgger | Expert relabeling on learner-induced states in a dataset aggregation procedure | Is expert access feasible and how is intervention recorded? |
| Affordance | Feasibility/usefulness of an action in a state | Does a language score measure physical feasibility? |
| Grounding | Connecting symbolic/language content to observations and behavior | Does instruction swapping change the action appropriately? |
| Action chunking | Predicting several future targets together | Which targets are committed and which may be revised? |
| Temporal ensembling | Combining overlapping action predictions | Are actions aligned to the same physical time before averaging? |
| Diffusion policy | Action generation through a learned denoising process | What objective and sampler does the implementation actually use? |
| Flow matching | Learning a vector field along specified probability paths | How many field evaluations are needed at deployment? |
| Action expert | A component specialized for generating actions | How does it condition on VLM features and robot state? |
| Cross-embodiment learning | Sharing information across robot interfaces | Is common tensor shape hiding incompatible semantics? |
| Co-training | Joint use of multiple data/objective types | Which parameters receive gradients from each source? |
| Adapter / LoRA | Restricted parameter update mechanism | Are deployed base weights, adapter, and normalizer compatible? |
| Offline RL | Policy learning from a fixed transition dataset | Can the critic extrapolate incorrectly beyond data support? |
| Online RL | Learning using newly collected interaction | How are resets, cost limits, and interventions handled? |
| World model | Model predicting aspects of environment evolution | Are plausible predictions also useful for control? |
| World-action model | Jointly modeled future observations and actions in some architectures | What additional compute and consistency constraints appear? |
| Hierarchical policy | Decisions across levels or timescales | Who verifies high-level plan completion? |
| Receding horizon | Execute part of a plan, then update from new observations | What changes when inference takes nonzero time? |
| System identification | Estimating dynamics/sensor parameters from observations | Does the model predict a held-out excitation trajectory? |
| Domain randomization | Sampling simulation variations during training | Are ranges realistic and evaluation conditions held out? |
| Action parity | Comparison against reference policy outputs/behavior | Does numerical parity translate into task parity? |
| Observation age | Time since the observation underlying a command | Which clock and event define the measurement? |
| Noninferiority | Evidence that degradation is smaller than a declared margin | Was the margin specified before observing results? |
| Calibration | Relating predictions/scores to measured outcomes or sensor geometry | Which meaning is intended and how was it tested? |
| Intervention | Human or supervisor changes to autonomous execution | Is it counted as failure, recovery, or assisted completion? |

## Paper-reading worksheet

For each paper, record its first publication and inspected revision; task and embodiment; observation/action contract; data source and scale; architecture; objective; training resources; inference path; evaluation split and horizon; strongest baseline; ablations; uncertainty; accessible artifacts; and one unresolved limitation.

Then write three sentences: the concrete problem, the mechanism that changed, and the evidence supporting the change. Reproduce one equation or numerical claim from the paper's definitions. Identify one result you cannot reproduce with the released artifacts and why. Compare dates and access using the [frontier register](Frontier-Register.md), rather than assuming every named model is downloadable.

## Oral-review prompts

Explain why a smaller action MSE may produce worse closed-loop success. Derive a masked loss and identify its denominator. Explain why a flow objective is not automatically a likelihood ratio. Trace a delayed camera frame to a physical command. Diagnose an inverted gripper signal. Design a language counterfactual. Choose an uncertainty resampling unit. Distinguish a paper's claimed performance from the behavior of one released checkpoint. These prompts connect mathematics to the evidence a professional engineer must supply.
