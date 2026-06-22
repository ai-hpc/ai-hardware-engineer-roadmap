# Lecture 09 - Transfer Learning: CV, NLP, and Prompting

**Collection:** [Practical Machine Learning (CS329P)](README.md) | **Previous:** [← Lecture 08](Lecture-08.md) | **Next:** [Lecture 10](Lecture-10.md)

---

Almost nobody trains a model from scratch anymore. If you join an ML team today and propose initializing a fresh network with random weights and a few thousand of your own labels, you will be politely asked why you are not starting from a pretrained model. The default workflow — across vision, language, audio, and beyond — is **"start from a model someone else trained on a mountain of data, and adapt it to your task."** This is transfer learning, and it is arguably the single biggest practical shift in modern ML.

The reason is economic. Deep networks are *data-hungry* and training them is *expensive*. A model that has already seen 1.2 million labeled images, or hundreds of billions of words, has paid the steep up-front cost of learning what edges, textures, shapes, syntax, and meaning look like. Those learned features are not specific to the original task — they **transfer**. Your job shrinks from "learn vision from nothing" to "tell an already-competent vision model what *your* ten classes are," and you can do that with 10–100× less data and a fraction of the compute.

This lecture walks the paradigm in three movements. First the idea itself — why features transfer and what the pretrain → fine-tune split looks like. Then the two domains where it took over: **computer vision** (fine-tuning ImageNet backbones like ResNet and ViT) and **NLP** (fine-tuning self-supervised models like BERT). Finally **prompt-based learning** — GPT-3's discovery that for a large enough model you often do not need to update *any* weights at all; you just *describe the task in text*. We teach the 2021 version faithfully, then map each piece forward to where it landed by 2026.

---

## Learning objectives

By the end of this lecture you should be able to:

* Explain *why* features learned on large datasets transfer to new tasks, and articulate the pretrain → fine-tune paradigm and the data/compute savings it buys.
* Fine-tune a pretrained CV backbone — choosing between **feature extraction** (freeze the backbone, train a new head) and **full fine-tuning**, deciding which layers to freeze, and setting a sensibly small learning rate for pretrained weights.
* Describe the self-supervised pretraining objectives behind NLP models — **masked language modeling** (BERT) and **autoregressive language modeling** (GPT) — and how downstream fine-tuning reuses their contextual embeddings.
* Explain **prompt-based / in-context learning**: zero-, one-, and few-shot prompting, why a large LM can solve tasks without weight updates, and the shift from "fine-tune weights" to "condition with text."
* Choose between **feature extraction, fine-tuning, and prompting** for a given task, data budget, and compute budget.
* Map the 2021 picture onto the 2026 stack: instruction tuning (RLHF/DPO), parameter-efficient fine-tuning (LoRA/QLoRA/adapters), and retrieval-augmented generation (RAG).

---

## 1. Why transfer learning

The motivation is one sentence: **exploit a model trained on one task for a related task.** It is especially popular in deep learning precisely because DNNs are data-hungry and the cost of training them from scratch is high. Why repeat that cost for every new problem when the expensive part — learning good features — generalizes?

There are several flavors of "reuse a model," and it helps to name them so you know which one this lecture is about:

* **Feature extraction** — run your data through a fixed pretrained model and use its internal representation as input features for a *separate* downstream model. Classic examples: Word2Vec embeddings for words, ResNet-50 penultimate-layer features for images, I3D features for video. The pretrained network is frozen; it is just a feature factory.
* **Train on a related task and reuse** — train a model on a task where labels *are* plentiful, then repurpose it for the task you actually care about.
* **Fine-tuning from a pretrained model** — initialize a new model with a pretrained model's weights and continue training on your data. **This is the focus of the lecture.**

Transfer learning also sits next to several adjacent ideas. It is **related to** semi-supervised learning (some labeled, lots of unlabeled), to **zero-shot / few-shot learning** in the extreme (adapt with almost no labeled examples — we get there in the prompting section), and to **multi-task learning**, where some labeled data is available for each of several tasks at once.

The mental model that makes everything downstream click: **a trained neural network is two parts stacked together.**

* A **feature extractor** (the encoder) — the bulk of the network — maps raw input (pixels, tokens) into a representation where the data becomes *linearly separable*.
* A **linear classifier** (the decoder/head) — typically the final layer — reads that representation and makes the decision.

A **pretrained model** is a network trained on a large-scale, general-enough dataset. The crucial empirical fact is that its *feature extractor generalizes well* — to other datasets (medical scans, satellite imagery), and even to other tasks (detection, segmentation). The head is task-specific and disposable; the feature extractor is the reusable asset. Hold onto that decomposition — it is the same idea in CV, in NLP, and in prompting.

---

## 2. Fine-tuning for CV

Computer vision is the cleanest place to see transfer learning work, because of a happy accident: **large-scale labeled CV datasets already exist.** Image classification in particular is among the cheapest things to label — a human glances at an image and types a class. ImageNet gives us roughly **1.2M images across 1,000 classes**. Your application probably has something like 50K images across 100 classes, or 60K across 10 — one or two orders of magnitude smaller. Transfer learning is precisely how you bridge that gap.

### Pretrained backbones

The standard backbones are convolutional networks like **ResNet** (ResNet-18/50/...) and, increasingly, **Vision Transformers (ViT)**, pretrained on ImageNet. You almost never build one yourself. Two common sources:

* **TensorFlow Hub** (`tfhub.dev`) — user-submitted TensorFlow models.
* **TIMM** (`pytorch-image-models`, originally by Ross Wightman) — a large, well-maintained collection of PyTorch image models.

### Feature extraction vs full fine-tuning

There are two ways to use a pretrained backbone, and the difference is *which parameters you allow to change.*

* **Feature extraction (freeze the backbone, train the head).** Freeze the entire pretrained feature extractor and train only a freshly initialized output layer (often just a linear classifier on top of the frozen features). The backbone never updates. This is fast, needs little data, and is strongly regularized — but it cannot adapt the features themselves.
* **Full fine-tuning.** Initialize the feature extractor from the pretrained weights, *randomly* initialize the new output layer, and then continue training the **whole network** on your data. Because you start near a good local minimum rather than from random noise, you train with a **small learning rate** for **just a few epochs** — this regularizes the search and keeps the model from forgetting what it learned.

The fine-tuning recipe in one picture: copy the pretrained layers into the target model, bolt on a randomly initialized head, and resume optimization from that already-good starting point.

```text
   Source (pretrained)            Target (your task)
   ┌───────────────┐              ┌───────────────┐
   │ Output layer  │              │ Output layer  │  ← random init (new head)
   ├───────────────┤   copy ─────►├───────────────┤
   │  Layer L-1    │ ───────────► │  Layer L-1    │  ← copied weights
   │      ...      │ ───────────► │      ...      │  ← copied weights
   │   Layer 1     │ ───────────► │   Layer 1     │  ← copied weights
   └───────────────┘              └───────────────┘
```

### Which layers to freeze

Neural networks learn **hierarchical features**, and that hierarchy tells you what to freeze:

* **Low-level features are universal** — curves, edges, blobs. They generalize across almost any natural image, so there is little reason to disturb them.
* **High-level features are task- and dataset-specific** — they encode things close to the original classification labels.

So the standard move is to **freeze the bottom layers and train the top layers**. This keeps the universal low-level features intact, focuses learning on the task-specific part, and acts as a **strong regularizer** — fewer free parameters means less overfitting on a small dataset. Where you draw the freeze line is a knob: freeze more when your data is tiny or very similar to the source; freeze less (or nothing) when you have more data or your images differ a lot from ImageNet.

### The minimal PyTorch recipe

With TIMM, full fine-tuning is almost no code — load a pretrained backbone, swap the classifier head for one with your number of classes, and train it like any normal job:

```python
import timm
from torch import nn

model = timm.create_model('resnet18', pretrained=True)
model.fc = nn.Linear(model.fc.in_features, n_classes)  # new head
# Train model as a normal training job (small LR, few epochs)
```

For feature extraction, you would additionally freeze the backbone's parameters (`requires_grad = False` on everything except `model.fc`) before training.

### When it helps — and when it doesn't

Fine-tuning ImageNet-pretrained models is used everywhere in CV: **detection and segmentation** (similar images, different targets) and **medical/satellite imagery** (same task, very different images). The most reliable benefit is **faster convergence** — you reach good accuracy in far fewer epochs.

The honest caveat from the slides: fine-tuning **does not always improve final accuracy.** If your target dataset is itself large, training from scratch can reach a similar accuracy. Transfer learning's biggest, most dependable win is in the *small-data* regime; as your data grows, the gap to from-scratch training shrinks.

**Section summary:** pretrain on a large dataset (usually image classification), initialize your downstream model's weights from it, and fine-tune. This accelerates convergence and *sometimes* improves accuracy — most when your own data is scarce.

---

## 3. Fine-tuning for NLP

NLP starts from the opposite resource situation than CV. There is **no large-scale labeled NLP dataset** comparable to ImageNet — but there are *enormous quantities of unlabeled text*: Wikipedia, ebooks, crawled web pages. The breakthrough was learning how to pretrain on that unlabeled text.

### Self-supervised pretraining

The trick is **self-supervised learning**: generate a "pseudo-label" from the raw text itself, then train with ordinary supervised learning against it. No human annotation required — the supervision is hidden in the data. The two canonical objectives:

* **Language model (LM) — predict the next word.** Given "I like your ___", predict "hat". This is **autoregressive**: it reads left to right. (This is the GPT family's objective.)
* **Masked language model (MLM) — predict a randomly masked word.** Take "I like your hat", hide a token → "I like your `[MASK]`", and predict the missing word from *both sides* of context. (This is BERT's objective.)

The payoff is **contextual embeddings**: unlike a static lookup table (Word2Vec/CBOW, where each word `w` gets fixed vectors learned by predicting it from the sum of its context words), a transformer produces a representation of each word *that depends on the sentence around it*. "Bank" near "river" and "bank" near "money" get different vectors. These representations are what transfer.

### Pretraining objectives, by model family

| Model | Architecture | Pretraining objective |
|---|---|---|
| **Word2Vec** (CBOW) | shallow embeddings | predict a word from the sum of its context word embeddings |
| **BERT** | transformer **encoder** | masked-token prediction + next-sentence prediction |
| **GPT** | transformer **decoder** | autoregressive next-token prediction (covered in §4) |
| **T5** | transformer **encoder-decoder** | fill in a masked *span* of text from documents |

### BERT and its fine-tuning recipe

**BERT** is a giant transformer **encoder**, pretrained on Wikipedia + BookCorpus (over 3 billion words) with two tasks: **masked token prediction** and **next-sentence prediction**. It ships in many versions — base/large, English/multilingual, cased/uncased — and spawned variants such as **RoBERTa, ALBERT, and ELECTRA**.

Fine-tuning BERT for a downstream task follows one consistent pattern: **randomly initialize a new last layer and train a few epochs with a small learning rate.** The input is formatted with special tokens — a `[CLS]` token at the front and `[SEP]` separators between segments — and *which* hidden states you read out depends on the task:

| Downstream task | What you read out of BERT |
|---|---|
| **Sentence classification** (e.g. sentiment) | the `[CLS]` embedding → a dense classifier |
| **Named-entity recognition** | each token's hidden state → predict that token's entity tag |
| **Question answering** | sentence 1 = question, sentence 2 = reference passage → predict the answer *span* in the reference |

BERT's "obtains new state-of-the-art results on eleven NLP tasks" included grammaticality judgments, movie-review sentiment, sentence-pair semantic equivalence, textual entailment, and answer-span extraction. T5, with its text-to-text framing, similarly topped summarization, QA, and classification benchmarks.

### Practical considerations

Two gotchas the slides call out, both worth remembering:

* **Fine-tuning on small datasets can be unstable.** Two common culprits: the original BERT removed the bias-correction step in Adam, and people train for *too few* epochs (3 is often not enough).
* **Re-initializing some of the *top* transformer layers can help.** The topmost layers' features are too specialized to the pretraining tasks; throwing them away and relearning them frees the model to adapt. How many layers to reset depends on the downstream task.

The practical home for all of this is **Hugging Face Transformers** — pretrained transformer models for both PyTorch and TensorFlow, behind a uniform API:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
inputs = tokenizer(sentences, padding="max_length", truncation=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=2)
# Train model on inputs as a normal training job
```

**Section summary:** self-supervised pretraining (a (masked) language-model objective) lets NLP learn from unlabeled text; BERT is a giant transformer encoder; downstream tasks fine-tune it in a consistent, low-LR, few-epoch manner.

---

## 4. Prompt-based learning

BERT-style fine-tuning has a subtle mismatch: its *pretraining* task (fill in masks) and its *downstream* task (classify a sentence) are different shapes, which is part of why it typically needs **thousands of labeled examples** to fine-tune well.

**Prompting** removes the mismatch by **converting the downstream task into the pretraining task itself** — a language-modeling problem. You phrase your task as text the model can simply continue:

* Sentiment analysis becomes: `"I like this movie. It was ___"` → the model fills `great` (positive) or `terrible` (negative).
* Machine translation becomes: `"Hello world! => ___"` → the model fills `Bonjour le monde!`.

GPT made this paradigm popular. The decisive demonstration was **GPT-3** — a giant transformer **decoder** (~175B parameters), trained on 500B+ tokens from CommonCrawl, WebText, and books, at a training cost reported around \$12M. It turned out to be a general-purpose language model with striking text generation and, crucially, **zero-shot / few-shot learning**: it can *understand a task specification given in plain language* and perform the task with no gradient updates at all.

### Zero-, one-, and few-shot

The "shots" are simply *how many worked examples you place in the prompt* before asking for the answer — the model conditions on them at inference time:

| Setting | What's in the prompt | Weight updates |
|---|---|---|
| **Zero-shot** | a task description only, then the input | none |
| **One-shot** | description + **one** example, then the input | none |
| **Few-shot** | description + a few (~10) examples, then the input | none |

This is **in-context learning**: the model "learns" the task from the context window on the fly, then forgets it the moment the request ends. Nothing is stored in the weights. With this, GPT-3 could write code from a description ("a table of the richest countries with column names and GDP"), generate thought experiments from classic examples, and power the hundreds of demos catalogued at the time (search engines, NPC dialogue, poetry).

### The conceptual shift

This is the pivot the whole lecture builds to: **from "fine-tune weights" to "condition with text."** In CV and BERT fine-tuning, adapting to a task means *changing parameters*. In prompting, adapting means *writing a better input*. The model is frozen; your leverage is the prompt — hence **prompt engineering**, the craft of choosing the wording, the template, and the example labels that make the model behave.

### Prompt-based fine-tuning (the hybrid)

There is a middle ground for *medium-sized* LMs (e.g. < 1B parameters), where pure prompting is weaker but full fine-tuning is wasteful. **Prompt-based fine-tuning** designs a task-specific *prompt template* (and the label words it maps to) instead of bolting on a brand-new output layer, then fine-tunes the model's weights through that template. Reported result: it is roughly **100× more example-efficient** than standard fine-tuning. Because hand-designing templates and label words is finicky, there is work on **automatic prompt search** — automatically selecting the template and the label words (Gao et al., 2021).

**Section summary:** prompt-based learning presents a downstream task in language-model format. A model large enough (GPT-3) uses its *pretrained* weights directly for downstream tasks **without updating parameters**; used in fine-tuning, prompting gives much better example efficiency.

---

## 5. Feature extraction vs fine-tuning vs prompting

The three strategies trace a line of *decreasing weight updates* and *increasing reliance on the pretrained model's raw capability*. Choosing among them is mostly a question of how much labeled data and compute you have, and how large/capable your base model is.

| | **Feature extraction** | **Fine-tuning** | **Prompting (in-context)** |
|---|---|---|---|
| **What changes** | only a new head; backbone frozen | new head **+** all (or top) backbone weights | nothing — weights frozen |
| **Data needed** | low (works with little labeled data) | medium → high (often 1000s of labels) | very low — zero/one/few examples |
| **Compute / memory** | low (no backbone gradients) | high (gradients + optimizer state for the model) | none for training; inference-only |
| **Adapts the features?** | no | yes | no (conditions behavior, not features) |
| **Typical model size** | any | small–large | very large (capability emerges with scale) |
| **Best when** | small data, source ≈ target, fast baseline | enough data and the task needs adapted features | a strong general LM exists and labels are scarce |

A rough decision rule: **start with the cheapest option that meets your accuracy bar.** Try feature extraction or prompting first (minutes, little data); reach for fine-tuning when the features genuinely need to move toward your domain and you have the labels to justify it.

> **2026 update:** The single slide titled "prompt-based learning" in 2021 has, by 2026, exploded into three of the largest subfields in applied ML. Learn the original framing first — *condition a frozen model with text* — then map it forward:
>
> 1. **Instruction tuning (RLHF / DPO).** GPT-3's raw few-shot prompting was clever but clunky; you had to coax the base model. The fix was to **fine-tune models to follow instructions** using human preference data — **RLHF** (reinforcement learning from human feedback, InstructGPT → ChatGPT) and its simpler successor **DPO** (direct preference optimization). Zero-shot prompting "just working" today is the *result* of this tuning, not a property of the base LM. This is still fine-tuning — it just changed *what* we fine-tune *for* (helpfulness/safety, not a single downstream task).
> 2. **Parameter-efficient fine-tuning (PEFT): LoRA / QLoRA / adapters.** "Full fine-tuning of a giant model" became infeasible for most teams, so we stopped updating all the weights. **LoRA** trains tiny low-rank update matrices while the base weights stay frozen; **QLoRA** does the same on top of a 4-bit-quantized base model so it fits on a single GPU; **adapters** insert small trainable modules between frozen layers. This is the *direct descendant* of "freeze the backbone, train only a small part" from §2 — now applied to billion-parameter LLMs.
> 3. **Retrieval-augmented generation (RAG).** Instead of putting task knowledge in the weights *or* hand-writing every example into the prompt, **retrieve** relevant documents at query time and stuff them into the context. RAG is prompting taken to its logical end: the prompt is *assembled programmatically* from a knowledge base, giving the frozen model fresh, grounded facts without retraining.
>
> This course has a sibling that walks the modern PEFT path end-to-end on a real model: **[Qwen3.5-4B Unsloth Fine-Tuning](../../5.%20LLM%20Application%20Development/Qwen3.5-4B%20Unsloth%20Fine-Tuning/Guide.md)** — LoRA/QLoRA fine-tuning of a small Qwen model with the Unsloth library. It is §2's "freeze most of it, train a little" idea, realized for 2026 LLMs.

> **Hardware lens:** PEFT is not just an elegance — it exists because **full fine-tuning is memory-bound.** Updating every weight means holding, per parameter, the weight, its gradient, and the optimizer's moment estimates (Adam keeps two) — in practice **~16 bytes/parameter** in fp32-ish mixed-precision training. A 7B model blows past a single consumer GPU's memory before you have loaded a batch. **LoRA** sidesteps this by training only a few million low-rank parameters, so gradients and optimizer state shrink by orders of magnitude. **QLoRA** goes further by **quantizing the frozen base to 4-bit (NF4)**, cutting the static weight footprint ~4× so a model that needed an A100 now fits on a 24 GB card — the LoRA adapters stay in higher precision because they are what actually trains. This is the same memory-bandwidth and footprint reasoning that governs *inference*; the quantization and memory-hierarchy mechanics live in **[Phase 5 — ML Systems Engineering](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/Guide.md)** and the **[MLSys Deep Dives](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/README.md)**, and the hands-on QLoRA payoff is the [Unsloth course](../../5.%20LLM%20Application%20Development/Qwen3.5-4B%20Unsloth%20Fine-Tuning/Guide.md) above.

---

## Current as of

The spine of this lecture is taught as the original **Stanford CS329P (2021 Fall)** material and tracks the slides directly: the transfer-learning motivation and the feature-extractor / classifier decomposition; CV fine-tuning of ImageNet backbones (feature extraction vs full fine-tuning, freezing the universal low-level layers, small LR / few epochs, TIMM and the ResNet recipe); NLP self-supervised pretraining (LM vs masked-LM objectives, contextual embeddings, BERT's encoder with its `[CLS]`/`[SEP]` downstream patterns and Hugging Face); and prompt-based learning (GPT-3, zero/one/few-shot in-context learning, prompt-based fine-tuning with automatic prompt search). All of that remains accurate and foundational.

What has moved, and what the **2026 refresh** flags, is almost entirely downstream of that final prompting section. The 2021 "prompt-based learning" slide is now three mature subfields: **instruction tuning** (RLHF → DPO) is why zero-shot prompting feels effortless today; **parameter-efficient fine-tuning** (LoRA/QLoRA/adapters) is how teams actually adapt billion-parameter models on commodity hardware — the direct heir of CV's "freeze most of the network"; and **retrieval-augmented generation** is prompting with a programmatically assembled, knowledge-grounded context. On the CV side, ViT backbones now sit alongside ResNets as default pretrained models, and large self-supervised / image-text foundation models have broadened "pretrained backbone" well beyond ImageNet classification. The throughline to remember: the field moved *from updating weights toward conditioning frozen models* — and where it still updates weights, it does so cheaply (LoRA) and in low precision (QLoRA). Reviewed June 2026.

*Adapted from [Stanford CS329P](https://c.d2l.ai/stanford-cs329p) — Huang, Li & Smola, CC-BY-SA-4.0.*
