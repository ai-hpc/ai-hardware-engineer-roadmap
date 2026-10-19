# Lab: a small quality grader you can trust

**Time:** 45–70 minutes. **Needs:** Python 3.10+, logarithms, lists. No model download or GPU. **Build:** a numerical oracle for masked NLL, corpus perplexity, full-distribution KL, and causal-window accounting.

Run from this course directory:

```bash
python3 labs/quality_metrics.py
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

## Work the example before running

Batch A has two scored tokens with total NLL 2 nats. Batch B has one scored token with NLL 4 nats. The corpus mean is `(2+4)/(2+1)=2` nats, so perplexity is `exp(2)=7.389056`. Averaging the batch means gives `(1+4)/2=2.5`, which incorrectly gives the short batch as much weight as the long one. Aggregate **NLL sums and scored-token counts**, then divide once.

`masked_nll` accepts prediction rows already aligned with their target token. `None` excludes a target. It does not shift labels secretly. When adapting a causal model, pair `logits[:, :-1]` with `labels[:, 1:]`, and count unmasked labels **after** that shift. Padding, prompts, overlapping context, and document boundaries need explicit policies.

For ten tokens, window six and stride three, the scored absolute target indices are:

```text
window [0..5]: targets [1,2,3,4,5]  -> 5 predictions
window [3..8]: targets [6,7,8]      -> 3 predictions
window [6..9]: targets [9]          -> 1 prediction
total: nine; token zero has no preceding context/BOS in this example
```

Subtracting one from every window's new-token count gives 5, 2, 0 and loses two predictions. Overlap provides context to predict each later window's first *new* token. The lab checks many lengths/windows/strides so every intended target appears exactly once.

## Two counterexamples that protect your conclusions

Take a teacher `p=(0.8,0.2)`, a candidate `q=(0.9,0.1)`, and held-out target token zero. Candidate NLL **improves**: `−ln(.9)+ln(.8)=−0.117783` nats. Yet teacher KL is **positive**: `Σ p ln(p/q)=0.044403` nats. Data NLL measures a recorded continuation; teacher KL measures a full probability distribution on the same prefix. They need not move in the same direction. Only an expectation over the teacher's token distribution turns that NLL difference into teacher KL.

Next use `q=(0.001,0.999)` and target zero. Perplexity is 1000 although the vocabulary has only two tokens. `PPL=V` describes a **uniform predictor**, not the largest possible perplexity. A confidently wrong predictor can be arbitrarily worse.

## Guided variations

1. Add 1000 to every logit. Predict the change in logprobs, then run `log_softmax`. **Answer:** none within floating-point tolerance; subtracting the maximum avoids overflow.
2. Add 100 padding rows with target `None`. **Answer:** neither NLL sum nor scored-token count changes.
3. Compute KL from top-two probabilities whose sum is 0.8. **Answer:** the oracle rejects them. Renormalizing changes the question to a conditional distribution; it does not recover full-vocabulary KL.
4. Set candidate probability to zero where the teacher has positive mass. **Answer:** forward KL is infinite. Where teacher probability is zero, its KL contribution is zero.
5. For 12 total nats on the same 20 UTF-8 bytes, compute `12/(ln(2)*20)=0.865617` bits/byte. **Answer:** the denominator is comparable across tokenizers; the numerator is still model-dependent. Equal text does not force equal sequence probabilities.

## Independent transfer and assessment

Construct three unequal batches with at least one masked target, including one almost impossible target. Produce NLL sum, count, mean, and PPL; compare with a deliberately wrong mean-of-means implementation. Then choose a new stream length/window/stride and prove each target is counted once. Finally give a two-token teacher/candidate pair where data NLL worsens and KL is positive.

Submit the inputs, JSON results, two hand calculations, and a short verdict that separates numerical drift from downstream task quality. Use the [course rubric](../../../Learning-Guide.md#completion-rubric): correctness, reasoning, measurement, reproduction, and transfer each score 0–2; pass at 8/10 with full correctness and reproduction. The tests validate the oracle, not the quality of any real model. GPU/framework evaluation remains a separate experiment.

The [Transformers perplexity guide](https://huggingface.co/docs/transformers/en/perplexity) explains context-limited scoring. This lab's synthetic cases isolate its accounting assumptions without the cost or variability of running an LLM. In a robot assistant, also measure task completion and incorrect tool actions: a low average language loss does not certify a safe or successful action sequence.
