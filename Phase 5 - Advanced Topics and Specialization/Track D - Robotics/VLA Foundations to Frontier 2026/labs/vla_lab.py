"""A tiny, trainable synthetic VLA grounding exercise; Python standard library only.

Images contain three colored points. A FIXED color-centroid encoder supplies
visual tokens. Nine learned attention logits ground three language instructions
in those tokens, producing a 2-D displacement from a supplied robot position.
This is neither a pretrained foundation model nor a physical robot controller.
"""
import argparse
import json
import math
import random
from dataclasses import dataclass


INSTRUCTIONS = ("reach red", "reach green", "reach blue")


@dataclass
class Sample:
    image: list
    instruction: int
    state: tuple
    target: tuple


def centroid_tokens(image):
    """Return three color centroids in [-1, 1] image coordinates, x then y."""
    size = len(image)
    if size < 2 or any(len(row) != size for row in image):
        raise ValueError("expected a square image with side >= 2")
    moments = [[0.0, 0.0, 0.0] for _ in range(3)]
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            if len(pixel) != 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in pixel):
                raise ValueError("expected finite RGB values in [0, 1]")
            for c, value in enumerate(pixel):
                moments[c][0] += value * (2 * x / (size - 1) - 1)
                moments[c][1] += value * (2 * y / (size - 1) - 1)
                moments[c][2] += value
    if any(m[2] == 0 for m in moments):
        raise ValueError("each color must be visible")
    return [tuple(v / m[2] for v in m[:2]) for m in moments]


def dataset(count, seed, size=8):
    if count < 1 or size < 2:
        raise ValueError("count must be positive and image side >= 2")
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        image = [[[0.0] * 3 for _ in range(size)] for _ in range(size)]
        positions = rng.sample(range(size * size), 3)
        for color, position in enumerate(positions):
            image[position // size][position % size][color] = 1.0
        instruction = rng.randrange(3)
        state = (rng.uniform(-1, 1), rng.uniform(-1, 1))
        goal = centroid_tokens(image)[instruction]
        result.append(Sample(image, instruction, state,
                             tuple(goal[d] - state[d] for d in range(2))))
    return result


def softmax(values):
    exponentials = [math.exp(v - max(values)) for v in values]
    return [v / sum(exponentials) for v in exponentials]


def predict(logits, sample, instruction=None):
    language = sample.instruction if instruction is None else instruction
    weights = softmax(logits[language])
    tokens = centroid_tokens(sample.image)
    return tuple(sum(w * token[d] for w, token in zip(weights, tokens))
                 - sample.state[d] for d in range(2))


def loss_gradient(logits, sample):
    """L = 1/2 sum_d(error_d^2); return loss and gradient for active row."""
    tokens = centroid_tokens(sample.image)
    weights = softmax(logits[sample.instruction])
    mean = [sum(w * token[d] for w, token in zip(weights, tokens)) for d in range(2)]
    errors = [mean[d] - sample.state[d] - sample.target[d] for d in range(2)]
    gradient = [weights[c] * sum(errors[d] * (tokens[c][d] - mean[d])
                               for d in range(2)) for c in range(3)]
    return sum(e * e for e in errors) / 2, gradient


def train(samples, epochs=30, learning_rate=0.2, seed=17):
    if not samples or epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("require samples, positive epochs and finite positive learning rate")
    logits = [[0.0] * 3 for _ in range(3)]
    rng = random.Random(seed)
    order = list(range(len(samples)))
    history = []
    for _ in range(epochs):
        rng.shuffle(order)
        total = 0.0
        for index in order:
            sample = samples[index]
            loss, gradient = loss_gradient(logits, sample)
            total += loss
            row = logits[sample.instruction]
            for c in range(3):
                row[c] -= learning_rate * gradient[c]
        history.append(total / len(samples))
    return logits, history


def evaluate(logits, samples, wrong_language=False):
    errors = []
    for sample in samples:
        instruction = (sample.instruction + 1) % 3 if wrong_language else sample.instruction
        action = predict(logits, sample, instruction)
        errors.append(sum((action[d] - sample.target[d]) ** 2 for d in range(2)) / 2)
    if not errors:
        raise ValueError("evaluation requires samples")
    return sum(errors) / len(errors)


def experiment(train_count=240, test_count=120, epochs=30, seed=17):
    # Different procedural streams, not a claim of disjoint visual support.
    training, testing = dataset(train_count, seed), dataset(test_count, seed + 1)
    logits, history = train(training, epochs=epochs, seed=seed)
    return {
        "scope": "synthetic 2-D grounding; fixed color encoder; 9 learned logits",
        "train_seed": seed, "test_seed": seed + 1, "epochs": epochs,
        "optimizer": "SGD", "learning_rate": 0.2,
        "train_count": train_count, "test_count": test_count,
        "first_epoch_loss": history[0], "last_epoch_loss": history[-1],
        "test_mse_per_coordinate": evaluate(logits, testing),
        "wrong_instruction_mse": evaluate(logits, testing, True),
        "uniform_attention_mse": evaluate([[0.0] * 3 for _ in range(3)], testing),
        "attention_rows_red_green_blue": [softmax(row) for row in logits],
        "logits": logits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-count", type=int, default=240)
    parser.add_argument("--test-count", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(experiment(args.train_count, args.test_count, args.epochs, args.seed), indent=2))


if __name__ == "__main__":
    main()
