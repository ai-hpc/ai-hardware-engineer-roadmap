# tinygrad for inference

Start with the [complete guide](Guide.md), then run the [CPU projects](projects/README.md). This course pins tinygrad **0.14.0** and includes executed operator, gradient, training, checkpoint, compiler/JIT and model-contract exercises.

The [mathematical references](labs/reference.py) use only the standard library. The framework path requires the [pinned wheel](labs/requirements-tinygrad.txt) and a supported CPU compiler/runtime. [Recorded results](evidence/tinygrad-cpu-2026-10-18.json) distinguish actual CPU execution from optional integrations.

| Read next | Purpose |
|---|---|
| [Concept overview](notes/overview.md) | Connect tensor semantics, compilation and deployment |
| [Source walkthrough](notes/internals.md) | Follow the actual v0.14.0 source layout |
| [Operation lessons](ops/README.md) | Work elementwise, reduction and layout examples |
| [Project commands](projects/README.md) | Reproduce the finite CPU progression |

The standalone package is separate from openpilot's pinned tinygrad submodule. Return to [the autonomous vehicle track](../Guide.md) for the observation/action contract and validation boundaries.
