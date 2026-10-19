# CPU collectives: two robots, one correct update

**Time:** 60–90 minutes. **Prerequisites:** gradients, a Python virtual environment, and [collective meanings](../01-Fundamentals.md). **Outcome:** run real two-process distributed training and prove that the result matches a serial reference. This lab runs on CPU with PyTorch/Gloo; it requires no GPU, model download or paid API.

## Predict before running

Robot 0 contributes x=[1,2]; robot 1 contributes x=[3,4,5]. Targets are y=2x+1. Start a scalar linear model `prediction=0.25x−0.5` and use mean squared error across **all five samples**. This original synthetic dataset is included in code under the repository license; it demonstrates gradient accounting, not robot-control quality. There is no train/test evaluation claim.

The per-sample weight derivative is `2(prediction−y)x = −3.5x²−3x`. Averaging x=1…5 gives **−47.5**. The bias derivative averages to **−13.5**. With SGD learning rate 0.01, the serial update yields weight **0.725**, bias **−0.365**.

Averaging the two **local means** instead gives weight gradient about **−41.791667**. Robot 0's two observations receive the same total weight as robot 1's three. Default DDP averages gradients across ranks, so the correct local loss is:

`local_sum_of_squared_errors × world_size / global_sample_count`.

For token models, use valid predicted-token counts; do not count padding. This lesson assumes every rank executes the same number of synchronized steps.

## Install in an isolated environment

Tested on Linux x86-64, Python **3.12.3**, PyTorch **2.14.0+cpu**. The course reuses the existing CPU practical's exact wheel URLs and SHA-256 lock; no CUDA wheels are needed. Installation needs network access once and local disk for the CPU framework. Other platforms need their own resolved lock and must not claim this exact environment.

From this lab directory:

```bash
python3.12 -m venv /tmp/course-collectives-venv
/tmp/course-collectives-venv/bin/python -m pip install --require-hashes -r "../../../../../../Phase 3 - Artificial Intelligence/2. Deep Learning Frameworks/PyTorch/CPU-Robot-Stop-Lab/labs/requirements-linux-cpu-py312.txt"
```

Use [the shared pinned requirements](../../../../../../Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/PyTorch/CPU-Robot-Stop-Lab/labs/requirements-linux-cpu-py312.txt) if following links instead of the shell. An existing environment installed from that lock can be reused. The course verification reused `/tmp/course-pytorch-cpu-venv`.

## Run the baseline and two variations

```bash
GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 /tmp/course-collectives-venv/bin/python \
    -m torch.distributed.run --standalone --nproc-per-node=2 \
    labs/distributed_demo.py --output /tmp/collectives-unequal.json

GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 /tmp/course-collectives-venv/bin/python \
    -m torch.distributed.run --standalone --nproc-per-node=2 \
    labs/distributed_demo.py --equal-batches --output /tmp/collectives-equal.json

GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 /tmp/course-collectives-venv/bin/python \
    -m torch.distributed.run --standalone --nproc-per-node=2 \
    labs/distributed_demo.py --mismatch --output /tmp/collectives-rejected.json
```

`torchrun` starts two processes. Each process receives a rank and joins one process group. Loopback confines this example to the same host. A same-shape SUM AllReduce turns each rank's `[rank+1,rank+1]` into `[3,3]`; AllGather returns `[1,2]` everywhere. We verify these operations before training.

The mismatch variation gathers trusted metadata and rejects a rank's incompatible shape **before** the payload AllReduce. It deliberately avoids launching a malformed collective that might crash or hang. This is not a test of NCCL failure recovery. Object collectives use pickle and are only appropriate for trusted local peers here.

## What actually ran

On 2026-10-18, the unequal/equal/mismatched variants ran as real two-process Gloo jobs. The [retained unequal-batch report](labs/measured-unequal-2026-10-18.json) records Python/framework versions, rank sample counts, actual gradients and a source SHA-256. Three optional integration tests passed, as did three standard-library accounting tests. The [source/execution manifest](labs/source-evidence-2026-10-18.json) distinguishes those runs from unexecuted hardware recipes.

| Case | Expected and observed behavior |
|---|---|
| Samples [2,3], naive local means | Weight gradient differs from serial by about **5.708333**; updated weight differs by about **0.0570833** |
| Samples [2,3], sample-weighted loss | Gradient and update match serial within 1e−12 |
| Samples [2,2] | Both local means and sample weighting match serial within 1e−12 |
| One mismatched shape | Both ranks reject the contract before payload reduction |

Floating-point results may differ in the last few digits; tests use tolerances. These are correctness results, **not GPU/NVLink/network throughput measurements**.

```bash
python3 -m unittest discover -s labs -p 'test_*.py' -v
/tmp/course-collectives-venv/bin/python -m unittest discover -s labs/optional -v
```

The integration suite launches subprocesses with finite timeouts; it takes about 15 seconds on the course machine. Optional framework tests are separate from the portable standard-library suite.

## Read bandwidth numbers without inventing them

The standard-library `accounting.py` checks a hypothetical timing: 8 ranks, a 1 GB decimal AllReduce in 4 ms gives algorithm bandwidth **250 GB/s** and normalized bus bandwidth **437.5 GB/s**. It performs arithmetic; it does not simulate a network or produce a measured bandwidth. AllGather's S is the **total gathered output**, not only one rank's contribution. [Official NCCL test accounting](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).

## Exercises with checks

1. **Explain the failure:** Why does the naive loss pass the equal-batch variation? **Answer:** with equal counts, the mean of rank means equals the mean of all samples; unequal counts break that identity.
2. **Independent change:** In `shard`, change robot 1 to x=[3,4,5,6], keeping y=2x+1. Predict the serial weight gradient before rerunning: mean x²=91/6 and mean x=3.5, so `−3.5×91/6−3×3.5 = −63.583333…`. The existing hard-coded baseline test is specific to the original fixture; save this variation separately and compare to the script's serial reference.
3. **Loss definition:** Switch to a sum objective. Should the global denominator remain? **Answer:** no; derive scaling from the intended sum and DDP's rank averaging. Do not silently change the objective when comparing runs.
4. **Rank failure boundary:** Why can a coordinated shape preflight not fix a rank that never launches? **Answer:** the preflight itself requires every rank; the launcher/process-group timeout must expose the missing participant.
5. **Transfer to robotics:** Two robots generate 200 and 20 samples. Decide whether each sample or each robot should have equal influence. Both can be valid objectives, but the loss and evaluation must state the choice explicitly.

**Evidence to submit:** hand-derived gradients, the three run reports, test output, and one independent dataset variation. A pass requires numerical agreement with the intended objective and a clear statement of what this CPU experiment does and does not validate. [PyTorch DDP semantics](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html), [distributed communication](https://docs.pytorch.org/docs/2.14/distributed.html).
