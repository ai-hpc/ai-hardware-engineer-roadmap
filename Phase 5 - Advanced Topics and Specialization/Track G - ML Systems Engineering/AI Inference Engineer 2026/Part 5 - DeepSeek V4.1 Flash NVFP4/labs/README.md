# Inference lab tools

These exercises use Python 3.10+ and only the standard library. Run from the parent part directory:

```bash
python3 labs/checkpoint_audit.py --demo
python3 labs/fp4_layout.py
python3 labs/inference_metrics.py --demo
python3 -m unittest discover -s labs -p 'test_*.py'
```

The [checkpoint audit](checkpoint_audit.py) should report 3 synthetic tensor entries, 2 shards, and 3 GiB declared payload. The [FP4 exercise](fp4_layout.py) checks packing and explicit scales. Neither loads model weights or proves runtime compatibility.

## Real metadata

```bash
python3 labs/checkpoint_audit.py --config /path/to/config.json \
  --index /path/to/model.safetensors.index.json
```

Save repository ID and immutable revision alongside the result. The tool hashes the two local metadata files. Payload bytes come from index metadata; missing size stays unknown. Packed parameter counts and runtime memory are deliberately not inferred.

## Request log schema

The [metric summarizer](inference_metrics.py) accepts one JSON object per line. Write timestamps in seconds using the same monotonic clock for all clients, or normalize to a synchronized run origin before merging logs.

```json
{"request_id":"r1","phase":"measurement","status":"ok","start_s":0.0,"first_token_s":0.2,"end_s":0.8,"output_tokens":4}
{"request_id":"r2","phase":"measurement","status":"timeout","start_s":1.0,"first_token_s":null,"end_s":3.0,"output_tokens":0}
```

- `phase`: `warmup` or `measurement`; warmup records require only a unique ID and phase.
- `status`: `ok`, `error`, or `timeout`. Log every attempt, including failures.
- `start_s`: measured request dispatch. Log scheduled arrival separately for open-loop studies.
- `first_token_s`: arrival of first content token, excluding headers and role-only chunks.
- `end_s`: last content token for success; final failure/timeout observation for failures. Keep stream-close time separately if measuring protocol completion.
- `output_tokens`: actual output token count; failed partial outputs are excluded from completed throughput.
- Single-token requests have undefined TPOT and are checked against TTFT only.

```bash
python3 labs/inference_metrics.py --input requests.jsonl \
  --ttft-slo-s 1.0 --tpot-slo-s 0.05
```

The finite-run window includes drain time and failures. Warmup must finish before measured traffic; the tool cannot detect hidden overlap from omitted warmup timings. Percentiles use nearest rank and successful requests only, with failures separately counted. TPOT is averaged within a request; this tool does not measure per-token stalls, censored requests, coordinated omission, server queue time, or model quality. Supply those separately when the experiment requires them.

The demo's expected throughput is 2 output tokens/s; goodput is 1.333333…; failure rate is 0.333333…. **These are fixtures, not hardware benchmarks.**

[Part overview](../README.md)
