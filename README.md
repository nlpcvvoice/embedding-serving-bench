# embedding-serving-bench

Benchmark harness for embedding serving, first run against `nvidia/nemotron-3-embed-1b` on hosted NIM (`integrate.api.nvidia.com`). OpenAI-compatible schema: point it at any serving endpoint by changing base URL and model.

## Why

Replicate the serving-performance methodology of the Nemotron-3-Embed blog: throughput versus input sequence length, batch size, and concurrency.

## Request shape

```
POST https://integrate.api.nvidia.com/v1/embeddings
Authorization: Bearer $NVIDIA_API_KEY
Content-Type: application/json

{ "model": "nvidia/nemotron-3-embed-1b",
  "input": ["..."],            # list of strings = in-batch batching
  "input_type": "passage",     # vs "query" for live search strings
  "encoding_format": "float" }
```

Response: `data[i]["embedding"]` (2048-dim float vector per input) + `usage.prompt_tokens`. Rate limiting surfaces as HTTP status 429 (status line, not body); quota preview in `x-ratelimit-{limit,remaining}-{requests,tokens}` headers.

## Experiments

| Script | Knob | Design |
|--------|------|--------|
| `nim_step1.py` | — | first call, response shape, rate-limit headers |
| `nim_step2_ratelimit.py` | — | 30 back-to-back requests to find the free-tier 429 ceiling |
| `nim_step3_isl.py` | ISL | same paragraph x4 / x16 (~270 / ~1100 tokens), only length varies |
| `nim_step4_batch.py` | batch N | 1 / 16 / 64 passages per request, 10 interleaved rounds, medians |
| `nim_step5_concurrency.py` | concurrency C | Python threads, C=1/2/4, 16 passages/request, pre-allocated per-thread result slots (no locks) |

## Results (2026-10-18, free tier)

### Batching — `d4_batch_results.txt`

| N | tok/passage | median ms | passages/s | tokens/s |
|---|-------------|-----------|------------|----------|
| 1 | 68 | 481 | 2.1 | 141 |
| 16 | 68 | 1037 | 15.4 | 1049 |
| 64 | 68 | 1583 | 40.4 | 2749 |

Throughput rises steeply because batching amortizes the fixed per-request cost — network, queueing, kernel launch — and lets the GPU process all passages in one forward pass.

### Concurrency — `d5_concurrency_results.txt`

| C | req/s | passages/s | lat p50 | 429s |
|---|-------|------------|---------|------|
| 1 | 1.17 | 18.7 | 994 ms | 0 |
| 2 | 1.90 | 30.5 | 968 ms | 0 |
| 4 | 5.67 | 90.8 | 448 ms | 0 |

Near-linear scaling, zero 429s, and p50 latency *dropping* from ~994 ms to ~448 ms at C=4. At C=4 the service delivered 90.8 passages/s — well above the ~40 passages/s ceiling of a single maximally-batched request — so **single-request saturation is not service saturation**: the engine batches across concurrent requests. Spare capacity remains beyond C=4; the client, not the server, was the bottleneck.

### Drift

Runs on different days show the same pattern with different absolute numbers (hosted free tier is shared). Pattern is the finding; absolute values are dated.

## Run

```
export NVIDIA_API_KEY=...    # never commit keys
python3 nim_step4_batch.py
python3 nim_step5_concurrency.py
```

## Next

- Keep doubling C until throughput flattens — the saturation knee

## NVFP4 vs BF16 on Blackwell (Modal B200, vLLM offline, 2026-10-20)

| N | BF16 (passages/s) | NVFP4 (passages/s) | ratio |
|---|-------------------|--------------------|-------|
| 1 | 118.3 | 70.3 | 0.59x |
| 16 | 723.1 | 560.2 | 0.77x |
| 64 | 1902.9 | 1743.1 | 0.92x |
| 256 | 2615.8 | 2657.6 | 1.02x |

No 2x at 1B + short ISL: the ratio climbs monotonically with batch size (0.59x at N=1 to 1.02x eager; up to 1.21x at ISL~1024 with CUDA graphs on). The blog's "up to 2x" sentence specifies no engine, ISL, or batch; on vLLM we could not reach it for a dense 1B encoder. Community measurements put large FP4 gains behind tuned kernels (SGLang 2.2-3.5x on MoE) and weight-bound workloads, while vLLM's generic CUTLASS path caps near 1.2x for dense models and goes sub-1x at batch 1.
