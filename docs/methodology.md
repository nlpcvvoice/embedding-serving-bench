# Methodology

## Design axes

| Axis | Knob | How it is isolated |
|------|------|--------------------|
| Input sequence length (ISL) | passage text length | same paragraph repeated, only token count varies |
| Batch size (N) | passages per request | one request carries 1 / 16 / 64 / 256 passages |
| Concurrency (C) | simultaneous requests | client threads hammering the endpoint |
| Precision | BF16 vs NVFP4 checkpoint | same engine, same hardware, same input set |

## Measurement rules

| Rule | Reason |
|------|--------|
| Medians over interleaved rounds | hosted endpoints drift day to day |
| One variable per experiment | engine, hardware, prompt set held fixed |
| Raw output archived to `data/` | every table in the README traces to a file |
| Rate-limit headers captured | 429 vs saturation must be distinguished |
| Offline batch vs online serve reported separately | they answer different claims (throughput ceiling vs serving efficiency) |

## Regime map (community + ours, NVFP4 vs BF16 on Blackwell)

| Regime | Ratio | Source |
|--------|-------|--------|
| 1B encoder, offline batch, eager, short ISL | 0.59x - 1.02x | ours |
| 1B encoder, offline batch, CUDA graphs, ISL~1024 | up to 1.21x | ours |
| 32B generative, online serve, concurrency 8-64 | 1.92x - 2.10x | Jarvislabs |
| 32B generative, online serve, unlimited | 2.41x - 2.50x | Jarvislabs |
| 20B MoE, offline, tuned kernels (SGLang), large batch | 2.23x - 3.54x | apsys |
| 20B MoE, offline, vLLM, batch 1 | 0.86x | apsys |

Reading: FP4 gains grow with GEMM fatness and occupancy; they shrink at batch 1. The blog's "up to 2x" sits at the fat-and-occupied end; our 1B offline runs sit at the other end.
