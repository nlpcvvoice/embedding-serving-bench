import os
import json
import time
import urllib.request

KEY = os.environ["NVIDIA_API_KEY"]  # never hardcode keys
URL = "https://integrate.api.nvidia.com/v1/embeddings"

PASSAGE = ("Retrieval augmented generation systems depend on embedding models to convert "
           "passages into dense vectors. A serving engine must process these passages at "
           "high throughput while keeping per request latency low. Benchmarking across "
           "input sequence lengths reveals where the engine saturates. Quantized variants "
           "such as NVFP4 target even higher throughput on modern GPU architectures.")


def call(n):
    body = json.dumps({
        "model": "nvidia/nemotron-3-embed-1b",
        "input": [PASSAGE] * n,
        "input_type": "passage",
        "encoding_format": "float",
    }).encode()
    req = urllib.request.Request(
        URL, data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        data = json.load(r)
    ms = (time.perf_counter() - t0) * 1000
    return ms, data["usage"]["prompt_tokens"]


lats = {1: [], 16: [], 64: []}
toks = {}
for r in range(10):  # interleaved rounds cancel drift
    for n in (1, 16, 64):
        ms, t = call(n)
        lats[n].append(ms)
        toks[n] = t

print(f"{'N':>3} | {'tok/req':>7} | {'tok/passage':>11} | {'median ms':>9} | {'passages/s':>10} | {'tokens/s':>8}")
for n in (1, 16, 64):
    s = sorted(lats[n])
    med = s[len(s) // 2]
    print(f"{n:3d} | {toks[n]:7d} | {toks[n] / n:11.0f} | {med:9.0f} | {n / (med / 1000):10.1f} | {toks[n] / (med / 1000):8.0f}")
