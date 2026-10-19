import os
import json
import time
import urllib.request

KEY = os.environ["NVIDIA_API_KEY"]  # never hardcode keys
URL = "https://integrate.api.nvidia.com/v1/embeddings"

# base paragraph ~53 words; repeated to build two controlled lengths
BASE = ("Retrieval augmented generation systems depend on embedding models to convert "
        "passages into dense vectors. A serving engine must process these passages at "
        "high throughput while keeping per request latency low. Benchmarking across "
        "input sequence lengths reveals where the engine saturates. Quantized variants "
        "such as NVFP4 target even higher throughput on modern GPU architectures.")

T_SHORT = (BASE + " ") * 4    # ~200 words -> ~270 tokens
T_LONG = (BASE + " ") * 16    # ~850 words -> ~1100 tokens


def call(text):
    body = json.dumps({
        "model": "nvidia/nemotron-3-embed-1b",
        "input": [text],
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


# probe first: ask the SERVER how many tokens it actually saw
ms, tok_s = call(T_SHORT)
ms, tok_l = call(T_LONG)
print(f"probe: short = {tok_s} tokens ({ms:.0f} ms) | long = {tok_l} tokens ({ms:.0f} ms)")

# 10 interleaved rounds (alternating short/long cancels time drift)
lat_s, lat_l = [], []
for r in range(10):
    ms, _ = call(T_SHORT)
    lat_s.append(ms)
    ms, _ = call(T_LONG)
    lat_l.append(ms)


def stats(name, toks, lats):
    s = sorted(lats)
    med = s[len(s) // 2]
    print(f"{name}: tokens={toks}  min={s[0]:.0f}  median={med:.0f}  max={s[-1]:.0f} ms  "
          f"| single-stream = {toks / (med / 1000):.0f} tokens/s")


print()
stats("short", tok_s, lat_s)
stats("long ", tok_l, lat_l)
