import os
import json
import time
import urllib.error
import urllib.request

KEY = os.environ["NVIDIA_API_KEY"]  # never hardcode keys
URL = "https://integrate.api.nvidia.com/v1/embeddings"

PAYLOAD = json.dumps({
    "model": "nvidia/nemotron-3-embed-1b",
    "input": [
        "An embedding inference engine written in Rust measures throughput in tokens per second on a GPU.",
        "Dense vectors let a semantic search engine find passages by meaning, not keywords.",
        "稍微替换一下A chocolate cake with vanilla buttercream and fresh strawberries.",
    ],
    "input_type": "passage",
    "encoding_format": "float",
}).encode()

ok = n429 = other = 0
lat = []
for i in range(30):
    req = urllib.request.Request(
        URL, data=PAYLOAD,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as r:
            body = r.read()
            ms = (time.perf_counter() - t0) * 1000
            lat.append(ms)
            print(f"[{i:2d}] HTTP {r.status}  {ms:6.0f} ms  {len(body):6d} bytes")
            ok += 1
    except urllib.error.HTTPError as e:
        ms = (time.perf_counter() - t0) * 1000
        detail = e.read()[:200]
        print(f"[{i:2d}] HTTP {e.code}  {ms:6.0f} ms  {detail}")
        if e.code == 429:
            n429 += 1
        else:
            other += 1

print()
print(f"summary: 200s={ok}  429s={n429}  other_errors={other}")
if lat:
    s = sorted(lat)
    print(f"latency ms over the 200s: min={s[0]:.0f}  median={s[len(s) // 2]:.0f}  max={s[-1]:.0f}")
