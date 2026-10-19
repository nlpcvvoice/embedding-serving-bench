import json
import os
import time
import urllib.request

KEY = os.environ["NVIDIA_API_KEY"]  # never hardcode keys
OUT = "nim_step1_response.json"

TEXTS = [
    "An embedding inference engine written in Rust measures throughput in tokens per second on a GPU.",
    "Dense vectors let a semantic search engine find passages by meaning, not keywords.",
    "稍微替换一下A chocolate cake with vanilla buttercream and fresh strawberries.",
]

body = json.dumps({
    "model": "nvidia/nemotron-3-embed-1b",
    "input": TEXTS,
    "input_type": "passage",
    "encoding_format": "float",
}).encode()

req = urllib.request.Request(
    "https://integrate.api.nvidia.com/v1/embeddings",
    data=body,
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
)

t0 = time.perf_counter()
with urllib.request.urlopen(req) as r:
    print("HTTP status:", r.status)
    for h in ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
              "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens"):
        v = r.headers.get(h)
        if v:
            print(f"  {h}: {v}")
    data = json.load(r)
wall = time.perf_counter() - t0
print(f"wall time (3 texts, 1 request): {wall*1000:.0f} ms")

json.dump(data, open(OUT, "w"))

vecs = [d["embedding"] for d in data["data"]]
print(f"num vectors: {len(vecs)}, dim: {len(vecs[0])}")
print("first 5 dims of vector 1:", [round(x, 4) for x in vecs[0][:5]])


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


print()
print("COSINE SIMILARITY (1.0 = identical meaning, ~0 = unrelated):")
print(f"  related  pair  (engine-in-Rust  vs  semantic-search): {cos(vecs[0], vecs[1]):.4f}")
print(f"  unrelated pair (engine-in-Rust  vs  chocolate-cake):  {cos(vecs[0], vecs[2]):.4f}")
