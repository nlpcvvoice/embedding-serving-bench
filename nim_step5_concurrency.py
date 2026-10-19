import os
import json
import time
import urllib.request
import urllib.error
import threading

KEY = os.environ["NVIDIA_API_KEY"]  # never hardcode keys
URL = "https://integrate.api.nvidia.com/v1/embeddings"
N = 16
R = 6

PASSAGE = ("Retrieval augmented generation systems depend on embedding models to convert "
           "passages into dense vectors. A serving engine must process these passages at "
           "high throughput while keeping per request latency low. Benchmarking across "
           "input sequence lengths reveals where the engine saturates. Quantized variants "
           "such as NVFP4 target even higher throughput on modern GPU architectures.")


def one_request(results, idx):
    body = json.dumps({
        "model": "nvidia/nemotron-3-embed-1b",
        "input": [PASSAGE] * N,
        "input_type": "passage",
        "encoding_format": "float",
    }).encode()
    req = urllib.request.Request(
        URL, data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as r:
            r.read()
        results[idx] = ("ok", (time.perf_counter() - t0) * 1000)
    except urllib.error.HTTPError as e:
        results[idx] = (str(e.code), (time.perf_counter() - t0) * 1000)


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


def worker(base):
    for r in range(R):
        one_request(results, base + r)


for C in (1, 2, 4):
    results = [None] * (C * R)
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(w * R,)) for w in range(C)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [ms for status, ms in results if status == "ok"]
    codes = {}
    for status, ms in results:
        codes[status] = codes.get(status, 0) + 1
    req_s = len(ok) / wall
    print(f"C={C} | reqs={C * R} | ok={len(ok)} | {codes} | wall={wall:5.1f}s | "
          f"req/s={req_s:4.2f} | passages/s={req_s * N:5.1f} | "
          f"lat p50={pct(ok, 0.5):5.0f} p95={pct(ok, 0.95):5.0f} ms")
    time.sleep(3)
