#!/usr/bin/env python3
"""Does batching corrupt output? Greedy decoding must be deterministic.

At temperature 0 a prompt has exactly one correct continuation. Run each
prompt alone, then run all of them together in one batch, and compare byte
for byte. If the runner leaks state between concurrent slots -- the class of
bug the blocklist was guarding against -- the batched answers diverge from
the solo ones, or answer the wrong question entirely.
"""
import json, sys, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11435"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.5:0.8b"

PROMPTS = [
    "List the first 8 prime numbers, comma separated. Answer only.",
    "Name the capital cities of France, Japan, Brazil and Egypt. Answer only.",
    "What is 17 * 23? Reply with only the number.",
    "Write the Greek alphabet's first 6 letters in order, comma separated. Answer only.",
]

def ask(prompt, barrier=None):
    body = json.dumps({
        "model": MODEL, "stream": False, "think": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_ctx": 4096, "num_predict": 120, "temperature": 0,
                    "top_k": 1, "seed": 42},
    }).encode()
    req = urllib.request.Request(URL + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    if barrier:
        barrier.wait(timeout=60)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"].strip()

print(f"model={MODEL}  url={URL}\n")
print("=== pass 1: each prompt alone (sequential, one slot busy) ===")
solo = [ask(p) for p in PROMPTS]
for i, (p, a) in enumerate(zip(PROMPTS, solo)):
    print(f"  [{i}] {a[:72]!r}")

print("\n=== pass 2: all prompts at once (batched across slots) ===")
bar = threading.Barrier(len(PROMPTS))
with ThreadPoolExecutor(max_workers=len(PROMPTS)) as ex:
    batched = list(ex.map(lambda p: ask(p, bar), PROMPTS))
for i, a in enumerate(batched):
    print(f"  [{i}] {a[:72]!r}")

print("\n=== comparison ===")
bad = 0
for i, (s, b) in enumerate(zip(solo, batched)):
    if s == b:
        print(f"  [{i}] IDENTICAL")
    else:
        bad += 1
        print(f"  [{i}] *** DIVERGED ***\n        solo:    {s[:100]!r}\n        batched: {b[:100]!r}")
print()
if bad:
    print(f"FAIL: {bad}/{len(PROMPTS)} answers changed under batching -- "
          "concurrent decoding is not safe on this build.")
    sys.exit(1)
print(f"PASS: all {len(PROMPTS)} answers byte-identical alone and batched.")
