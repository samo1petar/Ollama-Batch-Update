#!/usr/bin/env python3
"""
ollama_batch_probe.py — does this Ollama server actually batch concurrent requests?

Standard library only. No pip install. Python 3.8+.

WHAT IT ANSWERS
---------------
When N requests hit an Ollama server at the same moment, the server either

  (a) BATCHES them   — all N decode together in one forward pass per step.
                       Aggregate tokens/sec goes UP with N; each individual
                       request gets a bit slower. This is what you want.

  (b) SERIALIZES them — request 2 waits for request 1 to finish. Aggregate
                       tokens/sec stays FLAT no matter how many you send.
                       Wall-clock latency grows linearly with N.

Ollama serializes when it loads the model with only one parallel slot. The
usual causes, in order of how often they bite:

  1. OLLAMA_NUM_PARALLEL=1 set explicitly on the server.
  2. OLLAMA_NUM_PARALLEL unset (=auto) AND a large options.num_ctx: the
     scheduler multiplies num_ctx by the slot count and, when that KV cache
     will not fit alongside the weights, silently falls back to 1 slot.
     A 64k num_ctx is more than enough to trigger this on a big model.
  3. A reverse proxy / gateway in front of Ollama that serializes upstream
     requests, so Ollama never even sees them concurrently.

This script distinguishes all three.

HOW IT DECIDES (the core metric)
--------------------------------
Every Ollama response carries server-side timers in nanoseconds:
``prompt_eval_duration`` + ``eval_duration`` = time the server was actually
busy computing that request. Sum that across the N concurrent requests and
divide by the wall-clock time the whole batch took:

    overlap = sum(server_busy_per_request) / wall_clock

  * Serialized -> each request is busy only while the others idle, so the
    busy times add up to roughly the wall clock. overlap ~= 1, for any N.
  * Batched    -> all N requests are busy across the same window, so the
    busy times add up to roughly N * wall clock. overlap ~= N.

This works whether or not you stream, and it cannot be faked by network
latency. Wall-clock speedup of aggregate throughput is reported alongside it
as an independent confirmation.

In streaming mode the script also records time-to-first-token per request.
Under hard serialization the TTFTs come out staircased (0s, 4s, 8s, 12s);
under batching they all land within a few hundred ms of each other. That
staircase is the signature of cause 3 (a serializing proxy) as much as of
cause 1, so it is reported but not used for the verdict.

USAGE
-----
  # simplest: is my server batching at all?
  python3 ollama_batch_probe.py --model qwen3.5:0.8b

  # reproduce what the application actually sends (64k context!)
  python3 ollama_batch_probe.py --model qwen3.5:27b --num-ctx 65536

  # the money shot: does a big num_ctx kill batching on this box?
  python3 ollama_batch_probe.py --model qwen3.5:27b --ctx-sweep 4096,32768,65536

  # behind an authenticating proxy
  OLLAMA_TOKEN=xxx python3 ollama_batch_probe.py --url https://gpu.example.com

Exit status is 0 if batching was detected, 1 if the server serialized.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import string
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

NS = 1_000_000_000.0

# Verdict thresholds, expressed as a fraction of the requested concurrency.
BATCHING_MIN = 0.55   # normalized overlap at or above C*0.55 -> batching
SERIAL_MAX = 1.35     # normalized overlap at or below this  -> serialized

# Architectures hard-coded in Ollama's scheduler to run with exactly one slot.
# From server/sched.go on main (checked 2026-09-15):
#
#   if slices.Contains([]string{"mllama", "qwen3vl", "qwen3vlmoe", "qwen35",
#       "qwen35moe", "qwen3next", "lfm2", "lfm2moe", "nemotron_h",
#       "nemotron_h_moe", "nemotron_h_omni"}, req.model.Config.ModelFamily)
#       && numParallel != 1 {
#       numParallel = 1
#       slog.Warn("model architecture does not currently support parallel requests", ...)
#   }
#
# It is a static blocklist, not a capability probe: OLLAMA_NUM_PARALLEL cannot
# override it, and no amount of memory changes it. Tracking issue ollama#4165
# (open since May 2024); ollama#17144 proposes dropping qwen35/qwen35moe.
ARCH_REFUSES_PARALLEL = {
    a: "Ollama's scheduler blocklist in server/sched.go pins this architecture "
       "to n_slots=1; OLLAMA_NUM_PARALLEL is ignored for it."
    for a in (
        "mllama", "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "qwen3next",
        "lfm2", "lfm2moe", "nemotron_h", "nemotron_h_moe", "nemotron_h_omni",
    )
}

# Not blocklisted -- the scheduler grants these parallel slots -- but the
# runner behind them executes requests one at a time anyway, so the server
# reports runner.parallel=N while behaving exactly like n_slots=1. Continuous
# batching for the MLX engine is proposed in ollama#17317, unmerged.
ARCH_IGNORES_PARALLEL = {
    a: "this is an MLX-engine model; the MLX runner accepts parallel slots "
       "(runner.parallel=N) but still decodes requests serially."
    for a in ("qwen3_5", "qwen3_5_moe")
}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _headers(token: str) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def http_get(url: str, token: str, timeout: float = 15.0) -> Any:
    req = urllib.request.Request(url, headers=_headers(token), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class Result:
    """One request's measurements, on a clock shared with its siblings."""

    __slots__ = ("idx", "t_start", "t_ttft", "t_end", "timings", "text", "error",
                 "done_reason")

    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.t_start = 0.0
        self.t_ttft: Optional[float] = None
        self.t_end = 0.0
        self.timings: Dict[str, int] = {}
        self.text = ""
        self.done_reason = ""
        self.error: Optional[str] = None

    # server-side compute time for this request, in seconds
    @property
    def busy_s(self) -> float:
        t = self.timings
        return (t.get("prompt_eval_duration", 0) + t.get("eval_duration", 0)) / NS

    @property
    def eval_count(self) -> int:
        return self.timings.get("eval_count", 0)

    @property
    def prompt_eval_count(self) -> int:
        return self.timings.get("prompt_eval_count", 0)

    @property
    def load_s(self) -> float:
        return self.timings.get("load_duration", 0) / NS

    @property
    def wall_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def decode_tps(self) -> float:
        """This request's own generation rate, server-side."""
        ed = self.timings.get("eval_duration", 0)
        return (self.eval_count / (ed / NS)) if ed else 0.0


def one_request(
    idx: int,
    prompt: str,
    *,
    url: str,
    token: str,
    model: str,
    endpoint: str,
    num_ctx: Optional[int],
    num_predict: int,
    stream: bool,
    think: bool,
    timeout: float,
    barrier: Optional[threading.Barrier],
    t_zero: float,
) -> Result:
    """Issue one request. Blocks on `barrier` so siblings depart together."""
    res = Result(idx)

    options: Dict[str, Any] = {"num_predict": num_predict, "temperature": 0.0}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    if endpoint == "chat":
        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "think": think,
            "options": options,
        }
        path = "/api/chat"
    else:
        body = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": options,
        }
        path = "/api/generate"

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + path, data=data, headers=_headers(token), method="POST"
    )

    if barrier is not None:
        try:
            barrier.wait(timeout=60)
        except threading.BrokenBarrierError:
            pass

    res.t_start = time.perf_counter() - t_zero
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if stream:
                chunks: List[str] = []
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    piece = (
                        obj.get("message", {}).get("content", "")
                        if endpoint == "chat"
                        else obj.get("response", "")
                    )
                    if piece and res.t_ttft is None:
                        res.t_ttft = time.perf_counter() - t_zero
                    if piece:
                        chunks.append(piece)
                    if obj.get("done"):
                        res.timings = {
                            k: v for k, v in obj.items() if isinstance(v, int)
                        }
                        res.done_reason = obj.get("done_reason", "")
                res.text = "".join(chunks)
            else:
                obj = json.loads(resp.read().decode("utf-8"))
                res.timings = {k: v for k, v in obj.items() if isinstance(v, int)}
                res.done_reason = obj.get("done_reason", "")
                res.text = (
                    obj.get("message", {}).get("content", "")
                    if endpoint == "chat"
                    else obj.get("response", "")
                )
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:300]
        except Exception:
            pass
        res.error = f"HTTP {e.code}: {detail or e.reason}"
    except Exception as e:  # noqa: BLE001 - surface anything to the report
        res.error = f"{type(e).__name__}: {e}"

    res.t_end = time.perf_counter() - t_zero
    return res


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

def make_prompts(n: int, target_words: int, seed: int) -> List[str]:
    """`n` prompts of near-identical length, each with a unique prefix.

    Two things matter here.

    The unique prefix defeats Ollama's prompt cache. Identical prompts let
    later requests skip prompt evaluation entirely, which would inflate the
    apparent speedup and hide serialization.

    The task is an open-ended essay because the measurement needs every
    request to generate the SAME number of tokens -- num_predict of them --
    so the concurrency levels are comparable. Short or ragged generations
    make the timing noise larger than the effect being measured. A model
    asked to count, or to answer something factual, emits an end-of-sequence
    token after a few dozen tokens; asked for a long essay it runs until it
    hits the num_predict ceiling, which is what we want (done_reason
    "length").
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        tag = "".join(rng.choice(string.ascii_lowercase) for _ in range(12))
        filler = " ".join(
            "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
            for _ in range(target_words)
        )
        out.append(
            f"Reference id {tag}. Ignore this lookup key: {filler}\n\n"
            "Write a detailed, exhaustive technical essay on the history of "
            "maritime navigation. Cover dead reckoning, celestial navigation "
            "and the sextant, the marine chronometer and the longitude "
            "problem, radio navigation, inertial systems, and satellite "
            "constellations. Write continuous prose across many paragraphs, "
            "with specific dates, names and technical detail throughout. "
            "Be as thorough as possible and do not stop early."
        )
    return out


# --------------------------------------------------------------------------
# One concurrency level
# --------------------------------------------------------------------------

def run_level(concurrency: int, prompts: List[str], cfg: argparse.Namespace) -> Dict[str, Any]:
    barrier = threading.Barrier(concurrency) if concurrency > 1 else None
    t_zero = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(
                one_request,
                i,
                prompts[i],
                url=cfg.url,
                token=cfg.token,
                model=cfg.model,
                endpoint=cfg.endpoint,
                num_ctx=cfg.num_ctx,
                num_predict=cfg.num_predict,
                stream=cfg.stream,
                think=cfg.think,
                timeout=cfg.timeout,
                barrier=barrier,
                t_zero=t_zero,
            )
            for i in range(concurrency)
        ]
        results = [f.result() for f in futures]

    errors = [r.error for r in results if r.error]
    ok = [r for r in results if not r.error]
    if not ok:
        return {"concurrency": concurrency, "errors": errors, "ok": 0}

    wall = max(r.t_end for r in ok) - min(r.t_start for r in ok)
    busy_total = sum(r.busy_s for r in ok)
    gen_tokens = sum(r.eval_count for r in ok)

    ttfts = [r.t_ttft for r in ok if r.t_ttft is not None]

    return {
        "concurrency": concurrency,
        "ok": len(ok),
        "errors": errors,
        "wall_s": wall,
        "busy_total_s": busy_total,
        "overlap_raw": (busy_total / wall) if wall > 0 else 0.0,
        "gen_tokens": gen_tokens,
        "agg_decode_tps": (gen_tokens / wall) if wall > 0 else 0.0,
        "per_req_decode_tps": statistics.mean(r.decode_tps for r in ok),
        "mean_latency_s": statistics.mean(r.wall_s for r in ok),
        "p_eval_tokens": statistics.mean(r.prompt_eval_count for r in ok),
        "eval_tokens_mean": statistics.mean(r.eval_count for r in ok),
        "load_s_max": max(r.load_s for r in ok),
        "hit_token_ceiling": all(r.done_reason == "length" for r in ok),
        "ttft_min": min(ttfts) if ttfts else None,
        "ttft_max": max(ttfts) if ttfts else None,
        "ttft_spread_s": (max(ttfts) - min(ttfts)) if len(ttfts) > 1 else None,
        "ttfts": sorted(ttfts) if ttfts else [],
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def show_model(cfg: argparse.Namespace, model: str) -> Dict[str, Any]:
    """POST /api/show -- the only way to read a model's architecture over HTTP.

    Matters because Ollama's decision to batch is per-architecture, and a
    client inside a container cannot read the server's logs to see it.
    """
    body = json.dumps({"model": model}).encode("utf-8")
    req = urllib.request.Request(
        cfg.url.rstrip("/") + "/api/show", data=body,
        headers=_headers(cfg.token), method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def arch_of(show: Dict[str, Any]) -> str:
    return (show.get("model_info", {}) or {}).get("general.architecture", "") or ""


def server_info(cfg: argparse.Namespace) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    for key, path in (("version", "/api/version"), ("ps", "/api/ps")):
        try:
            info[key] = http_get(cfg.url.rstrip("/") + path, cfg.token)
        except Exception as e:  # noqa: BLE001
            info[key] = {"error": f"{type(e).__name__}: {e}"}
    return info


def describe_ps(ps: Dict[str, Any]) -> List[str]:
    """Render /api/ps, highlighting the loaded context size.

    `context_length` here is what the server ACTUALLY allocated at load
    time. Ollama sizes it as (per-request num_ctx x parallel slots), so a
    value that is an exact multiple of your requested num_ctx tells you the
    slot count directly -- e.g. num_ctx=8192 and context_length=32768 means
    4 slots. Equal values mean a single slot: no batching.
    """
    lines = []
    for m in ps.get("models", []) or []:
        name = m.get("name") or m.get("model") or "?"
        ctx = m.get("context_length")
        size = m.get("size")
        vram = m.get("size_vram")
        bits = [f"    - {name}"]
        if ctx:
            bits.append(f"context_length={ctx}")
        if size:
            bits.append(f"size={size / 2**30:.2f}GiB")
        if vram:
            bits.append(f"vram={vram / 2**30:.2f}GiB")
        lines.append("  ".join(bits))
    if not lines:
        lines.append("    (no model resident)")
    return lines


def verdict_for(levels: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = next((l for l in levels if l.get("concurrency") == 1 and l.get("ok")), None)
    top = max(
        (l for l in levels if l.get("ok")),
        key=lambda l: l["concurrency"],
        default=None,
    )
    if not base or not top or top["concurrency"] == 1:
        return {"verdict": "INCONCLUSIVE", "reason": "need at least C=1 and C>1 to succeed"}

    c = top["concurrency"]
    # Normalize by the C=1 overlap so network/HTTP overhead cancels out.
    norm = top["overlap_raw"] / base["overlap_raw"] if base["overlap_raw"] else 0.0
    speedup = top["agg_decode_tps"] / base["agg_decode_tps"] if base["agg_decode_tps"] else 0.0

    if norm >= BATCHING_MIN * c:
        v = "BATCHING"
    elif norm <= SERIAL_MAX:
        v = "SERIALIZED"
    else:
        v = "PARTIAL"
    return {
        "verdict": v,
        "at_concurrency": c,
        "effective_concurrency": norm,
        "throughput_speedup": speedup,
    }


def print_table(levels: List[Dict[str, Any]], base_overlap: float) -> None:
    hdr = (
        f"{'C':>3}  {'wall s':>7}  {'agg tok/s':>10}  {'per-req tok/s':>13}  "
        f"{'eff.conc':>8}  {'speedup':>7}  {'TTFT spread':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    base_tps = next((l["agg_decode_tps"] for l in levels if l["concurrency"] == 1 and l.get("ok")), 0.0)
    for l in levels:
        if not l.get("ok"):
            print(f"{l['concurrency']:>3}  FAILED: {l.get('errors', ['?'])[0][:60]}")
            continue
        eff = l["overlap_raw"] / base_overlap if base_overlap else 0.0
        spd = l["agg_decode_tps"] / base_tps if base_tps else 0.0
        spread = l["ttft_spread_s"]
        spread_s = f"{spread:>10.2f}s" if spread is not None else f"{'-':>11}"
        print(
            f"{l['concurrency']:>3}  {l['wall_s']:>7.2f}  {l['agg_decode_tps']:>10.1f}  "
            f"{l['per_req_decode_tps']:>13.1f}  {eff:>8.2f}  {spd:>6.2f}x  {spread_s}"
        )


def explain(v: Dict[str, Any], levels: List[Dict[str, Any]], cfg: argparse.Namespace) -> None:
    print()
    print("=" * 78)
    name = v.get("verdict")
    if name == "BATCHING":
        print(f"VERDICT: BATCHING IS ACTIVE")
        print(
            f"  At concurrency {v['at_concurrency']} the server ran "
            f"{v['effective_concurrency']:.2f} requests' worth of compute at once\n"
            f"  and delivered {v['throughput_speedup']:.2f}x the aggregate token throughput of a\n"
            f"  single request. Concurrent requests are sharing forward passes."
        )
    elif name == "SERIALIZED":
        print(f"VERDICT: NO BATCHING - REQUESTS ARE BEING SERIALIZED")
        print(
            f"  At concurrency {v['at_concurrency']} the server still only did "
            f"{v['effective_concurrency']:.2f} requests'\n"
            f"  worth of work at a time, and aggregate throughput moved by just "
            f"{v['throughput_speedup']:.2f}x.\n"
            f"  Sending more requests at once is buying you nothing."
        )
        print()
        arch = getattr(cfg, "arch", "")
        if arch in ARCH_REFUSES_PARALLEL or arch in ARCH_IGNORES_PARALLEL:
            why = ARCH_REFUSES_PARALLEL.get(arch) or ARCH_IGNORES_PARALLEL[arch]
            print(f"  MOST LIKELY CAUSE: the model architecture ({arch}).")
            print(f"  {why}")
            print("  This is a property of the model plus the Ollama build, NOT of your")
            print("  configuration: raising OLLAMA_NUM_PARALLEL will not change it, and")
            print("  neither will anything you do from the client side. Confirm by running")
            print("  this same probe with --control-model against a different architecture;")
            print("  if the control batches, the server is fine and the model is the limit.")
            print()
        print("  Ask the server admin to check, in this order:")
        print("    1. OLLAMA_NUM_PARALLEL on the ollama serve process.")
        print("       `launchctl getenv OLLAMA_NUM_PARALLEL`, or the Ollama.app")
        print("       settings, or the systemd unit. If it is 1, that is the cause.")
        print("       Set it to 4 (or unset it and shrink num_ctx, see 2).")
        print(f"    2. num_ctx. This run requested num_ctx={cfg.num_ctx}. Ollama reserves")
        print("       num_ctx x slots of KV cache up front; when that does not fit it")
        print("       drops silently to ONE slot. Re-run with --ctx-sweep to see the")
        print("       cliff, and compare context_length in /api/ps above.")
        print("    3. A reverse proxy in front of Ollama serializing upstream requests.")
        if cfg.stream:
            top = max((l for l in levels if l.get("ok")), key=lambda l: l["concurrency"])
            if top.get("ttfts") and len(top["ttfts"]) > 1:
                st = ", ".join(f"{t:.1f}s" for t in top["ttfts"])
                print(f"       First-token times this run: {st}")
                print("       Evenly spaced and growing => hard serialization (proxy or 1 slot).")
    elif name == "PARTIAL":
        print(f"VERDICT: PARTIAL BATCHING")
        print(
            f"  Effective concurrency {v['effective_concurrency']:.2f} out of "
            f"{v['at_concurrency']} requested, for {v['throughput_speedup']:.2f}x throughput.\n"
            f"  The server has more than one slot but fewer than you asked for --\n"
            f"  typically OLLAMA_NUM_PARALLEL set to a value below your concurrency,\n"
            f"  or memory-bandwidth saturation. Raise OLLAMA_NUM_PARALLEL and re-run."
        )
    else:
        print(f"VERDICT: {name} - {v.get('reason', '')}")
    print("=" * 78)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def unload(cfg: argparse.Namespace) -> None:
    """Evict the model so the next request reloads it with a fresh num_ctx."""
    body = json.dumps({"model": cfg.model, "keep_alive": 0}).encode("utf-8")
    req = urllib.request.Request(
        cfg.url.rstrip("/") + "/api/generate",
        data=body,
        headers=_headers(cfg.token),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
    except Exception as e:  # noqa: BLE001
        print(f"  (unload failed, continuing: {e})", file=sys.stderr)
    time.sleep(2.0)


def warmup(cfg: argparse.Namespace) -> None:
    print(f"  warming up {cfg.model} (loading weights, allocating KV cache)...")
    t0 = time.perf_counter()
    r = one_request(
        -1,
        "Reply with the single word: ready",
        url=cfg.url,
        token=cfg.token,
        model=cfg.model,
        endpoint=cfg.endpoint,
        num_ctx=cfg.num_ctx,
        num_predict=8,
        stream=False,
        think=cfg.think,
        timeout=cfg.timeout,
        barrier=None,
        t_zero=time.perf_counter(),
    )
    if r.error:
        print(f"\nFATAL: warmup request failed: {r.error}", file=sys.stderr)
        sys.exit(2)
    print(f"  ready in {time.perf_counter() - t0:.1f}s (model load {r.load_s:.1f}s)")


def run_suite(cfg: argparse.Namespace) -> Dict[str, Any]:
    levels: List[Dict[str, Any]] = []
    for c in cfg.concurrency:
        prompts = make_prompts(c, cfg.prompt_words, seed=1000 + c)
        print(f"  concurrency {c:>2} ... ", end="", flush=True)
        lvl = run_level(c, prompts, cfg)
        if lvl.get("ok"):
            print(
                f"{lvl['wall_s']:.2f}s wall, {lvl['gen_tokens']} tokens, "
                f"{lvl['agg_decode_tps']:.1f} tok/s aggregate"
            )
            if not lvl.get("hit_token_ceiling"):
                print(
                    f"      ! generations stopped before num_predict "
                    f"({lvl['eval_tokens_mean']:.0f} of {cfg.num_predict} tokens); "
                    "timings will be noisy -- raise --num-predict"
                )
            if lvl["load_s_max"] > 0.5:
                print(
                    f"      ! model reloaded mid-test ({lvl['load_s_max']:.1f}s); "
                    "result may be skewed"
                )
        else:
            print(f"all {c} requests failed: {lvl.get('errors', ['?'])[0][:80]}")
        levels.append(lvl)
        time.sleep(cfg.settle)
    return {"levels": levels}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Detect whether an Ollama server batches concurrent requests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    p.add_argument("--token", default=os.environ.get("OLLAMA_TOKEN", ""),
                   help="Bearer token, if Ollama sits behind an auth proxy")
    p.add_argument("--model", default=os.environ.get("QWEN_MODEL", "qwen3.5:0.8b"))
    p.add_argument("--concurrency", default="1,2,4,8",
                   help="comma-separated concurrency levels (default 1,2,4,8)")
    p.add_argument("--num-ctx", type=int, default=8192,
                   help="options.num_ctx per request; use the value your app sends")
    p.add_argument("--num-predict", type=int, default=256,
                   help="tokens to generate per request")
    p.add_argument("--prompt-words", type=int, default=120,
                   help="filler words per prompt, to give prompt_eval something to do")
    p.add_argument("--endpoint", choices=("chat", "generate"), default="chat")
    p.add_argument("--no-stream", dest="stream", action="store_false", default=True,
                   help="use non-streaming requests (matches an app that sets stream:false)")
    p.add_argument("--think", action="store_true", default=False,
                   help="leave model thinking enabled (default: disabled, like the app)")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--settle", type=float, default=3.0,
                   help="seconds to idle between concurrency levels")
    p.add_argument("--ctx-sweep", default="",
                   help="comma-separated num_ctx values; runs the whole suite at each, "
                        "unloading the model in between, to find the context size at "
                        "which batching collapses")
    p.add_argument("--control-model", default="",
                   help="if the target model does not batch, re-run the suite with "
                        "this model to prove whether the SERVER can batch at all "
                        "(pick a different architecture, e.g. qwen2.5:0.5b)")
    p.add_argument("--json", default="", help="write full results to this JSON file")
    cfg = p.parse_args()

    cfg.concurrency = sorted({int(x) for x in cfg.concurrency.split(",") if x.strip()})
    if 1 not in cfg.concurrency:
        cfg.concurrency = [1] + cfg.concurrency

    print("=" * 78)
    print("OLLAMA BATCHING PROBE")
    print("=" * 78)
    print(f"  url        {cfg.url}")
    print(f"  model      {cfg.model}")
    print(f"  endpoint   /api/{cfg.endpoint}   stream={cfg.stream}   think={cfg.think}")
    print(f"  num_ctx    {cfg.num_ctx}        num_predict={cfg.num_predict}")
    print(f"  auth       {'Bearer token supplied' if cfg.token else 'none'}")

    info = server_info(cfg)
    print(f"  server     ollama {info.get('version', {}).get('version', '?')}")

    show = show_model(cfg, cfg.model)
    arch = arch_of(show)
    caps = show.get("capabilities", [])
    print(f"  arch       {arch or '(unknown)'}   capabilities={caps}")
    if arch in ARCH_REFUSES_PARALLEL:
        print(f"  ! WARNING: {ARCH_REFUSES_PARALLEL[arch]}")
    elif arch in ARCH_IGNORES_PARALLEL:
        print(f"  ! WARNING: {ARCH_IGNORES_PARALLEL[arch]}")
    print()

    cfg.arch = arch

    sweeps = (
        [int(x) for x in cfg.ctx_sweep.split(",") if x.strip()]
        if cfg.ctx_sweep
        else [cfg.num_ctx]
    )

    all_runs = []
    exit_code = 0
    for ctx in sweeps:
        cfg.num_ctx = ctx
        print("-" * 78)
        print(f"RUN: num_ctx = {ctx}")
        print("-" * 78)
        if len(sweeps) > 1:
            print("  unloading model so it reloads at this context size...")
            unload(cfg)
        warmup(cfg)

        ps = server_info(cfg).get("ps", {})
        print("  resident after load:")
        for line in describe_ps(ps):
            print(line)
        print()

        suite = run_suite(cfg)
        levels = suite["levels"]
        base = next((l for l in levels if l["concurrency"] == 1 and l.get("ok")), None)
        print()
        if base:
            print_table(levels, base["overlap_raw"])
        v = verdict_for(levels)
        explain(v, levels, cfg)
        if v.get("verdict") != "BATCHING":
            exit_code = 1
        all_runs.append({"num_ctx": ctx, "ps": ps, "levels": levels, "verdict": v})
        print()

    if len(all_runs) > 1:
        print("=" * 78)
        print("CONTEXT SWEEP SUMMARY")
        print("=" * 78)
        print(f"{'num_ctx':>9}  {'loaded ctx':>11}  {'verdict':>11}  {'eff.conc':>8}  {'speedup':>7}")
        for r in all_runs:
            models = r["ps"].get("models", []) or []
            loaded = models[0].get("context_length", "?") if models else "?"
            v = r["verdict"]
            print(
                f"{r['num_ctx']:>9}  {str(loaded):>11}  {v.get('verdict', '?'):>11}  "
                f"{v.get('effective_concurrency', 0):>8.2f}  "
                f"{v.get('throughput_speedup', 0):>6.2f}x"
            )
        print()
        print("  A verdict that flips from BATCHING to SERIALIZED as num_ctx grows is")
        print("  the KV-cache cliff: Ollama could not fit num_ctx x slots, so it fell")
        print("  back to a single slot. Fix by lowering the app's num_ctx or raising")
        print("  the memory available to Ollama -- not by changing the client.")

    control = None
    if cfg.control_model and exit_code != 0:
        print("=" * 78)
        print(f"CONTROL RUN: {cfg.control_model}")
        print("=" * 78)
        print("  The target model did not batch. Running an identical suite against a")
        print("  second model to separate 'this server cannot batch' from 'this model")
        print("  cannot batch'.")
        print()
        target_model, cfg.model = cfg.model, cfg.control_model
        cshow = show_model(cfg, cfg.model)
        cfg.arch = arch_of(cshow)
        print(f"  arch       {cfg.arch or '(unknown)'}")
        unload(cfg)
        warmup(cfg)
        csuite = run_suite(cfg)
        clevels = csuite["levels"]
        cbase = next((l for l in clevels if l["concurrency"] == 1 and l.get("ok")), None)
        print()
        if cbase:
            print_table(clevels, cbase["overlap_raw"])
        cv = verdict_for(clevels)
        control = {"model": cfg.model, "arch": cfg.arch, "levels": clevels, "verdict": cv}
        print()
        print("=" * 78)
        if cv.get("verdict") == "BATCHING":
            print("CONCLUSION: THE SERVER IS FINE -- THE MODEL IS THE LIMIT")
            print(f"  {cfg.control_model} ({cfg.arch}) batched at "
                  f"{cv['effective_concurrency']:.2f}x effective concurrency on this")
            print(f"  same server, while {target_model} ({arch}) did not. Ollama is")
            print("  configured correctly and has the memory; it is declining to batch")
            print("  this specific architecture. Raising OLLAMA_NUM_PARALLEL will not")
            print("  help. The options are: run the model under a server that batches it")
            print("  (vLLM/SGLang), wait for Ollama to add support, or switch models.")
        else:
            print("CONCLUSION: THE SERVER ITSELF IS NOT BATCHING ANYTHING")
            print(f"  Neither {target_model} nor {cfg.control_model} batched. This points")
            print("  at the server or the network path, not at the model: check")
            print("  OLLAMA_NUM_PARALLEL, the KV-cache headroom at this num_ctx, and any")
            print("  reverse proxy between this client and Ollama.")
        print("=" * 78)
        cfg.model = target_model

    if cfg.json:
        with open(cfg.json, "w") as f:
            json.dump(
                {"config": {k: v for k, v in vars(cfg).items()}, "server": info,
                 "model_show": show, "runs": all_runs, "control": control},
                f,
                indent=2,
                default=str,
            )
        print(f"\n  full results written to {cfg.json}")

    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
