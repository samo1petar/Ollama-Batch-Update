# Ollama batching probe

A self-contained test that answers one question: **when N requests hit an
Ollama server at the same moment, does the server actually batch them, or does
it run them one after another?**

`ollama_batch_probe.py` is standard-library Python 3.8+. No `pip install`,
no dependencies, one file. It is safe to hand to a server admin as-is.

---

## TL;DR

**There is a fix, and it is verified working.** [PR #17144](https://github.com/ollama/ollama/pull/17144)
deletes `"qwen35", "qwen35moe"` from a blocklist in `server/sched.go`. Built and
measured on this MacBook against v0.34.0: qwen3.5 went from **1.03 effective
concurrency (serialized) to 7.79 at 8 slots**, aggregate throughput 170 -> 396
tok/s, with byte-identical greedy output and no crashes. See
[The fix, validated](#the-fix-validated) below for the exact recipe.

Without that patch, batching for **qwen3.5 is not happening, and it is not a
configuration mistake.** Two independent things block it, and they stack:

1. **`OLLAMA_NUM_PARALLEL` defaults to `1` in Ollama 0.32.5.** Not 4, not
   "auto-scale to fit". A stock `ollama serve` gives every model a single
   slot and serializes everything. This one the admin *can* fix.

2. **Ollama refuses to batch the `qwen35` architecture at all.** With
   `OLLAMA_NUM_PARALLEL=4` explicitly set, the server log says:

   ```
   WARN sched.go:510 msg="model architecture does not currently support
        parallel requests" architecture=qwen35
   srv load_model: initializing, n_slots = 1
   ```

   `runner.parallel=1` regardless of the environment variable. This one the
   admin **cannot** fix. `qwen3.5:27b` on the MacStudio is the same `qwen35`
   architecture as the `qwen3.5:0.8b` tested here.

The MLX build is not a way out either. `qwen3.5:0.8b-mlx` reports a *different*
architecture (`qwen3_5`), and the scheduler does grant it `runner.parallel=4` —
but the MLX runner still executes requests strictly one at a time. Measured
effective concurrency: **1.00**. This is the nastiest case, because every
server-side indicator says parallelism is enabled while throughput says
otherwise. Only measurement catches it.

### Measurements

All on the same MacBook Pro (M-series, 24 GB), 4 concurrent requests,
512 generated tokens each, `num_ctx=4096`.

| Server config | Model | Arch | `n_slots` | Eff. concurrency @4 | Aggregate tok/s (1 → 4) | Verdict |
|---|---|---|---|---|---|---|
| `NUM_PARALLEL=1` | `qwen3.5:0.8b` | `qwen35` | 1 | 1.03 | 167 → 172 | serialized |
| `NUM_PARALLEL=4` | `qwen3.5:0.8b` | `qwen35` | **1** | 1.03 | 165 → 174 | serialized |
| `NUM_PARALLEL=4` | `qwen3.5:0.8b-mlx` | `qwen3_5` | 4 | **1.00** | 193 → 200 | serialized |
| `NUM_PARALLEL=4` | `qwen2.5:0.5b` | `qwen2` | 4 | **3.99** | 295 → 699 | **batching, 2.37x** |
| default (`=1`) | `qwen2.5:0.5b` | `qwen2` | 1 | 1.05 | 290 → 303 | serialized |

The `qwen2.5:0.5b` row is the control that makes the result conclusive: on the
*same server, same settings, same probe*, a different architecture batches at
3.99x effective concurrency and 2.37x throughput. The hardware, the memory and
the configuration are all fine. Ollama is declining to batch `qwen3.5`
specifically.

Note the signature of real batching in that row: aggregate throughput more than
doubles (295 → 699 tok/s) while each individual request gets *slower*
(318 → 192 tok/s). That trade is what batching is. When a server serializes,
neither number moves.

### This is a known, tracked Ollama limitation (confirmed 2026-09-15)

Not a local quirk and not a misconfiguration. It is a **static blocklist in
Ollama's scheduler source**, `server/sched.go` on `main`:

```go
if slices.Contains([]string{"mllama", "qwen3vl", "qwen3vlmoe", "qwen35",
    "qwen35moe", "qwen3next", "lfm2", "lfm2moe", "nemotron_h",
    "nemotron_h_moe", "nemotron_h_omni"}, req.model.Config.ModelFamily)
    && numParallel != 1 {
    numParallel = 1
    slog.Warn("model architecture does not currently support parallel requests", ...)
}
```

`qwen35` is on that list literally. It is a hard-coded list, not a capability
check, so `OLLAMA_NUM_PARALLEL` cannot override it and no amount of RAM changes
it. Verified present in the installed 0.32.5 binary itself: the concatenated Go
string blob contains `qwen35moeqwen3nex` and `qwen3vlmoenemotro`.

Issue trail, all reporting exactly our symptom:

- **[#14510](https://github.com/ollama/ollama/issues/14510)** — "qwen3.5 35b/27b only have 1 active request", `qwen35`/`qwen35moe`, since 0.17.4. Closed as duplicate.
- **[#14621](https://github.com/ollama/ollama/issues/14621)** — Qwen3.5:9b concurrent call bug; also a SIGABRT on Linux ARM64. Closed as duplicate.
- **[#14879](https://github.com/ollama/ollama/issues/14879)** — same thing on a **Mac Studio M3 Ultra with `OLLAMA_NUM_PARALLEL=8`**. Closed as duplicate.
- **[#4165](https://github.com/ollama/ollama/issues/4165)** — the root tracking issue, opened by a maintainer **May 2024**, still **open**.

**The whole modern Qwen line shares one backbone.** Checked against each
model's raw `config.json` on HuggingFace and the arch string ollama.com reports
for the packaged blob:

| Ollama model | HF `model_type` | Ollama arch | Blocklisted? | Batches? |
|---|---|---|---|---|
| `qwen3:32b` / `qwen3:30b` | `qwen3` / `qwen3_moe` | `qwen3` / `qwen3moe` | no | **yes** — ollama#14510 reports Parallel=8 |
| `qwen3.5:27b` | `qwen3_5` | `qwen35` | **yes** | no (measured: 1.03) |
| `qwen3.6:27b` | `qwen3_5` | `qwen35` | **yes** | no |
| `qwen3.8:27b` | `qwen3_5` | `qwen35` | **yes** | no |
| any `*-mlx` variant | `qwen3_5` | `qwen3_5` | no | no — MLX runner decodes serially (measured: 1.00) |

Qwen 3.5, 3.6 and 3.8 are all `Qwen3_5ForConditionalGeneration`. The version
bumps are weights, not backbone, so every one of them lands on `qwen35` and gets
pinned to a single slot. The last Qwen generation that batches under Ollama is
**qwen3** (3.0), which predates the hybrid attention redesign.

Note the trap in the last row: an MLX variant reports `qwen3_5`, which is *not*
on the blocklist, so the scheduler grants it `runner.parallel=N` and every
server-side indicator says parallelism is on. It still decodes serially.

**Why this family is blocklisted.** The list is not arbitrary. Qwen3.8-27B's
config shows 48 of its 64 layers are `linear_attention` with
`full_attention_interval: 4`, plus `linear_conv_kernel_dim` and
`mamba_ssm_dtype` — a hybrid attention/state-space backbone that carries a
constant-size **recurrent state per sequence**. Batching those requires
per-sequence state handling that the slot machinery did not have. Every entry
on the blocklist is a hybrid/SSM or multimodal family (`qwen35`, `qwen35moe`,
`qwen3next`, `lfm2`, `nemotron_h`, `mllama`, `qwen3vl`). Worth weighing before
patching it out: the restriction is principled, even if PR #17144 argues the
underlying crash is now fixed.

**A fix is proposed but not merged: [PR #17144](https://github.com/ollama/ollama/pull/17144)** removes `qwen35`/`qwen35moe` from the blocklist. Its rationale is that the upstream llama.cpp crash the blocklist existed to avoid was fixed in March 2026, and Ollama's vendored llama.cpp already carries the fix. Community testers report ~1.37x throughput on CUDA/ROCm/Vulkan. It deliberately keeps `qwen3vl`/`qwen3vlmoe` blocked. Open as of September 2026, no maintainer merge timeline.

The MLX result is also a known gap, not a bug on our side:
**[PR #17317](https://github.com/ollama/ollama/pull/17317)** adds continuous
batching to the MLX runner (a `--parallel` flag, `MultiSeq` caches, a
continuous-batch decode loop). Also open, last activity July–September 2026.
Until it lands, MLX models report `runner.parallel=N` and decode serially —
precisely the 1.00 effective concurrency measured above.

Upgrading alone will not fix it — **verified against v0.34.0's own source**,
whose blocklist is byte-for-byte the list above and still contains `qwen35`.
Latest is v0.34.1 (2026-09-14); no release note mentions the blocklist.

### The fix, validated

Built Ollama **v0.34.0** from source with PR #17144's one-line change applied,
ran it on :11435 against the stock app's prebuilt native libraries, and pointed
the same probe at it. Same machine, same model, same settings as the table
above.

| | eff. conc @4 | agg tok/s (1 -> 4) | verdict |
|---|---|---|---|
| stock v0.32.5, `NUM_PARALLEL=4` | 1.03 | 165 -> 174 | serialized |
| **patched v0.34.0, `NUM_PARALLEL=4`** | **3.93** | **171 -> 335** | **batching, 1.96x** |

Scaling with `OLLAMA_NUM_PARALLEL=8`, `qwen3.5:0.8b`, 512 tokens/request:

| C | wall s | agg tok/s | per-req tok/s | eff. conc | speedup |
|---|---|---|---|---|---|
| 1 | 3.02 | 169.8 | 178.4 | 1.00 | 1.00x |
| 2 | 3.98 | 257.2 | 141.1 | 1.99 | 1.51x |
| 4 | 5.98 | 342.5 | 91.5 | 3.93 | 2.02x |
| 8 | 10.35 | 395.7 | 53.2 | 7.79 | 2.33x |

Effective concurrency tracks the slot count almost perfectly (7.79 of 8), and
aggregate throughput saturates near 2.3x — memory-bandwidth bound on a laptop,
which is the expected shape. `n_slots = 8` in the log, `runner.parallel=8`, and
**zero** "does not currently support parallel requests" warnings.

**Correctness held.** The blocklist existed to avoid a crash, so throughput
alone is not enough. `correctness_check.py` runs four prompts solo, then all
four concurrently, at `temperature=0, top_k=1, seed=42` where greedy decoding
has exactly one right continuation. All four answers came back **byte-identical**
alone and batched — no cross-slot state leakage. Zero panics, SIGABRT or
SIGSEGV across every run.

#### Recipe for the admin

```bash
git clone --depth 1 --branch v0.34.0 https://github.com/ollama/ollama.git
cd ollama
# the entire fix: drop the two qwen35 entries from the blocklist
git apply /path/to/this/repo/patch/0001-enable-parallel-requests-for-qwen35.patch
git diff --stat            # must show: server/sched.go | 2 +-

# build the Go binary only; reuse the installed app's native libraries
mkdir -p build/lib/ollama
cp /Applications/Ollama.app/Contents/Resources/*.dylib build/lib/ollama/
cp /Applications/Ollama.app/Contents/Resources/llama-server build/lib/ollama/
go build -o ollama-patched .

OLLAMA_NUM_PARALLEL=8 OLLAMA_HOST=127.0.0.1:11435 ./ollama-patched serve
```

No cmake and no Metal toolchain needed — `llama-server` and the `*.dylib` files
come from the already-installed app, so only the Go server is rebuilt. It runs
on a separate port and does not touch `/Applications/Ollama.app`, so the stock
install stays intact and rollback is deleting one directory.

Then verify, do not assume:

```bash
python3 ollama_batch_probe.py --url http://127.0.0.1:11435 --model qwen3.5:27b --num-ctx 65536
python3 correctness_check.py http://127.0.0.1:11435 qwen3.5:27b
```

#### Caveats before rolling this to the MacStudio

- **Measured on `qwen3.5:0.8b` at `num_ctx=4096`, not on the 27b at 65536.**
  The architecture and the code path are identical, but the throughput number
  will differ.
- **KV cache scales with the slot count.** Ollama reserves `num_ctx x slots`.
  The pipeline sends `num_ctx=65536`; at 8 slots that is 512k tokens of KV
  reserved up front. Run `--ctx-sweep 8192,32768,65536` on the MacStudio to find
  where it stops fitting, and expect to trade slot count against context size.
  qwen3.5's hybrid backbone helps here — only 16 of its 64 layers are full
  attention, the other 48 carry constant-size recurrent state.
- **This is an unmerged PR.** It is a patched build, so it must be rebuilt and
  re-validated on every Ollama upgrade until it lands upstream.

### What this means for the pipeline

Concurrency against `qwen3.5:27b` on the MacStudio buys nothing today — the
requests queue. Realistic options, in order of effort:

Ordered cheapest-first, now that the cause is known:

1. **Patch the blocklist and rebuild Ollama — verified working, see
   [The fix, validated](#the-fix-validated).** One line in `server/sched.go`,
   no client change, no model change. Measured 1.03 -> 7.79 effective
   concurrency with byte-identical greedy output. This is the recommended path.
2. ~~Try qwen3.8:27b~~ — **ruled out, do not bother.** Qwen3.8-27B reuses the
   Qwen3.5 architecture: its `config.json` is `model_type: "qwen3_5"` /
   `Qwen3_5ForConditionalGeneration`, and ollama.com lists the packaged model's
   architecture as **`qwen35`** — the blocklisted string. It would serialize
   exactly like qwen3.5:27b. Checked without downloading the 18 GB blob.
3. **Serve qwen3.5 under vLLM or SGLang.** Real continuous batching, best
   throughput, biggest operational change.
4. **Run several Ollama instances** behind the proxy and load-balance. Costs a
   full copy of the weights in RAM per instance.
5. **Keep it sequential** and drop the concurrent client, which is currently
   paying complexity for a server that queues.

Either way, ask the admin to set `OLLAMA_NUM_PARALLEL=4`. It is necessary but
not sufficient — and it is what makes the probe's control model batch.

---

## Running it

```bash
# Is my server batching this model?
python3 ollama_batch_probe.py --model qwen3.5:27b

# The conclusive version: also run a control model to separate
# "the server can't batch" from "this model can't batch".
python3 ollama_batch_probe.py --model qwen3.5:27b --control-model qwen2.5:0.5b

# Mirror exactly what the Insurance pipeline sends (64k context, no streaming)
python3 ollama_batch_probe.py --model qwen3.5:27b --num-ctx 65536 --no-stream

# Does a large num_ctx cost us the parallel slots on this box?
python3 ollama_batch_probe.py --model qwen3.5:27b --ctx-sweep 4096,32768,65536

# Behind the authenticating proxy (same token the pipeline uses)
OLLAMA_TOKEN=... python3 ollama_batch_probe.py --url https://<host> --model qwen3.5:27b
```

Exit status is `0` when batching was detected and `1` when it was not, so it
drops into CI or a health check.

Useful flags: `--concurrency 1,2,4,8`, `--num-predict`, `--endpoint chat|generate`,
`--no-stream`, `--json out.json`. `--help` documents the rest.

## How it decides (and why you can trust it)

Every Ollama response carries server-side nanosecond timers. Summing
`prompt_eval_duration + eval_duration` across N concurrent requests and
dividing by the wall-clock time the batch took gives:

```
overlap = sum(server_busy_time) / wall_clock
```

- **Serialized** → each request computes while the others idle, so the busy
  times add up to about the wall clock. `overlap ≈ 1`, for any N.
- **Batched** → all N compute across the same window, so the busy times add up
  to about N × wall clock. `overlap ≈ N`.

The figure is normalized against the N=1 run so HTTP and network overhead
cancel out, and it works identically with streaming on or off. It cannot be
faked by a slow network, and unlike `OLLAMA_NUM_PARALLEL` or
`/api/ps`, it measures what the server *did* rather than what it was told.

Supporting signals in the report:

- **Time-to-first-token spread** (streaming only). Under batching all N first
  tokens land together (measured spread: 0.10s at N=4). Under serialization
  they staircase — the measured qwen3.5 run gave `0.3s, 3.1s, 6.0s, 8.9s`.
  That staircase also fingerprints a serializing reverse proxy.
- **`context_length` from `/api/ps`**, printed after load. Ollama allocates
  `num_ctx × slots`, so a loaded context equal to the requested `num_ctx`
  means one slot. Four times it means four slots.
- **`general.architecture` from `/api/show`**, which is how the probe flags the
  `qwen35` / `qwen3_5` problem *from inside a container*, without log access.

### Validation

The probe's own client was verified against a mock HTTP server that is
genuinely concurrent (each request sleeps 2s and reports honest timings). It
reported effective concurrency **1.00 / 2.00 / 3.97** at N = 1 / 2 / 4 with a
0.00s TTFT spread — so when the probe reports serialization, that is the
server's behaviour and not an artifact of the test client.

Two details in the workload exist for measurement integrity:

- **Each prompt gets a unique random prefix.** Identical prompts would hit
  Ollama's prompt cache, letting later requests skip prompt evaluation and
  inflating the apparent speedup.
- **The task is an open-ended essay**, not a question. Every request must
  generate the same number of tokens for the concurrency levels to be
  comparable; a model asked something answerable emits EOS after a few dozen
  tokens (measured: 57 of a requested 1200) and the timing noise then swamps
  the effect. The essay prompt runs to the `num_predict` ceiling every time,
  which the probe verifies via `done_reason == "length"` and warns about
  otherwise.

## Files

- **`patch/0001-enable-parallel-requests-for-qwen35.patch`** — the fix. Apply
  with `git apply` against a v0.34.0 checkout.
- **`ollama_batch_probe.py`** — the batching probe. Stdlib only, no install.
- **`correctness_check.py`** — greedy-determinism test: same prompts solo vs
  batched, must come back byte-identical.
- `evidence/scheduler-decisions.txt` — the before/after scheduler log lines.
- `run_server.sh` — stock `ollama serve` on :11435 with a chosen
  `OLLAMA_NUM_PARALLEL`, for reproducing the broken behaviour.
- `run_patched_server.sh` — the patched build on :11435.
- `results/*.json` — full measurements behind every number in this README.
- `ollama-src/` — not committed (193 MB, own git repo). Recreate with the
  clone + `git apply` above.
- `run_server.sh` — starts a local `ollama serve` on :11435 with a chosen
  `OLLAMA_NUM_PARALLEL`, for reproducing the above locally.
- `results/*.json` — full measurements from the runs in the table.
- `logs/serve_np*.log` — Ollama debug logs containing the scheduler decisions.
