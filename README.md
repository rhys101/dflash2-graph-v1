# DFlash2 Graph v1

**A complete, copy-and-run recipe for approximately 77 output tokens/second at
concurrency 1 with Qwen3.8-27B on one NVIDIA DGX Spark.**

Repository: `https://github.com/rhysjones/dflash2-graph-v1`

This release takes the public Qwen3.8-27B NVFP4 DFlash2 stack, makes its
selector projection compatible with dynamic-shape compilation, and enables
vLLM CUDA graph execution. On the exact `r0b0bench` concurrency lane, the
result was **77.10 tok/s at C1**, compared with **69.23 tok/s** for the matched
eager runtime: **+11.37%**.

This repository deliberately contains only this guide, the standalone graph
patch, and an MIT license. The runtime overlay remains sourced from the
upstream repository; the patch is also embedded in the Docker build below so
the complete build stays copy-and-run.

## What “77 tok/s” means

It is the aggregate output-token throughput reported by `r0b0bench` with:

- one concurrent request (`C1`, so one active stream);
- a 512-output-token budget and a long digits-and-spaces response;
- temperature 0 and Qwen thinking disabled;
- one NVIDIA DGX Spark / GB10 / compute capability 12.1;
- Qwen3.8-27B NVFP4 target, DFlash2 K=8 draft, FP8 KV cache; and
- graph execution enabled in vLLM 0.27.2rc0.

It is not a promise that every short or interactive prompt will display 77
tokens/s. Prompt length, accepted draft length, output length, temperature,
tool use, and cold-start compilation all affect an individual request.

## Supported recipe

| Component | Pinned identity |
| --- | --- |
| Hardware | One NVIDIA DGX Spark, GB10, 128 GB unified memory |
| Target | `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` |
| Target revision | `36f717a22990e82c54c1d48ee77c491b87825680` |
| Draft | `z-lab/Qwen3.8-27B-DFlash2` |
| Draft revision | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Runtime source | `r0b0tlab/qwen38-27b-nvfp4-sm121-vllm` |
| Runtime revision | `a96186f8973023365bddb34b6b34fa889d4c40ae` |
| vLLM | `0.27.2rc0`, source-built SM121 image |
| Parent image manifest | `sha256:5bd3f329c531da4cd5f41f2c32d4ed9527b2131b88032b737364fb582e6b775f` |
| Benchmark | `r0b0tlab/r0b0bench` at `a8951f3e00f3c423c10077036c94dacc48adb35d` |

The shipped target is a hybrid ModelOpt checkpoint: NVFP4 weight-only layers,
FP8 layers, and BF16 remainder/MTP tensors. The measured route used Marlin
NVFP4 linear kernels, FlashInfer FP8 scaled matrix multiplication, and FP8 KV.
It should not be described as a pure W4A4 or native FP4 tensor-core result.

## Install and run on one DGX Spark

### 1. Prerequisites

You need:

- a DGX Spark with its NVIDIA driver working;
- Docker plus NVIDIA Container Toolkit (`docker run --gpus all` works);
- Git, Python 3, `curl`, and `jq`;
- approximately 55 GB of free disk for the two checkpoints and images; and
- no competing GPU/model workload during launch or measurement.

Set up a working directory:

```bash
export DFLASH2_WORK="$HOME/dflash2-graph-v1-work"
mkdir -p "$DFLASH2_WORK/models" "$DFLASH2_WORK/cache"
cd "$DFLASH2_WORK"
```

### 2. Fetch the pinned runtime and checkpoints

```bash
git clone https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-vllm runtime
git -C runtime checkout a96186f8973023365bddb34b6b34fa889d4c40ae

python3 -m venv .hf
.hf/bin/pip install --upgrade huggingface_hub

.hf/bin/hf download r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 \
  --revision 36f717a22990e82c54c1d48ee77c491b87825680 \
  --local-dir "$DFLASH2_WORK/models/Qwen3.8-27B-NVFP4-MTP-sm121"

.hf/bin/hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir "$DFLASH2_WORK/models/Qwen3.8-27B-DFlash2"
```

The target download is incomplete unless all four shards exist:

```bash
for shard in 1 2 3 4; do
  test -f "$DFLASH2_WORK/models/Qwen3.8-27B-NVFP4-MTP-sm121/$(printf 'model-%05d-of-00004.safetensors' "$shard")" || exit 1
done
```

### 3. Build the upstream DFlash2 vLLM image

First pin the SM121 parent by manifest digest, then build the upstream DFlash2
overlay without replacing the tuned parent:

```bash
docker pull \
  ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121@sha256:5bd3f329c531da4cd5f41f2c32d4ed9527b2131b88032b737364fb582e6b775f

docker tag \
  ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121@sha256:5bd3f329c531da4cd5f41f2c32d4ed9527b2131b88032b737364fb582e6b775f \
  ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121

docker build --pull=false \
  -f runtime/docker/Dockerfile.dflash2 \
  -t dflash2-sm121:eager-base \
  runtime/docker
```

Check the exact source that the graph patch expects:

```bash
docker run --rm --entrypoint /usr/bin/sha256sum \
  dflash2-sm121:eager-base \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py
```

The first field must be:

```text
801de92049fa749bcfdccb1a3179b7930d42cd433f425056815b9374990d87f2
```

If it differs, stop. The upstream source has moved and this v1 patch has not
been validated against it.

### 4. Apply the graph-compatibility patch

The original selector uses a Python loop over `flat.shape[0]`. AOT compilation
specializes that dynamic leading dimension, which prevents the DFlash2 graph
from serving the required shapes. The replacement performs the same
row-independent FP32 projection in one call and keeps that dimension symbolic.

The canonical source diff is [`dflash2-graph-v1.patch`](dflash2-graph-v1.patch).
It is suitable for review or application to a matching vLLM overlay tree:

```bash
# From a checkout of this repository, with the upstream runtime at ../runtime:
git -C ../runtime apply --check \
  --directory=docker/overlay \
  "$PWD/dflash2-graph-v1.patch"
```

The source-bound Docker step below performs the same change inside the already
audited upstream image. It refuses both an unexpected input and an unexpected
patched result.

Build the patched image directly from this embedded Dockerfile:

```bash
docker build --pull=false \
  --build-arg BASE_IMAGE=dflash2-sm121:eager-base \
  -t dflash2-sm121:graph-v1 - <<'DOCKERFILE'
# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN /usr/bin/python3 - <<'PY'
from hashlib import sha256
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py")
expected_before = "801de92049fa749bcfdccb1a3179b7930d42cd433f425056815b9374990d87f2"
expected_after = "a9b90731f3f2ed30066f607ff36c8060e594b81108ef4b72ee83c881bdb4da9b"

old = """        outs = []
        chunk = 128
        w32 = weight.float()
        for start in range(0, flat.shape[0], chunk):
            outs.append(torch.nn.functional.linear(flat[start : start + chunk].float(), w32))
        hidden = torch.cat(outs, 0).to(dtype=hidden_states.dtype).view(*orig[:-1], -1)
"""

new = """        # A single FP32 projection is mathematically row-independent and keeps
        # the leading dimension symbolic for torch.compile / CUDA graph capture.
        # The former Python chunk loop specialized flat.shape[0] during AOT.
        w32 = weight.float()
        projected = torch.nn.functional.linear(flat.float(), w32)
        hidden = projected.to(dtype=hidden_states.dtype).view(*orig[:-1], -1)
"""

before = path.read_bytes()
assert sha256(before).hexdigest() == expected_before, "unexpected upstream source"
text = before.decode("utf-8")
assert text.count(old) == 1, "selector block is not unique"
path.write_text(text.replace(old, new), encoding="utf-8")
assert sha256(path.read_bytes()).hexdigest() == expected_after, "patched hash mismatch"
print(f"patched {path}: {expected_after}")
PY
DOCKERFILE
```

Verify the installed result:

```bash
docker run --rm --entrypoint /usr/bin/sha256sum \
  dflash2-sm121:graph-v1 \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py
```

Expected first field:

```text
a9b90731f3f2ed30066f607ff36c8060e594b81108ef4b72ee83c881bdb4da9b
```

### 5. Launch vLLM with graph execution

The important detail is what is absent: **do not add `--enforce-eager`**.

```bash
docker rm -f qwen38-dflash2-graph >/dev/null 2>&1 || true

docker run -d \
  --name qwen38-dflash2-graph \
  --restart no \
  --gpus all \
  --cpus 14 \
  --memory 112g \
  -p 8000:8000 \
  -e VLLM_HOST_IP=127.0.0.1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_CACHE_ROOT=/cache/vllm \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -v "$DFLASH2_WORK/models/Qwen3.8-27B-NVFP4-MTP-sm121:/model:ro" \
  -v "$DFLASH2_WORK/models/Qwen3.8-27B-DFlash2:/draft:ro" \
  -v "$DFLASH2_WORK/cache:/cache" \
  dflash2-sm121:graph-v1 \
  --model /model \
  --served-model-name qwen38-27b \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.73 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":8}' \
  --no-enable-flashinfer-autotune \
  --kernel-config.enable_jit_warmup=false \
  --kernel-config.enable_cutedsl_warmup=false
```

The cold launch compiles and captures many graphs. Around four minutes is
normal; retain the mounted cache for later launches. Wait for readiness:

```bash
for attempt in $(seq 1 90); do
  curl -fsS http://127.0.0.1:8000/health && break
  sleep 10
done
curl -fsS http://127.0.0.1:8000/health
```

Confirm the logs describe graph mode, K=8, the V2 runner, FP8 KV, and DFlash2:

```bash
docker logs qwen38-dflash2-graph 2>&1 | \
  grep -E 'enforce_eager=False|num_spec_tokens=8|DFlash2DraftModel|kv_cache_dtype=fp8|V2 model runner|CUDAGraph|graph capture'
```

### 6. Run the semantic canary

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen38-27b",
    "messages":[
      {"role":"system","content":"Return only the final integer. No words or punctuation."},
      {"role":"user","content":"What is 19 multiplied by 23?"}
    ],
    "temperature":0,
    "max_tokens":16,
    "chat_template_kwargs":{"thinking":false,"enable_thinking":false}
  }' | tee /tmp/dflash2-canary.json

jq -r '.choices[0].message.content' /tmp/dflash2-canary.json
```

The answer must be `437`. A result of `417` is the known sign that the FP8 KV
route was not configured correctly.

### 7. Reproduce the approximately 77 tok/s C1 result

```bash
cd "$DFLASH2_WORK"
git clone https://github.com/r0b0tlab/r0b0bench benchmark
git -C benchmark checkout a8951f3e00f3c423c10077036c94dacc48adb35d

python3 -m venv .bench
.bench/bin/pip install --upgrade pip
.bench/bin/pip install -e "$DFLASH2_WORK/benchmark"

R0B0BENCH_CHAT_TEMPLATE_KWARGS='{"thinking":false,"enable_thinking":false}' \
  .bench/bin/python -m r0b0bench.cli run \
  --profile core-subset \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen38-27b \
  --output "$DFLASH2_WORK/results" \
  --only concurrency \
  --run-id graph-v1 \
  --timeout 600

jq '.rows[] | select(.concurrency == 1) | .aggregate_output_tok_s' \
  "$DFLASH2_WORK/results/graph-v1/lanes/concurrency/summary.json"
```

Expect run-to-run noise. On an otherwise idle Spark with a warm persistent
cache, the target is roughly **77 output tok/s**. If you see approximately
67–70 tok/s, check that the patched source hash is correct and that
`--enforce-eager` is absent.

## Why the patch helps

The target verifier is unchanged. DFlash2 still proposes a block of eight
tokens and the target still verifies it. The change is only in how one selector
projection is expressed:

```text
Eager/public path:   Python loop over dynamic rows -> several FP32 linear calls
Graph-v1 path:       one symbolic FP32 linear call -> capture once, replay
```

Removing Python shape specialization lets vLLM capture the DFlash2 decode
path. During serving, the GPU replays the captured work instead of repeatedly
paying the same launch and framework overhead.

## Memory and other trade-offs

Graph execution is faster here, but it is not free:

- minimum available unified memory fell from **24.31 GB** in matched eager mode
  to **21.66 GB** in graph mode;
- the observed headroom cost was **2.66 GB**;
- cold start is slower because compilation and graph capture take several
  minutes;
- captured shapes make this recipe less flexible than eager execution; and
- the source-bound patch intentionally refuses to apply after upstream drift.

Keep the Spark idle while loading. The evidence run used an 8% minimum-free
preflight and a 9% automatic-stop threshold. This README does not install a
memory watchdog, so monitor unified memory yourself when changing context,
batch, or model settings.

## SGLang status

SGLang now has its own DFlash2 implementation and CUDA-graph path. It is not
the same code path, and the vLLM selector patch above must **not** be copied into
SGLang.

For the supported SGLang DGX Spark package:

```bash
git clone https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang
cd qwen38-27b-nvfp4-sm121-sglang
bash scripts/click_run_dflash2.sh
```

That upstream package publishes approximately **68.6 tok/s at C1** in its
`r0b0bench` core-subset concurrency lane. The **77.10 tok/s** result in this
repository belongs only to the vLLM graph-v1 recipe. SGLang is therefore a
supported alternative deployment, not a reproduction of this headline result.

Useful upstream documentation:

- vLLM DFlash: <https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/>
- SGLang speculative decoding: <https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/speculative_decoding.mdx>

## Troubleshooting

### Server never becomes healthy

Inspect `docker logs qwen38-dflash2-graph`. The common causes are an incomplete
four-shard target, wrong draft architecture, another model occupying memory, or
a source/image mismatch.

### CUDA graph initialization fails

Recheck the installed `qwen3_dflash2.py` hash. The graph image must contain
`a9b90731...`; the upstream overlay before patching must contain
`801de920...`.

### Throughput is around 27–30 tok/s

That is commonly a different dedicated-C1, 2048-output measurement protocol.
Do not compare it directly with this repository's 512-output `r0b0bench`
concurrency lane.

### Throughput is around 67–70 tok/s

That is the expected eager/public neighborhood. Confirm all of the following:

- no `--enforce-eager` flag;
- `enforce_eager=False` appears in logs;
- K=8, FP8 KV, V2 model runner, and the graph image are active;
- persistent compilation caches are mounted; and
- no other GPU workload is running.

### Clean shutdown

```bash
docker rm -f qwen38-dflash2-graph
```

## Evidence addendum

This table is the exact matched comparison from 24 August 2026. Both matched
arms used the same patched image; removing `--enforce-eager` was the sole
runtime-arm change.

| Concurrency | Public eager | Matched eager | Graph v1 | Graph over matched eager |
| ---: | ---: | ---: | ---: | ---: |
| C1 | 67.057 | 69.230 | **77.101** | **+11.37%** |
| C2 | 121.492 | 125.462 | **140.403** | **+11.91%** |
| C4 | 211.523 | 215.999 | **240.715** | **+11.44%** |
| C6 | 279.250 | 285.735 | **316.291** | **+10.69%** |

Geometric-mean uplift across C1/C2/C4/C6: **11.35%**.

Evidence boundaries:

- the public eager baseline was independently replicated within 5% at every
  concurrency level;
- 26/26 paired endpoint responses matched on content, reasoning content,
  finish reason, and usage;
- the public endpoint did not expose internal selected-token IDs, so this is
  endpoint parity rather than raw internal-token proof; and
- the exact concurrency lane was rerun, not the entire `core-subset` suite.

Key measured identities:

| Item | SHA-256 / identity |
| --- | --- |
| Public DFlash2 image used as graph base | `sha256:4cade79e3ce5568ec0cc67d66db71dfdd60d35ca604ac82f22900d15b36677a2` |
| Matched graph-compatible image | `sha256:8e91ac63a65cfc0adecdb1980fd723c407bb4b4c895588113cffda8409e43c49` |
| Selector before patch | `801de92049fa749bcfdccb1a3179b7930d42cd433f425056815b9374990d87f2` |
| Selector after patch | `a9b90731f3f2ed30066f607ff36c8060e594b81108ef4b72ee83c881bdb4da9b` |
| Target manifest | `b9e4cabe3a072323718ac3def2eba085561d3dd9e98405aae7ee0e4a29be0fe2` |
| Draft manifest | `13122ec3d6a6cebe4b0a6b5197fbc797b0179e1e0ec908c75e12d09572770d02` |

The image IDs above document the measured run; rebuilding locally may produce
a different Docker image ID because build metadata can differ. The installed
source hashes, pinned inputs, flags, and benchmark protocol are the portable
reproduction contract.

## License

MIT. Checkpoint, container, vLLM, SGLang, and upstream overlay licenses remain
their respective owners' licenses and are not relicensed by this repository.

Maintainer: Rhys Jones, <rhys@axonzeta.ai>
