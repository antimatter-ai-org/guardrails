# Runbook: GPU Evals + HF “fast” Split Balancing

This is a practical memory dump of gotchas and working procedures that came up while:
- running `app/eval_v3` on remote GPU hosts (SSH), and
- rebalancing/publishing HF-stored `fast` splits for consistent cross-machine evaluation.

It is intentionally operational: copy-paste commands, failure modes, and what to check.

## Remote GPU Eval (SSH) Checklist

### Constraints We Actually Hit
- Local -> remote link can be slow. Do not move datasets/models over rsync; sync code only.
- Remote host may have `uv` installed but not on `PATH`.
- Remote shell might be `zsh`; globs like `reports/evaluations/*` can error when no matches exist.
- When HF `fast` splits are updated, existing remote caches may keep using the old parquet unless you force redownload / clear the cache.

### Minimal Working Environment (Remote)
- Python 3.12 worked reliably; Python 3.14 broke some deps (pydantic v1 vs v2 ecosystem assumptions).
- Use CUDA runtime:
  - `export GR_RUNTIME_MODE=cuda`
  - Pin to a specific GPU:
    - `export CUDA_VISIBLE_DEVICES=6` (so “GPU6 only” becomes “device 0” inside the process)
- Models:
  - `export GR_MODEL_DIR="$PWD/.models"`
  - `make download-models MODELS_DIR=./.models`
- HF auth for private datasets:
  - Either `HF_TOKEN=...` in env, or `hf auth login` on the remote host.

### Known-Good Example Run (SSH + H100 + GPU6 Only)
This is the exact setup that produced an ~8.5 minute full-suite run after rebalancing `fast` splits (2026-02-17):

```bash
cat <<'EOF' | ssh -o BatchMode=yes user@gpu-host 'bash -s'
set -euo pipefail
cd ~/guardrails
export CUDA_VISIBLE_DEVICES=6
export GR_RUNTIME_MODE=cuda
export GR_ENABLE_NEMOTRON=true
export GR_MODEL_DIR="$PWD/.models"

# Important if HF splits were updated recently:
rm -rf .eval_cache/hf

make download-models MODELS_DIR="$GR_MODEL_DIR"

uv run --extra eval --extra cuda python -m app.eval_v3.cli \
  --suite guardrails_ru \
  --split fast \
  --runtime cuda \
  --workers 16 \
  --policy-path configs/policy.yaml \
  --policy-name external_default \
  --output-dir reports/evaluations
EOF
```

### Recommended Way To Run Remote Commands
Avoid trying to inline complicated quoting through `ssh '...'` because it’s easy to:
- accidentally use the remote’s `zsh` (globbing behavior differs),
- accidentally break exports if your local environment gets interpolated.

Prefer a heredoc into `bash -s`:

```bash
cat <<'EOF' | ssh -o BatchMode=yes user@host 'bash -s'
set -euo pipefail
cd ~/guardrails
export PATH="$HOME/.local/bin:$PATH"  # if uv/hf are in ~/.local/bin
export CUDA_VISIBLE_DEVICES=6
export GR_RUNTIME_MODE=cuda
export GR_ENABLE_NEMOTRON=true
export GR_MODEL_DIR="$PWD/.models"
mkdir -p reports/evaluations
uv run --extra eval --extra cuda python -m app.eval_v3.cli --suite guardrails_ru --split fast --runtime cuda --workers 16 \
  --policy-path configs/policy.yaml --policy-name external_default --output-dir reports/evaluations
EOF
```

### Stopping A Remote Run
If the run is started via a non-interactive `ssh` command, you can’t reliably send Ctrl-C to the process.
Use `pkill` by user and `-f` match, and pass `--` so patterns aren’t parsed as flags:

```bash
cat <<'EOF' | ssh -o BatchMode=yes user@host 'bash -s'
set -euo pipefail
pkill -u "$USER" -TERM -f -- 'app.eval_v3.cli' || true
pkill -u "$USER" -TERM -f -- 'tritonserver' || true
sleep 2
pkill -u "$USER" -KILL -f -- 'app.eval_v3.cli' || true
pkill -u "$USER" -KILL -f -- 'tritonserver' || true
EOF
```

If you must be surgical, kill only the PIDs using GPU memory:

```bash
nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader,nounits
kill -TERM <pid> ...
```

### Embedded PyTriton Gotchas (CUDA Runtime)
Eval v3 uses the same runtime stack as production and may start an embedded Triton server.

Problems encountered and fixes that were applied in code:
- Embedded Triton wasn’t started in eval runs (fixed by starting `EmbeddedPyTritonManager` from eval CLI in CUDA mode).
- Triton python backend stub couldn’t find `libpython3.12.so.1.0` under `uv` Python (fixed by ensuring Python’s `LIBDIR` is on `LD_LIBRARY_PATH` when starting embedded Triton).

If you see “port already in use” or Triton refuses to start:
- kill the old `tritonserver` owned by the user (see “Stopping A Remote Run”)
- check ports `8000/8001/8002` (HTTP/GRPC/metrics)

### HF Auth Gotchas (Remote)
`datasets.load_dataset(..., token=True)` will use the local HF auth cache (from `hf auth login`) if present.
If the remote user is logged into a different HF account than the one that can access private datasets, loading will fail.

Verify on remote:

```bash
hf auth whoami
```

Also: if you change `HF_HOME` / cache roots, you may stop seeing the token that was stored by `hf auth login`.
In automated runs, prefer exporting `HF_TOKEN=...` explicitly.

## Throughput Reality (Why “fast” Needed Rebalancing)

Observed span_detection throughput on H100 (GPU) varied a lot by dataset:
- Some datasets ran ~35-45 samples/s.
- `meddies-pii-cleaned-v1` ran ~7 samples/s in our setup, so a 20k “fast” split implied ~45+ minutes.

Conclusion: HF-stored `fast` splits must be sized for a suite-level budget (we target ~10 minutes total).

## HF-Stored “fast” Split Balancing

### The Goal
- Make `split=fast` consistent across machines by storing it on HF.
- Keep total suite runtime around ~10 minutes on a single GPU.
- Keep label diversity: a label-balanced positive sample with some negatives (where the dataset supports negatives).

### Implemented Tooling
- `app/tools/rebalance_fast_splits.py`
  - Builds a deterministic subset from `full`
  - Writes and commits `data/fast-00000-of-00001.parquet`
  - Updates HF metadata so `datasets.load_dataset(..., split="fast")` works reliably:
    - **README.md front matter** `dataset_info.splits[fast].num_examples`
    - plus best-effort `dataset_info.json` and `dataset_infos.json`

- `app/tools/publish_dataset_info.py`
  - Emergency/repair tool to republish metadata (`README.md` + `dataset_infos.json`) if a repo gets out of sync.

### Critical HF Metadata Gotcha (We Hit This)
If you update only `data/fast-*.parquet` and do not update the dataset card front matter (`README.md` `dataset_info:`),
then `datasets` may throw:
- `datasets.exceptions.NonMatchingSplitsSizesError`

Reason: `datasets` treats expected split sizes as authoritative and verifies recorded sizes on download/prepare.

If you got into this state already, the fastest recovery is to run the metadata repair tool for the affected dataset(s):
`python -m app.tools.publish_dataset_info ...` (see `app/tools/publish_dataset_info.py`).

### Verifying That HF “fast” Loads Cleanly
Use a fresh cache dir and force redownload:

```bash
python - <<'PY'
from datasets import load_dataset
import tempfile

repo = "antimatter-ai/guardrails-ru-meddies-pii-cleaned-v1"
with tempfile.TemporaryDirectory() as td:
    ds = load_dataset(repo, split="fast", token=True, cache_dir=td, download_mode="force_redownload")
    print("len", len(ds))
PY
```

### Making Sure You Pick Up Updated Splits
If a dataset repo was updated (new `fast` parquet and metadata), but your machine keeps using old cached files, you can:
- delete the local HF dataset cache directory (recommended for one-off validation), or
- use `download_mode="force_redownload"` in a small verification script (see above), or
- run eval with a fresh `--cache-dir` (if using v3 CLI cache selection).

### Recommended Split Sizes (Current Defaults)
These were tuned to hit a ~10 minute suite budget on a single H100 in our remote runs:
- russian-pii-66k: 4000
- meddies: 800
- scanpatch: 1500
- rubai: 2500
- kaggle: 800
- hf-sample-en: 800
- presidio-test: 800

Notes:
- If a dataset has no negatives (or `entity_count` is always >0), the tool will still work; it will just sample positives.
- If a dataset is extremely slow, reduce its `fast` size aggressively; suite-level budget matters more than per-dataset parity.

## Operational Tips

### Keep Remote Runs Deterministic
- Always pin a GPU with `CUDA_VISIBLE_DEVICES`.
- Prefer a fixed `--workers` value (`16` worked well for our Triton-backed runtime).
- Use HF-stored splits, not local cached subsets, when comparing machines.

### Don’t Sync Data Over Slow Links
- Sync code only.
- Download datasets/models remotely if the host has fast internet.
