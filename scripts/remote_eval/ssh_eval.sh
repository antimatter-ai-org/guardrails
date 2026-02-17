#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/remote_eval/ssh_eval.sh \
    --host user@host \
    --remote-dir /path/to/workdir \
    --runtime cpu|cuda \
    --suite guardrails_ru \
    --split fast|full \
    --remote-has-internet yes|no \
    --models-dir /mnt/models \
    --datasets-dir /mnt/datasets
EOF
}

HOST=""
REMOTE_DIR=""
RUNTIME=""
SUITE="guardrails_ru"
SPLIT="fast"
REMOTE_HAS_INTERNET=""
MODELS_DIR=""
DATASETS_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --remote-has-internet) REMOTE_HAS_INTERNET="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --datasets-dir) DATASETS_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$HOST" || -z "$REMOTE_DIR" || -z "$RUNTIME" || -z "$REMOTE_HAS_INTERNET" || -z "$MODELS_DIR" || -z "$DATASETS_DIR" ]]; then
  usage
  exit 2
fi

EXTRA_UV_ARGS=(--extra eval)
if [[ "$RUNTIME" == "cuda" ]]; then
  EXTRA_UV_ARGS+=(--extra cuda)
fi

echo "[sync] rsync code -> $HOST:$REMOTE_DIR" >&2
rsync -az \
  --exclude '.git/' \
  --exclude '.env*' \
  --exclude '.venv/' \
  --exclude '.eval_cache/' \
  --exclude '.models/' \
  --exclude '.datasets/' \
  --exclude 'reports/' \
  --exclude 'baselines/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  ./ "$HOST:$REMOTE_DIR/"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
cd "$REMOTE_DIR"

echo "[deps] uv sync ${EXTRA_UV_ARGS[*]}" >&2
uv sync ${EXTRA_UV_ARGS[*]}

if [[ "$REMOTE_HAS_INTERNET" == "yes" ]]; then
  if [[ ! -f "$MODELS_DIR/manifest.json" ]]; then
    echo "[models] downloading into $MODELS_DIR" >&2
    make download-models MODELS_DIR="$MODELS_DIR"
  fi
  if [[ ! -f "$DATASETS_DIR/manifest.json" ]]; then
    echo "[datasets] downloading into $DATASETS_DIR" >&2
    make download-datasets DATASETS_DIR="$DATASETS_DIR"
  fi
else
  if [[ ! -f "$MODELS_DIR/manifest.json" ]]; then
    echo "[error] missing models manifest at $MODELS_DIR/manifest.json (remote-has-internet=no)" >&2
    exit 10
  fi
  if [[ ! -f "$DATASETS_DIR/manifest.json" ]]; then
    echo "[error] missing datasets manifest at $DATASETS_DIR/manifest.json (remote-has-internet=no)" >&2
    exit 11
  fi
fi

export GR_RUNTIME_MODE="$RUNTIME"
export GR_MODEL_DIR="$MODELS_DIR"

export HF_HOME="$DATASETS_DIR/hf_cache"
export HUGGINGFACE_HUB_CACHE="$DATASETS_DIR/hf_cache/hub"
export HF_DATASETS_CACHE="$DATASETS_DIR/hf_cache/datasets"

if [[ "$REMOTE_HAS_INTERNET" == "no" ]]; then
  export GR_OFFLINE_MODE=true
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

echo "[run] eval_v3 suite=$SUITE split=$SPLIT" >&2
# stdout is JSON (report paths). Progress logs are emitted to stderr by design.
uv run ${EXTRA_UV_ARGS[*]} python -m app.eval_v3.cli --suite "$SUITE" --split "$SPLIT" --env-file .env.eval --output-dir reports/evaluations
EOF
)

echo "[remote] running eval on $HOST" >&2
OUT_JSON="$(ssh "$HOST" "$REMOTE_CMD")"
echo "$OUT_JSON"

RUN_DIR="$(python - <<'PY'
import json,sys,os
payload=json.loads(sys.stdin.read())
print(payload["run_dir"])
PY
<<<"$OUT_JSON")"

RUN_BASENAME="$(basename "$RUN_DIR")"
LOCAL_RUN_DIR="reports/evaluations/$RUN_BASENAME"
mkdir -p "$LOCAL_RUN_DIR"

echo "[fetch] rsync report dir -> $LOCAL_RUN_DIR" >&2
rsync -az "$HOST:$RUN_DIR/" "$LOCAL_RUN_DIR/"

echo "[done] report synced to $LOCAL_RUN_DIR" >&2
