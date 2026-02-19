#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/remote_eval/k8s_eval.sh \
    --context <kube-context> \
    --namespace <ns> \
    --pod <pod-name> \
    [--container <container>] \
    --runtime cpu|cuda \
    --suite guardrails_ru \
    --split fast|full \
    --remote-has-internet yes|no \
    --models-dir /mnt/models \
    --datasets-dir /mnt/datasets \
    [--workdir /workdir/guardrails]
EOF
}

KCTX=""
NS=""
POD=""
CONTAINER=""
RUNTIME=""
SUITE="guardrails_ru"
SPLIT="fast"
REMOTE_HAS_INTERNET=""
MODELS_DIR=""
DATASETS_DIR=""
WORKDIR="/workdir/guardrails"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context) KCTX="$2"; shift 2 ;;
    --namespace) NS="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --remote-has-internet) REMOTE_HAS_INTERNET="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --datasets-dir) DATASETS_DIR="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$KCTX" || -z "$NS" || -z "$POD" || -z "$RUNTIME" || -z "$REMOTE_HAS_INTERNET" || -z "$MODELS_DIR" || -z "$DATASETS_DIR" ]]; then
  usage
  exit 2
fi

KUBECTL=(kubectl --context "$KCTX" -n "$NS")
EXEC=("${KUBECTL[@]}" exec "$POD")
if [[ -n "$CONTAINER" ]]; then
  EXEC+=(-c "$CONTAINER")
fi
EXEC+=(--)

EXTRA_UV_ARGS=(--extra eval)
if [[ "$RUNTIME" == "cuda" ]]; then
  EXTRA_UV_ARGS+=(--extra cuda)
fi

echo "[sync] streaming tar -> pod:$WORKDIR" >&2
tar -cf - \
  --exclude='.git' \
  --exclude='.env*' \
  --exclude='.venv' \
  --exclude='.eval_cache' \
  --exclude='.models' \
  --exclude='.datasets' \
  --exclude='reports' \
  --exclude='baselines' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  . \
| "${EXEC[@]}" sh -lc "mkdir -p '$WORKDIR' && tar -xf - -C '$WORKDIR'"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
cd "$WORKDIR"

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

echo "[run] eval suite=$SUITE split=$SPLIT" >&2
uv run ${EXTRA_UV_ARGS[*]} python -m app.eval.cli --suite "$SUITE" --split "$SPLIT" --env-file .env.eval --output-dir reports/evaluations
EOF
)

echo "[remote] running eval in pod=$POD" >&2
OUT_JSON="$("${EXEC[@]}" sh -lc "$REMOTE_CMD")"
echo "$OUT_JSON"

RUN_DIR="$(python - <<'PY'
import json,sys
payload=json.loads(sys.stdin.read())
print(payload["run_dir"])
PY
<<<"$OUT_JSON")"

RUN_BASENAME="$(basename "$RUN_DIR")"
LOCAL_RUN_DIR="reports/evaluations/$RUN_BASENAME"
mkdir -p "$LOCAL_RUN_DIR"

echo "[fetch] kubectl cp report dir -> $LOCAL_RUN_DIR" >&2
SRC="$POD:$RUN_DIR"
if [[ -n "$CONTAINER" ]]; then
  "${KUBECTL[@]}" cp -c "$CONTAINER" "$SRC" "$LOCAL_RUN_DIR/.."
else
  "${KUBECTL[@]}" cp "$SRC" "$LOCAL_RUN_DIR/.."
fi

echo "[done] report synced to $LOCAL_RUN_DIR" >&2
