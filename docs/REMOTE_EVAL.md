# Remote GPU Evaluation

This doc describes how to run eval v3 on remote GPU hosts via SSH or Kubernetes.

For operational gotchas and copy-paste commands (including HF `fast` split balancing), see:
- `docs/RUNBOOK_GPU_EVAL_AND_FAST_SPLITS.md`

Key constraints:
- Code sync should be lightweight: **sync only code**, never datasets/models over a slow link.
- Remote hosts may be online or air-gapped:
  - If online: download models/datasets on the remote host.
  - If offline: expect pre-mounted directories with `manifest.json` for both models and datasets.
- Fail explicitly when assets are missing and remote has no internet.

## Prerequisites

Local:
- `uv` installed
- `rsync` available (SSH runner)
- `kubectl` configured (K8s runner)

Remote (SSH host or K8s pod image):
- Python + `uv` (or prebuilt venv)
- The repo’s dependencies installable:
  - `uv sync --extra eval` (and `--extra cuda` for CUDA runtime)
- `HF_TOKEN` available on the remote host (env var or a local `.env.eval` created on the remote).

## Assets (Online vs Offline)

Models:
- Online: `make download-models MODELS_DIR=/path/to/models`
- Offline: mount `/path/to/models` with `/path/to/models/manifest.json` present

Datasets:
- Online: `make download-datasets DATASETS_DIR=/path/to/datasets`
- Offline: mount `/path/to/datasets` with `/path/to/datasets/manifest.json` present

## SSH Runner

Script:
- `scripts/remote_eval/ssh_eval.sh`

Notes:
- The SSH/K8s sync intentionally excludes `.env*` files. Provide `HF_TOKEN` on the remote host via environment or create `.env.eval` on the remote.

Example (remote has internet):

```bash
scripts/remote_eval/ssh_eval.sh \
  --host user@gpu-host \
  --remote-dir /home/user/guardrails-eval \
  --runtime cuda \
  --suite guardrails_ru \
  --split fast \
  --remote-has-internet yes \
  --models-dir /mnt/models \
  --datasets-dir /mnt/datasets
```

Example (air-gapped remote):

```bash
scripts/remote_eval/ssh_eval.sh \
  --host user@gpu-host \
  --remote-dir /home/user/guardrails-eval \
  --runtime cuda \
  --suite guardrails_ru \
  --split fast \
  --remote-has-internet no \
  --models-dir /mnt/models \
  --datasets-dir /mnt/datasets
```

## K8s Runner

Script:
- `scripts/remote_eval/k8s_eval.sh`

Example:

```bash
scripts/remote_eval/k8s_eval.sh \
  --context my-kube-context \
  --namespace guardrails \
  --pod guardrails-eval-pod-0 \
  --runtime cuda \
  --suite guardrails_ru \
  --split fast \
  --remote-has-internet yes \
  --models-dir /mnt/models \
  --datasets-dir /mnt/datasets
```
