#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    printf 'Run bash setup_lambda.sh first.\n' >&2
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    printf 'HF_TOKEN is unset. Export a Hugging Face read token before starting.\n' >&2
    exit 1
fi
if [[ ! -f billing_started_utc.txt ]]; then
    printf 'Missing setup billing marker. Run setup_lambda.sh or use the CLI with the actual billing start.\n' >&2
    exit 1
fi
# Allocator setting, not a scientific hyperparameter. No CUDA graph or KV cache.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export HF_HUB_DISABLE_TELEMETRY=1
bta_started="${BTA_BILLING_STARTED_UTC:-$(cat billing_started_utc.txt)}"
bta_command=run
if [[ "${1:-}" == profile ]]; then
    bta_command=profile
    shift
fi
exec .venv/bin/python -m bta "$bta_command" --config configs/h100_qwen38.json \
    --set "budget.billing_started_utc=$bta_started" "$@"
