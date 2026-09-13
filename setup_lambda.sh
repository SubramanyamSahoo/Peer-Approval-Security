#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != '--check' ) ]]; then
    printf 'Usage: bash setup_lambda.sh [--check]\n' >&2
    exit 2
fi

# A conservative start marker includes setup time. For accurate accounting, set
# BTA_BILLING_STARTED_UTC to the instance start time shown by Lambda BEFORE setup.
if [[ ! -f billing_started_utc.txt ]]; then
    if [[ -n "${BTA_BILLING_STARTED_UTC:-}" ]]; then
        printf '%s\n' "$BTA_BILLING_STARTED_UTC" > billing_started_utc.txt
    else
        date -u +%Y-%m-%dT%H:%M:%S+00:00 > billing_started_utc.txt
    fi
fi

if [[ -n "${BTA_PYTHON:-}" ]]; then
    bta_python="$BTA_PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
    bta_python="$(command -v python3.12)"
else
    bta_python="$(command -v python3)"
fi
"$bta_python" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Use Python 3.12 for the validated dependency set. Set BTA_PYTHON=/path/to/python3.12.')
PY
for program in nvidia-smi nvcc g++; do
    if ! command -v "$program" >/dev/null 2>&1; then
        printf 'Missing %s. Select a Lambda CUDA 12.8 development image with a C++ compiler.\n' "$program" >&2
        exit 1
    fi
done
nvcc --version
if [[ "${1:-}" == '--check' ]]; then
    nvidia-smi -L
    printf 'Compiler/interpreter checks passed; no packages or model weights downloaded. Kernel ABI is checked during setup/preflight.\n'
    exit 0
fi
if [[ ! -x .venv/bin/python ]]; then
    "$bta_python" -m venv .venv
fi
bta_pip=(.venv/bin/python -m pip)
"${bta_pip[@]}" install --upgrade pip setuptools wheel packaging ninja
"${bta_pip[@]}" install -c requirements-lambda.txt torch --index-url https://download.pytorch.org/whl/cu128
"${bta_pip[@]}" install -c requirements-lambda.txt -e '.[test]'

.venv/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('The CUDA wheel cannot use this NVIDIA driver. Use a CUDA 12.8 compatible Lambda image.')
props = torch.cuda.get_device_properties(0)
print(f'GPU: {props.name}; memory: {props.total_memory / 2**30:.2f} GiB; Torch CUDA: {torch.version.cuda}')
if 'H100' not in props.name:
    raise SystemExit('This setup targets one H100. Select that instance before running the full study.')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('BF16 is unavailable on the selected GPU.')
PY

# Let PyTorch provide Triton; do not independently upgrade it and break the ABI.
"${bta_pip[@]}" install -c requirements-lambda.txt flash-linear-attention
# Cache an ABI-matched wheel before any model download. Building, if necessary,
# is still billable; --check verifies prerequisites without attempting a build.
"${bta_pip[@]}" wheel -c requirements-lambda.txt causal-conv1d --no-deps --no-build-isolation --wheel-dir kernel_wheels
"${bta_pip[@]}" install -c requirements-lambda.txt causal-conv1d --no-index --find-links kernel_wheels --no-deps
"${bta_pip[@]}" check
.venv/bin/python - <<'PY'
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from causal_conv1d import causal_conv1d_fn
print('Fast-kernel imports passed. The model preflight also checks numerical behavior on this GPU.')
PY
"${bta_pip[@]}" freeze > installed_packages.txt
printf 'Setup complete. Read README.md, supply HF_TOKEN, then run: bash run_lambda.sh\n'
