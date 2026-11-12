#!/usr/bin/env bash
set -euo pipefail

task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${FASTTRELLIS2_CONFIG:-}" ]]; then
    task_config="$FASTTRELLIS2_CONFIG"
elif [[ -n "${1:-}" ]]; then
    task_config="$1"
elif [[ -f "$task_repo/deploy/config.json" ]]; then
    task_config="$task_repo/deploy/config.json"
else
    task_config="$task_repo/fasttrellis2/deploy/config.example.json"
fi
if [[ "$task_config" != /* ]]; then
    task_config="$(cd "$(dirname "$task_config")" && pwd)/$(basename "$task_config")"
fi

if [[ ! -f "$task_config" ]]; then
    echo "FastTrellis2 config not found: $task_config" >&2
    echo "Set FASTTRELLIS2_CONFIG to the deployment config file." >&2
    exit 1
fi

source "$task_repo/fasttrellis2/deploy/conda-env.sh"
fasttrellis2_activate_conda
export PYTHONNOUSERSITE=1
unset PYTHONPATH

task_cuda_home="${CUDA_HOME:-}"
if [[ -z "$task_cuda_home" ]] && command -v nvcc >/dev/null 2>&1; then
    task_cuda_home="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME="$task_cuda_home"
fi
if [[ -n "$task_cuda_home" && -d "$task_cuda_home/bin" ]]; then
    export PATH="$task_cuda_home/bin:$PATH"
fi
task_nvidia_libraries="$(python -c 'from pathlib import Path; import site; print(":".join(str(path) for root in site.getsitepackages() for path in Path(root).glob("nvidia/*/lib") if path.is_dir()))')"
task_nvidia_includes="$(python -c 'from pathlib import Path; import site; print(":".join(str(path) for root in site.getsitepackages() for path in Path(root).glob("nvidia/*/include") if path.is_dir()))')"
task_cuda_libraries="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$task_nvidia_libraries"
if [[ -n "$task_cuda_home" ]]; then
    task_cuda_libraries="$task_cuda_libraries:$task_cuda_home/lib64:$task_cuda_home/targets/x86_64-linux/lib"
fi
export LD_LIBRARY_PATH="$task_repo/.native/lib:$task_repo/.native/lib64:$task_cuda_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$task_repo/.native/lib:$task_repo/.native/lib64:$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$task_repo/.cache/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$task_repo/.cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$XDG_CACHE_HOME/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/cuda}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MAX_JOBS="${MAX_JOBS:-8}"
export CPATH="$task_nvidia_includes${CPATH:+:$CPATH}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TORCH_EXTENSIONS_DIR" \
    "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"

exec bash "$task_repo/fasttrellis2/start.sh" --config "$task_config"

