#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$repo/fasttrellis2/build"
if [[ ! -f "$build/libtrellis_cuda_arena.so" ]]; then
    echo "Native modules are missing. Run python fasttrellis2/build_extensions.py first." >&2
    exit 1
fi
for module in trellis_xatlas_bounded_whole trellis_telea_gpu trellis_telea_stream_seeded trellis_bvh_async; do
    if [[ ! -f "$build/$module.so" ]]; then
        echo "Missing $module.so. Run python fasttrellis2/build_extensions.py first." >&2
        exit 1
    fi
done

mkdir -p "$repo/.cache/huggingface" "$repo/.cache/triton" "$repo/.cache/flex_gemm"
export PYTHONPATH="$repo/fasttrellis2/runtime:$build:$repo:$repo/o-voxel${PYTHONPATH:+:$PYTHONPATH}"
export LD_PRELOAD="$build/libtrellis_cuda_arena.so${FASTTRELLIS2_LIBSTDCXX_PRELOAD:+:$FASTTRELLIS2_LIBSTDCXX_PRELOAD}${LD_PRELOAD:+:$LD_PRELOAD}"
if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/lib64" ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export ATTN_BACKEND="${ATTN_BACKEND:-xformers}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-xformers}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-ACTIVE}"
export GOMP_SPINCOUNT="${GOMP_SPINCOUNT:-300000}"
export HF_HOME="${HF_HOME:-$repo/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$repo/.cache/triton}"
export FLEX_GEMM_AUTOTUNE_CACHE_PATH="${FLEX_GEMM_AUTOTUNE_CACHE_PATH:-$repo/.cache/flex_gemm/autotune_cache.json}"

exec python "$repo/fasttrellis2/runtime/trellis_service.py" "$@"
