#!/usr/bin/env bash

fasttrellis2_activate_conda() {
    local task_conda_base="${FASTTRELLIS2_CONDA_BASE:-}"
    local task_conda_script
    local task_conda_candidate

    if [[ -z "$task_conda_base" && -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
        task_conda_base="$("$CONDA_EXE" info --base 2>/dev/null || true)"
    fi
    if [[ -z "$task_conda_base" ]] && command -v conda >/dev/null 2>&1; then
        task_conda_base="$(conda info --base 2>/dev/null || true)"
    fi
    if [[ -z "$task_conda_base" && "${FASTTRELLIS2_CONDA_PREFIX:-}" == */envs/* ]]; then
        task_conda_candidate="${FASTTRELLIS2_CONDA_PREFIX%%/envs/*}"
        if [[ -f "$task_conda_candidate/etc/profile.d/conda.sh" ]]; then
            task_conda_base="$task_conda_candidate"
        fi
    fi
    if [[ -z "$task_conda_base" ]]; then
        for task_conda_candidate in "$HOME/anaconda3" "$HOME/miniconda3" \
            "$HOME/miniforge3" "$HOME/mambaforge" /opt/conda; do
            if [[ -f "$task_conda_candidate/etc/profile.d/conda.sh" ]]; then
                task_conda_base="$task_conda_candidate"
                break
            fi
        done
    fi
    task_conda_script="$task_conda_base/etc/profile.d/conda.sh"
    if [[ -z "$task_conda_base" || ! -f "$task_conda_script" ]]; then
        echo 'Conda initialization script not found; set FASTTRELLIS2_CONDA_BASE.' >&2
        return 1
    fi

    source "$task_conda_script"
    if [[ -n "${FASTTRELLIS2_CONDA_PREFIX:-}" ]]; then
        conda activate "$FASTTRELLIS2_CONDA_PREFIX"
    else
        conda activate "${FASTTRELLIS2_CONDA_ENV:-fasttrellis2}"
    fi
}

