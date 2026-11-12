#!/usr/bin/env bash
set -euo pipefail

task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_runner="$task_repo/fasttrellis2/deploy/launch.sh"
if [[ -n "${FASTTRELLIS2_CONFIG:-}" ]]; then
    task_config="$FASTTRELLIS2_CONFIG"
elif [[ -f "$task_repo/deploy/config.json" ]]; then
    task_config="$task_repo/deploy/config.json"
else
    task_config="$task_repo/fasttrellis2/deploy/config.example.json"
fi
task_pid_file="${FASTTRELLIS2_PID_FILE:-$task_repo/deploy/fasttrellis2.pid}"
task_log_file="${FASTTRELLIS2_LOG_FILE:-$task_repo/deploy/logs/service.log}"

if [[ ! -f "$task_config" ]]; then
    echo "FastTrellis2 config not found: $task_config" >&2
    echo 'Set FASTTRELLIS2_CONFIG to the deployment config file.' >&2
    exit 1
fi
if [[ "$task_config" != /* ]]; then
    task_config="$(cd "$(dirname "$task_config")" && pwd)/$(basename "$task_config")"
fi
mkdir -p "$(dirname "$task_pid_file")" "$(dirname "$task_log_file")"

if [[ -s "$task_pid_file" ]]; then
    read -r task_old_pid < "$task_pid_file"
    if [[ "$task_old_pid" =~ ^[0-9]+$ ]] && kill -0 "$task_old_pid" 2>/dev/null; then
        task_old_cmdline="$(tr '\0' ' ' < "/proc/$task_old_pid/cmdline" 2>/dev/null || true)"
        if [[ "$task_old_cmdline" == *"$task_runner"* ||
              "$task_old_cmdline" == *"$task_repo/fasttrellis2/runtime/trellis_service.py"* ]]; then
            echo "FastTrellis2 is already running (pid $task_old_pid)."
            exit 0
        fi
        echo "PID file $task_pid_file points to unrelated process $task_old_pid; refusing to replace it." >&2
        exit 1
    fi
    rm -f "$task_pid_file"
fi

source "$task_repo/fasttrellis2/deploy/conda-env.sh"
fasttrellis2_activate_conda
task_port="$(python - "$task_config" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
port = value.get('port', 3956)
if type(port) is not int or not 1 <= port <= 65535:
    raise SystemExit('config port must be an integer from 1 to 65535')
print(port)
PY
)"
if python - "$task_port" <<'PY'
import socket
import sys
with socket.socket() as probe:
    probe.settimeout(.5)
    raise SystemExit(0 if probe.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0 else 1)
PY
then
    echo "Port $task_port is already accepting connections; refusing to start a second service." >&2
    exit 1
fi

nohup bash "$task_runner" "$task_config" >>"$task_log_file" 2>&1 </dev/null &
task_pid=$!
printf '%s\n' "$task_pid" > "$task_pid_file"
sleep 1
if ! kill -0 "$task_pid" 2>/dev/null; then
    rm -f "$task_pid_file"
    echo "FastTrellis2 exited during startup; inspect $task_log_file." >&2
    exit 1
fi
printf 'FastTrellis2 launched directly on %s (pid %s, port %s).\n' \
    "$(hostname -s)" "$task_pid" "$task_port"
printf 'Conda environment: %s\nConfig: %s\nLog: %s\n' \
    "${FASTTRELLIS2_CONDA_PREFIX:-${FASTTRELLIS2_CONDA_ENV:-fasttrellis2}}" \
    "$task_config" "$task_log_file"
echo 'The model loads in the background; poll /ready until state=ready.'

