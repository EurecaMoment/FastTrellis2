#!/usr/bin/env bash
set -euo pipefail

task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_runner="$task_repo/fasttrellis2/deploy/launch.sh"
task_pid_file="${FASTTRELLIS2_PID_FILE:-$task_repo/deploy/fasttrellis2.pid}"
task_log_file="${FASTTRELLIS2_LOG_FILE:-$task_repo/deploy/logs/service.log}"
if [[ ! -s "$task_pid_file" ]]; then
    echo "FastTrellis2 PID file not found: $task_pid_file" >&2
    exit 1
fi
read -r task_pid < "$task_pid_file"
if [[ ! "$task_pid" =~ ^[0-9]+$ ]]; then
    echo "Invalid PID in $task_pid_file." >&2
    exit 1
fi
if ! kill -0 "$task_pid" 2>/dev/null; then
    rm -f "$task_pid_file"
    echo 'FastTrellis2 is already stopped.'
    exit 0
fi
task_cmdline="$(tr '\0' ' ' < "/proc/$task_pid/cmdline" 2>/dev/null || true)"
if [[ "$task_cmdline" != *"$task_runner"* &&
      "$task_cmdline" != *"$task_repo/fasttrellis2/runtime/trellis_service.py"* ]]; then
    echo "PID $task_pid does not identify FastTrellis2; refusing to signal it." >&2
    exit 1
fi

kill -TERM "$task_pid"
for task_attempt in {1..60}; do
    if ! kill -0 "$task_pid" 2>/dev/null; then
        rm -f "$task_pid_file"
        echo 'FastTrellis2 stopped.'
        exit 0
    fi
    task_state="$(awk '{print $3}' "/proc/$task_pid/stat" 2>/dev/null || true)"
    if [[ "$task_state" == Z ]]; then
        rm -f "$task_pid_file"
        echo 'FastTrellis2 stopped.'
        exit 0
    fi
    sleep 1
done
echo "FastTrellis2 did not stop after SIGTERM; inspect pid $task_pid and $task_log_file." >&2
exit 1
