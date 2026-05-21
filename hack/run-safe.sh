#!/usr/bin/env bash
# Combined resource limiter: systemd cgroups (CPU + memory) + NVML (GPU memory).
#
# Usage:
#   ./run-safe.sh --memory 8192 --cpu 2 --gpu-memory 8192 --timeout 600 \
#     ~/pge-jax/.venv/bin python hack/benchmark.py
#
# All three limits are applied:
#   --memory   -> systemd MemoryMax (kills process if it exceeds)
#   --cpu      -> systemd CPUQuota (caps CPU time)
#   --gpu-memory -> pynvml deviceSetMemoryLimit (caps GPU VRAM)

set -euo pipefail

MEMORY_MB=8192
CPU_QUOTA="1"
GPU_MEMORY_MB=""
TIMEOUT_SECS=3600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --memory)      MEMORY_MB="$2"; shift 2 ;;
        --cpu)         CPU_QUOTA="$2"; shift 2 ;;
        --gpu-memory)  GPU_MEMORY_MB="$2"; shift 2 ;;
        --timeout)     TIMEOUT_SECS="$2"; shift 2 ;;
        *)             break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 --memory MB --cpu N --gpu-memory MB --timeout SECS <cmd> [args...]"
    exit 1
fi

CMD=("$@")

echo "=== Safe Run (kernel limits) ==="
echo "  Memory:    ${MEMORY_MB} MB   (systemd MemoryMax)"
echo "  CPU:       ${CPU_QUOTA} CPUs  (systemd CPUQuota)"
if [[ -n "$GPU_MEMORY_MB" ]]; then
    echo "  GPU Mem:   ${GPU_MEMORY_MB} MB  (pynvml limit)"
fi
echo "  Timeout:   ${TIMEOUT_SECS}s"
echo "  Command:   ${CMD[*]}"
echo "================================="

# Build the command chain:
# 1. gpu-limit.py caps GPU VRAM via NVML
# 2. systemd-run wraps in cgroup for CPU+memory
# 3. timeout kills after SECS

GPU_WRAPPER=""
if [[ -n "$GPU_MEMORY_MB" ]]; then
    GPU_WRAPPER="$(dirname "$0")/gpu-limit.py --memory $GPU_MEMORY_MB"
fi

# CPU quota in %
CPU_PERCENT=$(( CPU_QUOTA * 100000 ))

if [[ -n "$GPU_WRAPPER" ]]; then
    FINAL_CMD="$GPU_WRAPPER -- ${CMD[*]}"
else
    FINAL_CMD="${CMD[*]}"
fi

exec timeout "$TIMEOUT_SECS" \
    systemd-run --scope \
        --property="MemoryMax=${MEMORY_MB}M" \
        --property="MemoryHigh=${MEMORY_MB}M" \
        --property="CPUQuota=${CPU_PERCENT}%" \
    bash -c "$FINAL_CMD"
