#!/usr/bin/env python3
"""GPU memory limiter — caps GPU memory before running a subprocess.

Uses pynvml to set a hard memory limit on the GPU, so JAX cannot
allocate more than the specified amount.  This prevents OOM kills
of system processes.

Usage:
    python hack/gpu-limit.py --memory 8192 -- python3 hack/benchmark.py

Or set via env var:
    _GPU_LIMIT_MB=8192 python3 hack/benchmark.py
"""

import argparse
import os
import subprocess
import sys


def limit_gpu_memory(mb: int):
    """Cap GPU 0 to `mb` megabytes using NVML."""
    try:
        import pynvml
    except ImportError:
        print("ERROR: pynvml not installed. Run: pip install pynvml", file=sys.stderr)
        sys.exit(1)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
    limit_bytes = mb * 1024 * 1024

    if limit_bytes > total:
        print(f"WARNING: requested {mb}MB exceeds GPU total ({total // 1024 // 1024}MB), capping to total")
        limit_bytes = total

    pynvml.nvmlDeviceSetMemoryLimit(handle, limit_bytes)
    print(f"GPU memory limited to {mb}MB (total: {total // 1024 // 1024}MB)")

    return handle


def main():
    parser = argparse.ArgumentParser(description="Run a command with GPU memory limits")
    parser.add_argument("--memory", type=int, default=None, help="GPU memory limit in MB")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run")
    args = parser.parse_args()

    mb = args.memory or int(os.environ.get("_GPU_LIMIT_MB", "0"))

    if mb > 0:
        limit_gpu_memory(mb)

    if args.cmd:
        os.execvp(args.cmd[0], args.cmd)
    else:
        print("No command specified. Usage: gpu-limit.py --memory MB -- python3 script.py", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
