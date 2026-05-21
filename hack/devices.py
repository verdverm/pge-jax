"""Print JAX devices and backend info."""

import jax

print("Devices:", jax.devices())

for i, dev in enumerate(jax.devices()):
    print(f"  Device {i}: {dev}")
    if hasattr(dev, "device_kind"):
        print(f"    Kind: {dev.device_kind}")
    if hasattr(dev, "memory_limit"):
        print(f"    Memory limit: {dev.memory_limit}")
