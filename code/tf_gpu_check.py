import os
import time
import subprocess

def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"[ERR running {cmd}] {e}"

print("=== BASIC ENV ===")
print("python:", run(["python", "-c", "import sys; print(sys.executable)"]))
print("pip   :", run(["python", "-m", "pip", "--version"]))
print("conda :", run(["bash", "-lc", "conda info --envs | sed -n '1,5p'"]))  # optional

print("\n=== NVIDIA DRIVER ===")
print(run(["bash", "-lc", "nvidia-smi -L || true"]))
print(run(["bash", "-lc", "nvidia-smi | sed -n '1,15p' || true"]))

print("\n=== IMPORT CHECK (core libs) ===")
missing = []
for pkg in ["numpy", "pandas", "sklearn", "matplotlib"]:
    try:
        m = __import__(pkg)
        ver = getattr(m, "__version__", "unknown")
        print(f"OK  {pkg:10} {ver}")
    except Exception as e:
        print(f"MISS {pkg:10} -> {e}")
        missing.append(pkg)

print("\n=== TENSORFLOW GPU CHECK ===")
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("Built with ROCm:", tf.test.is_built_with_rocm())

    # List devices
    devices = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices("GPU")
    print("All devices:", devices)
    print("GPU devices:", gpus)

    # Build info (useful to confirm CUDA/cuDNN linkage)
    try:
        build = tf.sysconfig.get_build_info()
        print("Build info:", build)
    except Exception as e:
        print("Build info: (unavailable)", e)

    # Memory growth (prevents TF from grabbing all VRAM at once)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("set_memory_growth err:", e)

    if not gpus:
        print("\n❌ RESULT: TensorFlow does NOT see any GPU.")
        print("Common causes: wrong TF build, CUDA/cuDNN mismatch, driver issue, env not active.")
        raise SystemExit(1)

    print("\n✅ GPU is visible. Now verify actual computation runs on GPU...")

    # Force log device placement (optional)
    tf.debugging.set_log_device_placement(False)

    # Simple matmul test on GPU + timing
    import numpy as np

    with tf.device("/GPU:0"):
        a = tf.random.uniform((4096, 4096), dtype=tf.float32)
        b = tf.random.uniform((4096, 4096), dtype=tf.float32)

        # Warmup
        c = tf.matmul(a, b)
        _ = c.numpy()

        t0 = time.time()
        for _ in range(5):
            c = tf.matmul(a, b)
            _ = c.numpy()
        t1 = time.time()

    print(f"MatMul on GPU: OK  (avg {(t1-t0)/5:.4f} sec/iter)")

    # Extra: show TF thinks GPU is available
    print("tf.test.gpu_device_name():", tf.test.gpu_device_name())

    print("\n✅ RESULT: ENV OK + TensorFlow is using GPU (CUDA) successfully.")

except Exception as e:
    print("\n❌ TensorFlow check failed:", e)
    raise