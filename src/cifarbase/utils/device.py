"""Device selection that checks the device actually works.

`torch.cuda.is_available()` only reports that a driver and a device are present.
It says nothing about whether the installed torch was *compiled* for that
device's compute capability, and when it was not, every kernel launch fails with

    CUDA error: no kernel image is available for execution on the device

at whatever line first touches the GPU, a long way from the cause. Kaggle hits
this today: its default GPU accelerator is a Tesla P100 (sm_60) and recent torch
builds ship sm_70 and up. So probe with a real launch, and if a GPU is present
but unusable, say exactly what is wrong and stop.
"""
import os

import torch

ON_KAGGLE = os.path.isdir("/kaggle/working")

_UNUSABLE = (
    "\n  Refusing to fall back to CPU: inside a Kaggle session that spends the GPU"
    "\n  quota on CPU arithmetic and may hit the wall-clock limit before finishing."
    "\n"
    "\n  Fixes:"
    "\n    1. Set Accelerator to GPU T4 x2 (sm_75) on the kernel's settings page,"
    "\n       then Save & Run All from the UI. A CLI push resets the accelerator,"
    "\n       because `enable_gpu: true` resolves to the P100."
    "\n    2. Pin an older \"docker_image\" whose torch still ships sm_60."
    "\n    3. CIFAR_ALLOW_CPU=1 to run anyway (pair it with CIFAR_QUICK=1).")


def _probe():
    """Launch one real kernel. Returns None on success, else the exception."""
    try:
        x = torch.randn(64, 64, device="cuda")
        (x @ x).sum().item()          # forces a sync, so async errors surface here
        return None
    except Exception as exc:          # noqa: BLE001 - any failure means unusable
        return exc


def pick_device(allow_cpu=None):
    print(f"torch {torch.__version__}")
    if allow_cpu is None:
        allow_cpu = os.environ.get("CIFAR_ALLOW_CPU", "0") == "1"

    if not torch.cuda.is_available():
        if ON_KAGGLE and not allow_cpu:
            raise SystemExit("!! no CUDA device visible in a Kaggle session." + _UNUSABLE)
        print("device: cpu (no CUDA device visible -- expect this to be slow)")
        return torch.device("cpu")

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    failure = _probe()
    if failure is None:
        torch.backends.cudnn.benchmark = True
        # TF32 is a ~2x matmul win on Ampere+ and a no-op on T4/P100.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"device: cuda ({name}, {total:.1f} GB)")
        return torch.device("cuda")

    capability = torch.cuda.get_device_capability(0)
    print(f"!! {name} is visible but cannot run this torch build:")
    print(f"    {type(failure).__name__}: {str(failure).splitlines()[0]}")
    print(f"    device is sm_{capability[0]}{capability[1]}, this torch "
          f"({torch.__version__}) was built for {', '.join(torch.cuda.get_arch_list())}")
    if allow_cpu:
        print("    CIFAR_ALLOW_CPU=1 is set: continuing on CPU, which will be slow.")
        return torch.device("cpu")
    raise SystemExit(_UNUSABLE)
