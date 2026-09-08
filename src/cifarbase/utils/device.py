"""Device selection that checks the device actually works.

`torch.cuda.is_available()` only reports that a driver and a device are present.
It says nothing about whether the installed torch was *compiled* for that
device's compute capability, and when it was not, every kernel launch fails with

    CUDA error: no kernel image is available for execution on the device

at whatever line first touches the GPU, a long way from the cause. Kaggle hits
this today: its default GPU accelerator is a Tesla P100 (sm_60) and recent torch
builds ship sm_70 and up. So probe with a real launch, and if a GPU is present
but unusable, say exactly what is wrong and stop.

Apple Silicon (MPS) is tried after CUDA and before CPU, on the same terms: it is
probed rather than trusted. It is worth having -- measured at ~13x CPU on an
M-series part, which is the difference between a 60-epoch run taking 45 minutes
and taking 10 hours -- but two things about it are not the same as CUDA:

  * `deterministic` in the config pins cudnn's algorithm choice and does nothing
    at all here. A "deterministic" run on MPS is only seeded, not reproducible
    kernel-for-kernel.
  * float32 convolutions do not produce bit-identical results to a CUDA build,
    so a number produced here will not exactly reproduce the README's. It should
    land inside the seed-to-seed spread, and if it does not, that is the finding.

CIFAR_DEVICE overrides the whole search when you need a specific one, e.g.
CIFAR_DEVICE=cpu to check that a result is not an MPS artefact.
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


def _probe(device):
    """Launch one real kernel. Returns None on success, else the exception."""
    try:
        x = torch.randn(64, 64, device=device)
        (x @ x).sum().item()          # forces a sync, so async errors surface here
        return None
    except Exception as exc:          # noqa: BLE001 - any failure means unusable
        return exc


def _try_cuda(allow_cpu):
    """A usable cuda device, or None if there is none to be had.

    Raises rather than returning None when a GPU is present but unusable and no
    fallback was allowed: that is a misconfiguration to fix, not a slow path to
    take silently.
    """
    if not torch.cuda.is_available():
        return None

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    failure = _probe("cuda")
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
    if not allow_cpu:
        raise SystemExit(_UNUSABLE)
    print("    CIFAR_ALLOW_CPU=1 is set: falling back.")
    return None


def _try_mps():
    """Apple Silicon, probed the same way cuda is rather than trusted."""
    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        return None
    failure = _probe("mps")
    if failure is not None:
        print(f"!! mps is available but a test kernel failed: "
              f"{type(failure).__name__}: {str(failure).splitlines()[0]}")
        return None
    print("device: mps (Apple Silicon; unified memory, so the resident dataset "
          "costs host RAM)")
    return torch.device("mps")


def pick_device(allow_cpu=None):
    print(f"torch {torch.__version__}")
    if allow_cpu is None:
        allow_cpu = os.environ.get("CIFAR_ALLOW_CPU", "0") == "1"

    forced = os.environ.get("CIFAR_DEVICE")
    if forced:
        # No probe: an explicit request is a statement that you know what is
        # there, and a failure should surface at the first real kernel with a
        # message about that kernel rather than be second-guessed here.
        print(f"device: {forced} (forced by CIFAR_DEVICE)")
        return torch.device(forced)

    device = _try_cuda(allow_cpu)
    if device is not None:
        return device

    device = _try_mps()
    if device is not None:
        return device

    if ON_KAGGLE and not allow_cpu:
        raise SystemExit("!! no CUDA device visible in a Kaggle session." + _UNUSABLE)
    print("device: cpu (no GPU visible -- expect this to be slow)")
    return torch.device("cpu")
