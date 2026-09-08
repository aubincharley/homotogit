"""Minimal hardware probe.  No project imports, no installs, no correctness suite."""
import json, os, subprocess, sys

out = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
       "sys.executable": sys.executable}
for name, cmd in (("nvidia-smi -L", ["nvidia-smi", "-L"]),
                  ("nvidia-smi", ["nvidia-smi"])):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        out[name] = (r.stdout or r.stderr).strip()[:1500]
    except Exception as exc:
        out[name] = "FAILED: %r" % (exc,)

try:
    import torch
    out.update({"torch.__file__": torch.__file__,
                "torch.__version__": torch.__version__,
                "torch.version.cuda": torch.version.cuda,
                "torch.cuda.device_count()": torch.cuda.device_count(),
                "torch.cuda.is_available()": torch.cuda.is_available()})
    names, ops = [], []
    for i in range(torch.cuda.device_count()):
        names.append(torch.cuda.get_device_name(i))
        try:
            t = torch.ones(8, device="cuda:%d" % i)
            ops.append({"device": i, "sum": float((t + t).sum())})
        except Exception as exc:
            ops.append({"device": i, "error": repr(exc)})
    out["gpu_names"] = names
    out["tiny_cuda_op"] = ops
except Exception as exc:
    out["torch_import_error"] = repr(exc)

print("PROBE " + json.dumps(out, indent=2), flush=True)
with open("/kaggle/working/probe.json", "w") as fh:
    json.dump(out, fh, indent=2)
