import importlib.util
import json
import platform
import subprocess
import sys

mods = ["torch", "transformers", "datasets", "peft", "accelerate", "bitsandbytes", "trl"]
result = {"python": sys.version, "platform": platform.platform(), "modules": {}}
for name in mods:
    spec = importlib.util.find_spec(name)
    if spec is None:
        result["modules"][name] = None
        continue
    try:
        module = __import__(name)
        result["modules"][name] = getattr(module, "__version__", "present")
    except Exception as exc:
        result["modules"][name] = f"ERROR: {exc!r}"
if importlib.util.find_spec("torch"):
    import torch
    result["torch_cuda"] = torch.cuda.is_available()
    result["torch_cuda_version"] = torch.version.cuda
print(json.dumps(result, indent=2))
