"""Reuse the base image's compiled vLLM extensions with the pinned fork."""
import importlib.util
from pathlib import Path

installed = Path(importlib.util.find_spec("vllm").origin).parent
target = Path("/opt/arvq/runtime/vllm")
for source in installed.rglob("*.so"):
    destination = target / source.relative_to(installed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)
version = installed / "_version.py"
if version.exists():
    (target / "_version.py").write_bytes(version.read_bytes())
print(f"Linked native extensions from {installed}")
