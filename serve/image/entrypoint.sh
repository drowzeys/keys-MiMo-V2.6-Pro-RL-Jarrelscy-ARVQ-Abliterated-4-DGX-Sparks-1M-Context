#!/bin/bash
# The fork tree shadows the base vLLM package. Link extension packages the
# base image has and this tree does not, then start vLLM.
# Keep the fork's vllm_flash_attn/__init__.py. It exports
# compile_flash_attn_varlen_func_from_specs, which the base package does not.
# The stub is only missing the compiled subpackages (layers, cute, ops).
set -euo pipefail
sys=/usr/local/lib/python3.12/dist-packages/vllm
fork=/opt/arvq/runtime/vllm

link_children() {
  local src="$1" dst="$2" item base
  mkdir -p "$dst"
  for item in "$src"/*; do
    base=$(basename "$item")
    if [ ! -e "$dst/$base" ]; then
      ln -s "$item" "$dst/$base"
    fi
  done
}

if [ -d "$sys" ]; then
  for item in "$sys"/*; do
    base=$(basename "$item")
    if [ ! -e "$fork/$base" ]; then
      ln -s "$item" "$fork/$base"
    fi
  done
  # Real directories, not a symlink of the package. The fork __init__ treats
  # a symlinked cute/ as upstream flash_attn source and rewrites sys.modules.
  if [ -d "$fork/vllm_flash_attn" ]; then
    for sub in layers cute ops; do
      if [ -d "$sys/vllm_flash_attn/$sub" ] && [ ! -e "$fork/vllm_flash_attn/$sub" ]; then
        link_children "$sys/vllm_flash_attn/$sub" "$fork/vllm_flash_attn/$sub"
      fi
    done
  fi
fi

if [ "${1:-}" = "link-only" ]; then
  exit 0
fi

exec /opt/arvq/.venv/bin/python -m vllm.entrypoints.cli.main "$@"
