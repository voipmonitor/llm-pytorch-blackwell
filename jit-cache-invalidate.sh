#!/bin/bash
# Invalidate JIT caches when image changes (new BUILD_ID baked into ENV)
# Runs from entrypoint.sh and .bashrc to cover all launch modes.
BUILD_MARKER="/cache/jit/.build-id"
if [ -n "${JIT_BUILD_ID:-}" ] && [ "$JIT_BUILD_ID" != "unknown" ]; then
    if [ ! -f "$BUILD_MARKER" ] || [ "$(cat "$BUILD_MARKER" 2>/dev/null)" != "$JIT_BUILD_ID" ]; then
        echo "New image detected (build: $JIT_BUILD_ID). Clearing JIT caches..."
        rm -rf /cache/jit/flashinfer /cache/jit/tvm-ffi /cache/jit/triton /cache/jit/torch_extensions /cache/jit/vllm
        mkdir -p /cache/jit/triton /cache/jit/torch_extensions /cache/jit/vllm
        echo "$JIT_BUILD_ID" > "$BUILD_MARKER"
    fi
fi
