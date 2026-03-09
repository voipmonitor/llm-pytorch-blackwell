# =============================================================================
# Bleeding-edge LLM inference stack for NVIDIA RTX 5090 (Blackwell, sm_120a)
# Base: Ubuntu 24.04 + CUDA 13.1.1 + cuDNN
# Stack: PyTorch nightly (cu130) -> Triton (git) -> FlashInfer (git) ->
#        vLLM (source + PRs) -> SGLang (source) -> Transformers (git main)
# =============================================================================

# -- args (override at build time) -------------------------------------------
ARG CUDA_VERSION=13.1.1
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.12
ARG MAX_JOBS=128
ARG NVCC_THREADS=8
ARG TORCH_CUDA_ARCH_LIST="12.0a"
ARG FLASHINFER_CUDA_ARCH_LIST="12.0a"
ARG VLLM_FLASH_ATTN_VERSION=2

# =============================================================================
# Stage 1: base – system packages, Python, uv
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS base

ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libgomp1 \
    numactl \
    libnuma-dev \
    libibverbs-dev \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create venv with uv
RUN uv venv /opt/venv --python python${PYTHON_VERSION}
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# Ensure pip is available (needed for builds)
RUN uv pip install pip setuptools wheel

# ccache config for faster rebuilds
ENV CCACHE_DIR=/root/.ccache \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# =============================================================================
# Stage 2: deps – install vLLM dependencies (pulls in the dependency tree),
#          then override with nightly/bleeding-edge versions
# =============================================================================
FROM base AS deps

ARG TORCH_CUDA_ARCH_LIST
ARG FLASHINFER_CUDA_ARCH_LIST
ARG VLLM_FLASH_ATTN_VERSION
ARG MAX_JOBS
ARG NVCC_THREADS

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST} \
    VLLM_FLASH_ATTN_VERSION=${VLLM_FLASH_ATTN_VERSION} \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS}

# -- Step 1: Clone vLLM and apply unmerged PRs --------------------------------
WORKDIR /build
RUN git clone https://github.com/vllm-project/vllm.git

# Cherry-pick unmerged PRs before building:
#   PR #35219 - SKIPPED (merge conflict with main, waiting for author rebase)
#   PR #35675 - [Bug Fix] Qwen3.5-nvfp4 MTP Speculative Decoding Weight Shape Mismatch
#   PR #35687 - [Bugfix] Treat <tool_call> as implicit reasoning end in Qwen3 parser
#   PR #35936 - fix: tool_choice="required" falls back to tool_parser for non-JSON formats
WORKDIR /build/vllm
RUN git config user.email "build@docker" && git config user.name "build" && \
    git fetch origin \
        pull/35675/head:pr-35675 \
        pull/35687/head:pr-35687 \
        pull/35936/head:pr-35936 && \
    git cherry-pick --no-commit pr-35675 && \
    git cherry-pick --no-commit pr-35687 && \
    git cherry-pick --no-commit ac8975e1154e && \
    git cherry-pick --no-commit 0f3a813751ae && \
    git cherry-pick --no-commit 50662adb7bba

# -- Step 2: Install vLLM Python dependencies (pulls stable torch etc.) ------
RUN uv pip install -r requirements/build.txt && \
    uv pip install -r requirements/common.txt && \
    uv pip install setuptools_scm

# -- Step 3: Override with nightly PyTorch (cu130 for CUDA 13.x) --------------
RUN uv pip install --pre \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130 \
    --force-reinstall

# -- Step 4: Install latest NCCL for CUDA 13 ---------------------------------
RUN uv pip install --force-reinstall nvidia-nccl-cu13

# Verify nightly torch is installed
RUN python -c "import torch; v=torch.__version__; print(f'Nightly torch: {v}'); assert 'dev' in v, f'NOT NIGHTLY: {v}'"

# -- Step 5: Install Triton from git source ----------------------------------
WORKDIR /build
RUN git clone https://github.com/triton-lang/triton.git
WORKDIR /build/triton
# Install Triton build deps
RUN uv pip install pybind11 cmake lit

# Build Triton from source (regular install, not editable)
RUN --mount=type=cache,target=/root/.ccache \
    MAX_JOBS=${MAX_JOBS} pip install --no-build-isolation -v .

# -- Step 6: Install CUTLASS 4.x DSL (latest) --------------------------------
RUN uv pip install "nvidia-cutlass-dsl[cu13]"

# -- Step 7: Install transformers from git main -------------------------------
WORKDIR /build/vllm
RUN uv pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"

# -- Step 8: Install other bleeding-edge packages ----------------------------
# --no-deps to prevent uv from pulling stable torch back in
RUN uv pip install --no-deps --upgrade \
    accelerate \
    safetensors \
    tokenizers \
    sentencepiece \
    bitsandbytes

# DeepGEMM — FP8 GEMM kernels (JIT compiled at runtime)
ARG DEEPGEMM_CACHEBUST=2
RUN cd /tmp && git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git && \
    cd DeepGEMM && pip install --no-build-isolation --no-deps . && \
    rm -rf /tmp/DeepGEMM

# xformers nightly
RUN uv pip install --pre xformers --index-url https://download.pytorch.org/whl/nightly/cu130 --no-deps || true

# Verify nightly torch is still in place
RUN python -c "import torch; v=torch.__version__; print(f'PyTorch {v}'); assert 'dev' in v, f'STABLE TORCH DETECTED: {v}'"

# =============================================================================
# Stage 3: build FlashInfer from source (JIT mode, git main)
# =============================================================================
FROM deps AS flashinfer-build

WORKDIR /build
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git

WORKDIR /build/flashinfer
# Install FlashInfer's runtime deps (torch/cutlass-dsl already in deps stage)
RUN uv pip install "apache-tvm-ffi>=0.1.6" "nvidia-cudnn-frontend>=1.13.0" packaging
# Regular install (not editable) so we don't need /build at runtime
# --no-deps: prevents pulling different torch/triton versions
RUN pip install -v --no-build-isolation --no-deps .


# =============================================================================
# Stage 4: build vLLM from source (optimized for sm_120)
# =============================================================================
FROM flashinfer-build AS vllm-build

WORKDIR /build/vllm

# Verify nightly torch survived FlashInfer install
RUN python -c "import torch; v=torch.__version__; print(f'PyTorch {v}'); assert 'dev' in v, f'STABLE TORCH DETECTED: {v}'"

# Don't let vLLM install its own torch
RUN python use_existing_torch.py

# Compile vLLM from source with full parallelism
# Regular install (not editable) — result goes into /opt/venv/lib/...
RUN --mount=type=cache,target=/root/.ccache \
    VLLM_TARGET_DEVICE=cuda \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS} \
    pip install --no-build-isolation --no-deps . 2>&1

# Post-build: reinstall transformers from git main
RUN uv pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"

# Verify the build
RUN python -c "import vllm; print(f'vLLM {vllm.__version__}')"

# =============================================================================
# Stage 5: build SGLang from source (sgl-kernel + python package)
# =============================================================================
FROM vllm-build AS sglang-build

ARG MAX_JOBS
ARG NVCC_THREADS

RUN git clone --recursive https://github.com/sgl-project/sglang.git /opt/sglang

# Cherry-pick unmerged PRs (each guarded so it's skipped once merged into main):
#   PR #18904 - Pass kv scales to paged attention in flashinfer backend (FP8 cache fix)
#   PR #19948 - Auto-disable DEEPGEMM_SCALE_UE8M0 for non-ue8m0 checkpoints on Blackwell
#   PR #19951 - Remove spurious device arg from cutlass_fp4_group_mm calls
#   PR #19897 - Fix EAGLE-v2 NaN on radix cache prefix hits (zero-fill draft KV)
#   PR #19963 - Fix missing arch suffix in _get_cuda_arch_list for Blackwell/Hopper JIT
WORKDIR /opt/sglang
RUN git config user.email "build@docker" && git config user.name "build" && \
    git fetch origin \
        pull/18904/head:pr-18904 \
        pull/19897/head:pr-19897 \
        pull/19948/head:pr-19948 \
        pull/19951/head:pr-19951 \
        pull/19963/head:pr-19963 && \
    (git cherry-pick --no-commit 8912eb8801a3 || git cherry-pick --abort 2>/dev/null || true) && \
    (git cherry-pick --no-commit 97d666ae5109 || git cherry-pick --abort 2>/dev/null || true) && \
    (git cherry-pick --no-commit 45e03472fa46 || git cherry-pick --abort 2>/dev/null || true) && \
    (git cherry-pick --no-commit fc05fc6758aa || git cherry-pick --abort 2>/dev/null || true) && \
    (git cherry-pick --no-commit 340f619 b38295b 0ba073f || git cherry-pick --abort 2>/dev/null || true)

# Cherry-pick PCIe allreduce support (lukealonso fork, not yet in upstream PR)
RUN git remote add lukealonso https://github.com/lukealonso/sglang.git && \
    git fetch lukealonso 5bb89b03afe46fbd012da9f50bb5992673342123 && \
    (git cherry-pick --no-commit 5bb89b03afe46fbd012da9f50bb5992673342123 || git cherry-pick --abort 2>/dev/null || true)

# Build sgl-kernel for Blackwell (sm_120a) + Hopper FA3 (sm_90a for VisionAttention)
WORKDIR /opt/sglang/sgl-kernel
RUN uv pip install scikit-build-core
RUN --mount=type=cache,target=/root/.ccache \
    MAX_JOBS=${MAX_JOBS} \
    make build CMAKE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_ARCHITECTURES='90a;120a' -DENABLE_BELOW_SM90=0"

# Install SGLang Python package (editable so tests/benchmarks are accessible)
# --no-deps: prevents pip from pulling stable torch, sgl-kernel, flashinfer from PyPI
# which would overwrite our source-built versions (sm_120a)
WORKDIR /opt/sglang
RUN pip install --no-build-isolation --no-deps -e "./python[all]"

# Install SGLang's missing runtime deps (excluding packages we built from source:
# torch, torchvision, torchaudio, triton, sgl-kernel, flashinfer, vllm, transformers)
# Extract from pyproject.toml and filter out already-installed packages
RUN python -c "\
import tomllib, sys; \
data = tomllib.load(open('python/pyproject.toml', 'rb')); \
deps = data.get('project', {}).get('dependencies', []); \
extras = data.get('project', {}).get('optional-dependencies', {}); \
all_deps = list(deps); \
for k, v in extras.items(): \
    all_deps.extend(v); \
skip = {'torch', 'torchvision', 'torchaudio', 'triton', 'sgl-kernel', 'sgl_kernel', \
        'flashinfer', 'flashinfer-python', 'flashinfer_python', 'vllm', 'transformers', \
        'xformers', 'deepgemm', 'accelerate', 'safetensors', 'tokenizers', 'sentencepiece', \
        'bitsandbytes'}; \
import re; \
names = []; \
for d in all_deps: \
    name = re.split(r'[<>=!;\[@ ]', d)[0].strip().lower().replace('-','_'); \
    if name and name not in skip: \
        names.append(d); \
print('\n'.join(sorted(set(names)))); \
" > /tmp/sglang_deps.txt && \
    echo "--- SGLang deps to install ---" && cat /tmp/sglang_deps.txt && \
    uv pip install -r /tmp/sglang_deps.txt || true && \
    rm /tmp/sglang_deps.txt && \
    uv pip install ipython pynvml orjson "apache-tvm-ffi>=0.1.6" || true

# Verify nightly torch wasn't overwritten (sgl-kernel/vllm need GPU, can't import here)
RUN python -c "import torch; v=torch.__version__; print(f'PyTorch {v}'); assert 'dev' in v, f'TORCH DOWNGRADED: {v}'"

# Reinstall bleeding-edge packages (--no-deps to be safe)
RUN uv pip install --no-deps "transformers @ git+https://github.com/huggingface/transformers.git@main" && \
    uv pip install "nvidia-cutlass-dsl[cu13]"

# Verify SGLang
RUN python -c "import sglang; print(f'SGLang {sglang.__version__}')"

# Copy Triton MoE configs for Blackwell GPUs
COPY configs/ /tmp/triton-configs/
RUN TRITON_CONFIGS_DIR="/opt/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_6_0" && \
    mkdir -p "$TRITON_CONFIGS_DIR" && \
    for f in /tmp/triton-configs/*.json; do \
        base=$(basename "$f"); \
        cp "$f" "$TRITON_CONFIGS_DIR/$base"; \
        cp "$f" "${TRITON_CONFIGS_DIR}/$(echo "$base" | sed 's/\.json$/_down.json/')"; \
        nodt=$(echo "$base" | sed 's/,dtype=[^.]*//'); \
        cp "$f" "$TRITON_CONFIGS_DIR/$nodt"; \
        cp "$f" "${TRITON_CONFIGS_DIR}/$(echo "$nodt" | sed 's/\.json$/_down.json/')"; \
    done && \
    rm -rf /tmp/triton-configs && \
    ls -la "$TRITON_CONFIGS_DIR/"

# -- Patch FlashInfer: add missing GDC compile flags for PDL synchronization --
# Without these flags, wait_on_dependent_grids() compiles as a no-op, causing
# race conditions with PDL-enabled CUTLASS GEMM kernels on SM100/SM120.
# See: https://github.com/flashinfer-ai/flashinfer/pull/2716
ARG FLASHINFER_PATCH_CACHEBUST=1
RUN python -c "\
import flashinfer.jit.gemm.core as m; \
f = m.__file__; \
content = open(f).read(); \
GDC_FLAGS = '            \"-DCUTLASS_ENABLE_GDC_FOR_SM100=1\",\n            \"-DCUTLASS_ENABLE_GDC_FOR_SM90=1\",\n'; \
content = content.replace( \
    '\"-DENABLE_FP4\",\n', \
    '\"-DENABLE_FP4\",\n' + GDC_FLAGS); \
import re; \
content = re.sub( \
    r'(\"-DENABLE_BF16\",\n)(        \],\n        extra_cflags)', \
    r'\1' + GDC_FLAGS + r'\2', content); \
content = content.replace( \
    'extra_cuda_cflags=nvcc_flags,\n    )\n\n\ndef gen_trtllm', \
    'extra_cuda_cflags=nvcc_flags\n        + [\n' + GDC_FLAGS + '        ],\n    )\n\n\ndef gen_trtllm'); \
open(f, 'w').write(content); \
assert content.count('GDC_FOR_SM100') >= 7, f'Expected >=7, got {content.count(\"GDC_FOR_SM100\")}'; \
print(f'OK: {content.count(\"GDC_FOR_SM100\")} GDC_FOR_SM100 flags patched'); \
"

# -- Upgrade CUTLASS headers to 4.4.1 (PDL fixes, SM120 memory fences) --------
RUN FLASHINFER_CUTLASS="$(python -c "import flashinfer, os; \
    base = os.path.dirname(flashinfer.__file__); \
    candidates = [os.path.join(base, s) for s in ['data/cutlass', 'cutlass']]; \
    print(next(p for p in candidates if os.path.isdir(p)))")" && \
    echo "CUTLASS dir: ${FLASHINFER_CUTLASS}" && \
    cd /tmp && \
    curl -sL https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.4.1.tar.gz -o cutlass.tar.gz && \
    tar xzf cutlass.tar.gz && \
    rm -rf "${FLASHINFER_CUTLASS}/include" "${FLASHINFER_CUTLASS}/tools" && \
    cp -a cutlass-4.4.1/include "${FLASHINFER_CUTLASS}/include" && \
    cp -a cutlass-4.4.1/tools "${FLASHINFER_CUTLASS}/tools" && \
    rm -rf /tmp/cutlass* && \
    grep "CUTLASS_MAJOR\|CUTLASS_MINOR\|CUTLASS_PATCH" "${FLASHINFER_CUTLASS}/include/cutlass/version.h" | head -3

# -- Upgrade cuDNN to latest ------------------------------------------------
RUN uv pip install --upgrade nvidia-cudnn-cu13 && \
    python -c "import torch; print(f'cuDNN: {torch.backends.cudnn.version()}')"

# Clean up pip/uv caches before copying to final stage
RUN pip cache purge 2>/dev/null || true && \
    uv cache clean 2>/dev/null || true && \
    rm -rf /root/.cache/pip /root/.cache/uv /tmp/pip-* && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -name "*.pyc" -delete 2>/dev/null || true && \
    find /opt/venv -name "*.pyi" -delete 2>/dev/null || true && \
    find /opt/venv -name "*.a" -delete 2>/dev/null || true

# =============================================================================
# Stage 6: final – runtime image with CUDA devel (needed for FlashInfer JIT)
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS final

ARG PYTHON_VERSION=3.12
ARG VLLM_FLASH_ATTN_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG FLASHINFER_CUDA_ARCH_LIST
ARG DEBIAN_FRONTEND=noninteractive

# Runtime + JIT compilation deps (FlashInfer JIT needs nvcc, gcc, Python headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    gcc \
    g++ \
    libc6-dev \
    libgomp1 \
    numactl \
    libnuma1 \
    libibverbs1 \
    libjpeg8 \
    libpng16-16t64 \
    ninja-build \
    pciutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Copy venv and SGLang source (editable install needs the source tree)
COPY --from=sglang-build /opt/venv /opt/venv
COPY --from=sglang-build /opt/sglang /opt/sglang

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime env
ENV VLLM_FLASH_ATTN_VERSION=${VLLM_FLASH_ATTN_VERSION} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST} \
    NCCL_P2P_DISABLE=0 \
    NCCL_IB_DISABLE=0 \
    CUDA_DEVICE_MAX_CONNECTIONS=32

# JIT cache directories — mount as named volumes to persist across restarts
ARG BUILD_ID=unknown
ENV JIT_BUILD_ID=${BUILD_ID} \
    TRITON_CACHE_DIR=/cache/jit/triton \
    TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
    VLLM_CACHE_DIR=/cache/jit/vllm \
    FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
    TVM_FFI_CACHE_DIR=/cache/jit/tvm-ffi \
    XDG_CACHE_HOME=/cache/jit \
    HF_HOME=/root/.cache/huggingface

RUN mkdir -p /cache/jit/triton /cache/jit/torch_extensions /cache/jit/vllm

# Ensure CUDA libs are found at runtime
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# JIT cache invalidation — runs on any shell (bash -i, entrypoint, etc.)
COPY jit-cache-invalidate.sh /usr/local/bin/jit-cache-invalidate.sh
RUN chmod +x /usr/local/bin/jit-cache-invalidate.sh && \
    echo 'source /usr/local/bin/jit-cache-invalidate.sh' >> /root/.bashrc

# BASH_ENV ensures cache invalidation runs even with: docker run --entrypoint /bin/bash ... -c "cmd"
ENV BASH_ENV=/usr/local/bin/jit-cache-invalidate.sh

# Entrypoint with JIT warmup
COPY entrypoint.sh /entrypoint.sh
COPY warmup_jit.py /workspace/warmup_jit.py

WORKDIR /workspace

# Smoke test (import only, no GPU needed)
RUN python -c "\
import torch; \
import vllm; \
import flashinfer; \
import transformers; \
import sglang; \
print(f'PyTorch:      {torch.__version__}'); \
print(f'CUDA:         {torch.version.cuda}'); \
print(f'vLLM:         {vllm.__version__}'); \
print(f'FlashInfer:   {flashinfer.__version__}'); \
print(f'Transformers: {transformers.__version__}'); \
print(f'SGLang:       {getattr(sglang, \"__version__\", \"editable-install\")}'); \
print(f'DeepGEMM:     installed (JIT, needs GPU)'); \
assert 'dev' in torch.__version__, f'FINAL IMAGE HAS STABLE TORCH: {torch.__version__}'; \
print('OK: nightly torch confirmed in final image'); \
"

ENTRYPOINT ["/entrypoint.sh"]
