import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

# JIT build the extension (once). Reuses the cached build after the first run.
def _build_ext():
    this_dir = Path(__file__).resolve().parent
    src_dir = this_dir.parent / "csrc"
    sources = [str(src_dir / "dlgn.cpp"), str(src_dir / "dlgn_kernel.cu")]
    return load(
        name="dlgn_ext",
        sources=sources,
        verbose=True,
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "-lineinfo"],
    )

try:
    _ext = _build_ext()
except Exception as e:
    # Fallback: in case build fails (e.g., no NVCC), raise a clearer hint.
    raise RuntimeError(
        f"Failed to build CUDA extension. Ensure NVCC, a compiler, and a CUDA-enabled "
        f"PyTorch are installed. Original error:\n{e}"
    )

def dlgn_forward(x, idx_l, idx_r, alpha):
    return _ext.forward(x, idx_l, idx_r, alpha)

def dlgn_backward(grad_out, x, idx_l, idx_r, alpha):
    return _ext.backward(grad_out, x, idx_l, idx_r, alpha)
