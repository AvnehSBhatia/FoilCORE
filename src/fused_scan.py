"""Fused CPU affine scan ``h_t = a_t h_{t-1} + u_t`` as a JIT-built C++ extension.

Inference-only (no autograd). One sequential pass over N per (batch, channel)
instead of ceil(log2 N) full-tensor Hillis–Steele steps; at B=1, N=200, d=32 this
is ~1 µs vs ~150 µs. Build is cached under ``build/fused_scan``. Set
``FOILCORE_NO_FUSED=1`` to disable.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_EXT = None
_TRIED = False

_SRC = r"""
#include <torch/extension.h>

torch::Tensor affine_scan_seq(torch::Tensor a, torch::Tensor u) {
  TORCH_CHECK(a.dim() == 3 && u.sizes() == a.sizes(), "affine_scan_seq: expected (B, N, D) a and u");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32 && u.scalar_type() == torch::kFloat32, "float32 only");
  auto ac = a.contiguous();
  auto uc = u.contiguous();
  auto out = torch::empty_like(uc);
  const int64_t B = ac.size(0), N = ac.size(1), D = ac.size(2);
  const float* ap = ac.data_ptr<float>();
  const float* up = uc.data_ptr<float>();
  float* op = out.data_ptr<float>();
  at::parallel_for(0, B, 1, [&](int64_t b0, int64_t b1) {
    for (int64_t b = b0; b < b1; ++b) {
      const float* A = ap + b * N * D;
      const float* U = up + b * N * D;
      float* O = op + b * N * D;
      for (int64_t d = 0; d < D; ++d) O[d] = U[d];
      for (int64_t t = 1; t < N; ++t) {
        const float* At = A + t * D;
        const float* Ut = U + t * D;
        const float* Op = O + (t - 1) * D;
        float* Ot = O + t * D;
        for (int64_t d = 0; d < D; ++d) Ot[d] = At[d] * Op[d] + Ut[d];
      }
    }
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("affine_scan_seq", &affine_scan_seq); }
"""


def get_ext():
    """Return the compiled extension module, or None if unavailable/disabled."""
    global _EXT, _TRIED
    if _TRIED:
        return _EXT
    _TRIED = True
    if os.environ.get("FOILCORE_NO_FUSED", "").strip() in ("1", "true", "yes"):
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        build_dir = Path(__file__).resolve().parents[1] / "build" / "fused_scan"
        build_dir.mkdir(parents=True, exist_ok=True)
        _EXT = load_inline(
            name="foilcore_fused_scan",
            cpp_sources=_SRC,
            extra_cflags=["-O3", "-std=c++17"],
            build_directory=str(build_dir),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 — any build failure just disables the fast path
        if os.environ.get("FOILCORE_FUSED_VERBOSE"):
            print(f"[fused_scan] unavailable: {exc}")
        _EXT = None
    return _EXT


def can_use(a: torch.Tensor, u: torch.Tensor, dim: int) -> bool:
    return (
        a.device.type == "cpu"
        and a.dtype == torch.float32
        and u.dtype == torch.float32
        and a.dim() == 3
        and dim == 1
        and not (torch.is_grad_enabled() and (a.requires_grad or u.requires_grad))
    )


def affine_scan_fused(a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return get_ext().affine_scan_seq(a, u)
