"""Single-kernel C++ inference for FoilCORE (model_type="foilcore") at B=1 on CPU.

The whole forward — point encoder + PathNL, region means (+ profile memory), Mamba-style memory
query, error head, polar_in + sinusoidal α, K RPA corrector blocks (causal or bidirectional),
requery, optional ê-gated expert — runs in one C++ call over plain float32 buffers: no PyTorch
dispatch inside. Weights are read straight from the module. Build is JIT (torch cpp_extension,
cached under build/fused_forward). Verified against the PyTorch forward by ``check()``.

    from src.fused_forward import FusedFoilCORE
    ff = FusedFoilCORE(model)             # model: FoilCORE (foilcore type), eval mode
    cl_cd = ff(coords200x2, polar_Tx5, causal=True)   # (T, 2) normalized outputs (final)
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

_SRC = r"""
#include <torch/extension.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
static double g_prof[16]; static int g_prof_on = -1;
static inline double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define PROF(i, stmt) do { double _t0 = g_prof_on ? now_us() : 0; stmt; if (g_prof_on) g_prof[i] += now_us() - _t0; } while (0)
torch::Tensor prof_read() { auto t = torch::zeros({16}, torch::kFloat64); for (int i = 0; i < 16; ++i) { t[i] = g_prof[i]; g_prof[i] = 0; } return t; }
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define FOIL_HAVE_BLAS 1
#endif

// C[M,N] (+)= A[M,K] * B[N,K]^T  (all row-major, "NT" product); beta 0 or 1
static inline void gemm_nt(const float* A, const float* B, float* Cm, int M, int N, int K, float beta) {
#ifdef FOIL_HAVE_BLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.f, A, K, B, K, beta, Cm, N);
#else
  for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
    float acc = beta ? Cm[i * N + j] : 0.f;
    const float* a = A + i * K; const float* b = B + j * K;
    for (int k = 0; k < K; ++k) acc += a[k] * b[k];
    Cm[i * N + j] = acc;
  }
#endif
}
// C[M,N] = A[M,K] * B[K,N]  ("NN")
static inline void gemm_nn(const float* A, const float* B, float* Cm, int M, int N, int K) {
#ifdef FOIL_HAVE_BLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.f, A, K, B, N, 0.f, Cm, N);
#else
  for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
    float acc = 0.f; for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
    Cm[i * N + j] = acc;
  }
#endif
}

// ---------- helpers on raw float pointers ----------
static inline float softplus(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }
static inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

// linear interpolation of y on one contour half (points i0..i1 inclusive, x monotone in either direction)
static inline float interp_half(const float* C, int i0, int i1, float xs) {
  float best_y = 0.f, best_d = 1e9f; bool found = false;
  for (int i = i0; i < i1; ++i) {
    float xa = C[2 * i], xb = C[2 * (i + 1)];
    float lo = std::fmin(xa, xb), hi = std::fmax(xa, xb);
    if (xs >= lo && xs <= hi) {
      float w = (hi - lo) > 1e-6f ? (xs - xa) / (xb - xa) : 0.f;
      w = std::fmin(std::fmax(w, 0.f), 1.f);
      return C[2 * i + 1] + w * (C[2 * (i + 1) + 1] - C[2 * i + 1]);
    }
    float dd = std::fmin(std::fabs(xa - xs), std::fabs(xb - xs));
    if (dd < best_d) { best_d = dd; best_y = std::fabs(xa - xs) < std::fabs(xb - xs) ? C[2 * i + 1] : C[2 * (i + 1) + 1]; found = true; }
  }
  return found ? best_y : 0.f;
}

// Camber/thickness at 40 cosine stations (prof40[0..39]=camber, [40..79]=thickness),
// leading-edge radius r_le (y^2 = 2 r x parabola fit on x in (1e-3, 0.04), clamped [1e-4, 1]),
// and thin-airfoil zero-lift angle alpha0 = -(1/pi) * sum (dz/dtheta)(cos theta - 1) dtheta.
static inline void analytic_scalars(const float* C, int N, float* prof40, float& r_le, float& alpha0) {
  const int S = 40;
  int ile = 0; for (int i = 1; i < N; ++i) if (C[2 * i] < C[2 * ile]) ile = i;
  float xs[40];
  for (int s = 0; s < S; ++s) { float t = (s + 1.f) / (S + 1.f); xs[s] = 0.5f * (1.f - std::cos((float)M_PI * t)); }
  for (int s = 0; s < S; ++s) {
    float yu = interp_half(C, 0, ile, xs[s]);
    float yl = interp_half(C, ile, N - 1, xs[s]);
    prof40[s] = 0.5f * (yu + yl); prof40[S + s] = yu - yl;
  }
  double sx = 0., sy2 = 0.;
  for (int i = 0; i < N; ++i) { float x = C[2 * i]; if (x > 1e-3f && x < 0.04f) { sx += x; sy2 += (double)C[2 * i + 1] * C[2 * i + 1]; } }
  r_le = (float)(0.5 * sy2 / std::max(sx, 1e-4));
  r_le = std::fmin(std::fmax(r_le, 1e-4f), 1.f);
  double acc = 0.;
  for (int s = 1; s < S; ++s) {
    double th0 = std::acos(std::fmin(std::fmax(1.0 - 2.0 * xs[s - 1], -1.0), 1.0));
    double th1 = std::acos(std::fmin(std::fmax(1.0 - 2.0 * xs[s], -1.0), 1.0));
    double thm = 0.5 * (th0 + th1);
    double dth = std::max(th1 - th0, 1e-4);
    double dx = xs[s] - xs[s - 1];
    double dzdx = (prof40[s] - prof40[s - 1]) / std::max(dx, 1e-4);
    double dzdth = dzdx * 0.5 * std::sin(thm);
    acc += dzdth * (std::cos(thm) - 1.0) * dth;
  }
  alpha0 = (float)(-(1.0 / M_PI) * acc);
}

// y = W x + b ; W is (out, in) row-major
static inline void linear(const float* W, const float* b, const float* x, float* y, int out, int in) {
  for (int o = 0; o < out; ++o) {
    float acc = b ? b[o] : 0.f;
    const float* w = W + (size_t)o * in;
    for (int i = 0; i < in; ++i) acc += w[i] * x[i];
    y[o] = acc;
  }
}

// PathNonlinearity per channel: a_lin x + m + sum_k a_k cos(k acos(clamp(b_k x))) + A exp(-.5((x-mu)/sig)^2)
static inline void vlog1p(float* x, int n) {
#ifdef FOIL_HAVE_BLAS
  vvlog1pf(x, x, &n);
#else
  for (int i = 0; i < n; ++i) x[i] = std::log1p(x[i]);
#endif
}
// vectorized exp over n floats (in place)
static inline void vexp(float* x, int n) {
#ifdef FOIL_HAVE_BLAS
  vvexpf(x, x, &n);
#else
  for (int i = 0; i < n; ++i) x[i] = std::exp(x[i]);
#endif
}

// GELU(x) = 0.5 x (1 + erf(x/√2)) with erf via Abramowitz–Stegun 7.1.26 (|err| < 1.5e-7) and one vector exp
static inline void vgelu(float* x, int n) {
  std::vector<float> eb(n), tb(n), sb(n); float* e = eb.data(); float* tt = tb.data(); float* sgn = sb.data();
  for (int i = 0; i < n; ++i) {
    float z = x[i] * 0.70710678f;
    sgn[i] = z < 0 ? -1.f : 1.f; z = std::fabs(z);
    tt[i] = 1.f / (1.f + 0.3275911f * z);
    e[i] = -z * z;
  }
  vexp(e, n);
  for (int i = 0; i < n; ++i) {
    float t = tt[i];
    float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    float erf = sgn[i] * (1.f - poly * e[i]);
    x[i] = 0.5f * x[i] * (1.f + erf);
  }
}
static inline void vsincos(const float* ang, float* sn, float* cs, int n) {
#ifdef FOIL_HAVE_BLAS
  vvsincosf(sn, cs, ang, &n);
#else
  for (int i = 0; i < n; ++i) { sn[i] = std::sin(ang[i]); cs[i] = std::cos(ang[i]); }
#endif
}

struct PathNL {
  const float *a_lin, *m, *gamp, *gmu; std::vector<float> inv_sig, aT, bT; int d, K;
  PathNL(const float* al, const float* mm, const float* aa, const float* bb, const float* ga, const float* gm, const float* gls, int d_, int K_)
    : a_lin(al), m(mm), gamp(ga), gmu(gm), d(d_), K(K_) {
    inv_sig.resize(d); aT.resize(d * K); bT.resize(d * K);
    for (int c = 0; c < d; ++c) {
      inv_sig[c] = 1.f / (softplus(gls[c]) + 1e-3f);   // σ is constant: precompute once
      for (int k = 0; k < K; ++k) { aT[k * d + c] = aa[c * K + k]; bT[k * d + c] = bb[c * K + k]; }  // (K,d) layout: channel-inner loops vectorize
    }
  }
  // apply over `rows` consecutive d-vectors: channel-vectorized Chebyshev, one vectorized exp for the Gaussian bumps
  inline void apply_block(float* x, int rows, std::vector<float>& scratch) const {
    const int n = rows * d;
    if ((int)scratch.size() < n) scratch.resize(n);
    float* e = scratch.data();
    for (int r = 0; r < rows; ++r) { const float* xr = x + r * d; float* er = e + r * d; for (int c = 0; c < d; ++c) { float z = (xr[c] - gmu[c]) * inv_sig[c]; er[c] = -0.5f * z * z; } }
    vexp(e, n);
    float u[64], t0[64], t1[64], y[64];
    for (int r = 0; r < rows; ++r) {
      float* xr = x + r * d; const float* er = e + r * d;
      for (int c = 0; c < d; ++c) y[c] = a_lin[c] * xr[c] + m[c];
      for (int k = 0; k < K; ++k) {
        const float* bk = &bT[k * d]; const float* ak = &aT[k * d];
        for (int c = 0; c < d; ++c) { float uu = bk[c] * xr[c]; u[c] = std::fmin(std::fmax(uu, -1.f + 1e-4f), 1.f - 1e-4f); t0[c] = 1.f; t1[c] = u[c]; }
        for (int j = 1; j <= k; ++j)      // T_{k+1}(u) by recurrence, vectorized over channels
          for (int c = 0; c < d; ++c) { float t2 = 2.f * u[c] * t1[c] - t0[c]; t0[c] = t1[c]; t1[c] = t2; }
        for (int c = 0; c < d; ++c) y[c] += ak[c] * t1[c];
      }
      for (int c = 0; c < d; ++c) xr[c] = y[c] + gamp[c] * er[c];
    }
  }
  inline void apply(float* x) const { std::vector<float> sc; apply_block(x, 1, sc); }
};

torch::Tensor foilcore_forward(
    torch::Tensor coords,      // (N,2)
    torch::Tensor polar,       // (T,5) normalized [alpha, Re, Mach, Cl, Cd]; Cl/Cd unused (draft-only)
    bool causal,
    torch::Tensor mem_in,      // (M,d) precomputed geometry memories, or empty → computed from coords
    // geometry
    torch::Tensor pointW, torch::Tensor pointB,                    // (d,2),(d)
    torch::Tensor pa_lin, torch::Tensor pm, torch::Tensor pa, torch::Tensor pb,
    torch::Tensor pgamp, torch::Tensor pgmu, torch::Tensor pglogsig,   // PathNL
    double le_xmax, double te_xmin,
    int64_t profile_stations, torch::Tensor profW, torch::Tensor profB,   // (d,2S),(d) or empty
    // memory mix
    torch::Tensor logA, torch::Tensor WdW, torch::Tensor WdB, torch::Tensor WbW, torch::Tensor WbB,
    torch::Tensor WcW, torch::Tensor WcB,
    torch::Tensor headW, torch::Tensor headB,                     // (2,d),(2)
    // error head (optional): reduce (mdim,d),(mdim); mlp0 (h, M*mdim+5),(h); mlp2 (2,h),(2)
    bool use_err, torch::Tensor erW, torch::Tensor erB, torch::Tensor e0W, torch::Tensor e0B, torch::Tensor e2W, torch::Tensor e2B,
    torch::Tensor pinW, torch::Tensor pinB,                       // (d, 5[+2]),(d)
    // blocks: lists
    std::vector<torch::Tensor> s1, std::vector<torch::Tensor> s3, std::vector<torch::Tensor> ltau,
    std::vector<torch::Tensor> WvW, std::vector<torch::Tensor> WvB,
    std::vector<torch::Tensor> lnW, std::vector<torch::Tensor> lnB,
    // expert (optional): mlp0 (h, d+2),(h); mlp2 (2,h),(2); gate (1,2),(1)
    bool use_expert, torch::Tensor x0W, torch::Tensor x0B, torch::Tensor x2W, torch::Tensor x2B, torch::Tensor xgW, torch::Tensor xgB,
    // physics levers (optional)
    bool phys_base, double base_scale, double p_astall, double p_wrle,
    double a_sd, double a_mu, double m_sd, double m_mu, double cl_mu, double cl_sd,
    double cd_mu, double cd_sd, double re_sd, double re_mu,
    bool phys_res, bool phys_gated,
    torch::Tensor cd0W, torch::Tensor cd0B, torch::Tensor kW, torch::Tensor kB, torch::Tensor gW, torch::Tensor gB,
    bool bl_feat, bool inv_mem, torch::Tensor invW, torch::Tensor invB
) {
  const int N = coords.size(0), T = polar.size(0), d = pointW.size(0), K = pa.size(1);
  const int nblk = (int)WvW.size();
  const float* C = coords.data_ptr<float>();
  const float* P = polar.data_ptr<float>();
  TORCH_CHECK(d <= 64, "d_model <= 64");
  PathNL path(pa_lin.data_ptr<float>(), pm.data_ptr<float>(), pa.data_ptr<float>(), pb.data_ptr<float>(),
              pgamp.data_ptr<float>(), pgmu.data_ptr<float>(), pglogsig.data_ptr<float>(), d, K);

  // ---- geometry-derived analytic scalars (physics levers / invariant memory) ----
  const float* cd0Wp = phys_res ? cd0W.data_ptr<float>() : nullptr;
  const float* cd0Bp = phys_res ? cd0B.data_ptr<float>() : nullptr;
  const float* kWp = phys_res ? kW.data_ptr<float>() : nullptr;
  const float* kBp = phys_res ? kB.data_ptr<float>() : nullptr;
  const float* gWp = phys_gated ? gW.data_ptr<float>() : nullptr;
  const float* gBp = phys_gated ? gB.data_ptr<float>() : nullptr;
  std::vector<float> prof40(80, 0.f);
  float r_le = 0.f, alpha0 = 0.f;

  // ---- geometry: point features -> region means ----
  int M = 3 + (profile_stations > 0 ? 1 : 0) + (inv_mem ? 1 : 0);
  const bool have_mem = mem_in.numel() > 0;
  const bool have_ana_cache = have_mem && mem_in.size(0) == M + 2;   // last two rows cache [r_le, alpha0]
  if (have_ana_cache) M = (int)mem_in.size(0) - 2;
  std::vector<float> mem((size_t)M * d, 0.f);
  std::vector<float> feat(d);
  int n_le = 0, n_te = 0;
  if (g_prof_on < 0) g_prof_on = std::getenv("FOIL_PROFILE") ? 1 : 0;
  double _T0 = g_prof_on ? now_us() : 0;
  if (have_mem) {
    std::memcpy(mem.data(), mem_in.data_ptr<float>(), sizeof(float) * (size_t)M * d);
    if (have_ana_cache) {
      r_le = mem_in.data_ptr<float>()[(size_t)M * d];
      alpha0 = mem_in.data_ptr<float>()[(size_t)M * d + 1];
    } else if (phys_base || bl_feat) {
      analytic_scalars(C, N, prof40.data(), r_le, alpha0);
    }
  } else {
    if (phys_base || bl_feat || inv_mem) analytic_scalars(C, N, prof40.data(), r_le, alpha0);
  const float* pW = pointW.data_ptr<float>(); const float* pB = pointB.data_ptr<float>();
  std::vector<float> featN((size_t)N * d), scratch;
  for (int i = 0; i < N; ++i) {
    float x = C[2 * i], y = C[2 * i + 1];
    for (int c = 0; c < d; ++c) featN[i * d + c] = pW[2 * c] * x + pW[2 * c + 1] * y + pB[c];
  }
  path.apply_block(featN.data(), N, scratch);
  for (int i = 0; i < N; ++i) {
    float x = C[2 * i];
    bool le = x <= le_xmax, te = x >= te_xmin;
    n_le += le; n_te += te;
    const float* fi = &featN[i * d];
    for (int c = 0; c < d; ++c) {
      mem[c] += fi[c];
      if (le) mem[d + c] += fi[c];
      if (te) mem[2 * d + c] += fi[c];
    }
  }
  for (int c = 0; c < d; ++c) { mem[c] /= N; mem[d + c] /= std::max(n_le, 1); mem[2 * d + c] /= std::max(n_te, 1); }
  if (profile_stations > 0) {
    // camber/thickness at cosine stations; upper = idx <= argmin x, lower = idx >= argmin x
    int ile = 0; for (int i = 1; i < N; ++i) if (C[2 * i] < C[2 * ile]) ile = i;
    int S = (int)profile_stations;
    std::vector<float> prof(2 * S);
    for (int s = 0; s < S; ++s) {
      float t = (s + 1.f) / (S + 1.f);
      float xs = 0.5f * (1.f - std::cos((float)M_PI * t));
      // linear interpolation on each half (x is monotone on each half of a Selig contour)
      auto interp = [&](int i0, int i1) {   // inclusive index range, x monotone (either direction)
        // find bracketing pair with min |x - xs| on the sorted-by-x set: scan pairs of consecutive points
        float best_y = 0.f; float best_d = 1e9f; bool found = false;
        for (int i = i0; i < i1; ++i) {
          float xa = C[2 * i], xb = C[2 * (i + 1)];
          float lo = std::fmin(xa, xb), hi = std::fmax(xa, xb);
          if (xs >= lo && xs <= hi) {
            float w = (hi - lo) > 1e-6f ? (xs - xa) / (xb - xa) : 0.f;
            w = std::fmin(std::fmax(w, 0.f), 1.f);
            return C[2 * i + 1] + w * (C[2 * (i + 1) + 1] - C[2 * i + 1]);
          }
          float dd = std::fmin(std::fabs(xa - xs), std::fabs(xb - xs));
          if (dd < best_d) { best_d = dd; best_y = std::fabs(xa - xs) < std::fabs(xb - xs) ? C[2 * i + 1] : C[2 * (i + 1) + 1]; found = true; }
        }
        return found ? best_y : 0.f;
      };
      float yu = interp(0, ile), yl = interp(ile, N - 1);
      prof[s] = 0.5f * (yu + yl); prof[S + s] = yu - yl;
    }
    std::vector<float> pm4(d);
    linear(profW.data_ptr<float>(), profB.data_ptr<float>(), prof.data(), pm4.data(), d, 2 * S);
    path.apply(pm4.data());
    for (int c = 0; c < d; ++c) mem[3 * d + c] = pm4[c];
  }
  if (inv_mem) {
    // 5th memory: phi(Linear(6,d)) over [log1p(100 r_LE)/log31, theta_TE, t_max/0.25, x_tmax, c_max/0.15, x_cmax]
    float t_max = 0.f, x_tmax = 0.f, c_max = 0.f, x_cmax = 0.f;
    int i_t = 0, i_c = 0;
    for (int s = 0; s < 40; ++s) {
      float thk = prof40[40 + s];
      if (thk > t_max) { t_max = thk; i_t = s; }
      float ca = std::fabs(prof40[s]);
      if (ca > c_max) { c_max = ca; i_c = s; }
    }
    x_tmax = 0.5f * (1.f - std::cos((float)M_PI * (i_t + 1.f) / 41.f));
    x_cmax = 0.5f * (1.f - std::cos((float)M_PI * (i_c + 1.f) / 41.f));
    float yu_e = prof40[39] + 0.5f * prof40[79], yl_e = prof40[39] - 0.5f * prof40[79];
    float yu_p = prof40[38] + 0.5f * prof40[78], yl_p = prof40[38] - 0.5f * prof40[78];
    float xs39 = 0.5f * (1.f - std::cos((float)M_PI * 40.f / 41.f));
    float xs38 = 0.5f * (1.f - std::cos((float)M_PI * 39.f / 41.f));
    float dx_te = xs39 - xs38;
    float su = (yu_e - yu_p) / dx_te, sl = (yl_e - yl_p) / dx_te;
    float theta_te = std::fabs(std::atan(su) - std::atan(sl));
    float inv6[6] = { std::log1p(100.f * r_le) / std::log(31.f), theta_te, t_max / 0.25f, x_tmax, c_max / 0.15f, x_cmax };
    std::vector<float> inv4(d);
    linear(invW.data_ptr<float>(), invB.data_ptr<float>(), inv6, inv4.data(), d, 6);
    path.apply(inv4.data());
    for (int c = 0; c < d; ++c) mem[4 * d + c] = inv4[c];
  }
  }  // !have_mem
  if (g_prof_on) { g_prof[0] += now_us() - _T0; _T0 = now_us(); }

  // ---- memory query per token ----
  const float* A = logA.data_ptr<float>();
  std::vector<float> Aexp(d); for (int c = 0; c < d; ++c) Aexp[c] = std::exp(A[c]);
  std::vector<float> delta(T * d), Bv(T * d), Cv(T * d), hmem(T * d), ctx(T * d), draft(T * 2);
  std::vector<float> tmp(d);
  {
    const float *Wd = WdW.data_ptr<float>(), *bd = WdB.data_ptr<float>(), *Wb = WbW.data_ptr<float>(), *bb_ = WbB.data_ptr<float>(), *Wc = WcW.data_ptr<float>(), *bc = WcB.data_ptr<float>();
    for (int t = 0; t < T; ++t) {   // cond linears, channel-vectorized (in=3)
      const float a0 = P[5 * t], a1 = P[5 * t + 1], a2 = P[5 * t + 2];
      float *dt = &delta[t * d], *bt = &Bv[t * d], *ct = &Cv[t * d];
      for (int c = 0; c < d; ++c) {
        dt[c] = Wd[3 * c] * a0 + Wd[3 * c + 1] * a1 + Wd[3 * c + 2] * a2 + bd[c];
        bt[c] = Wb[3 * c] * a0 + Wb[3 * c + 1] * a1 + Wb[3 * c + 2] * a2 + bb_[c];
        ct[c] = Wc[3 * c] * a0 + Wc[3 * c + 1] * a1 + Wc[3 * c + 2] * a2 + bc[c];
      }
    }
    // dA = exp(-(softplus(delta)+1e-4)·A): softplus = log1p(exp(x)) (x>20 → x), all vectorized over T·d
    const int n = T * d;
    std::vector<float> dA(n);
    for (int i = 0; i < n; ++i) dA[i] = std::fmin(delta[i], 20.f);
    vexp(dA.data(), n); vlog1p(dA.data(), n);
    for (int t = 0; t < T; ++t) for (int c = 0; c < d; ++c) { int i = t * d + c; float sp = delta[i] > 20.f ? delta[i] : dA[i]; dA[i] = -(sp + 1e-4f) * Aexp[c]; }
    vexp(dA.data(), n);
    for (int t = 0; t < T; ++t) {
      for (int c = 0; c < d; ++c) {
        int i = t * d + c; float a = dA[i], h = 0.f, bm = Bv[i];
        for (int j = 0; j < M; ++j) h = a * h + (1.f - a) * bm * mem[j * d + c];
        hmem[i] = h; ctx[i] = Cv[i] * h;
      }
      linear(headW.data_ptr<float>(), headB.data_ptr<float>(), &ctx[t * d], &draft[2 * t], 2, d);
      if (phys_base || phys_res) {
        const float* cx = &ctx[t * d];
        double a_deg = (double)P[5 * t] * a_sd + a_mu;
        double a_rad = a_deg * M_PI / 180.0;
        if (phys_base) {
          double m_phys = std::fmax((double)P[5 * t + 2] * m_sd + m_mu, 0.0);
          double m_c = std::fmin(m_phys, 0.95);
          double pg = 1.0 / std::sqrt(1.0 - m_c * m_c);
          double astall = softplus((float)p_astall) + softplus((float)p_wrle) * std::sqrt((double)r_le);
          double aeff = a_rad - (double)alpha0;
          double taper = std::cos(a_rad) * std::exp(-std::pow(aeff / std::fmax(astall, 0.05), 4.0));
          double base_phys = 2.0 * M_PI * std::sin(aeff) * pg * taper;
          double base_z = (base_phys - cl_mu) / cl_sd;
          draft[2 * t] += (float)(base_scale * base_z);
        }
        if (phys_res) {
          float cd0 = cd0Bp[0], k = kBp[0], g = 0.f;
          for (int c = 0; c < d; ++c) { cd0 += cd0Wp[c] * cx[c]; k += kWp[c] * cx[c]; if (phys_gated) g += gWp[c] * cx[c]; }
          cd0 = softplus(cd0); k = softplus(k);
          double cl_phys = (double)draft[2 * t] * cl_sd + cl_mu;
          double cd_phys = (double)cd0 + (double)k * cl_phys * cl_phys + (double)draft[2 * t + 1] * cd_sd;
          if (phys_gated) {
            g = sigmoid(g + gBp[0]);
            double cd_learned = (double)draft[2 * t + 1] * cd_sd + cd_mu;
            cd_phys = (double)g * cd_phys + (1.0 - (double)g) * cd_learned;
          }
          draft[2 * t + 1] = (float)((cd_phys - cd_mu) / cd_sd);
        }
      }
    }
  }

  if (g_prof_on) { g_prof[1] += now_us() - _T0; _T0 = now_us(); }
  // ---- error head ----
  const int mdim = use_err ? erW.size(0) : 0;
  const int ehid = use_err ? e0W.size(0) : 0;
  std::vector<float> ered((size_t)M * mdim), err(T * 2, 0.f), ein(M * mdim + 5), ehid_v(ehid);
  if (use_err) {
    for (int j = 0; j < M; ++j) linear(erW.data_ptr<float>(), erB.data_ptr<float>(), &mem[j * d], &ered[j * mdim], mdim, d);
    std::vector<float> H1((size_t)T * ehid);
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < M * mdim; ++i) ein[i] = ered[i];
      for (int i = 0; i < 3; ++i) ein[M * mdim + i] = P[5 * t + i];
      ein[M * mdim + 3] = draft[2 * t]; ein[M * mdim + 4] = draft[2 * t + 1];
      linear(e0W.data_ptr<float>(), e0B.data_ptr<float>(), ein.data(), &H1[t * ehid], ehid, M * mdim + 5);
    }
    vgelu(H1.data(), T * ehid);
    for (int t = 0; t < T; ++t) {
      linear(e2W.data_ptr<float>(), e2B.data_ptr<float>(), &H1[t * ehid], &err[2 * t], 2, ehid);
      err[2 * t] = softplus(err[2 * t]); err[2 * t + 1] = softplus(err[2 * t + 1]);
    }
  }

  if (g_prof_on) { g_prof[2] += now_us() - _T0; _T0 = now_us(); }
  // ---- polar tokens ----
  const int fin = 5 + (use_err ? 2 : 0) + (bl_feat ? 4 : 0);
  std::vector<float> f(fin), tok(T * d), h(T * d), divs(d / 2), angT(T * (d / 2));
  for (int i = 0; i < d / 2; ++i) divs[i] = std::exp((2 * i) * (-std::log(10000.f) / d));
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < 3; ++i) f[i] = P[5 * t + i];
    f[3] = draft[2 * t]; f[4] = draft[2 * t + 1];
    if (use_err) { f[5] = err[2 * t]; f[6] = err[2 * t + 1]; }
    if (bl_feat) {
      double re_phys = std::fmax((double)P[5 * t + 1] * re_sd + re_mu, 5e3);
      double a_deg = (double)P[5 * t] * a_sd + a_mu;
      double a_rad = a_deg * M_PI / 180.0;
      double rl = std::sqrt((double)r_le);
      f[7] = (float)(std::log10(re_phys) / 7.0);
      f[8] = (float)(200.0 / std::sqrt(re_phys));
      f[9] = (float)(20.0 / std::pow(re_phys, 0.2));
      f[10] = (float)std::tanh(std::fmin(std::fabs(a_rad) / rl, 20.0) / 10.0);
    }
    linear(pinW.data_ptr<float>(), pinB.data_ptr<float>(), f.data(), &tok[t * d], d, fin);
    float alpha = P[5 * t];
    for (int c = 0; c < d; ++c) tok[t * d + c] += ctx[t * d + c];
    { int hn = d / 2; for (int i = 0; i < hn; ++i) angT[t * hn + i] = alpha * divs[i]; }
  }
  { int hn = d / 2; std::vector<float> sn(T * hn), cs(T * hn);
    vsincos(angT.data(), sn.data(), cs.data(), T * hn);
    for (int t = 0; t < T; ++t) for (int i = 0; i < hn; ++i) { tok[t * d + 2 * i] += sn[t * hn + i]; tok[t * d + 2 * i + 1] += cs[t * hn + i]; } }
  h = tok;
  if (g_prof_on) { g_prof[3] += now_us() - _T0; _T0 = now_us(); }

  // ---- corrector blocks ----
  const float scale = std::sqrt((float)d);
  std::vector<float> Q(T * d), V(T * d), y(T * d), S((size_t)T * T), xn(d), scratchT;
  for (int b = 0; b < nblk; ++b) {
    float s1v = s1[b].item<float>(), s3v = s3[b].item<float>();
    float tau = softplus(ltau[b].item<float>()) + 1e-3f;
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < d; ++c) Q[t * d + c] = s1v * h[t * d + c] + s3v;   // s2·tanh(x−x)=0
    // V = h · W_v^T + b   (T×d)
    gemm_nt(h.data(), WvW[b].data_ptr<float>(), V.data(), T, d, d, 0.f);
    { const float* vb = WvB[b].data_ptr<float>(); for (int t = 0; t < T; ++t) for (int c = 0; c < d; ++c) V[t * d + c] += vb[c]; }
    // S = Q · Q^T / scale − Δα²/τ   (T×T), then row softmax (causal: lower triangle)
    gemm_nt(Q.data(), Q.data(), S.data(), T, T, d, 0.f);
    const float inv_scale = 1.f / scale, inv_tau = 1.f / tau;
    for (int i = 0; i < T; ++i) {
      int jmax = causal ? i + 1 : T;
      float* row = &S[i * T];
      float mx = -1e30f;
      for (int j = 0; j < jmax; ++j) { float da = P[5 * i] - P[5 * j]; row[j] = row[j] * inv_scale - da * da * inv_tau; if (row[j] > mx) mx = row[j]; }
      for (int j = 0; j < jmax; ++j) row[j] -= mx;
      vexp(row, jmax);
      float z = 0.f;
      for (int j = 0; j < jmax; ++j) z += row[j];
      float iz = 1.f / z;
      for (int j = 0; j < jmax; ++j) row[j] *= iz;
      for (int j = jmax; j < T; ++j) row[j] = 0.f;
    }
    // y = S · V   (T×d)
    gemm_nn(S.data(), V.data(), y.data(), T, d, T);
    if (g_prof_on) { g_prof[4] += now_us() - _T0; _T0 = now_us(); }
    // residual + LayerNorm + PathNL
    const float* g = lnW[b].data_ptr<float>(); const float* bb = lnB[b].data_ptr<float>();
    for (int t = 0; t < T; ++t) {
      float* ht = &h[t * d]; const float* yt = &y[t * d];
      float mean = 0.f;
      for (int c = 0; c < d; ++c) { ht[c] += yt[c]; mean += ht[c]; }
      mean /= d;
      float var = 0.f; for (int c = 0; c < d; ++c) { float q = ht[c] - mean; var += q * q; }
      var /= d;
      float inv = 1.f / std::sqrt(var + 1e-5f);
      for (int c = 0; c < d; ++c) ht[c] = (ht[c] - mean) * inv * g[c] + bb[c];
    }
    path.apply_block(h.data(), T, scratchT);
    if (g_prof_on) { g_prof[5] += now_us() - _T0; _T0 = now_us(); }
  }

  // ---- requery + head (+ expert) ----
  auto out = torch::empty({T, 2}, torch::kFloat32);
  float* O = out.data_ptr<float>();
  const int xhid = use_expert ? x0W.size(0) : 0;
  std::vector<float> XH((size_t)T * xhid), xin(d + 2), dlt(2);
  if (use_expert) {
    for (int t = 0; t < T; ++t) {
      for (int c = 0; c < d; ++c) xin[c] = h[t * d + c];
      xin[d] = err[2 * t]; xin[d + 1] = err[2 * t + 1];
      linear(x0W.data_ptr<float>(), x0B.data_ptr<float>(), xin.data(), &XH[t * xhid], xhid, d + 2);
    }
    vgelu(XH.data(), T * xhid);
  }
  const float g0 = use_expert ? xgW.data_ptr<float>()[0] : 0.f;
  const float g1 = use_expert ? xgW.data_ptr<float>()[1] : 0.f;
  const float gb = use_expert ? xgB.data_ptr<float>()[0] : 0.f;
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < d; ++c) tmp[c] = (Cv[t * d + c] + h[t * d + c]) * hmem[t * d + c];
    linear(headW.data_ptr<float>(), headB.data_ptr<float>(), tmp.data(), &O[2 * t], 2, d);
    if (use_expert) {
      linear(x2W.data_ptr<float>(), x2B.data_ptr<float>(), &XH[t * xhid], dlt.data(), 2, xhid);
      float gate = sigmoid(g0 * err[2 * t] + g1 * err[2 * t + 1] + gb);
      O[2 * t] += gate * dlt[0]; O[2 * t + 1] += gate * dlt[1];
    }
    if (phys_base || phys_res) {
      const float* cx = tmp.data();
      double a_deg = (double)P[5 * t] * a_sd + a_mu;
      double a_rad = a_deg * M_PI / 180.0;
      if (phys_base) {
        double m_phys = std::fmax((double)P[5 * t + 2] * m_sd + m_mu, 0.0);
        double m_c = std::fmin(m_phys, 0.95);
        double pg = 1.0 / std::sqrt(1.0 - m_c * m_c);
        double astall = softplus((float)p_astall) + softplus((float)p_wrle) * std::sqrt((double)r_le);
        double aeff = a_rad - (double)alpha0;
        double taper = std::cos(a_rad) * std::exp(-std::pow(aeff / std::fmax(astall, 0.05), 4.0));
        double base_phys = 2.0 * M_PI * std::sin(aeff) * pg * taper;
        double base_z = (base_phys - cl_mu) / cl_sd;
        O[2 * t] += (float)(base_scale * base_z);
      }
      if (phys_res) {
        float cd0 = cd0Bp[0], k = kBp[0], g = 0.f;
        for (int c = 0; c < d; ++c) { cd0 += cd0Wp[c] * cx[c]; k += kWp[c] * cx[c]; if (phys_gated) g += gWp[c] * cx[c]; }
        cd0 = softplus(cd0); k = softplus(k);
        double cl_phys = (double)O[2 * t] * cl_sd + cl_mu;
        double cd_phys = (double)cd0 + (double)k * cl_phys * cl_phys + (double)O[2 * t + 1] * cd_sd;
        if (phys_gated) {
          g = sigmoid(g + gBp[0]);
          double cd_learned = (double)O[2 * t + 1] * cd_sd + cd_mu;
          cd_phys = (double)g * cd_phys + (1.0 - (double)g) * cd_learned;
        }
        O[2 * t + 1] = (float)((cd_phys - cd_mu) / cd_sd);
      }
    }
  }
  if (g_prof_on) { g_prof[6] += now_us() - _T0; }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("foilcore_forward", &foilcore_forward); m.def("prof_read", &prof_read); }
"""

_EXT = None


def get_ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        bd = Path(__file__).resolve().parents[1] / "build" / "fused_forward"
        bd.mkdir(parents=True, exist_ok=True)
        import platform
        ldflags = ["-framework", "Accelerate"] if platform.system() == "Darwin" else []
        _EXT = load_inline(name="foilcore_fused_forward", cpp_sources=_SRC,
                           extra_cflags=["-O3", "-std=c++17", "-ffast-math", "-march=native"], extra_ldflags=ldflags,
                           build_directory=str(bd), verbose=False)
    return _EXT


class FusedFoilCORE:
    """Wraps a FoilCORE (model_type foilcore, no MoE / softbin) for single-call C++ inference."""

    def __init__(self, model):
        cfg = model.config
        assert cfg.model_type == "foilcore" and cfg.geom_pool == "region3" and int(cfg.moe_experts) == 0, "unsupported config"
        assert cfg.share_path, "shared PathNL only"
        m = model.eval().cpu()
        f = lambda t: t.detach().contiguous().float()
        self.d = int(cfg.d_model)
        p = m.path
        self.args_geo = [f(m.geom_encoder.point.weight), f(m.geom_encoder.point.bias), f(p.a_lin), f(p.m), f(p.a), f(p.b),
                         f(p.gauss_amp), f(p.gauss_mu), f(p.gauss_log_sigma), float(cfg.le_xmax), float(cfg.te_xmin)]
        S = int(cfg.profile_stations)
        self.args_prof = [S, f(m.profile_in.weight) if S else torch.zeros(0), f(m.profile_in.bias) if S else torch.zeros(0)]
        mm = m.mem_mix
        self.args_mix = [f(mm.log_A), f(mm.W_delta.weight), f(mm.W_delta.bias), f(mm.W_B.weight), f(mm.W_B.bias),
                         f(mm.W_C.weight), f(mm.W_C.bias), f(m.head.weight), f(m.head.bias)]
        eh = getattr(m, "err_head", None)
        z = torch.zeros(0)
        self.args_err = [eh is not None] + ([f(eh.reduce.weight), f(eh.reduce.bias), f(eh.mlp[0].weight), f(eh.mlp[0].bias),
                                             f(eh.mlp[2].weight), f(eh.mlp[2].bias)] if eh is not None else [z] * 6)
        self.args_pin = [f(m.polar_in.weight), f(m.polar_in.bias)]
        blocks = list(m.agent)
        self.args_blk = [[f(b.attn.s1) for b in blocks], [f(b.attn.s3) for b in blocks], [f(b.attn.local_log_tau) for b in blocks],
                         [f(b.attn.W_v.weight) for b in blocks], [f(b.attn.W_v.bias) for b in blocks],
                         [f(b.norm.weight) for b in blocks], [f(b.norm.bias) for b in blocks]]
        ex = getattr(m, "err_expert", None)
        self.args_exp = [ex is not None] + ([f(ex.mlp[0].weight), f(ex.mlp[0].bias), f(ex.mlp[2].weight), f(ex.mlp[2].bias),
                                             f(ex.gate.weight), f(ex.gate.bias)] if ex is not None else [z] * 6)
        pb = bool(cfg.phys_base)
        pr = bool(cfg.phys_residual)
        pg = bool(cfg.phys_residual_gated)
        self.args_phys = (
            [pb] + ([float(m.base_scale), float(m.alpha_stall_base), float(m.w_r_le)] if pb else [0.0, 0.0, 0.0])
            + [float(cfg.a_sd), float(cfg.a_mu), float(cfg.m_sd), float(cfg.m_mu),
               float(cfg.cl_mu), float(cfg.cl_sd), float(cfg.cd_mu), float(cfg.cd_sd),
               float(cfg.re_sd), float(cfg.re_mu)]
            + [pr, pg]
            + ([f(m.cd0_head.weight), f(m.cd0_head.bias), f(m.k_head.weight), f(m.k_head.bias)] if pr else [z, z, z, z])
            + ([f(m.gate_head.weight), f(m.gate_head.bias)] if pg else [z, z])
            + [bool(cfg.bl_features), bool(cfg.invariant_memory)]
            + ([f(m.inv_in.weight), f(m.inv_in.bias)] if cfg.invariant_memory else [z, z])
        )
        self.ext = get_ext()

    def __call__(self, coords: torch.Tensor, polar: torch.Tensor, causal: bool = True, mem: torch.Tensor | None = None) -> torch.Tensor:
        mem_in = mem.contiguous().float() if mem is not None else torch.zeros(0)
        return self.ext.foilcore_forward(coords.contiguous().float(), polar.contiguous().float(), bool(causal), mem_in,
                                         *self.args_geo, *self.args_prof, *self.args_mix, *self.args_err, *self.args_pin,
                                         *self.args_blk, *self.args_exp, *self.args_phys)

    @torch.no_grad()
    def encode(self, model, coords: torch.Tensor) -> torch.Tensor:
        """Geometry memories (M, d) plus one cache row [r_LE, alpha0, 0...] — computed once per airfoil for streaming use."""
        from src.model import le_radius, thin_airfoil_alpha0
        mem = model.encode_geom(coords[None])[0].contiguous().float()
        cache = torch.zeros(mem.shape[1], dtype=torch.float32)
        cache[0] = le_radius(coords[None])[0, 0]
        cache[1] = thin_airfoil_alpha0(coords[None])[0, 0]
        return torch.cat((mem, cache.unsqueeze(0)), dim=0)


@torch.no_grad()
def check(model, coords: torch.Tensor, polar: torch.Tensor, causal: bool = True) -> float:
    """Max abs difference between fused and PyTorch outputs for one polar (normalized units)."""
    ff = FusedFoilCORE(model)
    out = ff(coords, polar, causal)
    mask = torch.ones(1, polar.shape[0], dtype=torch.bool)
    ref, _ = model(coords[None], polar[None], mask, causal=causal)
    return float((out - ref[0]).abs().max())
