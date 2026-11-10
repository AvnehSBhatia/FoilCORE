"""Single-kernel C++ inference for seqbaseline models (transformer / TSMixer / Mamba).

Same geometry front-end and pointwise draft as ``fused_forward.FusedFoilCORE``; the RPA
corrector + requery is replaced by the trained backbone. B=1 CPU, float32, JIT-built
under ``build/fused_baselines``. Verified against the PyTorch forward by ``check()``.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

_SRC = r"""
#include <torch/extension.h>
#include <cmath>
#include <cstring>
#include <vector>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define FOIL_HAVE_BLAS 1
#endif

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
static inline float softplus(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }
static inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
static inline void linear(const float* W, const float* b, const float* x, float* y, int out, int in) {
  for (int o = 0; o < out; ++o) {
    float acc = b ? b[o] : 0.f;
    const float* w = W + (size_t)o * in;
    for (int i = 0; i < in; ++i) acc += w[i] * x[i];
    y[o] = acc;
  }
}
static inline void vlog1p(float* x, int n) {
#ifdef FOIL_HAVE_BLAS
  vvlog1pf(x, x, &n);
#else
  for (int i = 0; i < n; ++i) x[i] = std::log1p(x[i]);
#endif
}
static inline void vexp(float* x, int n) {
#ifdef FOIL_HAVE_BLAS
  vvexpf(x, x, &n);
#else
  for (int i = 0; i < n; ++i) x[i] = std::exp(x[i]);
#endif
}
static inline void vgelu(float* x, int n) {
  std::vector<float> eb(n), tb(n), sb(n);
  for (int i = 0; i < n; ++i) {
    float z = x[i] * 0.70710678f;
    sb[i] = z < 0 ? -1.f : 1.f; z = std::fabs(z);
    tb[i] = 1.f / (1.f + 0.3275911f * z);
    eb[i] = -z * z;
  }
  vexp(eb.data(), n);
  for (int i = 0; i < n; ++i) {
    float t = tb[i];
    float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    x[i] = 0.5f * x[i] * (1.f + sb[i] * (1.f - poly * eb[i]));
  }
}
static inline void vsincos(const float* ang, float* sn, float* cs, int n) {
#ifdef FOIL_HAVE_BLAS
  vvsincosf(sn, cs, ang, &n);
#else
  for (int i = 0; i < n; ++i) { sn[i] = std::sin(ang[i]); cs[i] = std::cos(ang[i]); }
#endif
}
static inline void layernorm(const float* x, const float* g, const float* b, float* y, int d) {
  float mean = 0.f; for (int i = 0; i < d; ++i) mean += x[i]; mean /= d;
  float var = 0.f; for (int i = 0; i < d; ++i) { float q = x[i] - mean; var += q * q; } var /= d;
  float inv = 1.f / std::sqrt(var + 1e-5f);
  for (int i = 0; i < d; ++i) y[i] = (x[i] - mean) * inv * g[i] + b[i];
}

struct PathNL {
  const float *a_lin, *m, *gamp, *gmu; std::vector<float> inv_sig, aT, bT; int d, K;
  PathNL(const float* al, const float* mm, const float* aa, const float* bb, const float* ga, const float* gm, const float* gls, int d_, int K_)
    : a_lin(al), m(mm), gamp(ga), gmu(gm), d(d_), K(K_) {
    inv_sig.resize(d); aT.resize(d * K); bT.resize(d * K);
    for (int c = 0; c < d; ++c) {
      inv_sig[c] = 1.f / (softplus(gls[c]) + 1e-3f);
      for (int k = 0; k < K; ++k) { aT[k * d + c] = aa[c * K + k]; bT[k * d + c] = bb[c * K + k]; }
    }
  }
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
        for (int j = 1; j <= k; ++j)
          for (int c = 0; c < d; ++c) { float t2 = 2.f * u[c] * t1[c] - t0[c]; t0[c] = t1[c]; t1[c] = t2; }
        for (int c = 0; c < d; ++c) y[c] += ak[c] * t1[c];
      }
      for (int c = 0; c < d; ++c) xr[c] = y[c] + gamp[c] * er[c];
    }
  }
  inline void apply(float* x) const { std::vector<float> sc; apply_block(x, 1, sc); }
};

// tokens (T,d), draft (T,2). polar_in is Linear(5,d): [α,Re,Ma,draft Cl, draft Cd].
static void front_end(
    torch::Tensor coords, torch::Tensor polar, torch::Tensor mem_in,
    torch::Tensor pointW, torch::Tensor pointB,
    torch::Tensor pa_lin, torch::Tensor pm, torch::Tensor pa, torch::Tensor pb,
    torch::Tensor pgamp, torch::Tensor pgmu, torch::Tensor pglogsig,
    double le_xmax, double te_xmin,
    int64_t profile_stations, torch::Tensor profW, torch::Tensor profB,
    torch::Tensor logA, torch::Tensor WdW, torch::Tensor WdB, torch::Tensor WbW, torch::Tensor WbB,
    torch::Tensor WcW, torch::Tensor WcB, torch::Tensor headW, torch::Tensor headB,
    torch::Tensor pinW, torch::Tensor pinB,
    std::vector<float>& tokens, std::vector<float>& draft, int& T_out, int& d_out) {
  const int N = coords.size(0), T = polar.size(0), d = pointW.size(0), K = pa.size(1);
  TORCH_CHECK(d <= 64, "d_model <= 64");
  const float* C = coords.data_ptr<float>();
  const float* P = polar.data_ptr<float>();
  PathNL path(pa_lin.data_ptr<float>(), pm.data_ptr<float>(), pa.data_ptr<float>(), pb.data_ptr<float>(),
              pgamp.data_ptr<float>(), pgmu.data_ptr<float>(), pglogsig.data_ptr<float>(), d, K);
  int M = 3 + (profile_stations > 0 ? 1 : 0);
  std::vector<float> mem((size_t)M * d, 0.f);
  if (mem_in.numel() > 0) { std::memcpy(mem.data(), mem_in.data_ptr<float>(), sizeof(float) * M * d); }
  else {
    const float* pW = pointW.data_ptr<float>(); const float* pB = pointB.data_ptr<float>();
    std::vector<float> featN((size_t)N * d), scratch;
    int n_le = 0, n_te = 0;
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
      int ile = 0; for (int i = 1; i < N; ++i) if (C[2 * i] < C[2 * ile]) ile = i;
      int S = (int)profile_stations;
      std::vector<float> prof(2 * S);
      for (int s = 0; s < S; ++s) {
        float t = (s + 1.f) / (S + 1.f);
        float xs = 0.5f * (1.f - std::cos((float)M_PI * t));
        auto interp = [&](int i0, int i1) {
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
  }
  const float* A = logA.data_ptr<float>();
  std::vector<float> Aexp(d); for (int c = 0; c < d; ++c) Aexp[c] = std::exp(A[c]);
  std::vector<float> delta(T * d), Bv(T * d), Cv(T * d), hmem(T * d), ctx(T * d);
  draft.assign((size_t)T * 2, 0.f);
  {
    const float *Wd = WdW.data_ptr<float>(), *bd = WdB.data_ptr<float>(), *Wb = WbW.data_ptr<float>(), *bb_ = WbB.data_ptr<float>(), *Wc = WcW.data_ptr<float>(), *bc = WcB.data_ptr<float>();
    for (int t = 0; t < T; ++t) {
      const float a0 = P[5 * t], a1 = P[5 * t + 1], a2 = P[5 * t + 2];
      float *dt = &delta[t * d], *bt = &Bv[t * d], *ct = &Cv[t * d];
      for (int c = 0; c < d; ++c) {
        dt[c] = Wd[3 * c] * a0 + Wd[3 * c + 1] * a1 + Wd[3 * c + 2] * a2 + bd[c];
        bt[c] = Wb[3 * c] * a0 + Wb[3 * c + 1] * a1 + Wb[3 * c + 2] * a2 + bb_[c];
        ct[c] = Wc[3 * c] * a0 + Wc[3 * c + 1] * a1 + Wc[3 * c + 2] * a2 + bc[c];
      }
    }
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
    }
  }
  tokens.assign((size_t)T * d, 0.f);
  std::vector<float> f(5), divs(d / 2), angT(T * (d / 2));
  for (int i = 0; i < d / 2; ++i) divs[i] = std::exp((2 * i) * (-std::log(10000.f) / d));
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < 3; ++i) f[i] = P[5 * t + i];
    f[3] = draft[2 * t]; f[4] = draft[2 * t + 1];
    linear(pinW.data_ptr<float>(), pinB.data_ptr<float>(), f.data(), &tokens[t * d], d, 5);
    for (int c = 0; c < d; ++c) tokens[t * d + c] += ctx[t * d + c];
    { int hn = d / 2; for (int i = 0; i < hn; ++i) angT[t * hn + i] = P[5 * t] * divs[i]; }
  }
  { int hn = d / 2; std::vector<float> sn(T * hn), cs(T * hn);
    vsincos(angT.data(), sn.data(), cs.data(), T * hn);
    for (int t = 0; t < T; ++t) for (int i = 0; i < hn; ++i) { tokens[t * d + 2 * i] += sn[t * hn + i]; tokens[t * d + 2 * i + 1] += cs[t * hn + i]; } }
  T_out = T; d_out = d;
}

static void project_bb(const std::vector<float>& tokens, int T, int d,
                       torch::Tensor bbInW, torch::Tensor bbInB,
                       std::vector<float>& h, int& bd) {
  if (bbInW.numel() == 0) { h = tokens; bd = d; return; }
  bd = bbInW.size(0);
  h.assign((size_t)T * bd, 0.f);
  gemm_nt(tokens.data(), bbInW.data_ptr<float>(), h.data(), T, bd, d, 0.f);
  const float* b = bbInB.data_ptr<float>();
  for (int t = 0; t < T; ++t) for (int c = 0; c < bd; ++c) h[t * bd + c] += b[c];
}

static torch::Tensor pack_out(const std::vector<float>& draft, const std::vector<float>& h,
                              torch::Tensor hdW, torch::Tensor hdB, int T, int bd) {
  auto out = torch::empty({T, 2}, torch::kFloat32);
  float* O = out.data_ptr<float>();
  std::vector<float> delta(2);
  for (int t = 0; t < T; ++t) {
    linear(hdW.data_ptr<float>(), hdB.data_ptr<float>(), &h[t * bd], delta.data(), 2, bd);
    O[2 * t] = draft[2 * t] + delta[0];
    O[2 * t + 1] = draft[2 * t + 1] + delta[1];
  }
  return out;
}

static void transformer_layers(std::vector<float>& h, int T, int bd, int n_heads, bool causal,
    const std::vector<torch::Tensor>& ln1W, const std::vector<torch::Tensor>& ln1B,
    const std::vector<torch::Tensor>& qkvW, const std::vector<torch::Tensor>& qkvB,
    const std::vector<torch::Tensor>& projW, const std::vector<torch::Tensor>& projB,
    const std::vector<torch::Tensor>& ln2W, const std::vector<torch::Tensor>& ln2B,
    const std::vector<torch::Tensor>& ff0W, const std::vector<torch::Tensor>& ff0B,
    const std::vector<torch::Tensor>& ff2W, const std::vector<torch::Tensor>& ff2B,
    torch::Tensor fnW, torch::Tensor fnB) {
  const int nL = (int)qkvW.size();
  TORCH_CHECK(bd % n_heads == 0, "bb_d must divide n_heads");
  const int dh = bd / n_heads;
  const float inv_s = 1.f / std::sqrt((float)dh);
  const int hid = ff0W[0].size(0);
  std::vector<float> qkv((size_t)T * 3 * bd), Q((size_t)T * bd), K((size_t)T * bd), V((size_t)T * bd);
  std::vector<float> S((size_t)n_heads * T * T), y((size_t)T * bd), y2((size_t)T * bd), ffh((size_t)T * hid);
  for (int L = 0; L < nL; ++L) {
    std::vector<float> lnX((size_t)T * bd);
    for (int t = 0; t < T; ++t) layernorm(&h[t * bd], ln1W[L].data_ptr<float>(), ln1B[L].data_ptr<float>(), &lnX[t * bd], bd);
    gemm_nt(lnX.data(), qkvW[L].data_ptr<float>(), qkv.data(), T, 3 * bd, bd, 0.f);
    { const float* b = qkvB[L].data_ptr<float>(); for (int t = 0; t < T; ++t) for (int c = 0; c < 3 * bd; ++c) qkv[t * 3 * bd + c] += b[c]; }
    // split to Q,K,V as (T, h, dh) packed (T, bd)
    for (int t = 0; t < T; ++t) {
      for (int p = 0; p < 3; ++p) {
        float* dst = p == 0 ? &Q[t * bd] : p == 1 ? &K[t * bd] : &V[t * bd];
        std::memcpy(dst, &qkv[t * 3 * bd + p * bd], sizeof(float) * bd);
      }
    }
    std::memset(S.data(), 0, sizeof(float) * n_heads * T * T);
    for (int hd = 0; hd < n_heads; ++hd) {
      for (int i = 0; i < T; ++i) {
        int jmax = causal ? i + 1 : T;
        float mx = -1e30f;
        float* row = &S[(hd * T + i) * T];
        for (int j = 0; j < jmax; ++j) {
          float acc = 0.f;
          const float* qi = &Q[i * bd + hd * dh];
          const float* kj = &K[j * bd + hd * dh];
          for (int c = 0; c < dh; ++c) acc += qi[c] * kj[c];
          row[j] = acc * inv_s;
          if (row[j] > mx) mx = row[j];
        }
        for (int j = 0; j < jmax; ++j) row[j] -= mx;
        vexp(row, jmax);
        float z = 0.f; for (int j = 0; j < jmax; ++j) z += row[j];
        float iz = z > 0.f ? 1.f / z : 0.f;
        for (int j = 0; j < jmax; ++j) row[j] *= iz;
        for (int j = jmax; j < T; ++j) row[j] = 0.f;
      }
    }
    std::memset(y.data(), 0, sizeof(float) * T * bd);
    for (int hd = 0; hd < n_heads; ++hd) {
      for (int i = 0; i < T; ++i) {
        const float* row = &S[(hd * T + i) * T];
        float* yi = &y[i * bd + hd * dh];
        for (int c = 0; c < dh; ++c) {
          float acc = 0.f;
          for (int j = 0; j < T; ++j) acc += row[j] * V[j * bd + hd * dh + c];
          yi[c] = acc;
        }
      }
    }
    gemm_nt(y.data(), projW[L].data_ptr<float>(), y2.data(), T, bd, bd, 0.f);
    { const float* b = projB[L].data_ptr<float>(); for (int t = 0; t < T; ++t) for (int c = 0; c < bd; ++c) h[t * bd + c] += y2[t * bd + c] + b[c]; }
    std::vector<float> ln2((size_t)T * bd);
    for (int t = 0; t < T; ++t) layernorm(&h[t * bd], ln2W[L].data_ptr<float>(), ln2B[L].data_ptr<float>(), &ln2[t * bd], bd);
    gemm_nt(ln2.data(), ff0W[L].data_ptr<float>(), ffh.data(), T, hid, bd, 0.f);
    { const float* b = ff0B[L].data_ptr<float>(); for (int t = 0; t < T; ++t) for (int c = 0; c < hid; ++c) ffh[t * hid + c] += b[c]; }
    vgelu(ffh.data(), T * hid);
    gemm_nt(ffh.data(), ff2W[L].data_ptr<float>(), y2.data(), T, bd, hid, 0.f);
    { const float* b = ff2B[L].data_ptr<float>(); for (int t = 0; t < T; ++t) for (int c = 0; c < bd; ++c) h[t * bd + c] += y2[t * bd + c] + b[c]; }
  }
  std::vector<float> nf((size_t)T * bd);
  for (int t = 0; t < T; ++t) layernorm(&h[t * bd], fnW.data_ptr<float>(), fnB.data_ptr<float>(), &nf[t * bd], bd);
  h.swap(nf);
}

static void tsmixer_layers(std::vector<float>& h, int T, int bd, int t_max, bool causal,
    const std::vector<torch::Tensor>& tnW, const std::vector<torch::Tensor>& tnB,
    const std::vector<torch::Tensor>& timeW, const std::vector<torch::Tensor>& timeB,
    const std::vector<torch::Tensor>& fnW, const std::vector<torch::Tensor>& fnB,
    const std::vector<torch::Tensor>& ff0W, const std::vector<torch::Tensor>& ff0B,
    const std::vector<torch::Tensor>& ff2W, const std::vector<torch::Tensor>& ff2B,
    torch::Tensor nW, torch::Tensor nB) {
  TORCH_CHECK(T <= t_max, "T > t_max");
  const int nL = (int)timeW.size();
  const int hid = ff0W[0].size(0);
  std::vector<float> H((size_t)t_max * bd, 0.f);
  for (int t = 0; t < T; ++t) for (int c = 0; c < bd; ++c) H[t * bd + c] = h[t * bd + c];
  std::vector<float> lnX((size_t)t_max * bd), Y((size_t)t_max * bd), W((size_t)t_max * t_max), ffh((size_t)t_max * hid);
  for (int L = 0; L < nL; ++L) {
    for (int t = 0; t < t_max; ++t) layernorm(&H[t * bd], tnW[L].data_ptr<float>(), tnB[L].data_ptr<float>(), &lnX[t * bd], bd);
    std::memcpy(W.data(), timeW[L].data_ptr<float>(), sizeof(float) * t_max * t_max);
    if (causal) for (int i = 0; i < t_max; ++i) for (int j = i + 1; j < t_max; ++j) W[i * t_max + j] = 0.f;
    gemm_nn(W.data(), lnX.data(), Y.data(), t_max, bd, t_max);
    { const float* b = timeB[L].data_ptr<float>(); for (int t = 0; t < t_max; ++t) for (int c = 0; c < bd; ++c) Y[t * bd + c] += b[t]; }
    vgelu(Y.data(), t_max * bd);
    for (int t = 0; t < t_max; ++t) for (int c = 0; c < bd; ++c) H[t * bd + c] += Y[t * bd + c];
    for (int t = 0; t < t_max; ++t) layernorm(&H[t * bd], fnW[L].data_ptr<float>(), fnB[L].data_ptr<float>(), &lnX[t * bd], bd);
    gemm_nt(lnX.data(), ff0W[L].data_ptr<float>(), ffh.data(), t_max, hid, bd, 0.f);
    { const float* b = ff0B[L].data_ptr<float>(); for (int t = 0; t < t_max; ++t) for (int c = 0; c < hid; ++c) ffh[t * hid + c] += b[c]; }
    vgelu(ffh.data(), t_max * hid);
    gemm_nt(ffh.data(), ff2W[L].data_ptr<float>(), Y.data(), t_max, bd, hid, 0.f);
    { const float* b = ff2B[L].data_ptr<float>(); for (int t = 0; t < t_max; ++t) for (int c = 0; c < bd; ++c) H[t * bd + c] += Y[t * bd + c] + b[c]; }
  }
  std::vector<float> nf((size_t)T * bd);
  for (int t = 0; t < T; ++t) layernorm(&H[t * bd], nW.data_ptr<float>(), nB.data_ptr<float>(), &nf[t * bd], bd);
  h.swap(nf);
}

static void mamba_scan(const float* xc, float* y, int T, int di, int ds, int dtr,
                       const float* xpW, const float* dtW, const float* dtB,
                       const float* Alog, const float* D) {
  std::vector<float> h((size_t)di * ds, 0.f), dbc(dtr + 2 * ds), dt(dtr), Bv(ds), Cv(ds), delta(di);
  for (int t = 0; t < T; ++t) {
    linear(xpW, nullptr, xc + t * di, dbc.data(), dtr + 2 * ds, di);
    for (int i = 0; i < dtr; ++i) dt[i] = dbc[i];
    for (int i = 0; i < ds; ++i) { Bv[i] = dbc[dtr + i]; Cv[i] = dbc[dtr + ds + i]; }
    linear(dtW, dtB, dt.data(), delta.data(), di, dtr);
    for (int c = 0; c < di; ++c) delta[c] = softplus(delta[c]);
    const float* xt = xc + t * di;
    float* yt = y + t * di;
    for (int c = 0; c < di; ++c) {
      float acc = 0.f;
      for (int s = 0; s < ds; ++s) {
        float A = -std::exp(Alog[c * ds + s]);
        float da = std::exp(delta[c] * A);
        float du = delta[c] * xt[c] * Bv[s];
        h[c * ds + s] = da * h[c * ds + s] + du;
        acc += h[c * ds + s] * Cv[s];
      }
      yt[c] = acc + D[c] * xt[c];
    }
  }
}

static void mamba_layers(std::vector<float>& x, int T, int bd, bool causal,
    const std::vector<torch::Tensor>& inW, const std::vector<torch::Tensor>& convW, const std::vector<torch::Tensor>& convB,
    const std::vector<torch::Tensor>& xpW, const std::vector<torch::Tensor>& dtW, const std::vector<torch::Tensor>& dtB,
    const std::vector<torch::Tensor>& Alog, const std::vector<torch::Tensor>& D,
    const std::vector<torch::Tensor>& outW, const std::vector<torch::Tensor>& lnW, const std::vector<torch::Tensor>& lnB,
    torch::Tensor nW, torch::Tensor nB) {
  const int nL = (int)inW.size();
  const int di = inW[0].size(0) / 2;
  const int ds = Alog[0].size(1);
  const int dtr = dtW[0].size(1);
  const int k = convW[0].dim() == 3 ? (int)convW[0].size(2) : (int)convW[0].size(1);
  std::vector<float> u((size_t)T * bd), xz((size_t)T * 2 * di), xs((size_t)T * di), z((size_t)T * di);
  std::vector<float> xc((size_t)T * di), y((size_t)T * di), yb((size_t)T * di), xr((size_t)T * di), yp((size_t)T * bd);
  for (int L = 0; L < nL; ++L) {
    for (int t = 0; t < T; ++t) layernorm(&x[t * bd], lnW[L].data_ptr<float>(), lnB[L].data_ptr<float>(), &u[t * bd], bd);
    gemm_nt(u.data(), inW[L].data_ptr<float>(), xz.data(), T, 2 * di, bd, 0.f);
    for (int t = 0; t < T; ++t) {
      std::memcpy(&xs[t * di], &xz[t * 2 * di], sizeof(float) * di);
      std::memcpy(&z[t * di], &xz[t * 2 * di + di], sizeof(float) * di);
    }
    const float* cw = convW[L].data_ptr<float>();
    const float* cb = convB[L].data_ptr<float>();
    for (int t = 0; t < T; ++t) for (int c = 0; c < di; ++c) {
      float acc = cb[c];
      for (int kk = 0; kk < k; ++kk) {
        int src = t - (k - 1) + kk;
        if (src >= 0) acc += cw[c * k + kk] * xs[src * di + c];
      }
      float s = acc * sigmoid(acc);
      xc[t * di + c] = s;
    }
    mamba_scan(xc.data(), y.data(), T, di, ds, dtr,
               xpW[L].data_ptr<float>(), dtW[L].data_ptr<float>(), dtB[L].data_ptr<float>(),
               Alog[L].data_ptr<float>(), D[L].data_ptr<float>());
    if (!causal) {
      for (int t = 0; t < T; ++t) std::memcpy(&xr[t * di], &xc[(T - 1 - t) * di], sizeof(float) * di);
      mamba_scan(xr.data(), yb.data(), T, di, ds, dtr,
                 xpW[L].data_ptr<float>(), dtW[L].data_ptr<float>(), dtB[L].data_ptr<float>(),
                 Alog[L].data_ptr<float>(), D[L].data_ptr<float>());
      for (int t = 0; t < T; ++t) for (int c = 0; c < di; ++c)
        y[t * di + c] = 0.5f * (y[t * di + c] + yb[(T - 1 - t) * di + c]);
    }
    for (int t = 0; t < T; ++t) for (int c = 0; c < di; ++c) y[t * di + c] *= z[t * di + c] * sigmoid(z[t * di + c]);
    gemm_nt(y.data(), outW[L].data_ptr<float>(), yp.data(), T, bd, di, 0.f);
    for (int t = 0; t < T; ++t) for (int c = 0; c < bd; ++c) x[t * bd + c] += yp[t * bd + c];
  }
  std::vector<float> nf((size_t)T * bd);
  for (int t = 0; t < T; ++t) layernorm(&x[t * bd], nW.data_ptr<float>(), nB.data_ptr<float>(), &nf[t * bd], bd);
  x.swap(nf);
}

#define FRONT_ARGS \
  torch::Tensor coords, torch::Tensor polar, bool causal, torch::Tensor mem_in, \
  torch::Tensor pointW, torch::Tensor pointB, torch::Tensor pa_lin, torch::Tensor pm, torch::Tensor pa, torch::Tensor pb, \
  torch::Tensor pgamp, torch::Tensor pgmu, torch::Tensor pglogsig, double le_xmax, double te_xmin, \
  int64_t profile_stations, torch::Tensor profW, torch::Tensor profB, \
  torch::Tensor logA, torch::Tensor WdW, torch::Tensor WdB, torch::Tensor WbW, torch::Tensor WbB, \
  torch::Tensor WcW, torch::Tensor WcB, torch::Tensor headW, torch::Tensor headB, \
  torch::Tensor pinW, torch::Tensor pinB, torch::Tensor bbInW, torch::Tensor bbInB

torch::Tensor transformer_forward(
    FRONT_ARGS, int64_t n_heads,
    std::vector<torch::Tensor> ln1W, std::vector<torch::Tensor> ln1B,
    std::vector<torch::Tensor> qkvW, std::vector<torch::Tensor> qkvB,
    std::vector<torch::Tensor> projW, std::vector<torch::Tensor> projB,
    std::vector<torch::Tensor> ln2W, std::vector<torch::Tensor> ln2B,
    std::vector<torch::Tensor> ff0W, std::vector<torch::Tensor> ff0B,
    std::vector<torch::Tensor> ff2W, std::vector<torch::Tensor> ff2B,
    torch::Tensor fnW, torch::Tensor fnB, torch::Tensor hdW, torch::Tensor hdB) {
  std::vector<float> tokens, draft, h; int T, d, bd;
  front_end(coords, polar, mem_in, pointW, pointB, pa_lin, pm, pa, pb, pgamp, pgmu, pglogsig,
            le_xmax, te_xmin, profile_stations, profW, profB, logA, WdW, WdB, WbW, WbB, WcW, WcB,
            headW, headB, pinW, pinB, tokens, draft, T, d);
  project_bb(tokens, T, d, bbInW, bbInB, h, bd);
  transformer_layers(h, T, bd, (int)n_heads, causal, ln1W, ln1B, qkvW, qkvB, projW, projB, ln2W, ln2B, ff0W, ff0B, ff2W, ff2B, fnW, fnB);
  return pack_out(draft, h, hdW, hdB, T, bd);
}

torch::Tensor tsmixer_forward(
    FRONT_ARGS, int64_t t_max,
    std::vector<torch::Tensor> tnW, std::vector<torch::Tensor> tnB,
    std::vector<torch::Tensor> timeW, std::vector<torch::Tensor> timeB,
    std::vector<torch::Tensor> fnW, std::vector<torch::Tensor> fnB,
    std::vector<torch::Tensor> ff0W, std::vector<torch::Tensor> ff0B,
    std::vector<torch::Tensor> ff2W, std::vector<torch::Tensor> ff2B,
    torch::Tensor nW, torch::Tensor nB, torch::Tensor hdW, torch::Tensor hdB) {
  std::vector<float> tokens, draft, h; int T, d, bd;
  front_end(coords, polar, mem_in, pointW, pointB, pa_lin, pm, pa, pb, pgamp, pgmu, pglogsig,
            le_xmax, te_xmin, profile_stations, profW, profB, logA, WdW, WdB, WbW, WbB, WcW, WcB,
            headW, headB, pinW, pinB, tokens, draft, T, d);
  project_bb(tokens, T, d, bbInW, bbInB, h, bd);
  tsmixer_layers(h, T, bd, (int)t_max, causal, tnW, tnB, timeW, timeB, fnW, fnB, ff0W, ff0B, ff2W, ff2B, nW, nB);
  return pack_out(draft, h, hdW, hdB, T, bd);
}

torch::Tensor mamba_forward(
    FRONT_ARGS,
    std::vector<torch::Tensor> inW, std::vector<torch::Tensor> convW, std::vector<torch::Tensor> convB,
    std::vector<torch::Tensor> xpW, std::vector<torch::Tensor> dtW, std::vector<torch::Tensor> dtB,
    std::vector<torch::Tensor> Alog, std::vector<torch::Tensor> Dv,
    std::vector<torch::Tensor> outW, std::vector<torch::Tensor> lnW, std::vector<torch::Tensor> lnB,
    torch::Tensor nW, torch::Tensor nB, torch::Tensor hdW, torch::Tensor hdB) {
  std::vector<float> tokens, draft, h; int T, d, bd;
  front_end(coords, polar, mem_in, pointW, pointB, pa_lin, pm, pa, pb, pgamp, pgmu, pglogsig,
            le_xmax, te_xmin, profile_stations, profW, profB, logA, WdW, WdB, WbW, WbB, WcW, WcB,
            headW, headB, pinW, pinB, tokens, draft, T, d);
  project_bb(tokens, T, d, bbInW, bbInB, h, bd);
  mamba_layers(h, T, bd, causal, inW, convW, convB, xpW, dtW, dtB, Alog, Dv, outW, lnW, lnB, nW, nB);
  return pack_out(draft, h, hdW, hdB, T, bd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transformer_forward", &transformer_forward);
  m.def("tsmixer_forward", &tsmixer_forward);
  m.def("mamba_forward", &mamba_forward);
}
"""

_EXT = None


def get_ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        import platform

        bd = Path(__file__).resolve().parents[1] / "build" / "fused_baselines"
        bd.mkdir(parents=True, exist_ok=True)
        ldflags = ["-framework", "Accelerate"] if platform.system() == "Darwin" else []
        _EXT = load_inline(
            name="foilcore_fused_baselines",
            cpp_sources=_SRC,
            extra_cflags=["-O3", "-std=c++17", "-ffast-math", "-march=native"],
            extra_ldflags=ldflags,
            build_directory=str(bd),
            verbose=False,
        )
    return _EXT


def _f(t):
    return t.detach().contiguous().float()


class FusedSeqBaseline:
    """Wraps a seqbaseline FoilCORE (transformer / tsmixer / mamba) for one-call C++ inference."""

    def __init__(self, model):
        cfg = model.config
        assert cfg.model_type == "seqbaseline" and cfg.geom_pool == "region3" and cfg.share_path
        m = model.eval().cpu()
        self.kind = str(cfg.seq_backbone)
        self.d = int(cfg.d_model)
        p = m.path
        self.args_geo = [
            _f(m.geom_encoder.point.weight), _f(m.geom_encoder.point.bias),
            _f(p.a_lin), _f(p.m), _f(p.a), _f(p.b), _f(p.gauss_amp), _f(p.gauss_mu), _f(p.gauss_log_sigma),
            float(cfg.le_xmax), float(cfg.te_xmin),
        ]
        S = int(cfg.profile_stations)
        z = torch.zeros(0)
        self.args_prof = [S, _f(m.profile_in.weight) if S else z, _f(m.profile_in.bias) if S else z]
        mm = m.mem_mix
        self.args_mix = [
            _f(mm.log_A), _f(mm.W_delta.weight), _f(mm.W_delta.bias),
            _f(mm.W_B.weight), _f(mm.W_B.bias), _f(mm.W_C.weight), _f(mm.W_C.bias),
            _f(m.head.weight), _f(m.head.bias),
        ]
        self.args_pin = [_f(m.polar_in.weight), _f(m.polar_in.bias)]
        if isinstance(m.bb_in, nn.Identity):
            self.args_bbin = [z, z]
        else:
            self.args_bbin = [_f(m.bb_in.weight), _f(m.bb_in.bias)]
        bb = m.backbone
        if self.kind == "transformer":
            layers = list(bb.layers)
            self.n_heads = int(layers[0].h)
            self.args_bb = [
                self.n_heads,
                [_f(L.ln1.weight) for L in layers], [_f(L.ln1.bias) for L in layers],
                [_f(L.qkv.weight) for L in layers], [_f(L.qkv.bias) for L in layers],
                [_f(L.proj.weight) for L in layers], [_f(L.proj.bias) for L in layers],
                [_f(L.ln2.weight) for L in layers], [_f(L.ln2.bias) for L in layers],
                [_f(L.ff[0].weight) for L in layers], [_f(L.ff[0].bias) for L in layers],
                [_f(L.ff[2].weight) for L in layers], [_f(L.ff[2].bias) for L in layers],
                _f(bb.norm.weight), _f(bb.norm.bias),
                _f(m.bb_head.weight), _f(m.bb_head.bias),
            ]
        elif self.kind == "tsmixer":
            n = len(bb.time_w)
            self.args_bb = [
                int(bb.t_max),
                [_f(bb.time_norm[i].weight) for i in range(n)], [_f(bb.time_norm[i].bias) for i in range(n)],
                [_f(bb.time_w[i]) for i in range(n)], [_f(bb.time_b[i]) for i in range(n)],
                [_f(bb.feat_norm[i].weight) for i in range(n)], [_f(bb.feat_norm[i].bias) for i in range(n)],
                [_f(bb.feat_mlp[i][0].weight) for i in range(n)], [_f(bb.feat_mlp[i][0].bias) for i in range(n)],
                [_f(bb.feat_mlp[i][2].weight) for i in range(n)], [_f(bb.feat_mlp[i][2].bias) for i in range(n)],
                _f(bb.norm.weight), _f(bb.norm.bias),
                _f(m.bb_head.weight), _f(m.bb_head.bias),
            ]
        elif self.kind == "mamba":
            layers = list(bb.layers)
            self.args_bb = [
                [_f(L.in_proj.weight) for L in layers],
                [_f(L.conv.weight) for L in layers], [_f(L.conv.bias) for L in layers],
                [_f(L.x_proj.weight) for L in layers],
                [_f(L.dt_proj.weight) for L in layers], [_f(L.dt_proj.bias) for L in layers],
                [_f(L.A_log) for L in layers], [_f(L.D) for L in layers],
                [_f(L.out_proj.weight) for L in layers],
                [_f(L.norm.weight) for L in layers], [_f(L.norm.bias) for L in layers],
                _f(bb.norm.weight), _f(bb.norm.bias),
                _f(m.bb_head.weight), _f(m.bb_head.bias),
            ]
        else:
            raise ValueError(self.kind)
        self.ext = get_ext()

    def __call__(self, coords: torch.Tensor, polar: torch.Tensor, causal: bool = True, mem: torch.Tensor | None = None) -> torch.Tensor:
        mem_in = mem.contiguous().float() if mem is not None else torch.zeros(0)
        args = (coords.contiguous().float(), polar.contiguous().float(), bool(causal), mem_in,
                *self.args_geo, *self.args_prof, *self.args_mix, *self.args_pin, *self.args_bbin, *self.args_bb)
        if self.kind == "transformer":
            return self.ext.transformer_forward(*args)
        if self.kind == "tsmixer":
            return self.ext.tsmixer_forward(*args)
        return self.ext.mamba_forward(*args)

    @torch.no_grad()
    def encode(self, model, coords: torch.Tensor) -> torch.Tensor:
        return model.encode_geom(coords[None])[0].contiguous().float()


@torch.no_grad()
def check(model, coords: torch.Tensor, polar: torch.Tensor, causal: bool = True) -> float:
    ff = FusedSeqBaseline(model)
    out = ff(coords, polar, causal)
    mask = torch.ones(1, polar.shape[0], dtype=torch.bool)
    ref, _ = model(coords[None], polar[None], mask, causal=causal)
    return float((out - ref[0]).abs().max())
