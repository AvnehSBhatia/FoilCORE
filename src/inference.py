import torch

from src.data import N_GEOM_PATCHES, patches_to_coords, resample_airfoil, normalize_coords
from src.model import FoilCORE, config_from_checkpoint, remap_foilcore_state_dict


class FoilCOREGenerator:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.stats = ckpt["stats"]
        cfg = config_from_checkpoint(ckpt)
        self.model = FoilCORE(config=cfg).to(device)
        m_sd = remap_foilcore_state_dict(
            dict(ckpt["model"]), target_shared_qk_scalars=cfg.shared_qk_scalars
        )
        model_sd = self.model.state_dict()
        compatible = {
            k: v for k, v in m_sd.items()
            if k in model_sd and getattr(v, "shape", None) == model_sd[k].shape
        }
        self.model.load_state_dict(compatible, strict=False)
        self.model.eval()
        self.config = cfg

    def _normalize_polar_grid(self, alpha, re, mach):
        n = len(alpha)
        mean = torch.as_tensor(self.stats["polar_mean"], dtype=torch.float32)
        std = torch.as_tensor(self.stats["polar_std"], dtype=torch.float32)
        tok = torch.zeros(n, 5, dtype=torch.float32)
        tok[:, 0] = torch.as_tensor(alpha, dtype=torch.float32)
        tok[:, 1] = float(re)
        tok[:, 2] = float(mach)
        return (tok - mean) / std

    @torch.no_grad()
    def predict(
        self,
        coords,
        alpha,
        re,
        mach,
        *,
        causal: bool = False,
        seed_len: int = 0,
        seed_clcd=None,
    ):
        """Predict Cl, Cd for an α schedule. ``coords`` is raw or 200-point chord-normalized."""
        import numpy as np

        coords = np.asarray(coords, dtype=np.float64)
        if coords.shape[0] != 200:
            coords = resample_airfoil(coords, n_points=200)
        coords_n, _, _ = normalize_coords(coords)
        geom = torch.tensor(coords_n, dtype=torch.float32, device=self.device).unsqueeze(0)
        polar = self._normalize_polar_grid(alpha, re, mach).unsqueeze(0).to(self.device)
        mask = torch.ones(1, polar.shape[1], dtype=torch.bool, device=self.device)
        inject = None
        if seed_clcd is not None:
            inject = torch.as_tensor(seed_clcd, dtype=polar.dtype, device=self.device)
            if inject.dim() == 2:
                inject = inject.unsqueeze(0)
        final, draft = self.model(
            geom, polar, mask, causal=causal, seed_len=seed_len, clcd_inject=inject
        )
        mean = torch.as_tensor(self.stats["polar_mean"], device=self.device, dtype=final.dtype)
        std = torch.as_tensor(self.stats["polar_std"], device=self.device, dtype=final.dtype)
        cl = final[0, :, 0] * std[3] + mean[3]
        cd = final[0, :, 1] * std[4] + mean[4]
        return {
            "Cl": cl.cpu(),
            "Cd": cd.cpu(),
            "draft_Cl": (draft[0, :, 0] * std[3] + mean[3]).cpu(),
            "draft_Cd": (draft[0, :, 1] * std[4] + mean[4]).cpu(),
            "alpha": torch.as_tensor(alpha, dtype=torch.float32),
        }

    def generate(self, prompt, max_len=200, temperature=1.0):
        _ = (max_len, temperature)
        geom_patches = [t["value"] for t in prompt if t.get("type") == "geom"]
        polar_ctx = [t for t in prompt if t.get("type") == "polar"]
        if not geom_patches:
            raise ValueError("prompt must include geom tokens (or call predict() with coords)")
        stacked = torch.stack(
            [p if torch.is_tensor(p) else torch.as_tensor(p) for p in geom_patches]
        )
        if stacked.shape[0] == N_GEOM_PATCHES:
            coords = patches_to_coords(stacked)
        else:
            coords = stacked.reshape(-1, 2)
        if not polar_ctx:
            raise ValueError("prompt must include polar tokens with AoA/Re/Mach")
        pol = torch.stack([t["value"] for t in polar_ctx])
        mean = torch.as_tensor(self.stats["polar_mean"])
        std = torch.as_tensor(self.stats["polar_std"])
        phys = pol * std + mean
        out = self.predict(
            coords.numpy(),
            phys[:, 0].tolist(),
            float(phys[0, 1]),
            float(phys[0, 2]),
        )
        generated = list(prompt)
        for i in range(len(phys)):
            val = pol[i].clone()
            val[3] = (out["Cl"][i] - mean[3]) / std[3]
            val[4] = (out["Cd"][i] - mean[4]) / std[4]
            generated.append(
                {
                    "type": "polar",
                    "value": val,
                    "physical": torch.stack(
                        [phys[i, 0], phys[i, 1], phys[i, 2], out["Cl"][i], out["Cd"][i]]
                    ),
                }
            )
        return {"polar": torch.stack([g["physical"] for g in generated if "physical" in g])}
