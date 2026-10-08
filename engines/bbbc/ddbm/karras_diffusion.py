"""
Based on: https://github.com/crowsonkb/k-diffusion
"""

import math

import numpy as np
import torch
from tqdm.auto import tqdm
import torch.distributed as dist


from .auxiliary_losses import cfg_distillation_loss
from .nn import mean_flat, append_dims, append_zero
from .random_util import BatchedSeedGenerator


def _capture_torch_rng_state(device):
    """Capture CPU and current-device RNG without touching other DDP ranks."""
    cpu_state = torch.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device=device)
    return cpu_state, cuda_state


def _restore_torch_rng_state(device, state):
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device=device)


ECSI_SAMPLER_OPTIONS = {
    "ecsi": {"eta_schedule": "constant", "second_order_x0": False},
    "ecsi_sde_exp": {
        "eta_schedule": "constant",
        "second_order_x0": False,
        "sde_integrator": "exponential",
        "solver_order": 1,
        "reconstruction_tail_eta": 0.0,
    },
    "ecsi_sde_dpmpp_2m": {
        "eta_schedule": "constant",
        "second_order_x0": False,
        "sde_integrator": "exponential",
        "solver_order": 2,
        "solver_type": "midpoint",
        "terminal_lower_order": True,
        "reconstruction_tail_eta": 0.0,
    },
    "ecsi_sde_dpmpp_2m_lower_order_tail": {
        "eta_schedule": "constant",
        "second_order_x0": False,
        "sde_integrator": "exponential",
        "solver_order": 2,
        "solver_type": "midpoint",
        "solver_lower_order_tail": True,
        "terminal_lower_order": True,
        "reconstruction_tail_eta": 0.0,
    },
    "ecsi_cosine": {"eta_schedule": "cosine", "second_order_x0": False},
    "ecsi_2m": {"eta_schedule": "constant", "second_order_x0": True},
    "ecsi_cosine_2m": {"eta_schedule": "cosine", "second_order_x0": True},
    "ecsi_cosine_recon2": {
        "eta_schedule": "cosine",
        "second_order_x0": False,
    },
    "ecsi_2m_lower_order_tail": {
        "eta_schedule": "constant",
        "second_order_x0": True,
        "second_order_lower_order_tail": True,
    },
    "ecsi_cosine_2m_lower_order_tail": {
        "eta_schedule": "cosine",
        "second_order_x0": True,
        "second_order_lower_order_tail": True,
    },
    "ecsi_lambda": {
        "eta_schedule": "constant",
        "second_order_x0": False,
        "time_grid": "bridge_lambda",
    },
    "ecsi_2m_lower_order_tail_lambda": {
        "eta_schedule": "constant",
        "second_order_x0": True,
        "second_order_lower_order_tail": True,
        "time_grid": "bridge_lambda",
    },
    "ecsi_cosine_2m_lower_order_tail_lambda": {
        "eta_schedule": "cosine",
        "second_order_x0": True,
        "second_order_lower_order_tail": True,
        "time_grid": "bridge_lambda",
    },
    **{
        f"ecsi_2m_damped_{tag}": {
            "eta_schedule": "constant",
            "second_order_x0": True,
            "second_order_lower_order_tail": True,
            "second_order_scale": scale,
        }
        for tag, scale in (
            ("001", 0.01),
            ("002", 0.02),
            ("003", 0.03),
            ("004", 0.04),
            ("005", 0.05),
            ("01", 0.1),
            ("02", 0.2),
            ("03", 0.3),
            ("05", 0.5),
            ("075", 0.75),
        )
    },
    **{
        f"ecsi_2m_window_{name}": {
            "eta_schedule": "constant",
            "second_order_x0": True,
            "second_order_lower_order_tail": True,
            "second_order_scale": 0.1,
            "second_order_progress_window": window,
        }
        for name, window in (
            ("early", [0.0, 1 / 3]),
            ("middle", [1 / 3, 2 / 3]),
            ("late", [2 / 3, 1.0]),
        )
    },
    **{
        f"ecsi_cosine_power{power}": {
            "eta_schedule": "cosine",
            "eta_schedule_power": float(power),
            "second_order_x0": False,
        }
        for power in (2, 4, 8, 16)
    },
    **{
        f"ecsi_2m_reverse_{tag}": {
            "eta_schedule": "constant",
            "second_order_x0": True,
            "second_order_lower_order_tail": True,
            "second_order_scale": -scale,
        }
        for tag, scale in (
            ("001", 0.01),
            ("003", 0.03),
            ("005", 0.05),
            ("01", 0.1),
            ("02", 0.2),
            ("03", 0.3),
            ("035", 0.35),
            ("0375", 0.375),
            ("04", 0.4),
            ("0415", 0.415),
            ("0425", 0.425),
            ("04275", 0.4275),
            ("043", 0.43),
            ("04325", 0.4325),
            ("0435", 0.435),
            ("04375", 0.4375),
            ("044", 0.44),
            ("04425", 0.4425),
            ("0445", 0.445),
            ("045", 0.45),
            ("0475", 0.475),
            ("05", 0.5),
            ("075", 0.75),
            ("1", 1.0),
            ("15", 1.5),
        )
    },
    **{
        f"ecsi_2m_reverse_late_{tag}": {
            "eta_schedule": "constant",
            "second_order_x0": True,
            "second_order_lower_order_tail": True,
            "second_order_scale": -scale,
            "second_order_progress_window": [2 / 3, 1.0],
        }
        for tag, scale in (
            ("005", 0.05),
            ("01", 0.1),
            ("02", 0.2),
            ("03", 0.3),
            ("04", 0.4),
            ("05", 0.5),
            ("075", 0.75),
            ("1", 1.0),
        )
    },
    **{
        f"ecsi_2m_reverse_piecewise_{tag}": {
            "eta_schedule": "constant",
            "second_order_x0": True,
            "second_order_lower_order_tail": True,
            "second_order_scale": scales[0],
            "second_order_scale_schedule": scales,
        }
        for tag, scales in (
            ("035_04_05", [-0.35, -0.4, -0.5]),
            ("04_035_05", [-0.4, -0.35, -0.5]),
            ("04_04_05", [-0.4, -0.4, -0.5]),
            ("04_045_05", [-0.4, -0.45, -0.5]),
            ("045_04_05", [-0.45, -0.4, -0.5]),
            ("04_04_06", [-0.4, -0.4, -0.6]),
        )
    },
    **{
        f"ecsi_2m_reverse_blend_{tag}": {
            "eta_schedule": "constant",
            "second_order_x0": True,
            "second_order_lower_order_tail": True,
            "second_order_blend_weight": blend_weight,
        }
        for tag, blend_weight in (
            ("010", 0.10),
            ("015", 0.15),
            ("020", 0.20),
            ("025", 0.25),
            ("030", 0.30),
            ("035", 0.35),
        )
    },
    **{
        f"ecsi_2m_reverse_0425_tail_{tag}": {
            "eta_schedule": "constant",
            "second_order_x0": True,
            "second_order_scale": -0.425,
            "second_order_tail_scale": -tail_scale,
        }
        for tag, tail_scale in (
            ("0005", 0.005),
            ("001", 0.01),
            ("0015", 0.015),
            ("0018", 0.018),
            ("00185", 0.0185),
            ("0019", 0.019),
            ("001925", 0.01925),
            ("00195", 0.0195),
            ("001975", 0.01975),
            ("002", 0.02),
            ("002025", 0.02025),
            ("00205", 0.0205),
            ("0025", 0.025),
            ("003", 0.03),
            ("0035", 0.035),
            ("004", 0.04),
            ("005", 0.05),
        )
    },
    "ecsi_2m_reverse_02_tail_001": {
        "eta_schedule": "constant",
        "second_order_x0": True,
        "second_order_scale": -0.2,
        "second_order_tail_scale": -0.01,
    },
    **{
        f"ecsi_2m_reverse_{main_tag}_tail_{tail_tag}": {
            "eta_schedule": "constant",
            "second_order_x0": True,
            "second_order_scale": -main_scale,
            "second_order_tail_scale": -tail_scale,
        }
        for main_tag, main_scale, tail_tag, tail_scale in (
            ("0415", 0.415, "002", 0.02),
            ("042", 0.42, "002", 0.02),
            ("04225", 0.4225, "002", 0.02),
            ("04275", 0.4275, "002", 0.02),
            ("043", 0.43, "002", 0.02),
            ("0435", 0.435, "002", 0.02),
            ("04375", 0.4375, "002", 0.02),
            ("044", 0.44, "00175", 0.0175),
            ("044", 0.44, "00185", 0.0185),
            ("044", 0.44, "002", 0.02),
            ("044", 0.44, "00205", 0.0205),
            ("044", 0.44, "00215", 0.0215),
            ("044", 0.44, "00225", 0.0225),
            ("04425", 0.4425, "002", 0.02),
        )
    },
}
ECSI_SAMPLERS = frozenset(ECSI_SAMPLER_OPTIONS)
ECSI_SECOND_ORDER_STEP_GUARD = 1e-5


class NoiseSchedule:
    def __init__(self):
        raise NotImplementedError

    def get_f_g2(self, t):
        raise NotImplementedError

    def get_alpha_rho(self, t):
        raise NotImplementedError

    def get_abc(self, t):
        alpha_t, alpha_bar_t, rho_t, rho_bar_t = self.get_alpha_rho(t)
        a_t, b_t, c_t = (
            (alpha_bar_t * rho_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t * rho_t) / self.rho_T,
        )
        return a_t, b_t, c_t


class VPNoiseSchedule(NoiseSchedule):
    def __init__(self, beta_d=2, beta_min=0.1):
        self.beta_d, self.beta_min = beta_d, beta_min
        self.alpha_fn = lambda t: np.e ** (-0.5 * beta_min * t - 0.25 * beta_d * t**2)
        self.alpha_T = self.alpha_fn(1)
        self.rho_fn = lambda t: (np.e ** (beta_min * t + 0.5 * beta_d * t**2) - 1).sqrt()
        self.rho_T = self.rho_fn(torch.DoubleTensor([1])).item()

        self.f_fn = lambda t: (-0.5 * beta_min - 0.5 * beta_d * t)
        self.g2_fn = lambda t: (beta_min + beta_d * t)

    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class VENoiseSchedule(NoiseSchedule):
    def __init__(self, sigma_max=80.0):
        self.sigma_max = sigma_max
        self.alpha_fn = lambda t: torch.ones_like(t)
        self.alpha_T = 1
        self.rho_fn = lambda t: t
        self.rho_T = sigma_max

        self.f_fn = lambda t: torch.zeros_like(t)
        self.g2_fn = lambda t: 2 * t

    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class LinearBridgeNoiseSchedule(NoiseSchedule):
    """ECSI's symmetric linear bridge path.

    ``NoiseSchedule.get_abc`` uses the DDBM endpoint convention
    ``a(t) * xT + b(t) * x0 + c(t) * z``.  ECSI writes the same path as
    ``alpha(t) * x0 + beta(t) * xT + gamma(t) * z``.  Consequently this
    class returns ``(beta, alpha, gamma)`` from :meth:`get_abc`.
    """

    def __init__(self, gamma_max=0.125):
        if gamma_max <= 0:
            raise ValueError(f"gamma_max must be positive, got {gamma_max}")
        self.gamma_max = float(gamma_max)

    def get_abc(self, t):
        t = t.to(torch.float64)
        a_t = t
        b_t = 1 - t
        c_t = 2 * self.gamma_max * torch.sqrt((t * (1 - t)).clamp_min(0))
        return a_t, b_t, c_t

    def get_abc_derivatives(self, t):
        t = t.to(torch.float64)
        da_t = torch.ones_like(t)
        db_t = -torch.ones_like(t)
        denominator = torch.sqrt((t * (1 - t)).clamp_min(torch.finfo(t.dtype).tiny))
        dc_t = self.gamma_max * (1 - 2 * t) / denominator
        return da_t, db_t, dc_t


class I2SBNoiseSchedule(NoiseSchedule):
    def __init__(self, n_timestep=1000, beta_min=0.1, beta_max=1.0):
        self.n_timestep, self.linear_start, self.linear_end = (
            n_timestep,
            beta_min / n_timestep,
            beta_max / n_timestep,
        )
        betas = (
            torch.linspace(
                self.linear_start**0.5,
                self.linear_end**0.5,
                n_timestep,
                dtype=torch.float64,
            ).cuda()
            ** 2
        )
        betas = torch.cat(
            [
                betas[: self.n_timestep // 2],
                torch.flip(betas[: self.n_timestep // 2], dims=(0,)),
            ]
        )
        std_fwd = torch.sqrt(torch.cumsum(betas, dim=0))
        std_bwd = torch.sqrt(torch.flip(torch.cumsum(torch.flip(betas, dims=(0,)), dim=0), dims=(0,)))

        self.alpha_fn = lambda t: torch.ones_like(t).float()
        self.alpha_T = 1
        self.rho_fn = lambda t: std_fwd[t]
        self.rho_T = std_fwd[-1]
        self.rho_bar_fn = lambda t: std_bwd[t]

        self.f_fn = lambda t: torch.zeros_like(t).float()
        self.g2_fn = lambda t: betas[t]

    def get_f_g2(self, t):
        t = ((self.n_timestep - 1) * t).round().long()
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = ((self.n_timestep - 1) * t).round().long()
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = self.rho_bar_fn(t)
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class PreCond:
    def __init__(self, ns):
        raise NotImplementedError

    def _get_scalings_and_weightings(self, t):
        raise NotImplementedError

    def get_scalings_and_weightings(self, t, ndim):
        c_skip, c_in, c_out, c_noise, weightings = self._get_scalings_and_weightings(t)
        c_skip, c_in, c_out, weightings = [append_dims(item, ndim) for item in [c_skip, c_in, c_out, weightings]]
        return c_skip, c_in, c_out, c_noise, weightings


class I2SBPreCond(PreCond):
    def __init__(self, ns, n_timestep=1000, t0=1e-4, T=1.0):
        self.ns = ns
        self.n_timestep = n_timestep
        self.noise_levels = torch.linspace(t0, T, n_timestep).cuda() * n_timestep

    def _get_scalings_and_weightings(self, t):
        _, _, rho_t, _ = self.ns.get_alpha_rho(t)
        c_skip = torch.ones_like(t)
        c_in = torch.ones_like(t)
        c_out = -rho_t
        c_noise = self.noise_levels[((self.n_timestep - 1) * t).round().long()]
        weightings = 1 / c_out**2
        return c_skip, c_in, c_out, c_noise, weightings


class DDBMPreCond(PreCond):
    def __init__(self, ns, sigma_data, cov_xy):
        self.ns, self.sigma_data, self.cov_xy = ns, sigma_data, cov_xy
        self.sigma_data_end = sigma_data

    def _get_scalings_and_weightings(self, t):
        a_t, b_t, c_t = self.ns.get_abc(t)
        A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2 * a_t * b_t * self.cov_xy + c_t**2
        c_in = 1 / (A) ** 0.5
        c_skip = (b_t * self.sigma_data**2 + a_t * self.cov_xy) / A
        c_out = (
            a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * c_t**2
        ) ** 0.5 * c_in
        c_noise = 1000 * 0.25 * torch.log(t + 1e-44)
        weightings = 1 / c_out**2
        return c_skip, c_in, c_out, c_noise, weightings


class KarrasDenoiser:
    def __init__(
        self,
        noise_schedule,
        precond,
        t_max=1.0,
        t_min=0.0001,
        ecsi_sample_t_min=0.001,
        ecsi_sample_t_max=1 - 1e-4,
        loss_norm="mse",
    ):

        self.t_max = t_max
        self.t_min = t_min
        self.ecsi_sample_t_min = float(ecsi_sample_t_min)
        self.ecsi_sample_t_max = float(ecsi_sample_t_max)
        if not 0 < self.t_min < self.t_max <= 1:
            raise ValueError(f"invalid training time range [{self.t_min}, {self.t_max}]")
        if not 0 < self.ecsi_sample_t_min < self.ecsi_sample_t_max < 1:
            raise ValueError(
                "invalid ECSI sampling time range "
                f"[{self.ecsi_sample_t_min}, {self.ecsi_sample_t_max}]"
            )

        self.noise_schedule = noise_schedule
        self.precond = precond
        self.auxiliary_loss_manager = None
        self.auxiliary_gate_thresh = None
        self.auxiliary_gate_slope = None
        self.cfg_distill_teacher_scale = None
        self.cfg_distill_student_scale = None
        self.cfg_distill_clip_denoised = True
        self.cfg_distill_teacher_model = None
        self.cfg_distill_pair_student_dropout = False

        self.loss_norm = loss_norm
        if loss_norm == "lpips":
            try:
                from piq import LPIPS
            except ImportError as exc:
                raise ImportError("loss_norm='lpips' requires the optional piq package") from exc
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")

    def configure_auxiliary_losses(self, manager, *, gate_thresh, gate_slope):
        """Attach frozen feature losses outside the trainable U-Net/DDP tree."""
        if manager is None:
            raise ValueError("auxiliary loss manager must not be None")
        if gate_thresh <= 0 or gate_slope <= 0:
            raise ValueError("auxiliary c_out gate threshold and slope must be positive")
        self.auxiliary_loss_manager = manager
        self.auxiliary_gate_thresh = float(gate_thresh)
        self.auxiliary_gate_slope = float(gate_slope)

    def configure_cfg_distillation(
        self,
        *,
        student_cfg_scale,
        teacher_cfg_scale,
        clip_denoised=True,
        teacher_model=None,
        pair_student_dropout=False,
    ):
        """Attach an online or fixed guidance teacher to the denoiser path."""
        if student_cfg_scale < 0 or teacher_cfg_scale <= student_cfg_scale:
            raise ValueError("CFG distillation requires 0 <= student scale < teacher scale")
        if teacher_model is not None:
            if not isinstance(teacher_model, torch.nn.Module):
                raise TypeError("fixed CFG teacher must be a torch module")
            teacher_model.eval()
            teacher_model.requires_grad_(False)
        self.cfg_distill_student_scale = float(student_cfg_scale)
        self.cfg_distill_teacher_scale = float(teacher_cfg_scale)
        self.cfg_distill_clip_denoised = bool(clip_denoised)
        self.cfg_distill_teacher_model = teacher_model
        self.cfg_distill_pair_student_dropout = bool(pair_student_dropout)

    def auxiliary_runtime_state_dict(self):
        if self.auxiliary_loss_manager is None:
            return {}
        return self.auxiliary_loss_manager.runtime_state_dict()

    def load_auxiliary_runtime_state_dict(self, state):
        if self.auxiliary_loss_manager is None:
            if state:
                raise ValueError("checkpoint contains auxiliary state but no auxiliary loss is configured")
            return
        self.auxiliary_loss_manager.load_runtime_state_dict(state)

    def bridge_sample(self, x0, xT, t, noise):
        a_t, b_t, c_t = [append_dims(item, x0.ndim) for item in self.noise_schedule.get_abc(t)]
        samples = a_t * xT + b_t * x0 + c_t * noise
        return samples

    def denoise(self, model, x_t, t, **model_kwargs):
        c_skip, c_in, c_out, c_noise, weightings = self.precond.get_scalings_and_weightings(t, x_t.ndim)
        model_output = model(c_in * x_t, c_noise, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised, weightings

    def denoise_with_cfg(self, model, x_t, t, cfg_scale=0.0, **model_kwargs):
        """Apply CellFlux classifier-free guidance to the predicted ``x0``.

        CellFlux defines ``cfg_scale=s`` as ``(1+s) * conditional -
        s * unconditional``. The unconditional branch removes only the
        molecular condition; the bridge endpoint ``xT`` remains present in
        both branches. Applying guidance after DDBM preconditioning is exactly
        equivalent to applying it to the raw model output because the skip
        coefficients sum to one under this affine combination.
        """
        if cfg_scale == 0.0:
            return self.denoise(model, x_t, t, **model_kwargs)
        if "condition" not in model_kwargs:
            raise ValueError("cfg_scale != 0 requires a conditional 'condition' tensor")

        cond_output, cond_denoised, weightings = self.denoise(model, x_t, t, **model_kwargs)
        uncond_kwargs = dict(model_kwargs)
        del uncond_kwargs["condition"]
        uncond_output, uncond_denoised, _ = self.denoise(model, x_t, t, **uncond_kwargs)
        model_output = (1.0 + cfg_scale) * cond_output - cfg_scale * uncond_output
        denoised = (1.0 + cfg_scale) * cond_denoised - cfg_scale * uncond_denoised
        return model_output, denoised, weightings

    def training_bridge_losses(
        self,
        model,
        x_start,
        t,
        model_kwargs=None,
        noise=None,
        auxiliary_context=None,
        auxiliary_loss_weights=None,
    ):
        assert model_kwargs is not None
        xT = model_kwargs["xT"]
        mask = model_kwargs.pop("mask", None)
        if noise is None:
            noise = torch.randn_like(x_start)
        t = torch.minimum(t, torch.ones_like(t) * self.t_max)
        terms = {}

        x_t = self.bridge_sample(x_start, xT, t, noise)

        auxiliary_loss_weights = auxiliary_loss_weights or {}
        moment_weight = float(auxiliary_loss_weights.get("moment", 0.0))
        infonce_weight = float(auxiliary_loss_weights.get("infonce", 0.0))
        cfg_distill_weight = float(auxiliary_loss_weights.get("cfg_distill", 0.0))
        paired_student_rng = None
        if cfg_distill_weight > 0.0 and self.cfg_distill_pair_student_dropout:
            paired_student_rng = _capture_torch_rng_state(x_t.device)
        _, denoised, weights = self.denoise(model, x_t, t, **model_kwargs)

        if mask is not None:
            terms["xs_mse"] = mean_flat(mask * (denoised - x_start) ** 2)
            terms["mse"] = mean_flat(weights * mask * (denoised - x_start) ** 2)
        else:
            terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
            terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        terms["loss"] = terms["mse"]

        auxiliary_total = torch.zeros_like(terms["mse"])
        if moment_weight > 0.0 or infonce_weight > 0.0:
            if self.auxiliary_loss_manager is None:
                raise RuntimeError("positive auxiliary weight requested without an auxiliary loss manager")
            if not auxiliary_context or "condition_id" not in auxiliary_context:
                raise RuntimeError("conditional auxiliary losses require per-sample condition_id")
            # Use the scalar c_out before append_dims.  The weight belongs
            # inside each condition's moment aggregation, not outside its loss.
            c_out = self.precond._get_scalings_and_weightings(t)[2]
            gate_weights = torch.sigmoid(
                (self.auxiliary_gate_thresh - c_out.to(denoised.dtype).abs())
                / self.auxiliary_gate_slope
            )
            auxiliary = self.auxiliary_loss_manager(
                denoised,
                auxiliary_context,
                gate_weights,
                {"moment": moment_weight, "infonce": infonce_weight},
            )
            for key, value in auxiliary.items():
                if key != "total":
                    terms[key] = value.expand_as(terms["mse"])
            auxiliary_total = auxiliary_total + auxiliary["total"]
            terms["aux_gate_weight_mean"] = gate_weights.mean().expand_as(terms["mse"])

        if cfg_distill_weight > 0.0:
            if self.cfg_distill_teacher_scale is None:
                raise RuntimeError(
                    "positive CFG distillation weight requested without a configured teacher"
                )
            if "condition" not in model_kwargs:
                raise RuntimeError("CFG distillation requires a conditional model branch")
            uncond_kwargs = dict(model_kwargs)
            del uncond_kwargs["condition"]
            if paired_student_rng is None:
                _, uncond_denoised, _ = self.denoise(model, x_t, t, **uncond_kwargs)
            else:
                continuation_rng = _capture_torch_rng_state(x_t.device)
                _restore_torch_rng_state(x_t.device, paired_student_rng)
                try:
                    _, uncond_denoised, _ = self.denoise(model, x_t, t, **uncond_kwargs)
                finally:
                    # The paired auxiliary forward must not consume a second
                    # independent dropout stream or rewind moment-loss dither.
                    _restore_torch_rng_state(x_t.device, continuation_rng)

            teacher_conditional = None
            teacher_unconditional = None
            if self.cfg_distill_teacher_model is not None:
                if self.cfg_distill_teacher_model.training:
                    raise RuntimeError("fixed CFG teacher unexpectedly entered training mode")
                with torch.no_grad():
                    _, teacher_conditional, _ = self.denoise(
                        self.cfg_distill_teacher_model,
                        x_t,
                        t,
                        **model_kwargs,
                    )
                    _, teacher_unconditional, _ = self.denoise(
                        self.cfg_distill_teacher_model,
                        x_t,
                        t,
                        **uncond_kwargs,
                    )
            cfg_auxiliary = cfg_distillation_loss(
                denoised,
                uncond_denoised,
                weights.reshape(-1),
                student_cfg_scale=self.cfg_distill_student_scale,
                teacher_cfg_scale=self.cfg_distill_teacher_scale,
                clip_denoised=self.cfg_distill_clip_denoised,
                teacher_conditional=teacher_conditional,
                teacher_unconditional=teacher_unconditional,
            )
            auxiliary_total = auxiliary_total + cfg_distill_weight * cfg_auxiliary["total"]
            for key, value in cfg_auxiliary.items():
                if key != "total":
                    terms[f"aux_cfg_distill_{key}"] = value

        if moment_weight > 0.0 or infonce_weight > 0.0 or cfg_distill_weight > 0.0:
            terms["aux_total"] = auxiliary_total
            terms["loss"] = terms["loss"] + auxiliary_total

        return terms


def karras_sample(
    diffusion,
    model,
    x_T,
    x_0,
    steps,
    mask=None,
    clip_denoised=True,
    model_kwargs=None,
    device=None,
    rho=7.0,
    sampler="heun",
    churn_step_ratio=0.0,
    eta=0.0,
    order=2,
    seed=None,
    cfg_scale=0.0,
    progress=False,
    ecsi_reconstruction_steps=2,
):
    assert sampler in [
        "heun",
        "ground_truth",
        "dbim",
        "dbim_high_order",
        *ECSI_SAMPLERS,
    ], "only these sampler is supported currently"

    if sampler in ECSI_SAMPLERS:
        if ECSI_SAMPLER_OPTIONS[sampler].get("time_grid") == "bridge_lambda":
            ts = get_sigmas_bridge_lambda(
                steps,
                diffusion.ecsi_sample_t_min,
                diffusion.ecsi_sample_t_max,
                diffusion.noise_schedule.gamma_max,
                device=device,
            )
        else:
            ts = get_sigmas_karras(
                steps,
                diffusion.ecsi_sample_t_min,
                diffusion.ecsi_sample_t_max,
                rho,
                device=device,
            )
    elif sampler == "heun":
        ts = get_sigmas_karras(steps, diffusion.t_min, diffusion.t_max - 1e-4, rho, device=device)
    else:
        ts = get_sigmas_uniform(steps, diffusion.t_min, diffusion.t_max - 1e-3, device=device)

    sample_fn = {
        "heun": sample_heun,
        "ground_truth": sample_ground_truth,
        "dbim": sample_dbim,
        "dbim_high_order": sample_dbim_high_order,
        **{
            name: (
                sample_ecsi_exponential
                if options.get("sde_integrator") == "exponential"
                else sample_ecsi
            )
            for name, options in ECSI_SAMPLER_OPTIONS.items()
        },
    }[sampler]

    sampler_args = dict(
        churn_step_ratio=churn_step_ratio,
        mask=mask,
        eta=eta,
        x_0=x_0,
        order=order,
        seed=seed,
        progress=progress,
    )
    if sampler in ECSI_SAMPLERS:
        sampler_args["ecsi_reconstruction_steps"] = ecsi_reconstruction_steps
        sampler_args.update(ECSI_SAMPLER_OPTIONS[sampler])

    def denoiser(x_t, sigma):
        _, denoised, _ = diffusion.denoise_with_cfg(
            model,
            x_t,
            sigma,
            cfg_scale=cfg_scale,
            **model_kwargs,
        )
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0, path, nfe, pred_x0, sigmas, noise = sample_fn(
        denoiser,
        diffusion,
        x_T,
        ts,
        **sampler_args,
    )
    if progress and dist.get_rank() == 0:
        print("nfe:", nfe)

    return (
        x_0.clamp(-1, 1),
        [x.clamp(-1, 1) for x in path],
        nfe,
        [x.clamp(-1, 1) for x in pred_x0],
        sigmas,
        noise,
    )


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    return torch.linspace(t_max, t_min, n + 1).to(device)


def get_sigmas_bridge_lambda(n, t_min, t_max, gamma_max, device="cpu"):
    """Uniform finite nodes in ECSI's lambda=log(b/c) coordinate."""
    if n < 2:
        raise ValueError(f"Bridge-lambda ECSI requires at least two steps, got {n}")
    if not 0 < t_min < t_max < 1:
        raise ValueError(f"Expected 0 < t_min < t_max < 1, got {t_min}, {t_max}")
    if gamma_max <= 0:
        raise ValueError(f"gamma_max must be positive, got {gamma_max}")

    endpoints = torch.tensor([t_max, t_min], dtype=torch.float64)
    lambdas = 0.5 * torch.log((1 - endpoints) / endpoints) - math.log(2 * gamma_max)
    lambda_grid = torch.linspace(lambdas[0], lambdas[1], n, dtype=torch.float64)
    ratio = (2 * gamma_max * torch.exp(lambda_grid)).square()
    finite_times = 1 / (1 + ratio)
    return append_zero(finite_times).to(device=device, dtype=torch.float32)


def _ecsi_step_eta(eta, step_index, total_steps, schedule, schedule_power=1.0):
    """Return the stochasticity assigned to one ECSI transition."""
    if schedule == "constant":
        return eta
    if schedule != "cosine":
        raise ValueError(f"Unsupported ECSI eta schedule: {schedule}")
    if total_steps < 2:
        raise ValueError("Cosine ECSI stochasticity requires at least two transitions")
    if not math.isfinite(schedule_power) or schedule_power <= 0:
        raise ValueError(
            f"Cosine ECSI schedule power must be finite and positive, got {schedule_power}"
        )
    if not 0 <= step_index < total_steps:
        raise ValueError(
            f"ECSI step index {step_index} is outside [0, {total_steps})"
        )
    if step_index == total_steps - 1:
        return 0.0
    progress = (step_index / (total_steps - 1)) ** schedule_power
    return eta * (1 + math.cos(math.pi * progress)) / 2


def _ecsi_second_order_correct_x0(
    x0_hat,
    prev_x0_hat,
    lambda_u,
    lambda_s,
    lambda_t,
    step_guard=ECSI_SECOND_ORDER_STEP_GUARD,
    correction_scale=1.0,
    blend_weight=None,
):
    """Free DPM-Solver++(2M)-style extrapolation in bridge lambda space.

    The denoiser value at ``s`` is extrapolated toward ``t`` using the previous
    value at ``u``.  Singular/degenerate steps, including the exact ``t=0``
    endpoint where ``log(b/c)`` diverges, deliberately fall back to first
    order.  No denoiser call is added.
    """
    if prev_x0_hat is None:
        return x0_hat
    h = lambda_t - lambda_s
    h2 = lambda_s - lambda_u
    safe = (
        (h2.abs() > step_guard)
        & torch.isfinite(h)
        & torch.isfinite(h2)
        & torch.isfinite(lambda_s)
    )
    if blend_weight is None:
        scaled_coefficient = correction_scale * h / (2 * h2)
    else:
        scaled_coefficient = torch.full_like(h, -blend_weight)
    coefficient = torch.where(safe, scaled_coefficient, torch.zeros_like(h))
    return (x0_hat + coefficient * (x0_hat - prev_x0_hat)).clamp(-1, 1)


def _ecsi_second_order_scale_at_progress(base_scale, scale_schedule, progress):
    """Select a piecewise-constant correction scale over equal path segments."""
    if scale_schedule is None:
        return base_scale
    segment = min(int(progress * len(scale_schedule)), len(scale_schedule) - 1)
    return scale_schedule[segment]


def _ecsi_exponential_update(
    x,
    x_T,
    x0_hat,
    a_s,
    b_s,
    c_s,
    a_t,
    b_t,
    c_t,
    step_eta,
    noise=None,
):
    """Exact frozen-denoiser ECSI-SDE transition in bridge lambda space.

    For ``v=(x-a*x_T)/b`` and ``lambda=log(b/c)``, ECSI is the same
    exponential SDE solved by DPM-Solver++ SDE samplers.  Freezing the x0
    prediction over one finite transition gives

      v_t = exp(-(1+eta)h) v_s + (1-exp(-(1+eta)h)) x0_hat
            + exp(-lambda_t) sqrt(1-exp(-2 eta h)) z,

    where ``h=lambda_t-lambda_s``.  The caller handles the singular exact
    endpoint ``c_t=0`` by returning the current denoiser prediction.
    """
    if step_eta < 0:
        raise ValueError(f"ECSI step eta must be non-negative, got {step_eta}")
    if torch.any(c_s <= 0) or torch.any(c_t <= 0):
        raise ValueError("The exponential ECSI transition requires finite positive c_s/c_t")

    lambda_s = torch.log(b_s.double() / c_s.double())
    lambda_t = torch.log(b_t.double() / c_t.double())
    h = lambda_t - lambda_s
    if torch.any(~torch.isfinite(h)) or torch.any(h <= 0):
        raise FloatingPointError("ECSI bridge lambda must increase on every finite transition")

    eta_h = step_eta * h
    decay = torch.exp(-h - eta_h).to(dtype=x.dtype)
    denoised_weight = (-torch.expm1(-h - eta_h)).to(dtype=x.dtype)
    v_s = (x - a_s * x_T) / b_s
    v_t = decay * v_s + denoised_weight * x0_hat
    if step_eta:
        if noise is None:
            raise ValueError("A stochastic exponential ECSI transition requires noise")
        noise_weight = torch.sqrt(-torch.expm1(-2 * eta_h)).to(dtype=x.dtype)
        v_t = v_t + (c_t / b_t) * noise_weight * noise
    return a_t * x_T + b_t * v_t


def _ecsi_dpmpp_2m_denoised(
    x0_hat,
    prev_x0_hat,
    lambda_s,
    prev_lambda_s,
    lambda_t,
    step_guard=ECSI_SECOND_ORDER_STEP_GUARD,
):
    """DPM-Solver++(2M) midpoint denoiser combination for one finite step."""
    if prev_x0_hat is None:
        return x0_hat
    h = lambda_t - lambda_s
    h_last = lambda_s - prev_lambda_s
    safe = (
        (h_last.abs() > step_guard)
        & torch.isfinite(h)
        & torch.isfinite(h_last)
        & (h > 0)
        & (h_last > 0)
    )
    coefficient = torch.where(safe, h / (2 * h_last), torch.zeros_like(h))
    return x0_hat + coefficient.to(dtype=x0_hat.dtype) * (x0_hat - prev_x0_hat)


@torch.no_grad()
def sample_ecsi_exponential(
    denoiser,
    diffusion,
    x,
    ts,
    eta=1.0,
    mask=None,
    seed=None,
    progress=False,
    ecsi_reconstruction_steps=2,
    eta_schedule="constant",
    solver_order=1,
    solver_type=None,
    solver_lower_order_tail=False,
    **kwargs,
):
    """Exact ECSI-SDE exponential solver, optionally DPM-Solver++(2M).

    The final configured transitions retain ECSI's deterministic
    reconstruction convention by setting their eta to zero.  The singular
    terminal transition to ``t=0`` is always lower order, matching the
    standard DPM-Solver++ endpoint rule and avoiding an infinite lambda step.
    """
    if not isinstance(diffusion.noise_schedule, LinearBridgeNoiseSchedule):
        raise TypeError("The ECSI sampler requires LinearBridgeNoiseSchedule")
    if eta < 0:
        raise ValueError(f"eta must be non-negative, got {eta}")
    if eta_schedule != "constant":
        raise ValueError("The exact ECSI-SDE diagnostic currently requires constant eta")
    if solver_order not in (1, 2):
        raise ValueError(f"The exponential ECSI solver supports order 1 or 2, got {solver_order}")
    if solver_order == 1 and solver_type is not None:
        raise ValueError("A first-order exponential ECSI solver has no solver_type")
    if solver_order == 2 and solver_type != "midpoint":
        raise ValueError("The second-order exponential ECSI solver requires midpoint type")
    if solver_lower_order_tail and solver_order != 2:
        raise ValueError("A lower-order tail applies only to the order-2 solver")
    if len(ts) < 3:
        raise ValueError("ECSI requires at least two function evaluations")
    if (
        isinstance(ecsi_reconstruction_steps, bool)
        or not isinstance(ecsi_reconstruction_steps, int)
        or not 1 <= ecsi_reconstruction_steps <= len(ts) - 1
    ):
        raise ValueError(
            "ecsi_reconstruction_steps must be an integer between 1 and "
            f"the realized NFE ({len(ts) - 1}), got {ecsi_reconstruction_steps!r}"
        )

    x_T = x
    path = []
    pred_x0 = []
    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(not progress or dist.get_rank() != 0))
    generator = BatchedSeedGenerator(seed)
    first_noise = None
    prev_x0_hat = None
    prev_lambda_s = None

    for i in indices:
        s = ts[i]
        t = ts[i + 1]
        raw_x0_hat = denoiser(x, s * ones)
        if mask is not None:
            raw_x0_hat = raw_x0_hat * mask + x_T * (1 - mask)

        a_s, b_s, c_s = [
            append_dims(item, x.ndim).to(dtype=x.dtype)
            for item in diffusion.noise_schedule.get_abc(s * ones)
        ]
        a_t, b_t, c_t = [
            append_dims(item, x.ndim).to(dtype=x.dtype)
            for item in diffusion.noise_schedule.get_abc(t * ones)
        ]
        lambda_s = torch.log(b_s.double() / c_s.double())
        in_reconstruction_tail = i >= len(ts) - 1 - ecsi_reconstruction_steps
        terminal_step = bool(torch.all(c_t == 0))

        x0_hat = raw_x0_hat
        if terminal_step:
            # lambda_t is infinite at t=0.  The exact limiting update is the
            # current denoiser prediction, and 2M must fall back to order one.
            x = raw_x0_hat
        else:
            lambda_t = torch.log(b_t.double() / c_t.double())
            use_second_order = (
                solver_order == 2
                and prev_x0_hat is not None
                and not (solver_lower_order_tail and in_reconstruction_tail)
            )
            if use_second_order:
                x0_hat = _ecsi_dpmpp_2m_denoised(
                    raw_x0_hat,
                    prev_x0_hat,
                    lambda_s,
                    prev_lambda_s,
                    lambda_t,
                )

            step_eta = 0.0 if in_reconstruction_tail else eta
            noise = generator.randn_like(x) if step_eta else None
            if first_noise is None and noise is not None:
                first_noise = noise
            x = _ecsi_exponential_update(
                x,
                x_T,
                x0_hat,
                a_s,
                b_s,
                c_s,
                a_t,
                b_t,
                c_t,
                step_eta,
                noise,
            )

        prev_x0_hat = raw_x0_hat.detach()
        prev_lambda_s = lambda_s.detach()
        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())

    return x, path, len(ts) - 1, pred_x0, ts, first_noise


@torch.no_grad()
def sample_ecsi(
    denoiser,
    diffusion,
    x,
    ts,
    eta=1.0,
    mask=None,
    seed=None,
    progress=False,
    ecsi_reconstruction_steps=2,
    eta_schedule="constant",
    eta_schedule_power=1.0,
    second_order_x0=False,
    second_order_lower_order_tail=False,
    second_order_scale=1.0,
    second_order_scale_schedule=None,
    second_order_blend_weight=None,
    second_order_tail_scale=None,
    second_order_progress_window=(0.0, 1.0),
    **kwargs,
):
    """Euler-Maruyama ECSI sampler with a deterministic reconstruction tail.

    This is Eq. (11) followed by Eq. (12) in *Exploring the Design Space of
    Diffusion Bridge Models*.  The base density is the unmodified endpoint
    distribution (``b=0`` in the paper), which produced its lowest FID. The
    paper and official implementation use two reconstruction transitions;
    ``ecsi_reconstruction_steps`` exposes that boundary for controlled
    ablations while preserving two as the default.
    """
    if not isinstance(diffusion.noise_schedule, LinearBridgeNoiseSchedule):
        raise TypeError("The ECSI sampler requires LinearBridgeNoiseSchedule")
    if eta < 0:
        raise ValueError(f"eta must be non-negative, got {eta}")
    if eta_schedule not in {"constant", "cosine"}:
        raise ValueError(f"Unsupported ECSI eta schedule: {eta_schedule}")
    if not math.isfinite(eta_schedule_power) or eta_schedule_power <= 0:
        raise ValueError(
            f"eta_schedule_power must be finite and positive, got {eta_schedule_power}"
        )
    if not isinstance(second_order_x0, bool):
        raise TypeError("second_order_x0 must be a bool")
    if not isinstance(second_order_lower_order_tail, bool):
        raise TypeError("second_order_lower_order_tail must be a bool")
    if second_order_lower_order_tail and not second_order_x0:
        raise ValueError("A lower-order ECSI tail requires second-order x0 correction")
    if not math.isfinite(second_order_scale) or second_order_scale == 0:
        raise ValueError(
            f"second_order_scale must be finite and nonzero, got {second_order_scale}"
        )
    if second_order_scale_schedule is not None and (
        not isinstance(second_order_scale_schedule, (tuple, list))
        or not second_order_scale_schedule
        or not all(
            math.isfinite(scale) and scale != 0
            for scale in second_order_scale_schedule
        )
    ):
        raise ValueError(
            "second_order_scale_schedule must be a non-empty sequence of finite, "
            f"nonzero scales, got {second_order_scale_schedule!r}"
        )
    if second_order_blend_weight is not None and (
        not math.isfinite(second_order_blend_weight)
        or not 0 < second_order_blend_weight < 1
    ):
        raise ValueError(
            "second_order_blend_weight must be finite and in (0, 1), "
            f"got {second_order_blend_weight!r}"
        )
    if second_order_tail_scale is not None and (
        not math.isfinite(second_order_tail_scale) or second_order_tail_scale == 0
    ):
        raise ValueError(
            "second_order_tail_scale must be finite and nonzero, "
            f"got {second_order_tail_scale!r}"
        )
    if second_order_tail_scale is not None and second_order_lower_order_tail:
        raise ValueError(
            "second_order_tail_scale cannot be combined with a lower-order tail"
        )
    if (
        not isinstance(second_order_progress_window, (tuple, list))
        or len(second_order_progress_window) != 2
        or not all(math.isfinite(bound) for bound in second_order_progress_window)
        or not 0 <= second_order_progress_window[0] <= second_order_progress_window[1] <= 1
    ):
        raise ValueError(
            "second_order_progress_window must be a finite [start, end] subset of [0, 1], "
            f"got {second_order_progress_window!r}"
        )
    if len(ts) < 3:
        raise ValueError("ECSI requires at least two function evaluations")
    if (
        isinstance(ecsi_reconstruction_steps, bool)
        or not isinstance(ecsi_reconstruction_steps, int)
        or not 1 <= ecsi_reconstruction_steps <= len(ts) - 1
    ):
        raise ValueError(
            "ecsi_reconstruction_steps must be an integer between 1 and "
            f"the realized NFE ({len(ts) - 1}), got {ecsi_reconstruction_steps!r}"
        )

    x_T = x
    path = []
    pred_x0 = []
    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(not progress or dist.get_rank() != 0))
    generator = BatchedSeedGenerator(seed)
    first_noise = None
    prev_x0_hat = None
    prev_lambda = None
    total_steps = len(ts) - 1

    for i in indices:
        s = ts[i]
        t = ts[i + 1]
        raw_x0_hat = denoiser(x, s * ones)
        if mask is not None:
            raw_x0_hat = raw_x0_hat * mask + x_T * (1 - mask)

        a_s, b_s, c_s = [
            append_dims(item, x.ndim).to(dtype=x.dtype)
            for item in diffusion.noise_schedule.get_abc(s * ones)
        ]
        a_t, b_t, c_t = [
            append_dims(item, x.ndim).to(dtype=x.dtype)
            for item in diffusion.noise_schedule.get_abc(t * ones)
        ]

        x0_hat = raw_x0_hat
        lambda_s = None
        in_reconstruction_tail = i >= len(ts) - 1 - ecsi_reconstruction_steps
        second_order_progress = i / (total_steps - 1)
        in_second_order_window = (
            second_order_progress_window[0]
            <= second_order_progress
            <= second_order_progress_window[1]
        )
        if second_order_x0:
            lambda_s = torch.log(b_s / c_s)
            lambda_t = torch.log(b_t / c_t)
            if prev_x0_hat is not None and not (
                second_order_lower_order_tail and in_reconstruction_tail
            ) and in_second_order_window:
                x0_hat = _ecsi_second_order_correct_x0(
                    raw_x0_hat,
                    prev_x0_hat,
                    prev_lambda,
                    lambda_s,
                    lambda_t,
                    correction_scale=_ecsi_second_order_scale_at_progress(
                        (
                            second_order_tail_scale
                            if in_reconstruction_tail
                            and second_order_tail_scale is not None
                            else second_order_scale
                        ),
                        (
                            None
                            if in_reconstruction_tail
                            and second_order_tail_scale is not None
                            else second_order_scale_schedule
                        ),
                        second_order_progress,
                    ),
                    blend_weight=second_order_blend_weight,
                )
        step_eta = _ecsi_step_eta(
            eta, i, total_steps, eta_schedule, eta_schedule_power
        )

        if in_reconstruction_tail:
            # ECSI disables epsilon and reconstructs the path for the configured
            # final transitions. Here b=alpha and a=beta in the paper's notation.
            z_hat = (x - b_s * x0_hat - a_s * x_T) / c_s
            x = b_t * x0_hat + a_t * x_T + c_t * z_hat
        else:
            da_s, db_s, dc_s = [
                append_dims(item, x.ndim).to(dtype=x.dtype)
                for item in diffusion.noise_schedule.get_abc_derivatives(s * ones)
            ]
            epsilon = step_eta * (c_s * dc_s - (db_s / b_s) * c_s.square())
            if torch.any(epsilon < -1e-12):
                raise FloatingPointError("ECSI epsilon became negative")
            epsilon = epsilon.clamp_min(0)
            z_hat = (x - b_s * x0_hat - a_s * x_T) / c_s
            drift = db_s * x0_hat + da_s * x_T + (dc_s + epsilon / c_s) * z_hat
            noise = generator.randn_like(x)
            if first_noise is None:
                first_noise = noise
            x = x + drift * (t - s) + noise * torch.sqrt((s - t) * 2 * epsilon)

        if second_order_x0:
            prev_x0_hat = raw_x0_hat.detach()
            prev_lambda = lambda_s.detach()
        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())

    return x, path, len(ts) - 1, pred_x0, ts, first_noise


@torch.no_grad()
def sample_dbim_high_order(
    denoiser,
    diffusion,
    x,
    ts,
    mask=None,
    order=2,
    lower_order_final=True,
    seed=None,
    progress=False,
    **kwargs,
):
    if order not in [2, 3]:
        raise NotImplementedError("Not supported")
    x_T = x
    path = []
    pred_x0 = []

    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(not progress or dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    u = diffusion.t_max
    if u == 1.0:
        u -= 5e-5
    u = [u for _ in range(order - 1)]
    xu_hat = [x0_hat.detach().clone() for _ in range(order - 1)]

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        # First Order Update, t < s
        if (lower_order_final and i + 1 == len(ts) - 1) or (i == 0):
            if progress and dist.get_rank() == 0:
                print("Step order 1")
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]

            tmp_var = c_t / c_s
            coeff_xs = tmp_var
            coeff_x0_hat = b_t - tmp_var * b_s
            coeff_xT = a_t - tmp_var * a_s

            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            x_old = x
            x = coeff_xs * x_old + coeff_x0_hat * x0_hat + coeff_xT * x_T

        # Second Order Update, t < s < u
        elif order == 2 or i == 1:
            if progress and dist.get_rank() == 0:
                print("Step order 2")
            a_u, b_u, c_u = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u, lambda_s, lambda_t = (
                torch.log(b_u / c_u),
                torch.log(b_s / c_s),
                torch.log(b_t / c_t),
            )

            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            h = lambda_t - lambda_s
            h2 = lambda_s - lambda_u
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat + (torch.exp(-h) + h - 1) * (x0_hat - xu_hat[-1]) / h2
            )
            x_old = x
            x = x_old * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        elif order == 3:
            if progress and dist.get_rank() == 0:
                print("Step order 3")
            a_u1, b_u1, c_u1 = [
                append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-1] * ones)
            ]
            a_u2, b_u2, c_u2 = [
                append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-2] * ones)
            ]
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u2, lambda_u1, lambda_s, lambda_t = (
                torch.log(b_u2 / c_u2),
                torch.log(b_u1 / c_u1),
                torch.log(b_s / c_s),
                torch.log(b_t / c_t),
            )
            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1

            h = lambda_t - lambda_s
            h1 = lambda_s - lambda_u1
            h2 = lambda_u1 - lambda_u2
            dx0_hat = ((x0_hat - xu_hat[-1]) * (2 * h1 + h2) / h1 - (xu_hat[-1] - xu_hat[-2]) * h1 / h2) / (h1 + h2)
            d2x0_hat = 2 * ((x0_hat - xu_hat[-1]) / h1 - (xu_hat[-1] - xu_hat[-2]) / h2) / (h1 + h2)
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat
                + (torch.exp(-h) + h - 1) * dx0_hat
                + (h**2 / 2 - h + 1 - torch.exp(-h)) * d2x0_hat
            )
            x_old = x
            x = x_old * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        u.append(s)
        u.pop(0)
        xu_hat.append(x0_hat)
        xu_hat.pop(0)

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())

    return x, path, nfe, pred_x0, ts, first_noise


@torch.no_grad()
def sample_dbim(
    denoiser,
    diffusion,
    x,
    ts,
    eta=1.0,
    mask=None,
    seed=None,
    progress=False,
    **kwargs,
):
    x_T = x
    path = []
    pred_x0 = []

    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(not progress or dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        x0_hat = denoiser(x, s * ones)
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)

        a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
        a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]

        _, _, rho_s, _ = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_alpha_rho(s * ones)]
        alpha_t, _, rho_t, _ = [
            append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_alpha_rho(t * ones)
        ]

        omega_st = eta * (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()
        tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s
        coeff_xs = tmp_var
        coeff_x0_hat = b_t - tmp_var * b_s
        coeff_xT = a_t - tmp_var * a_s

        noise = generator.randn_like(x0_hat)

        x = coeff_x0_hat * x0_hat + coeff_xT * x_T + coeff_xs * x + (1 if i != len(ts) - 2 else 0) * omega_st * noise

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1

    return x, path, nfe, pred_x0, ts, first_noise


@torch.no_grad()
def sample_ground_truth(
    denoiser,
    diffusion,
    x,
    ts,
    x0=None,
    **kwargs,
):
    assert x0 is not None
    x_T = x
    path = []
    pred_x0 = []

    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    noise = torch.randn_like(x0)
    first_noise = noise
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        x0_hat = denoiser(x, s * ones)
        noise = torch.randn_like(x0)
        x = diffusion.bridge_sample(x0, x_T, t * ones, noise)

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1

    return x, path, nfe, pred_x0, ts, first_noise


def get_d(denoiser, noise_schedule, x, x_T, t, stochastic):
    ones = x.new_ones([x.shape[0]])
    f_t, g2_t = [append_dims(item, x.ndim) for item in noise_schedule.get_f_g2(t * ones)]
    alpha_t, alpha_bar_t, _, rho_bar_t = [append_dims(item, x.ndim) for item in noise_schedule.get_alpha_rho(t * ones)]
    a_t, b_t, c_t = [append_dims(item, x.ndim) for item in noise_schedule.get_abc(t * ones)]
    denoised = denoiser(x, t * ones)
    grad_logq = -(x - (a_t * x_T + b_t * denoised)) / c_t**2
    grad_logpxTlxt = -(x - alpha_bar_t * x_T) / (alpha_t**2 * rho_bar_t**2)
    d = f_t * x - g2_t * ((0.5 if not stochastic else 1) * grad_logq - grad_logpxTlxt)
    return d, g2_t, denoised


def ddbm_simulate(denoiser, noise_schedule, x, x_T, t_cur, t_next, stochastic, second_order=False):
    dt = t_next - t_cur
    if isinstance(noise_schedule, I2SBNoiseSchedule):
        dt = dt * (noise_schedule.n_timestep - 1)
    d, g2_t, pred_x0 = get_d(denoiser, noise_schedule, x, x_T, t_cur, stochastic)
    x_new = x + d * dt + (0 if not stochastic else 1) * torch.randn_like(x) * ((dt).abs() ** 0.5) * g2_t.sqrt()
    if second_order:
        d_2, _, pred_x0 = get_d(denoiser, noise_schedule, x_new, x_T, t_next, stochastic)
        d_prime = (d + d_2) / 2
        x_new = (
            x + d_prime * dt + (0 if not stochastic else 1) * torch.randn_like(x) * ((dt).abs() ** 0.5) * g2_t.sqrt()
        )
    return x_new, pred_x0


@torch.no_grad()
def sample_heun(
    denoiser,
    diffusion,
    x,
    ts,
    churn_step_ratio=0.0,
    **kwargs,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    x_T = x
    path = []
    pred_x0 = []

    indices = range(len(ts) - 1)

    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    assert churn_step_ratio < 1

    for _, i in enumerate(indices):

        if churn_step_ratio > 0:
            # 1 step euler
            t_hat = (ts[i + 1] - ts[i]) * churn_step_ratio + ts[i]
            x, _pred_x0 = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                ts[i],
                t_hat,
                stochastic=True,
            )
            nfe += 1
            path.append(x.detach().cpu())
            pred_x0.append(_pred_x0.detach().cpu())
        else:
            t_hat = ts[i]

        # heun step
        if ts[i + 1] == 0:
            x, _pred_x0 = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                t_hat,
                ts[i + 1],
                stochastic=False,
            )
            nfe += 1
        else:
            # Heun's method
            x, _pred_x0 = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                t_hat,
                ts[i + 1],
                stochastic=False,
                second_order=True,
            )
            nfe += 2

        path.append(x.detach().cpu())
        pred_x0.append(_pred_x0.detach().cpu())

    return x, path, nfe, pred_x0, ts, None
