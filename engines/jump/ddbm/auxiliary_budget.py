"""Optional scalar-loss budget split; this is not a gradient-norm constraint."""

def protected_moment_budget(zoe, moment, main, *, ratio, fraction):
    """Blend individually capped losses, preserving (1-fraction) of Zoe's cap.

    All cap references are detached. Even a zero-valued moment remains in the
    backward graph, as required by distributed moment all-gather collectives.
    The fraction must be ramped by the caller, and zero reproduces Zoe alone.
    """
    if not 0 < ratio <= 1 or not 0 <= fraction <= 1:
        raise ValueError("invalid protected auxiliary budget ratio/fraction")
    budget = main.detach().mean() * ratio
    zoe_scale = (budget / zoe.detach().abs().mean().clamp_min(1e-12)).clamp(max=1)
    moment_scale = (budget / moment.detach().abs().mean().clamp_min(1e-12)).clamp(max=1)
    zoe_coefficient = (1 - fraction) * zoe_scale.detach()
    moment_coefficient = fraction * moment_scale.detach()
    return zoe * zoe_coefficient + moment * moment_coefficient, {
        "aux_budget": budget,
        "aux_zoe_budget_coefficient": zoe_coefficient,
        "aux_moment_budget_coefficient": moment_coefficient,
        "aux_moment_budget_fraction": budget.new_tensor(fraction),
        "aux_zoe_budgeted": zoe * zoe_coefficient,
        "aux_moment_budgeted": moment * moment_coefficient,
    }
