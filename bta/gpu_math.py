from __future__ import annotations

import math

import torch


def finite_scalar(value: torch.Tensor) -> float | None:
    return float(value.item()) if bool(torch.isfinite(value)) else None


def correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.to(torch.float64), y.to(torch.float64)
    x, y = x - x.mean(), y - y.mean()
    denominator = x.norm() * y.norm()
    if len(x) < 2 or not bool(denominator > 0):
        return x.new_tensor(float("nan"))
    return (x @ y) / denominator


def group_weights(groups: list[str], device) -> torch.Tensor:
    counts = {group: groups.count(group) for group in set(groups)}
    weights = torch.tensor([1 / counts[group] for group in groups], device=device, dtype=torch.float64)
    return weights / weights.sum()


def ridge_candidates(x: torch.Tensor, y: torch.Tensor, groups: list[str], x_cal: torch.Tensor,
                     y_cal: torch.Tensor, cal_groups: list[str]) -> dict | None:
    """Dual ridge; penalties come from the empirical spectrum, selection from calibration loss."""
    if len(x) < 2 or len(x_cal) == 0 or bool((y == y[0]).all()):
        return None
    x, y, x_cal, y_cal = [v.to(torch.float64) for v in (x, y, x_cal, y_cal)]
    w = group_weights(groups, x.device)
    wc = group_weights(cal_groups, x.device)
    mean_x, mean_y = (x * w[:, None]).sum(0), (y * w).sum()
    centered = x - mean_x
    scale = (centered.square() * w[:, None]).sum(0).mean().sqrt()
    if not bool(scale > 0):
        return None
    z = centered / scale
    design = z * w.sqrt()[:, None]
    response = (y - mean_y) * w.sqrt()
    gram = design @ design.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    tolerance = torch.finfo(gram.dtype).eps * len(gram) * eigenvalues.abs().max()
    penalties = torch.unique(eigenvalues[eigenvalues > tolerance])
    if penalties.numel() == 0:
        return None
    eigenvalues = eigenvalues.clamp_min(0)
    projections = eigenvectors.T @ response
    coefficients = eigenvectors @ (projections[:, None] / (eigenvalues[:, None] + penalties[None, :]))
    calibration_kernel = ((x_cal - mean_x) / scale) @ design.T
    predictions = calibration_kernel @ coefficients + mean_y
    losses = ((predictions - y_cal[:, None]).square() * wc[:, None]).sum(0)
    best = int(losses.argmin().item())
    weight = design.T @ coefficients[:, best] / scale
    bias = mean_y - mean_x @ weight
    return {"weight": weight, "bias": bias, "penalty": float(penalties[best].item()),
            "calibration_mse": float(losses[best].item()), "candidate_count": len(penalties)}


def calibrate_threshold(negative_episode_maxima: torch.Tensor, target_fpr: float) -> float | None:
    if negative_episode_maxima.numel() == 0:
        return None
    if not bool(torch.isfinite(negative_episode_maxima).all()):
        raise ValueError("Nonfinite calibration scores")
    maxima = negative_episode_maxima.sort(descending=True).values
    allowed = math.floor(target_fpr * len(maxima))
    # Strict score > threshold handles ties conservatively; allowed is always < n for target_fpr < 1.
    return float(maxima[allowed].item())


def bootstrap_mean(values: torch.Tensor, cfg: dict, seed: int) -> dict:
    values = values.to(torch.float64)
    if values.numel() == 0:
        return {"n_scenarios": 0, "estimate": None, "ci_low": None, "ci_high": None, "status": "no_matched_scenarios"}
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Statistics contain nonfinite input")
    out = {"n_scenarios": len(values), "estimate": float(values.mean().item()), "ci_low": None, "ci_high": None}
    if len(values) < 2:
        return {**out, "status": "insufficient_independent_scenarios_for_interval"}
    # Worst-case Bernoulli variance 1/4 bounds the MC standard error of an empirical quantile rank.
    count = math.ceil((1 / 4) / cfg["bootstrap_mc_standard_error"] ** 2)
    generator = torch.Generator(device=values.device).manual_seed(seed)
    indices = torch.randint(len(values), (count, len(values)), device=values.device, generator=generator)
    samples = values[indices].mean(-1)
    tail = (1 - cfg["confidence_level"]) / 2
    bounds = torch.quantile(samples, values.new_tensor([tail, 1 - tail]))
    return {**out, "ci_low": float(bounds[0].item()), "ci_high": float(bounds[1].item()),
            "bootstrap_resamples": count, "confidence_level": cfg["confidence_level"], "status": "estimated"}


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    positive, negative = scores[labels.bool()], scores[~labels.bool()]
    if not len(positive) or not len(negative):
        return None
    # Exact GPU pair comparisons with tie credit; the pilot's episode matrix is small.
    pairs = positive[:, None]
    total = (pairs > negative).to(torch.float64).sum() + (pairs == negative).to(torch.float64).sum() / 2
    return float((total / (len(positive) * len(negative))).item())


def random_equal_norm(direction: torch.Tensor, magnitude: torch.Tensor, seed: int) -> torch.Tensor:
    direction = direction.to(torch.float64)
    generator = torch.Generator(device=direction.device).manual_seed(seed)
    noise = torch.randn(direction.shape, generator=generator, device=direction.device, dtype=torch.float64)
    noise = noise - (noise @ direction) * direction
    norm = noise.norm()
    if not bool(norm > 0):
        raise ArithmeticError("Degenerate random control direction")
    return (noise / norm * magnitude.abs()).to(torch.float32)


def max_correlation_permutation(projections, target, mc_standard_error, seed):
    """Rows are independent SCENARIO summaries; the same shuffle is used for all layers."""
    x, y = projections.double(), target.double()
    if x.ndim != 2 or len(x) != len(y) or len(y) < 2:
        return None
    x, y = x - x.mean(0), y - y.mean()
    norms, ynorm = x.norm(dim=0), y.norm()
    valid = norms > 0
    if not bool(valid.any()) or not bool(ynorm > 0):
        return None
    normalized, y = x[:, valid] / norms[valid], y / ynorm
    observed = (normalized.T @ y).abs().max()
    count = math.ceil((1 / 4) / mc_standard_error ** 2)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    permutations = torch.stack([torch.randperm(len(y), device=x.device, generator=generator) for _ in range(count)])
    null = (y[permutations] @ normalized).abs().amax(-1)
    rank_tolerance = torch.finfo(x.dtype).eps * len(y)
    p = ((null >= observed - rank_tolerance).sum().double() + 1) / (count + 1)
    return {"max_absolute_correlation": float(observed.item()),
            "max_statistic_permutation_p": float(p.item()), "permutation_draws": count,
            "independent_scenarios": len(y), "tested_layers": int(valid.sum().item()),
            "unit": "one authorized public-trajectory summary per scenario",
            "layer_selection_repeated_in_every_permutation": True,
            "null_hypothesis": "scenario-level association is absent conditional on fit-derived directions",
            "no_automatic_significance_or_mechanism_claim": True}
