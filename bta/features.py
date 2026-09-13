"""CPU-backed, per-layer safetensors reads; scientific calculations stay on CUDA."""
from safetensors import safe_open
import torch


def feature_vector(row, position, device, *, condition=None):
    path = row["diagnostic_path"] if condition is not None else row["feature_path"]
    index = row.get("feature_condition") if condition is None else row["order"].index(condition)
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        source = reader.get_slice("hidden")
        value = source[position, :] if index is None else source[index, position, :]
    return value.to(device=device)


def feature_matrix(rows, position, device):
    if not rows:
        raise ValueError("Cannot construct a feature matrix without observations")
    cpu = torch.stack([feature_vector(row, position, "cpu") for row in rows])
    return cpu.to(device=device)
