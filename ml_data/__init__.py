"""
Leakage-free ML data pipeline for the chip heat-dissipation dataset.

Heavy imports (torch Dataset) are lazy so aux-cache / stats scripts can run
without PyTorch installed.
"""

from .channels import CHANNEL_NAMES, TARGET_NAME, N_INPUT_CHANNELS

__all__ = [
    "CHANNEL_NAMES",
    "TARGET_NAME",
    "N_INPUT_CHANNELS",
    "ChipThermalDataset",
    "make_dataloader",
    "validate_no_target_leakage",
    "inverse_kirchhoff_from_sample",
    "build_n_map",
]


def __getattr__(name):
    if name in ("ChipThermalDataset", "make_dataloader"):
        from .dataset import ChipThermalDataset, make_dataloader
        return {"ChipThermalDataset": ChipThermalDataset, "make_dataloader": make_dataloader}[name]
    if name == "validate_no_target_leakage":
        from .leakage import validate_no_target_leakage
        return validate_no_target_leakage
    if name in ("inverse_kirchhoff_from_sample", "build_n_map"):
        from .kirchhoff_utils import inverse_kirchhoff_from_sample, build_n_map
        return {
            "inverse_kirchhoff_from_sample": inverse_kirchhoff_from_sample,
            "build_n_map": build_n_map,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
