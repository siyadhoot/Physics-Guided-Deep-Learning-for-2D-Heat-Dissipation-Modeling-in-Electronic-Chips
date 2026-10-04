"""models package — neural architectures for chip thermal modeling."""

from .fno import FNO2d, SpectralConv2d, build_fno_from_config

__all__ = ["FNO2d", "SpectralConv2d", "build_fno_from_config"]
