"""Transform pipeline for waveform compression and preprocessing."""

# Import Transform framework from dingo-svd
from dingo_svd import Transform, ComposeTransforms, ApplySVD

# Keep domain-specific transforms in dingo-waveform
from .whitening import WhitenAndUnwhiten

__all__ = [
    "Transform",
    "ComposeTransforms",
    "ApplySVD",
    "WhitenAndUnwhiten",
]
