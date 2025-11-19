"""
Comparison utilities for validating dingo-waveform against dingo (dingo-gw).

This sub-package provides:
- Core comparison functions for waveforms and SVD compression
- CLI tools for verification (dingo-verify, dingo-verify-batch)
- Performance benchmarking tool (dingo-benchmark)
- Test utilities for comparing generated waveforms
"""

from .core import (
    WaveformComparisonResult,
    SVDComparisonResult,
    compare_waveforms,
    compare_waveforms_modes,
    compare_svd_compression,
    generate_waveform_dingo,
    generate_waveform_refactored,
    create_uniform_domain_dingo,
    create_uniform_domain_refactored,
    create_multibanded_domain_dingo,
    create_multibanded_domain_refactored,
)

__all__ = [
    'WaveformComparisonResult',
    'SVDComparisonResult',
    'compare_waveforms',
    'compare_waveforms_modes',
    'compare_svd_compression',
    'generate_waveform_dingo',
    'generate_waveform_refactored',
    'create_uniform_domain_dingo',
    'create_uniform_domain_refactored',
    'create_multibanded_domain_dingo',
    'create_multibanded_domain_refactored',
]
