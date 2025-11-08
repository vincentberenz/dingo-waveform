# Domains

Domains define how gravitational waveforms are sampled in either frequency or time.

## Why Domains Matter

The choice of domain affects:

- **Memory usage** - How much data is stored
- **Computational efficiency** - How fast waveforms are generated
- **Neural network training** - Dimensionality of input data

## Domain Types

### UniformFrequencyDomain

Standard uniform frequency sampling with constant spacing Δf.

**When to use:**
- Standard parameter estimation
- Matched filtering
- General-purpose waveform generation

**Parameters:**
- `f_min` - Minimum frequency (Hz)
- `f_max` - Maximum frequency (Hz)
- `delta_f` - Frequency spacing (Hz)

### MultibandedFrequencyDomain

Adaptive frequency binning that reduces dimensionality while preserving waveform information.

**When to use:**
- Neural network training
- Large-scale inference
- Memory-constrained applications

**Parameters:**
- `f_min`, `f_max` - Frequency range
- `delta_f_initial` - Initial frequency spacing
- `n_bins_per_harmonic` - Bins per harmonic band

### TimeDomain

Time-domain sampling for waveforms.

**When to use:**
- Time-domain approximants
- Whitening operations
- Time-frequency analysis

**Parameters:**
- `duration` - Total duration (seconds)
- `delta_t` - Time step (seconds)

## See Also

- [API Reference: Domains](../api/domains.md)
- [Quick Start](../getting-started/quickstart.md)
