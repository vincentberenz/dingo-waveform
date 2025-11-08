# Core Concepts

## Overview

dingo-waveform is built around several key concepts that work together to generate gravitational wave waveforms efficiently.

## The Waveform Generation Pipeline

```
Domain → Approximant → WaveformParameters → WaveformGenerator → Polarizations
```

### 1. Domain

The **domain** defines how the waveform is sampled - either in frequency or time:

- **UniformFrequencyDomain**: Standard uniform frequency sampling
- **MultibandedFrequencyDomain**: Adaptive frequency binning for efficiency
- **TimeDomain**: Time-domain sampling

[Learn more about Domains →](domains.md)

### 2. Approximant

The **approximant** is the waveform model used to compute the gravitational wave signal:

- **IMRPhenomXPHM**: Precessing binaries with higher modes
- **SEOBNRv5PHM**: Effective-one-body model with higher modes
- **SEOBNRv4PHM**: Previous generation model

[Learn more about Approximants →](approximants.md)

### 3. Waveform Parameters

**WaveformParameters** describe the binary system:

- Masses (m₁, m₂)
- Spins (a₁, a₂, tilt angles)
- Distance and orientation
- Orbital phase

### 4. Waveform Generator

The **WaveformGenerator** combines domain, approximant, and parameters to produce waveforms using LALSimulation under the hood.

### 5. Polarizations

The output **Polarizations** contain the two gravitational wave polarization states:

- **h₊** (h_plus): Plus polarization
- **h×** (h_cross): Cross polarization

[Learn more about Polarizations →](polarizations.md)

## Mode-Separated Waveforms

For higher-mode approximants, waveforms can be decomposed into spherical harmonic modes:

\\[
h(t) = \\sum_{\\ell,m} h_{\\ell m}(t) Y_{\\ell m}(\\theta, \\phi)
\\]

In dingo-waveform, modes are represented by their **m** value (azimuthal quantum number).

[Learn more about Modes →](modes.md)

## Workflow Examples

### Basic Workflow

```python
# 1. Define domain
domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)

# 2. Create generator
wfg = WaveformGenerator(
    approximant=Approximant("IMRPhenomXPHM"),
    domain=domain,
    f_ref=20.0
)

# 3. Set parameters
params = WaveformParameters(mass_1=36.0, mass_2=29.0, ...)

# 4. Generate
polarization = wfg.generate_hplus_hcross(params)
```

### Mode-Separated Workflow

```python
# Same setup as above, then:
modes = wfg.generate_hplus_hcross_m(params)

# Access individual modes
for m, pol in modes.items():
    print(f"Mode m={m}")
```

## Computational Considerations

### CPU vs GPU

- **Waveform generation**: CPU-only (via LALSimulation)
- **Domain transformations**: Can use GPU via PyTorch tensors
- **Neural network inference** (in main dingo package): GPU-accelerated

### Performance Tips

1. **Use MultibandedFrequencyDomain** for neural network training to reduce dimensionality
2. **Batch generate** multiple waveforms when creating datasets
3. **Cache** domain objects when generating many waveforms with same sampling

## Next Steps

Dive deeper into specific concepts:

- [Domains](domains.md) - Understand frequency and time sampling
- [Approximants](approximants.md) - Learn about waveform models
- [Polarizations](polarizations.md) - Understand waveform outputs
- [Modes](modes.md) - Work with spherical harmonic decomposition
