# Domains API

Domains define how waveforms are sampled in frequency or time.

## UniformFrequencyDomain

::: dingo_waveform.domains.UniformFrequencyDomain
    options:
      show_root_heading: true
      show_source: false

## MultibandedFrequencyDomain

::: dingo_waveform.domains.MultibandedFrequencyDomain
    options:
      show_root_heading: true
      show_source: false

## TimeDomain

::: dingo_waveform.domains.TimeDomain
    options:
      show_root_heading: true
      show_source: false

## Usage Examples

### Uniform Frequency Domain

```python
from dingo_waveform import UniformFrequencyDomain

domain = UniformFrequencyDomain(
    f_min=20.0,      # Minimum frequency (Hz)
    f_max=1024.0,    # Maximum frequency (Hz)
    delta_f=0.125    # Frequency spacing (Hz)
)

# Get frequency array
freqs = domain()
print(f"Number of frequency bins: {len(freqs)}")
```

### Multibanded Frequency Domain

```python
from dingo_waveform import MultibandedFrequencyDomain

domain = MultibandedFrequencyDomain(
    f_min=20.0,
    f_max=1024.0,
    delta_f_initial=0.125,
    n_bins_per_harmonic=6
)

# Adaptive binning for efficiency
freqs = domain.sample_frequencies
print(f"Reduced to {len(freqs)} bins (from ~{int(1024/0.125)})")
```

### Time Domain

```python
from dingo_waveform import TimeDomain

domain = TimeDomain(
    duration=8.0,       # Total duration (seconds)
    delta_t=1/2048.0    # Time step (seconds)
)

# Get time array
times = domain()
print(f"Number of time samples: {len(times)}")
```

## See Also

- [Concepts: Domains](../concepts/domains.md) - Detailed explanation of domains
- [Waveform Generator API](waveform-generator.md) - Using domains with generators
