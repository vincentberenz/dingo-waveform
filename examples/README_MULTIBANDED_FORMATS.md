# MultibandedFrequencyDomain Configuration Formats

dingo-waveform supports **two formats** for MultibandedFrequencyDomain configuration:

## 1. Simple Format (dingo-waveform native)

The simplest way to specify a multibanded domain - just provide the base frequency spacing:

```yaml
domain:
  type: MultibandedFrequencyDomain
  nodes: [20.0, 128.0, 512.0, 1024.0]
  delta_f_initial: 0.125
  base_delta_f: 0.125  # Simple: just the frequency spacing
```

**Advantages:**
- Cleaner, more concise syntax
- Faster to write
- Sufficient for most use cases

**Example:** `benchmark_mfd_simple.yaml`

## 2. dingo-gw Compatible Format

Full specification including base domain details (compatible with dingo-gw configs):

```yaml
domain:
  type: MultibandedFrequencyDomain
  nodes: [20.0, 128.0, 512.0, 1024.0]
  delta_f_initial: 0.125
  base_domain:
    type: UniformFrequencyDomain
    f_min: 20.0
    f_max: 1024.0
    delta_f: 0.125
```

**Advantages:**
- Compatible with existing dingo-gw configuration files
- Explicitly specifies base domain bounds
- Can be used interchangeably between dingo-gw and dingo-waveform

**Example:** `benchmark_multibanded.yaml`

## Both Formats Are Equivalent

Both formats produce **identical results** in dingo-waveform. The `base_domain` format is automatically converted internally:
- Extract `delta_f` from `base_domain` dict
- Use it as `base_delta_f` for the multibanded domain

## Benchmarking

The benchmark tool (`dingo-benchmark`) automatically handles both formats:
- When benchmarking against dingo-gw, simple format is automatically converted to dingo-gw compatible format
- Both packages produce identical waveforms regardless of which format is used

## Migration Guide

**From dingo-gw to dingo-waveform:**
- No changes needed! Existing configs with `base_domain` work directly

**New configs for dingo-waveform only:**
- Use the simpler `base_delta_f` format for cleaner syntax
- Convert to `base_domain` format only if you need dingo-gw compatibility
