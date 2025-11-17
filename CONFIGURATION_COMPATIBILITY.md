# Configuration File Compatibility

**Last Updated:** 2025-11-12

## Summary

✅ **dingo-waveform now supports both configuration formats:**
1. **dingo-gw compatible format** - Full compatibility with existing dingo-gw configs
2. **Simple format** - Native to dingo-waveform, with cleaner syntax

Both formats produce **identical results** and can be used interchangeably.

---

## MultibandedFrequencyDomain Formats

### Format 1: Simple (dingo-waveform native)

```yaml
domain:
  type: MultibandedFrequencyDomain
  nodes: [20.0, 128.0, 512.0, 1024.0]
  delta_f_initial: 0.125
  base_delta_f: 0.125  # Simple: just the frequency spacing
```

**Advantages:**
- Cleaner, more concise syntax
- Sufficient for most use cases
- Faster to write

**Example:** `examples/benchmark_mfd_simple.yaml`

### Format 2: dingo-gw Compatible

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
- Can be used with both dingo-gw and dingo-waveform

**Example:** `examples/benchmark_multibanded.yaml`

---

## Implementation Details

### dingo-waveform

**Natively supports both formats:**
- When `base_delta_f` is provided → use it directly
- When `base_domain` dict is provided → extract `delta_f` from it
- If both provided → `base_delta_f` takes precedence

Implementation in `MultibandedFrequencyDomain.from_parameters()`:
```python
# Try base_delta_f first (simple format)
base_delta_f = domain_parameters.base_delta_f

# Fall back to extracting from base_domain (dingo-gw format)
if base_delta_f is None and domain_parameters.base_domain is not None:
    base_delta_f = domain_parameters.base_domain["delta_f"]
```

### dingo-gw

**Only supports `base_domain` format:**
- Requires nested dict with `type`, `f_min`, `f_max`, `delta_f`
- Does not support simple `base_delta_f` format

### Benchmark Tool (benchmark.py)

**Automatically handles both formats:**
- When benchmarking against dingo-gw with simple format:
  - Converts `base_delta_f` → `base_domain` dict automatically
  - Extracts `f_min`/`f_max` from `nodes` array
- Both packages receive appropriate format for their API

---

## Benchmark Results

Both formats produce **identical performance:**

**Simple format** (5 waveforms, IMRPhenomXPHM):
```
dingo-gw:       171.52 waveforms/s
dingo-waveform: 189.24 waveforms/s
Speedup:        10.3% faster ✅
```

**dingo-gw compatible format** (5 waveforms, IMRPhenomXPHM):
```
dingo-gw:       161.41 waveforms/s
dingo-waveform: 173.86 waveforms/s
Speedup:        7.6% faster ✅
```

---

## Migration Guide

### From dingo-gw to dingo-waveform

**No changes needed!** Existing configs work directly:
```yaml
# This dingo-gw config works in dingo-waveform without modification
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

### New configs for dingo-waveform

**Use simpler format** for cleaner syntax:
```yaml
# Simpler dingo-waveform-only config
domain:
  type: MultibandedFrequencyDomain
  nodes: [20.0, 128.0, 512.0, 1024.0]
  delta_f_initial: 0.125
  base_delta_f: 0.125  # Just this!
```

---

## Python API

All formats work with `build_domain()`:

```python
from dingo_waveform.domains import build_domain, DomainParameters

# Method 1: From dict (simple format)
domain = build_domain({
    "type": "MultibandedFrequencyDomain",
    "nodes": [20.0, 128.0, 512.0, 1024.0],
    "delta_f_initial": 0.125,
    "base_delta_f": 0.125,
})

# Method 2: From dict (dingo-gw format)
domain = build_domain({
    "type": "MultibandedFrequencyDomain",
    "nodes": [20.0, 128.0, 512.0, 1024.0],
    "delta_f_initial": 0.125,
    "base_domain": {
        "type": "UniformFrequencyDomain",
        "f_min": 20.0,
        "f_max": 1024.0,
        "delta_f": 0.125,
    },
})

# Method 3: From DomainParameters
params = DomainParameters(
    type="MultibandedFrequencyDomain",
    nodes=[20.0, 128.0, 512.0, 1024.0],
    delta_f_initial=0.125,
    base_delta_f=0.125,
)
domain = build_domain(params)

# Method 4: From YAML/JSON file
domain = build_domain("config.yaml")
```

---

## Tests

All configuration formats verified by:
- `tests/test_multibanded_frequency_domain.py` - Unit tests for both formats
- `tests/test_dingo_compatibility.py` - Comparison with dingo-gw
- `dingo-benchmark` - Performance benchmarking tool

**Test coverage:** ✅ All 38 MultibandedFrequencyDomain tests pass
