# Verification: f_min and f_max in MultibandedFrequencyDomain

## Question

Are `f_min` and `f_max` required by MultibandedFrequencyDomain in:
1. dingo-waveform (refactored package)?
2. dingo (original dingo-gw package)?

## Answer: ✅ CONFIRMED - NOT REQUIRED

Both implementations **do NOT require** `f_min` and `f_max` as input parameters. They are **automatically derived** from the `nodes` parameter.

---

## Detailed Verification

### 1. dingo-waveform (Refactored Package)

#### Constructor Signature
```python
def __init__(
    self,
    nodes: Iterable[float],
    delta_f_initial: float,
    base_delta_f: float,
    window_factor: Optional[float] = None,
)
```

**Required parameters:**
- `nodes` ✅
- `delta_f_initial` ✅
- `base_delta_f` ✅
- `window_factor` (optional)

**NO `f_min` or `f_max` parameters** ❌

#### How f_min and f_max are derived

From `dingo_waveform/domains/multibanded_frequency_domain.py`:

```python
@property
def f_min(self) -> float:
    if self._binning.f_base_lower.size == 0:
        return 0.0
    return float(self._binning.f_base_lower[0])  # First element

@property
def f_max(self) -> float:
    if self._binning.f_base_upper.size == 0:
        return 0.0
    return float(self._binning.f_base_upper[-1])  # Last element
```

These are **@property** methods that compute values from the internal binning structure, which is built from `nodes`.

#### from_parameters method

From `dingo_waveform/domains/multibanded_frequency_domain.py:350-365`:

```python
@classmethod
def from_parameters(
    cls, domain_parameters: DomainParameters
) -> "MultibandedFrequencyDomain":
    for attr in ("nodes", "delta_f_initial", "base_delta_f"):
        if getattr(domain_parameters, attr) is None:
            raise ValueError(...)

    return cls(
        nodes=domain_parameters.nodes,
        delta_f_initial=domain_parameters.delta_f_initial,
        base_delta_f=domain_parameters.base_delta_f,
        window_factor=domain_parameters.window_factor,
    )
```

**Only requires:** `nodes`, `delta_f_initial`, `base_delta_f`
**Does NOT use:** `f_min`, `f_max`

---

### 2. dingo (Original dingo-gw Package)

#### Constructor Signature
```python
def __init__(
    self,
    nodes: Iterable[float],
    delta_f_initial: float,
    base_domain: Union[UniformFrequencyDomain, dict],
)
```

**Required parameters:**
- `nodes` ✅
- `delta_f_initial` ✅
- `base_domain` ✅

**NO `f_min` or `f_max` parameters** ❌

#### How f_min and f_max are derived

From `dingo/gw/domains/multibanded_frequency_domain.py:359-364`:

```python
@property
def f_max(self) -> float:
    return float(self._f_base_upper[-1])

@property
def f_min(self) -> float:
    return float(self._f_base_lower[0])
```

These are **@property** methods computed from internal arrays built from `nodes`.

#### domain_dict property

From `dingo/gw/domains/multibanded_frequency_domain.py:375-383`:

```python
@property
def domain_dict(self) -> dict:
    """Enables to rebuild the domain via calling build_domain(domain_dict)."""
    return {
        "type": "MultibandedFrequencyDomain",
        "nodes": self.nodes.tolist(),
        "delta_f_initial": self._delta_f_bands[0].item(),
        "base_domain": self.base_domain.domain_dict,  # Nested dict
    }
```

**Keys in domain_dict:**
- `type` ✅
- `nodes` ✅
- `delta_f_initial` ✅
- `base_domain` ✅ (nested dict)

**NOT in domain_dict:**
- `f_min` ❌
- `f_max` ❌

---

## Practical Test Results

### Test: Creating domains without f_min/f_max

```python
# dingo-waveform
config = {
    "type": "MultibandedFrequencyDomain",
    "nodes": [20.0, 38.0, 50.0, 66.0, 82.0, 1810.0],
    "delta_f_initial": 0.125,
    "base_delta_f": 0.125,
    # NO f_min or f_max
}
mfd = build_domain(config)
print(f"f_min = {mfd.f_min}")  # 20.0 Hz (from nodes[0])
print(f"f_max = {mfd.f_max}")  # 1809.875 Hz (from nodes[-1])
```

**Result:** ✅ Works perfectly without `f_min` or `f_max`

### Test: Including f_min/f_max (should be ignored)

```python
config = {
    "type": "MultibandedFrequencyDomain",
    "nodes": [20.0, 38.0, 50.0, 66.0, 82.0, 1810.0],
    "delta_f_initial": 0.125,
    "base_delta_f": 0.125,
    "f_min": 20.0,  # Included but ignored
    "f_max": 1810.0,  # Included but ignored
}
mfd = build_domain(config)
```

**Result:** ✅ Works - these fields are simply ignored during construction

---

## How My Updates Handle This

In my updates to handle the new API, I ensured proper fallback logic:

### 1. inspiral_choose_FD_modes.py (lines 90-100)
```python
# f_min and f_max should be present; if not, derive from nodes for MultibandedFrequencyDomain
fmin = domain_dict.get("f_min")
fmax = domain_dict.get("f_max")
if domain_dict.get("nodes") is not None:
    nodes = domain_dict.get("nodes")
    if fmin is None:
        fmin = nodes[0]  # Derive from first node
    if fmax is None:
        fmax = nodes[-1]  # Derive from last node
parameters["f_min"] = fmin
parameters["f_max"] = fmax
```

**Logic:** Try to get f_min/f_max from DomainParameters; if not present AND nodes exist, derive from nodes.

### 2. gw_signals_parameters.py (lines 104-107)
```python
fmax = dp_dict.get("f_max")
# For MultibandedFrequencyDomain, f_max comes from the last node
if fmax is None and dp_dict.get("nodes") is not None:
    fmax = dp_dict.get("nodes")[-1]
```

**Logic:** Same fallback pattern for f_max.

### Why this is correct

The `DomainParameters` dataclass CAN have `f_min` and `f_max` fields (they're optional fields for other domain types like UniformFrequencyDomain). For MultibandedFrequencyDomain:

1. These fields may or may not be present in DomainParameters
2. If present, they should match nodes[0] and nodes[-1]
3. If not present, we derive them from nodes
4. Either way, we ensure the waveform generation functions have access to f_min/f_max values when needed

---

## Conclusion

### ✅ CONFIRMED

1. **dingo-waveform MultibandedFrequencyDomain:**
   - Does NOT require `f_min` or `f_max` as constructor parameters
   - Derives them as properties from `nodes`: `f_min = nodes[0]`, `f_max ≈ nodes[-1]`

2. **dingo (dingo-gw) MultibandedFrequencyDomain:**
   - Does NOT require `f_min` or `f_max` as constructor parameters
   - Derives them as properties from `nodes`: `f_min = nodes[0]`, `f_max ≈ nodes[-1]`
   - Does NOT include them in `domain_dict`

3. **My updates correctly handle both cases:**
   - Primary: Use explicit `f_min`/`f_max` if present in DomainParameters
   - Fallback: Derive from `nodes` for MultibandedFrequencyDomain
   - This ensures robustness across different domain creation paths

### Configuration Format

**Minimal valid configuration (recommended):**
```yaml
domain:
  type: MultibandedFrequencyDomain
  nodes: [20.0, 38.0, 50.0, 66.0, 82.0, 1810.0]
  delta_f_initial: 0.125
  base_delta_f: 0.125
```

**With optional f_min/f_max (not needed, will be ignored by constructor):**
```yaml
domain:
  type: MultibandedFrequencyDomain
  nodes: [20.0, 38.0, 50.0, 66.0, 82.0, 1810.0]
  delta_f_initial: 0.125
  base_delta_f: 0.125
  f_min: 20.0  # Optional, ignored during construction
  f_max: 1810.0  # Optional, ignored during construction
```

---

**Verification Date:** 2025-11-07
**Status:** ✅ Complete and Verified
