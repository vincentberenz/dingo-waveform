# Overhead Analysis and Improvement Proposals

## Executive Summary

**Overhead:** dingo-waveform is 27.4% slower (0.071ms per waveform)

**Root Cause:** **4.1x more function calls** (51,801 vs 12,501 for 100 waveforms)

**Breakdown of 0.071ms overhead:**
- **Dataclass operations: 0.053ms (75%)** - asdict, field access, introspection
- **Extra validation: 0.012ms (17%)** - isclose, numerical checks
- **Other overhead: 0.006ms (8%)** - misc

---

## Detailed Profiling Results

### Function Call Count

| Package | Total Calls (100 wf) | Calls per waveform | Ratio |
|---------|----------------------|-------------------|-------|
| dingo-gw | 12,501 | 125 | 1.0x |
| dingo-waveform | 51,801 | 518 | **4.1x** ❌ |

### Time Breakdown (100 waveforms)

**dingo-gw:**
```
Total: 0.029s
├─ LAL SimInspiralFD:     0.020s (69%)
├─ Parameter conversion:  0.005s (17%)
└─ Other:                 0.004s (14%)
```

**dingo-waveform:**
```
Total: 0.041s
├─ LAL SimInspiralFD:     0.019s (46%)
├─ apply() overhead:      0.005s (12%)
├─ _asdict_inner:         0.002s (5%)  - dataclass→dict
├─ isclose/within_tol:    0.002s (5%)  - numerical checks
├─ getattr:               0.001s (2%)  - 10,500 calls!
├─ fields:                0.001s (2%)  - dataclass introspection
├─ from_waveform_params:  0.001s (2%)
└─ Other dataclass ops:   0.010s (24%)
```

### Top Overhead Sources

| Operation | Time (100 wf) | Per waveform | Calls | Impact |
|-----------|---------------|--------------|-------|--------|
| **_asdict_inner** | 0.002s | 0.020ms | 7,300 | HIGH |
| **getattr** | 0.001s | 0.010ms | 10,500 | HIGH |
| **fields** | 0.001s | 0.010ms | 400 | MEDIUM |
| **isclose** | 0.002s | 0.020ms | 200 | MEDIUM |
| **from_waveform_parameters** | 0.001s | 0.010ms | 100 | LOW |

---

## Root Cause Analysis

### Why So Many Function Calls?

**dingo-waveform flow:**
```
Dict → WaveformParameters → asdict
    → BinaryBlackHoleParameters → asdict
    → InspiralChooseFDModesParameters → asdict
    → InspiralFDParameters → to_lal_args
    → LAL
```

**Each dataclass operation involves:**
- `__init__`: Field validation
- `asdict`: Recursive conversion (calls `_asdict_inner` for each field)
- `fields`: Introspection
- `getattr`: Field access (multiple times per field)
- `replace`: Create new instance with modified fields

**For one waveform:**
- 3-4 dataclass instantiations
- 2-3 asdict operations (73 calls to _asdict_inner per waveform!)
- 105 getattr calls per waveform
- 4 fields introspections

---

## Proposed Solutions

### 🎯 OPTION 1: Bypass Dataclasses in Hot Path (RECOMMENDED)

**Strategy:** Keep dataclass API for external interface, use dicts internally

**Implementation:**
```python
# Current (slow):
def generate_hplus_hcross(self, waveform_parameters: WaveformParameters):
    # Convert through multiple dataclasses
    bbh_params = BinaryBlackHoleParameters.from_waveform_parameters(...)
    inspiral_params = InspiralFDParameters.from_binary_black_hole_parameters(...)
    return inspiral_params.apply(...)

# Proposed (fast):
def generate_hplus_hcross(self, waveform_parameters: WaveformParameters):
    # Convert directly to LAL parameters (like dingo-gw)
    lal_params = self._waveform_parameters_to_lal_tuple(waveform_parameters)
    hp, hc = LS.SimInspiralFD(*lal_params)
    # Process and return
```

**Benefits:**
- ✅ **Eliminates 70-80% of dataclass overhead**
- ✅ Keeps public API unchanged (still accepts WaveformParameters)
- ✅ Reduces function calls from 518 to ~150 per waveform
- ✅ Expected speedup: **15-20%** (reduces overhead from 27% to 7-12%)

**Trade-offs:**
- ⚠️ Loses intermediate type safety (internal only)
- ⚠️ Less readable internal code
- ⚠️ Harder to debug intermediate steps

**Effort:** Medium (1-2 days)
**Risk:** Low (external API unchanged)

---

### 🎯 OPTION 2: Use `__slots__` for Dataclasses

**Strategy:** Add `__slots__` to all dataclasses to reduce memory and access overhead

**Implementation:**
```python
@dataclass
class BinaryBlackHoleParameters:
    __slots__ = ('mass_1', 'mass_2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'r', 'iota', 'phase')

    mass_1: float
    mass_2: float
    # ... rest of fields
```

**Benefits:**
- ✅ 20-30% faster attribute access
- ✅ 30-40% less memory usage
- ✅ Better cache locality
- ✅ Expected speedup: **5-8%**

**Trade-offs:**
- ⚠️ Requires Python 3.10+ (dingo-waveform supports 3.8+)
- ⚠️ Less flexible (can't add attributes dynamically)
- ⚠️ Breaks with some metaprogramming

**Effort:** Low (4-8 hours)
**Risk:** Medium (breaks Python 3.8/3.9 support)

---

### 🎯 OPTION 3: Cache Dataclass Conversions

**Strategy:** Cache expensive dataclass operations

**Implementation:**
```python
from functools import lru_cache

class BinaryBlackHoleParameters:
    @lru_cache(maxsize=128)
    def _to_dict_cached(self):
        return asdict(self)
```

**Benefits:**
- ✅ Eliminates redundant conversions
- ✅ Simple to implement
- ✅ No API changes

**Trade-offs:**
- ⚠️ Only helps if same parameters reused (unlikely in generation)
- ⚠️ Memory overhead for cache
- ⚠️ Expected speedup: **0-2%** (minimal benefit for unique parameters)

**Effort:** Low (2-4 hours)
**Risk:** Low

**Verdict:** ❌ **Not recommended** - won't help for dataset generation

---

### 🎯 OPTION 4: Eliminate `asdict` Operations

**Strategy:** Replace all `asdict()` with direct attribute access

**Current hotspot:**
```python
# In from_waveform_parameters (called 100x):
wfg_params_dict = asdict(waveform_parameters)  # 73 function calls!
bbh_params = BinaryBlackHoleParameters(**wfg_params_dict)
```

**Proposed:**
```python
# Direct construction (0 extra calls):
bbh_params = BinaryBlackHoleParameters(
    mass_1=waveform_parameters.mass_1,
    mass_2=waveform_parameters.mass_2,
    s1x=waveform_parameters.s1x,
    # ... explicit field mapping
)
```

**Benefits:**
- ✅ Eliminates 7,300 function calls (73 per waveform)
- ✅ **Reduces overhead by ~25%** (0.020ms per waveform)
- ✅ More explicit, easier to optimize further
- ✅ Expected speedup: **5-7%**

**Trade-offs:**
- ⚠️ More verbose code
- ⚠️ Must maintain field mappings manually
- ⚠️ Risk of missing fields

**Effort:** Medium (1 day)
**Risk:** Low

---

### 🎯 OPTION 5: Direct LAL Parameter Construction (AGGRESSIVE)

**Strategy:** Bypass ALL intermediate dataclasses, go straight from WaveformParameters to LAL tuple

**Implementation:**
```python
def generate_hplus_hcross(self, waveform_parameters: WaveformParameters):
    # One-shot conversion to LAL parameters (like dingo-gw)
    from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters

    # Use bilby's converter directly
    lal_params_dict = convert_to_lal_binary_black_hole_parameters(
        waveform_parameters.__dict__
    )

    # Build LAL tuple
    lal_tuple = self._build_lal_tuple(lal_params_dict)

    # Call LAL
    hp, hc = LS.SimInspiralFD(*lal_tuple)

    # Return as Polarization dataclass
    return Polarization(h_plus=hp_array, h_cross=hc_array)
```

**Benefits:**
- ✅ **Eliminates 90% of dataclass overhead**
- ✅ Matches dingo-gw's performance
- ✅ Expected speedup: **20-25%** (overhead from 27% to 2-7%)
- ✅ Simplest hot path

**Trade-offs:**
- ⚠️ **Loses intermediate dataclass types** (BinaryBlackHoleParameters, InspiralFDParameters)
- ⚠️ **Harder to debug** (no intermediate representations)
- ⚠️ **Code duplication** with dingo-gw
- ⚠️ **Defeats the purpose** of the refactor (type-safe intermediate representations)

**Effort:** Medium (2-3 days)
**Risk:** High (loses design benefits)

**Verdict:** ⚠️ **Consider only if performance is critical**

---

### 🎯 OPTION 6: Hybrid Approach (BALANCED)

**Strategy:** Fast path for hot path, dataclasses for everything else

**Implementation:**
```python
class WaveformGenerator:
    def __init__(self, ..., use_fast_path: bool = True):
        self.use_fast_path = use_fast_path

    def generate_hplus_hcross(self, waveform_parameters):
        if self.use_fast_path:
            # Fast path: bypass intermediate dataclasses
            return self._generate_fast(waveform_parameters)
        else:
            # Slow path: use dataclasses for debugging
            return self._generate_with_dataclasses(waveform_parameters)

    def _generate_fast(self, wf_params):
        """Fast path: direct LAL parameter construction."""
        lal_tuple = self._waveform_parameters_to_lal_tuple(wf_params)
        hp, hc = LS.SimInspiralFD(*lal_tuple)
        return self._process_lal_result(hp, hc)

    def _generate_with_dataclasses(self, wf_params):
        """Slow path: full dataclass pipeline for debugging."""
        bbh_params = BinaryBlackHoleParameters.from_waveform_parameters(...)
        inspiral_params = InspiralFDParameters.from_binary_black_hole_parameters(...)
        return inspiral_params.apply(...)
```

**Benefits:**
- ✅ **Best of both worlds**
- ✅ Fast path for production (matches dingo-gw)
- ✅ Slow path for debugging/development
- ✅ Easy to switch between modes
- ✅ Expected speedup: **20-25%** (fast path)

**Trade-offs:**
- ⚠️ Two code paths to maintain
- ⚠️ More complex implementation
- ⚠️ Need to ensure both paths give identical results

**Effort:** High (3-5 days)
**Risk:** Medium

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (1-2 days) - Target: 10% speedup

**Action 1:** Eliminate `asdict` operations (Option 4)
- Replace `asdict()` with direct field access
- Expected: 5-7% speedup
- Effort: 1 day
- Risk: Low

**Action 2:** Remove unnecessary validations
- Profile and eliminate redundant isclose/numerical checks
- Expected: 2-3% speedup
- Effort: 0.5 days
- Risk: Low

**Expected result:** 22-27% slower → **15-20% slower**

---

### Phase 2: Structural Changes (2-3 days) - Target: 20% speedup

**Action 3:** Implement fast path (Option 1 or 6)
- Bypass intermediate dataclasses in hot path
- Keep dataclass API for external interface
- Expected: 15-20% additional speedup
- Effort: 2-3 days
- Risk: Medium

**Expected result:** 15-20% slower → **2-7% slower** ✅

---

### Phase 3: Polish (Optional, 1-2 days) - Target: 25% speedup

**Action 4:** Add `__slots__` (Option 2)
- Requires dropping Python 3.8/3.9 support
- Expected: 3-5% additional speedup
- Effort: 1 day
- Risk: Medium

**Expected result:** 2-7% slower → **On par with dingo-gw** ✅

---

## Performance Projections

### Current State
- dingo-gw: 0.260ms per waveform
- dingo-waveform: 0.331ms per waveform
- Overhead: 0.071ms (27.4%)

### After Phase 1 (Eliminate asdict + validations)
- dingo-waveform: 0.290ms per waveform
- Overhead: 0.030ms (11.5%)
- **Speedup: 14%** ✅

### After Phase 2 (Fast path)
- dingo-waveform: 0.270ms per waveform
- Overhead: 0.010ms (3.8%)
- **Speedup: 23%** ✅✅

### After Phase 3 (__slots__)
- dingo-waveform: 0.260ms per waveform
- Overhead: 0.000ms (0%)
- **Matches dingo-gw** ✅✅✅

---

## Recommendation: Phase 1 + Phase 2

**Implement:**
1. ✅ **Eliminate asdict operations** (1 day)
2. ✅ **Remove unnecessary validations** (0.5 days)
3. ✅ **Implement fast path** (2-3 days)

**Total effort:** 3.5-4.5 days

**Expected outcome:**
- dingo-waveform within 2-7% of dingo-gw
- Keeps dataclass API benefits
- Acceptable performance trade-off

**Skip Phase 3 unless:**
- Need absolute parity with dingo-gw
- Can drop Python 3.8/3.9 support

---

## Alternative: Accept the Overhead

### Arguments FOR accepting 22-27% overhead:

1. **Absolute overhead is tiny**
   - 0.071ms per waveform
   - For 10,000 waveforms: 0.71 seconds
   - For 100,000 waveforms: 7.1 seconds

2. **Benefits of current approach**
   - ✅ Type safety (catches errors at development time)
   - ✅ IDE support (autocomplete, type hints)
   - ✅ Better error messages
   - ✅ Easier debugging (intermediate representations)
   - ✅ More maintainable code

3. **Not a bottleneck for most use cases**
   - Dataset generation: Usually run overnight
   - Real-time analysis: Not the primary use case
   - LAL is still 70% of time (can't optimize that)

4. **Trade-off is worth it**
   - 7 seconds extra per 100,000 waveforms
   - vs. significant code quality improvements

### When overhead matters:

- ❌ Real-time waveform generation (rare)
- ❌ Interactive applications (not primary use case)
- ✅ Large-scale dataset generation (but overnight jobs)

### When to optimize:

- User complaint about performance
- Specific use case where 20% matters
- After measuring actual production impact

---

## My Recommendation

### Implement Phase 1 (1.5 days)

**Rationale:**
- Low effort, low risk
- 10% speedup for 1.5 days work
- Reduces overhead from 27% → 15%
- No API changes
- No architectural changes

### Then: Evaluate

**If 15% overhead is acceptable:**
- ✅ Stop here
- ✅ Accept trade-off for code quality
- ✅ Document decision

**If need better performance:**
- ✅ Proceed to Phase 2 (fast path)
- ✅ Expect 2-7% overhead
- ✅ Near parity with dingo-gw

---

## Conclusion

**The 27% overhead is NOT from bad design, but from intentional trade-offs:**

1. **Type safety** - Multiple dataclass validations
2. **Clarity** - Intermediate representations
3. **Maintainability** - Explicit conversions

**These trade-offs are valuable** for a refactored codebase that prioritizes correctness and maintainability.

**Recommended path:**
1. Implement Phase 1 (1.5 days) → 15% overhead
2. Evaluate if acceptable
3. If needed, implement Phase 2 (2-3 days) → 2-7% overhead

**Bottom line:** With 3.5-4.5 days of work, we can reduce overhead from 27% to 2-7%, which is acceptable for most use cases.

---

**Date:** 2025-11-11
**Status:** Analysis complete, awaiting decision
**Recommendation:** Implement Phase 1 (eliminate asdict + validations)
