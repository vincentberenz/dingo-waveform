# Optimization Analysis Summary

## Quick Reference

### Current Performance
- **Simple approximants (IMRPhenomD):** 25-27% slower than dingo-gw
- **Complex approximants (IMRPhenomXPHM):** Only 4.8% slower than dingo-gw
- **Absolute overhead:** 0.16ms per waveform (simple) to 0.3ms per waveform (complex)

### Potential Improvements
With all optimizations: **10-15% slower** (vs current 25-27%)

---

## Top 7 Optimization Opportunities

### 1. Eliminate `deepcopy` Operations ⭐⭐⭐ HIGH PRIORITY
- **Current:** 43 deepcopy calls per waveform
- **Impact:** 15-20% faster
- **Fix:** Replace with direct tuple construction or `dataclasses.replace()`
- **Location:** `polarization_functions/inspiral_FD.py:197`

### 2. Reduce `asdict` Calls ⭐⭐⭐ HIGH PRIORITY
- **Current:** 85 asdict calls per waveform
- **Impact:** 10-15% faster
- **Fix:** Use `dataclasses.replace()` instead of dict conversion
- **Location:** `polarization_functions/inspiral_FD.py:99`

### 3. Optimize Domain Parameter Handling ⭐⭐⭐ HIGH PRIORITY
- **Current:** Dataclass → dict → dataclass round-trip
- **Impact:** 3-5% faster
- **Fix:** Direct attribute access with `getattr()`
- **Location:** `polarization_functions/inspiral_FD.py:125-133`

### 4. Eliminate `astuple` Overhead ⭐⭐ MEDIUM PRIORITY
- **Current:** 21 astuple calls per waveform
- **Impact:** 5-8% faster
- **Fix:** Cached tuple conversion method
- **Location:** `polarization_functions/inspiral_FD.py:210`

### 5. Add `__slots__` to Dataclasses ⭐⭐ MEDIUM PRIORITY
- **Current:** Using default `__dict__` storage
- **Impact:** 3-5% faster + 30-40% less memory
- **Fix:** Add `__slots__` to all parameter dataclasses
- **Location:** All dataclass definitions

### 6. Optimize `convert_to_float` ⭐ LOW PRIORITY
- **Current:** 23 calls per waveform with type checks
- **Impact:** 2-3% faster
- **Fix:** Add fast path for common cases
- **Location:** `binary_black_holes_parameters.py:20`

### 7. Cache Domain Conversions ⭐ LOW PRIORITY
- **Current:** No caching of sanitized domains
- **Impact:** 1-2% faster
- **Fix:** Add LRU cache for domain parameter objects
- **Location:** `waveform_generator.py`

---

## Implementation Roadmap

### Phase 1: High-Priority Optimizations (1-2 days)
Implement #1, #2, #3 → **30-40% overhead reduction**

**Expected result:** 15-18% slower instead of 25-27%

### Phase 2: Medium-Priority Optimizations (1 day)
Implement #4, #5 → **Additional 8-13% reduction**

**Expected result:** 10-15% slower instead of 25-27%

### Phase 3: Polish (0.5 day)
Implement #6, #7 → **Additional 3-5% reduction**

**Expected result:** 10-12% slower instead of 25-27%

---

## Recommendation

**For simple approximants:** Implement Phase 1 optimizations to achieve ~15-18% overhead (vs current 27%)

**For complex approximants:** No optimization needed - already within 5% of dingo-gw!

---

## Key Insight

The bottleneck is **not the dataclass design** but rather the **defensive programming patterns**:
- Excessive deepcopy to prevent mutation
- Round-trip conversions (dataclass → dict → dataclass)
- Lack of caching for repeated operations

These can be optimized while maintaining the benefits of type-safe dataclasses.

---

**Full details:** See `OPTIMIZATION_OPPORTUNITIES.md`
