# Comprehensive Optimization and Testing Plan

## Executive Summary

**Issue:** Only `inspiral_FD` has been optimized with Phase 2 fast path. All other waveform generation functions still have 20-27% overhead compared to dingo-gw.

**Root Cause:** Assumed usage patterns from biased test/example sampling rather than treating all public API functions equally.

**Solution:** Optimize ALL public waveform generation functions and ensure comprehensive test/benchmark coverage.

---

## Current State Analysis

### Polarization Functions (return h+ and h×)

| Function | Status | Used By | Test Coverage | Benchmark Coverage |
|----------|--------|---------|---------------|-------------------|
| **inspiral_FD** | ✅ OPTIMIZED | IMRPhenomD, IMRPhenomXAS, IMRPhenomXPHM (UFD) | ✅ test_dingo_compatibility | ✅ benchmark_quick.yaml |
| **inspiral_TD** | ❌ NOT OPTIMIZED | Time-domain approximants | ❌ NO TESTS | ❌ NO BENCHMARKS |
| **generate_FD_modes** | ❌ NOT OPTIMIZED | SEOBNRv5PHM/HM (FD, new interface) | ⚠️ Partial (via MFD test) | ❌ NO BENCHMARKS |
| **generate_TD_modes** | ❌ NOT OPTIMIZED | SEOBNRv5PHM/HM (TD, new interface) | ✅ test_dingo_compatibility | ❌ NO BENCHMARKS |

### Polarization Modes Functions (return individual (l,m) modes)

| Function | Status | Used By | Test Coverage | Benchmark Coverage |
|----------|--------|---------|---------------|-------------------|
| **inspiral_choose_FD_modes** | ❌ NOT OPTIMIZED | Mode-separated FD waveforms | ❌ NO TESTS | ❌ NO BENCHMARKS |
| **inspiral_choose_TD_modes** | ❌ NOT OPTIMIZED | Mode-separated TD waveforms | ❌ NO TESTS | ❌ NO BENCHMARKS |
| **generate_FD_modes_LO** | ❌ NOT OPTIMIZED | Leading-order FD modes | ❌ NO TESTS | ❌ NO BENCHMARKS |
| **generate_TD_modes_LO** | ❌ NOT OPTIMIZED | Leading-order TD modes | ❌ NO TESTS | ❌ NO BENCHMARKS |
| **generate_TD_modes_LO_cond_extra_time** | ❌ NOT OPTIMIZED | SEOBNRv5 with extra conditioning | ✅ test_dingo_compatibility | ❌ NO BENCHMARKS |

---

## Architecture Analysis

### How Functions Are Called

```
User → WaveformGenerator.generate_hplus_hcross()
        ↓
        ├─ For most approximants:
        │   ├─ FD domain → inspiral_FD() ✅ OPTIMIZED
        │   └─ TD domain → inspiral_TD() ❌ NOT OPTIMIZED
        │
        ├─ For SEOBNRv5PHM/HM (new interface):
        │   ├─ FD domain → generate_FD_modes() ❌ NOT OPTIMIZED
        │   └─ TD domain → generate_TD_modes() ❌ NOT OPTIMIZED
        │
        └─ For SEOBNRv5HM specifically:
            └─ TD domain → generate_TD_modes_LO_cond_extra_time() ❌ NOT OPTIMIZED
```

**Mode-separated functions are NOT called by generate_hplus_hcross()** - they're separate entry points for advanced users who need individual (l,m) modes.

### Internal Dependencies

Each polarization function follows similar pattern:

```python
# Current slow path (applies to all except inspiral_FD)
def inspiral_TD(...):
    # Step 1: Convert to BinaryBlackHoleParameters
    bbh_params = BinaryBlackHoleParameters.from_waveform_parameters(...)

    # Step 2: Convert to InspiralTDParameters
    inspiral_params = _InspiralTDParameters.from_binary_black_hole_parameters(...)

    # Step 3: Call LAL
    return inspiral_params.apply(...)
```

**Optimization strategy:** Create fast path for each function similar to what we did for `inspiral_FD`.

---

## Optimization Plan

### Phase 3A: Core Polarization Functions (HIGH PRIORITY)

**Functions to optimize:**
1. **inspiral_TD** - Time domain equivalent of inspiral_FD
2. **generate_FD_modes** - SEOBNRv5 FD generation
3. **generate_TD_modes** - SEOBNRv5 TD generation

**Why high priority:**
- Called by WaveformGenerator.generate_hplus_hcross()
- Users can't avoid these functions for certain approximants
- Already have test coverage (except inspiral_TD)

**Effort:** ~1 day (3 functions × 3-4 hours each)

---

### Phase 3B: Mode-Separated Functions (MEDIUM PRIORITY)

**Functions to optimize:**
1. **inspiral_choose_FD_modes** - FD mode decomposition
2. **inspiral_choose_TD_modes** - TD mode decomposition
3. **generate_FD_modes_LO** - Leading order FD modes
4. **generate_TD_modes_LO** - Leading order TD modes
5. **generate_TD_modes_LO_cond_extra_time** - SEOBNRv5 with conditioning

**Why medium priority:**
- Separate entry points for advanced users
- Not called by standard generate_hplus_hcross() workflow
- BUT: We can't assume users don't need these

**Effort:** ~1-1.5 days (5 functions × 2-3 hours each)

---

## Testing Plan

### Current Gaps

**Missing test coverage:**
- ❌ inspiral_TD with any approximant
- ❌ inspiral_choose_FD_modes
- ❌ inspiral_choose_TD_modes
- ❌ generate_FD_modes_LO
- ❌ generate_TD_modes_LO
- ⚠️ generate_FD_modes (only tested indirectly via MFD)

**Missing benchmark coverage:**
- ❌ ALL functions except inspiral_FD

---

### Test Coverage Expansion

**Add to test_dingo_compatibility.py:**

```python
class TestTimeDomainCompatibility:
    """Test time domain waveform generation."""

    @pytest.mark.parametrize("approximant", [
        "SEOBNRv4PHM", "IMRPhenomTPHM", "TEOBResumS"
    ])
    def test_td_polarizations(self, approximant, ...):
        # Test inspiral_TD directly
        ...

class TestModeSeparatedCompatibility:
    """Test mode-separated waveform generation."""

    @pytest.mark.parametrize("approximant", [
        "IMRPhenomXPHM", "SEOBNRv4PHM"
    ])
    def test_fd_modes(self, approximant, ...):
        # Test inspiral_choose_FD_modes
        ...

    @pytest.mark.parametrize("approximant", [
        "SEOBNRv4PHM", "SEOBNRv5PHM"
    ])
    def test_td_modes(self, approximant, ...):
        # Test inspiral_choose_TD_modes
        ...

    def test_leading_order_modes(self, ...):
        # Test generate_FD_modes_LO, generate_TD_modes_LO
        ...
```

**Effort:** ~0.5 days

---

### Benchmark Coverage Expansion

**Create benchmark configurations:**

```yaml
# benchmark_time_domain.yaml
approximant: SEOBNRv4PHM
domain:
  type: TimeDomain
  duration: 16.0
  sample_rate: 4096.0

# benchmark_modes.yaml
# (need to extend benchmark tool to support mode-separated generation)
approximant: IMRPhenomXPHM
return_modes: true  # New flag
modes_to_return: [(2,2), (3,3), (4,4)]
```

**Extend benchmark.py to support:**
- Time domain benchmarks
- Mode-separated benchmarks
- Per-function benchmarking (not just end-to-end)

**Effort:** ~0.5 days

---

## Implementation Strategy

### Approach: Template-Based Fast Path

Since all functions follow similar pattern, we can create a template:

```python
# Template for fast path optimization
def _waveform_parameters_to_lal_args_fast_FUNCTION(
    waveform_params: WaveformParameters,
    domain_params: DomainParameters,
    f_ref: float,
    ...
) -> Tuple[...]:
    """
    Fast path for FUNCTION_NAME.
    Bypasses intermediate dataclasses for maximum performance.
    """
    # 1. Build parameter dict with explicit field access
    params = {field: getattr(waveform_params, field) for field in FIELDS}

    # 2. Convert to LAL parameters
    p, _ = convert_to_lal_binary_black_hole_parameters(params)

    # 3. Unit conversions
    # 4. Spin conversions
    # 5. Domain parameter extraction
    # 6. Build LAL tuple for specific function

    return lal_args_tuple
```

Each function needs:
1. Copy template
2. Adjust LAL tuple construction for specific LAL function signature
3. Handle function-specific parameters (e.g., conditioning for SEOBNRv5)
4. Update main function to call fast path

---

## Validation Strategy

For each optimized function:

1. ✅ **Correctness:** Compare output with dingo-gw (machine precision)
2. ✅ **Performance:** Measure speedup with detailed profiling
3. ✅ **Compatibility:** All existing tests must pass
4. ✅ **Documentation:** Update docstrings to mention fast path

---

## Risk Assessment

### Low Risk
- **Phase 3A (core polarizations):** Following proven pattern from inspiral_FD
- **Testing expansion:** Only adds coverage, doesn't change code

### Medium Risk
- **Phase 3B (modes):** Mode-separated functions have more complex LAL signatures
- **Benchmark expansion:** Requires extending benchmark tool

### Mitigation
- Implement one function at a time
- Test each function before moving to next
- Keep old code path as fallback (comment out, don't delete)

---

## Recommended Execution Order

### Week 1: Core Functions + Testing

**Day 1-2: Phase 3A Core Functions**
- [ ] Optimize inspiral_TD
- [ ] Optimize generate_TD_modes
- [ ] Optimize generate_FD_modes
- [ ] Test each function thoroughly

**Day 3: Testing Expansion**
- [ ] Add TD tests to test_dingo_compatibility.py
- [ ] Add mode-separated tests
- [ ] Verify all tests pass

**Day 4: Benchmarking**
- [ ] Create TD benchmark configs
- [ ] Extend benchmark tool for modes
- [ ] Run comprehensive benchmarks
- [ ] Document results

### Week 2: Mode Functions

**Day 5-6: Phase 3B Mode Functions**
- [ ] Optimize inspiral_choose_FD_modes
- [ ] Optimize inspiral_choose_TD_modes
- [ ] Optimize generate_FD_modes_LO
- [ ] Optimize generate_TD_modes_LO
- [ ] Optimize generate_TD_modes_LO_cond_extra_time

**Day 7: Final Validation**
- [ ] Run all tests
- [ ] Run all benchmarks
- [ ] Create comprehensive performance report
- [ ] Update documentation

---

## Success Criteria

**All functions within 1% of dingo-gw performance:**
- [ ] inspiral_FD: ✅ 0.1% (DONE)
- [ ] inspiral_TD: Target <1%
- [ ] generate_FD_modes: Target <1%
- [ ] generate_TD_modes: Target <1%
- [ ] inspiral_choose_FD_modes: Target <1%
- [ ] inspiral_choose_TD_modes: Target <1%
- [ ] generate_FD_modes_LO: Target <1%
- [ ] generate_TD_modes_LO: Target <1%
- [ ] generate_TD_modes_LO_cond_extra_time: Target <1%

**100% test coverage:**
- [ ] All public functions tested
- [ ] All approximants covered
- [ ] Both UFD and MFD tested
- [ ] Both FD and TD tested

**100% benchmark coverage:**
- [ ] All public functions benchmarked
- [ ] Results documented

---

## Questions Before Starting

1. **Should we proceed with full optimization of all functions?**
   - Effort: ~2-2.5 days total
   - Benefit: Complete performance parity across entire API

2. **Should we implement all test/benchmark coverage?**
   - Effort: ~1 day total
   - Benefit: Confidence in API quality and performance

3. **Any specific approximants or use cases to prioritize?**
   - We should test what users actually use
   - But we can't know without comprehensive coverage

4. **Timeline preference?**
   - All at once (1 week sprint)?
   - Incremental (1-2 functions per day)?

---

**Recommendation:** Proceed with full optimization plan. The API should perform consistently across all functions, not just the ones that happen to be in our test suite.

---

**Date:** 2025-11-12
**Status:** ⏳ Awaiting approval to proceed
**Estimated Effort:** 2-3 days for complete optimization
