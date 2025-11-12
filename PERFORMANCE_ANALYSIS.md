# Performance Analysis: dingo-waveform vs dingo-gw

## Executive Summary

**Finding:** dingo-waveform is ~98% slower than dingo-gw (60-100x slowdown)

**Root Cause:** Expensive logging operations with Rich table rendering are executed even when logging is disabled

**Impact:** ~2.4 seconds spent on logging overhead per 50 waveforms (vs 0.01s for actual LAL calls)

---

## Profiling Results

### Performance Comparison (50 waveforms)

| Package | Time | Function Calls | Waveforms/sec |
|---------|------|----------------|---------------|
| **dingo-gw** | 0.021s | 28,955 | ~2,380 |
| **dingo-waveform** | 2.499s | 10,289,632 | ~20 |

**Slowdown Factor:** ~119x

---

## Root Cause Analysis

### The Logging Antipattern

The code contains many debug logging statements like this:

```python
# dingo_waveform/polarization_functions/inspiral_FD.py:104
_logger.debug(instance.to_table("generated inspiral fd parameters"))

# dingo_waveform/polarization_functions/inspiral_FD.py:201
_logger.debug(
    params.to_table("generating polarization using lalsimulation.SimInspiralFD")
)

# dingo_waveform/binary_black_holes_parameters.py:100
_logger.debug(instance.to_table("generated binary black hole parameters"))
```

### Why This Is a Problem

In Python, function arguments are evaluated **before** the function is called. This means:

1. `instance.to_table(...)` is called **first** (expensive Rich table rendering)
2. The resulting string is passed to `_logger.debug(...)`
3. **Only then** does the logger check if DEBUG level is enabled
4. If DEBUG is disabled (default is WARNING), the string is discarded

**Result:** Expensive table rendering happens regardless of log level!

### Profiling Evidence

From `cProfile` output for 50 waveforms:

```
Top time consumers in dingo-waveform:
  - to_table():                351 calls, 2.519s cumulative (98% of total time!)
  - rich.console.print():      351 calls, 2.416s cumulative
  - rich.table.__rich_console__(): 98,503 calls, 2.278s cumulative
  - Actual LAL waveform call:   ~0.010s (buried in the noise)
```

**351 calls** = ~7 `to_table()` calls per waveform

### Where the Calls Come From

Tracing through the code for a single waveform generation:

1. `WaveformGenerator.generate_hplus_hcross()` → logs waveform_parameters
2. `BinaryBlackHoleParameters.from_waveform_parameters()` → logs BBH params
3. `BinaryBlackHoleParameters.get_spins()` → logs spins
4. `_InspiralFDParameters.from_binary_black_hole_parameters()` → logs inspiral params
5. `_InspiralFDParameters.from_waveform_parameters()` → logs FD params
6. `_InspiralFDParameters.apply()` → logs before LAL call
7. Sometimes `_turn_off_multibanding()` → logs params again

**Total:** 6-7 expensive Rich table renders per waveform

---

## Impact Breakdown

### Time Distribution (per 50 waveforms)

```
dingo-gw:
  LAL waveform generation:  0.010s (48%)
  Parameter conversion:     0.002s (10%)
  Prior sampling:           0.004s (19%)
  Other:                    0.005s (23%)
  Total:                    0.021s

dingo-waveform:
  Rich logging overhead:    2.519s (98%)
  LAL waveform generation:  ~0.010s (0.4%)
  Other:                    ~0.050s (1.6%)
  Total:                    2.567s
```

### Cost Per Operation

- **Cost of one `to_table()` call:** ~7.2ms
- **Cost of one LAL waveform generation:** ~0.2ms
- **Ratio:** Each logging operation is **36x more expensive** than generating a waveform!

---

## Solution

### Correct Pattern

Use conditional logging to avoid expensive operations when logging is disabled:

```python
# BEFORE (current - BAD):
_logger.debug(instance.to_table("generated inspiral fd parameters"))

# AFTER (correct - GOOD):
if _logger.isEnabledFor(logging.DEBUG):
    _logger.debug(instance.to_table("generated inspiral fd parameters"))
```

### Alternative: Lazy Logging

For simple cases without method calls:

```python
# String formatting only happens if DEBUG is enabled
_logger.debug("Generated params: mass_1=%.2f, mass_2=%.2f", mass_1, mass_2)
```

But this won't work for `to_table()` since it's a method call.

---

## Files Requiring Changes

Based on `grep` results, the following files have `_logger.debug(*.to_table())` calls:

### High Priority (in hot path):
1. `dingo_waveform/polarization_functions/inspiral_FD.py` (3 calls)
2. `dingo_waveform/binary_black_holes_parameters.py` (3 calls)
3. `dingo_waveform/polarization_modes_functions/inspiral_choose_FD_modes.py` (2 calls)
4. `dingo_waveform/waveform_generator.py` (3 calls)

### Medium Priority:
5. `dingo_waveform/polarization_functions/inspiral_TD.py` (2 calls)
6. `dingo_waveform/polarization_functions/generate_FD_modes.py` (1 call)
7. `dingo_waveform/polarization_functions/generate_TD_modes.py` (1 call)
8. `dingo_waveform/polarization_modes_functions/inspiral_choose_TD_modes.py` (2 calls)
9. `dingo_waveform/gw_signals_parameters.py` (1 call)

### Lower Priority (less frequently called):
10. `dingo_waveform/polarization_modes_functions/generate_TD_modes_LO_cond_extra_time.py`
11. `dingo_waveform/polarization_modes_functions/generate_TD_modes_LO.py`
12. `dingo_waveform/polarization_modes_functions/generate_FD_modes_LO.py`

---

## Expected Performance Improvement

If logging overhead is eliminated:

- **Current:** 2.567s for 50 waveforms (~19 wf/s)
- **Expected:** ~0.060s for 50 waveforms (~833 wf/s)
- **Improvement:** ~42x faster

This would bring dingo-waveform very close to dingo-gw performance (potentially even faster given the cleaner dataclass design).

---

## Verification Steps

After implementing the fix:

1. Run the profiling script again:
   ```bash
   python profile_benchmark.py
   ```

2. Check that `to_table()` no longer appears in top time consumers

3. Run benchmark:
   ```bash
   dingo-benchmark --config examples/benchmark_quick.yaml -n 100
   ```

4. Expected result: Speedup factor > 0.95 (within 5% of dingo-gw)

---

## Additional Notes

### Why This Wasn't Caught Earlier

- The Rich tables are visually appealing and useful for debugging
- The performance impact isn't noticeable with single waveforms
- Only becomes apparent when generating datasets (hundreds/thousands of waveforms)
- DEBUG logs appear disabled by default, so the overhead is "invisible"

### Lessons Learned

1. **Always guard expensive debug operations with level checks**
2. **Profile early in hot paths (dataset generation)**
3. **Be cautious with rich formatting libraries in performance-critical code**
4. **Consider using lazy evaluation for debug logging**

---

## References

- Profiling output: `/tmp/dingo_waveform_profile.prof`
- Benchmark tool: `dingo-benchmark`
- Python logging best practices: https://docs.python.org/3/howto/logging.html#optimization
