# Waveform Generation Benchmarking

## Overview

The `dingo-benchmark` tool compares waveform generation performance between `dingo-waveform` (refactored) and `dingo-gw` (original implementation).

## Installation

After installing dingo-waveform in development mode, the benchmark command is available:

```bash
pip install -e .
dingo-benchmark --help
```

## Usage

### Quick Test (10 waveforms, ~1 second)
```bash
dingo-benchmark --config examples/benchmark_quick.json --num-waveforms 10 --seed 42 --verbose
```

### Standard Benchmark (100 waveforms)
```bash
dingo-benchmark --config examples/benchmark_quick.json --num-waveforms 100 --seed 42
```

### Production Scale (1000 waveforms, longer approximant)
```bash
dingo-benchmark --config examples/benchmark_production.json --num-waveforms 1000 --seed 42
```

### MultibandedFrequencyDomain Test
```bash
dingo-benchmark --config examples/benchmark_multibanded.json --num-waveforms 100
```

### Save Results to JSON
```bash
dingo-benchmark --config config.json -n 500 --output results.json
```

### Per-Waveform Timing Statistics
```bash
dingo-benchmark --config config.json -n 100 --per-waveform-timing
```

## Configuration Files

Three example configurations are provided:

1. **`benchmark_quick.json`**: Fast test
   - IMRPhenomD approximant (simpler, faster)
   - UniformFrequencyDomain with delta_f=0.25 Hz
   - Domain size: 2049 bins
   - Good for quick testing

2. **`benchmark_production.json`**: Production-like
   - IMRPhenomXPHM approximant (higher-order modes)
   - UniformFrequencyDomain with delta_f=0.125 Hz
   - Domain size: 8193 bins
   - Realistic parameter ranges

3. **`benchmark_multibanded.json`**: MultibandedFrequencyDomain
   - IMRPhenomXPHM approximant
   - 4 bands with dyadic spacing
   - Tests decimation performance

## Benchmark Results

### Current Findings (as of 2025-11-10)

**⚠️ PERFORMANCE REGRESSION DETECTED**

Initial benchmarking reveals that **dingo-waveform is significantly slower** than dingo-gw:

```
Configuration:
  Waveforms:      100
  Domain:         UniformFrequencyDomain (2049 bins)
  Approximant:    IMRPhenomD

Package              Setup (s)    Generation (s)  Total (s)    Wf/s
--------------------------------------------------------------------------------
dingo-gw             0.0000       0.0250          0.0251       3992.64
dingo-waveform       0.0021       1.5262          1.5282       65.52

Comparison:
  Total speedup:      0.016x (~60x slower)
  Generation speedup: 0.016x
  Time saved:         -1.503s
  ⚠️  dingo-waveform is 98.4% slower
```

**Analysis:**
- **~60x slowdown** in waveform generation
- Overhead appears to scale with number of waveforms
- Setup time is negligible for both
- Generation time dominates

### Potential Causes

Several factors may contribute to the slowdown:

1. **Parameter Conversion Overhead**
   - Converting dict → WaveformParameters dataclass for each waveform
   - Validation and type checking overhead

2. **Function Call Overhead**
   - Additional abstraction layers (multipledispatch, dataclasses)
   - Type annotations and validation

3. **Domain/Frequency Array Caching**
   - Different caching strategies between implementations
   - Possible recomputation of cached values

4. **LAL Interface Differences**
   - Different ways of calling LALSimulation functions
   - Parameter preparation and conversion

5. **Python Overhead**
   - More Pythonic code may have overhead vs optimized loops
   - Dataclass operations vs dict operations

### Recommended Investigations

To identify and fix the performance regression:

1. **Profile with cProfile**
   ```bash
   python -m cProfile -o profile.stats -m dingo_waveform.benchmark --config config.json -n 100
   ```

2. **Line-level profiling** with line_profiler
   - Profile `generate_hplus_hcross()` method
   - Identify hot spots in the call stack

3. **Compare LAL call patterns**
   - Check if LAL functions are called the same number of times
   - Verify parameter preparation is efficient

4. **Isolate dataclass overhead**
   - Test with direct LAL calls vs wrapped calls
   - Measure dict → WaveformParameters conversion time

5. **Check domain caching**
   - Verify sample_frequencies are cached properly
   - Ensure no redundant computations per waveform

## Understanding the Output

### Timing Breakdown

- **Setup time**: Time to initialize domain and waveform generator
- **Generation time**: Time to generate all waveforms
- **Total time**: Setup + Generation
- **Wf/s**: Waveforms per second (throughput)

### Speedup Metrics

- **Total speedup**: Ratio of total times (>1 means dingo-waveform is faster)
- **Generation speedup**: Ratio of generation times only
- **Time saved**: Difference in total time (positive = faster)

### Exit Codes

- `0`: Success (speedup >= 0.95, within 5% of dingo-gw)
- `1`: Performance regression (>5% slower than dingo-gw)

## JSON Output Format

When using `--output`, results are saved in JSON format:

```json
{
  "configuration": {
    "num_waveforms": 100,
    "domain_type": "UniformFrequencyDomain",
    "approximant": "IMRPhenomD",
    "domain_size": 2049
  },
  "dingo_gw": {
    "setup_time_s": 0.0000,
    "generation_time_s": 0.0250,
    "total_time_s": 0.0251,
    "time_per_waveform_s": 0.00025,
    "waveforms_per_second": 3992.64
  },
  "dingo_waveform": {
    "setup_time_s": 0.0021,
    "generation_time_s": 1.5262,
    "total_time_s": 1.5282,
    "time_per_waveform_s": 0.01526,
    "waveforms_per_second": 65.52
  },
  "comparison": {
    "total_speedup": 0.016,
    "generation_speedup": 0.016,
    "time_difference_s": -1.503,
    "percent_faster": -98.4
  }
}
```

## Continuous Monitoring

The benchmark tool can be integrated into CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Performance benchmark
  run: |
    dingo-benchmark --config examples/benchmark_quick.json -n 100 --output benchmark.json
    # Will fail if >5% slower than dingo-gw
```

## Next Steps

1. **Profile and optimize** the identified performance bottleneck
2. **Run comprehensive benchmarks** across:
   - Different approximants (IMRPhenomD, IMRPhenomXPHM, SEOBNRv5PHM)
   - Different domain types (Uniform, Multibanded)
   - Different domain sizes
   - Different parameter ranges
3. **Investigate LAL interface** for optimization opportunities
4. **Consider caching strategies** for parameter conversion
5. **Benchmark SVD compression** separately (already verified for correctness)

## Contributing

When making performance-related changes:

1. Run benchmarks before and after
2. Document performance impact
3. Include benchmark results in PR description
4. Consider adding specific benchmarks for new features

## Notes

- **Timing precision**: Uses `time.perf_counter()` for high-resolution timing
- **Reproducibility**: Use `--seed` for consistent results
- **Warm-up**: First waveform may be slower due to LAL initialization
- **System load**: Run on dedicated machine for accurate results
- **Python version**: Performance may vary with Python version

## References

- LALSimulation documentation: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
- Python profiling: https://docs.python.org/3/library/profile.html
- Performance optimization guide: (to be added)
