#!/usr/bin/env python3
"""
Demonstrates the logging performance issue with to_table() calls.

Shows that expensive operations in logger.debug() calls are executed
even when DEBUG logging is disabled.
"""

import logging
import time
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table


@dataclass
class ExampleParams:
    """Example dataclass similar to waveform parameters."""
    mass_1: float
    mass_2: float
    distance: float
    spin_1: float
    spin_2: float

    def to_table(self, header: str = "") -> str:
        """Expensive table rendering operation."""
        from io import StringIO
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=False)
        table = Table(title=header)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("mass_1", str(self.mass_1))
        table.add_row("mass_2", str(self.mass_2))
        table.add_row("distance", str(self.distance))
        table.add_row("spin_1", str(self.spin_1))
        table.add_row("spin_2", str(self.spin_2))

        # Render to string
        console.print(table)
        return buffer.getvalue()


def test_bad_logging_pattern(logger: logging.Logger, iterations: int = 100):
    """BAD: Expensive operation called regardless of log level."""
    params = ExampleParams(36.0, 29.0, 1000.0, 0.5, 0.3)

    start = time.perf_counter()
    for _ in range(iterations):
        # This is the ANTIPATTERN - to_table() is called even if DEBUG is disabled!
        logger.debug(params.to_table("waveform parameters"))
    elapsed = time.perf_counter() - start

    return elapsed


def test_good_logging_pattern(logger: logging.Logger, iterations: int = 100):
    """GOOD: Expensive operation only called when DEBUG is enabled."""
    params = ExampleParams(36.0, 29.0, 1000.0, 0.5, 0.3)

    start = time.perf_counter()
    for _ in range(iterations):
        # Check log level BEFORE expensive operation
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(params.to_table("waveform parameters"))
    elapsed = time.perf_counter() - start

    return elapsed


def main():
    print("=" * 80)
    print("DEMONSTRATING THE LOGGING PERFORMANCE ISSUE")
    print("=" * 80)

    iterations = 100

    # Test with WARNING level (DEBUG disabled - typical production scenario)
    logger_warning = logging.getLogger("test_warning")
    logger_warning.setLevel(logging.WARNING)

    print(f"\nTest scenario: {iterations} waveforms, DEBUG logging DISABLED (WARNING level)")
    print("-" * 80)

    print("\n1. BAD pattern: logger.debug(expensive_operation())")
    time_bad = test_bad_logging_pattern(logger_warning, iterations)
    print(f"   Time: {time_bad:.4f}s ({time_bad/iterations*1000:.2f}ms per waveform)")
    print(f"   Note: expensive_operation() is called {iterations} times!")

    print("\n2. GOOD pattern: if logger.isEnabledFor(DEBUG): logger.debug(...)")
    time_good = test_good_logging_pattern(logger_warning, iterations)
    print(f"   Time: {time_good:.4f}s ({time_good/iterations*1000:.2f}ms per waveform)")
    print(f"   Note: expensive_operation() is called 0 times (skipped!)")

    print(f"\n{'Speedup:':<20} {time_bad/time_good:.1f}x faster")
    print(f"{'Time saved:':<20} {time_bad - time_good:.4f}s ({(time_bad-time_good)/iterations*1000:.2f}ms per waveform)")

    # Test with DEBUG level enabled (to show it works correctly)
    logger_debug = logging.getLogger("test_debug")
    logger_debug.setLevel(logging.DEBUG)
    logger_debug.addHandler(logging.NullHandler())  # Don't actually print

    print("\n" + "=" * 80)
    print(f"Test scenario: {iterations} waveforms, DEBUG logging ENABLED")
    print("-" * 80)

    print("\n1. BAD pattern: logger.debug(expensive_operation())")
    time_bad_debug = test_bad_logging_pattern(logger_debug, iterations)
    print(f"   Time: {time_bad_debug:.4f}s")

    print("\n2. GOOD pattern: if logger.isEnabledFor(DEBUG): logger.debug(...)")
    time_good_debug = test_good_logging_pattern(logger_debug, iterations)
    print(f"   Time: {time_good_debug:.4f}s")

    print(f"\n{'Overhead difference:':<20} {abs(time_bad_debug - time_good_debug):.4f}s (minimal)")
    print(f"   Note: Both patterns have similar cost when DEBUG is enabled")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"When DEBUG is disabled (typical production):")
    print(f"  - BAD pattern wastes {time_bad:.4f}s per {iterations} waveforms")
    print(f"  - GOOD pattern wastes {time_good:.4f}s per {iterations} waveforms")
    print(f"  - Speedup: {time_bad/time_good:.1f}x")
    print(f"\nFor 1000 waveforms, this means:")
    print(f"  - BAD: ~{time_bad*10:.1f}s wasted on disabled logging")
    print(f"  - GOOD: ~{time_good*10:.1f}s wasted")
    print(f"  - Time saved: ~{(time_bad-time_good)*10:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
