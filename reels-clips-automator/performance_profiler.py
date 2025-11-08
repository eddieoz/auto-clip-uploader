"""
Performance Profiler Module

This module provides performance monitoring and profiling capabilities
for the talking face detection system.

Usage:
    from performance_profiler import PerformanceProfiler

    profiler = PerformanceProfiler()

    with profiler.measure("face_evaluation"):
        # ... code to profile ...
        pass

    report = profiler.get_report()
"""

import time
import json
from collections import defaultdict
from contextlib import contextmanager


class PerformanceProfiler:
    """
    Performance profiler for tracking timing metrics.

    Features:
    - Context manager for timing code blocks
    - Aggregates timing statistics (min, max, avg, total)
    - Tracks call counts
    - JSON export for reports
    - Cache hit/miss tracking
    """

    def __init__(self):
        """Initialize the performance profiler."""
        self.timings = defaultdict(list)
        self.counts = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()

    @contextmanager
    def measure(self, operation_name):
        """
        Context manager to measure execution time of a code block.

        Args:
            operation_name: Name of the operation being measured

        Usage:
            with profiler.measure("face_detection"):
                # ... code ...
                pass
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.timings[operation_name].append(elapsed)
            self.counts[operation_name] += 1

    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1

    def get_cache_hit_rate(self):
        """
        Calculate cache hit rate.

        Returns:
            float: Hit rate (0.0 to 1.0), or 0.0 if no cache operations
        """
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def get_stats(self, operation_name):
        """
        Get statistics for a specific operation.

        Args:
            operation_name: Name of the operation

        Returns:
            dict: Statistics including min, max, avg, total, count
        """
        timings = self.timings.get(operation_name, [])

        if not timings:
            return {
                'count': 0,
                'total_ms': 0.0,
                'avg_ms': 0.0,
                'min_ms': 0.0,
                'max_ms': 0.0
            }

        return {
            'count': len(timings),
            'total_ms': sum(timings) * 1000,
            'avg_ms': (sum(timings) / len(timings)) * 1000,
            'min_ms': min(timings) * 1000,
            'max_ms': max(timings) * 1000
        }

    def get_report(self):
        """
        Generate a complete performance report.

        Returns:
            dict: Complete performance report with all metrics
        """
        total_runtime = time.time() - self.start_time

        report = {
            'total_runtime_seconds': total_runtime,
            'cache': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.get_cache_hit_rate()
            },
            'operations': {}
        }

        # Add stats for each operation
        for operation in self.timings.keys():
            report['operations'][operation] = self.get_stats(operation)

        return report

    def save_report(self, filepath):
        """
        Save performance report to JSON file.

        Args:
            filepath: Path to save the report
        """
        report = self.get_report()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Performance report saved to: {filepath}")

    def print_summary(self):
        """Print a human-readable performance summary."""
        report = self.get_report()

        print("\n" + "=" * 70)
        print("Performance Summary")
        print("=" * 70)

        print(f"\n⏱️  Total Runtime: {report['total_runtime_seconds']:.2f}s")

        # Cache stats
        cache = report['cache']
        print(f"\n📦 Cache Performance:")
        print(f"   Hits: {cache['hits']}")
        print(f"   Misses: {cache['misses']}")
        print(f"   Hit Rate: {cache['hit_rate']*100:.1f}%")

        # Operation stats
        if report['operations']:
            print(f"\n📊 Operation Timing:")
            for op_name, stats in report['operations'].items():
                print(f"\n   {op_name}:")
                print(f"      Calls: {stats['count']}")
                print(f"      Total: {stats['total_ms']:.2f}ms")
                print(f"      Average: {stats['avg_ms']:.2f}ms")
                print(f"      Min: {stats['min_ms']:.2f}ms")
                print(f"      Max: {stats['max_ms']:.2f}ms")

        print("\n" + "=" * 70)


# Example usage and testing
if __name__ == "__main__":
    print("PerformanceProfiler Module")
    print("=" * 50)

    profiler = PerformanceProfiler()

    # Simulate some operations
    print("\nSimulating operations...")

    for i in range(10):
        with profiler.measure("face_detection"):
            time.sleep(0.01)  # Simulate 10ms operation

        if i % 2 == 0:
            profiler.record_cache_hit()
        else:
            profiler.record_cache_miss()

    for i in range(5):
        with profiler.measure("movement_detection"):
            time.sleep(0.005)  # Simulate 5ms operation

    # Print summary
    profiler.print_summary()

    # Save report
    profiler.save_report("performance_test_report.json")

    print("\n" + "=" * 50)
    print("Module test complete!")
