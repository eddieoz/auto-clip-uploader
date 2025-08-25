#!/usr/bin/env python3
"""
Phase 5: Testing & Validation - Performance Benchmark Suite

Story 5.2: Performance optimization and benchmarks.

This suite provides comprehensive performance analysis and optimization:
- Processing speed benchmarks across content types
- Memory usage profiling and optimization
- Scalability analysis under various loads
- Performance regression detection
- Optimization recommendations
"""

import sys
import os
import time
import psutil
import gc
import json
import statistics
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topic_analyzer import TopicAnalyzer
from tests.test_data.sample_transcripts import get_test_scenario, get_all_scenarios


class PerformanceBenchmark:
    """Comprehensive performance benchmarking and optimization suite."""
    
    def __init__(self):
        self.benchmark_results = []
        self.optimization_recommendations = []
        
    def run_processing_speed_benchmarks(self) -> Dict:
        """Benchmark processing speed across different content types and sizes."""
        print("⚡ Running Processing Speed Benchmarks")
        print("=" * 50)
        
        scenarios = get_all_scenarios()
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        
        speed_results = {}
        
        for scenario_name, scenario in scenarios.items():
            print(f"  📊 Benchmarking: {scenario_name}")
            
            # Warm up
            analyzer.analyze_transcript(scenario['transcript'])
            
            # Run multiple iterations for accurate timing
            times = []
            for i in range(10):  # 10 iterations for statistical reliability
                start_time = time.perf_counter()
                try:
                    results = analyzer.analyze_transcript(scenario['transcript'])
                    end_time = time.perf_counter()
                    
                    if 'error' not in results:
                        times.append(end_time - start_time)
                except Exception as e:
                    print(f"    ❌ Error in iteration {i}: {str(e)}")
            
            if times:
                avg_time = statistics.mean(times)
                min_time = min(times)
                max_time = max(times)
                std_dev = statistics.stdev(times) if len(times) > 1 else 0
                
                # Calculate content processing metrics
                content_size = len(scenario['transcript'])
                chars_per_second = content_size / avg_time if avg_time > 0 else 0
                
                speed_results[scenario_name] = {
                    'avg_time_sec': avg_time,
                    'min_time_sec': min_time,
                    'max_time_sec': max_time,
                    'std_dev_sec': std_dev,
                    'content_size_chars': content_size,
                    'chars_per_second': chars_per_second,
                    'iterations': len(times)
                }
                
                print(f"    ✅ Avg: {avg_time*1000:.2f}ms, Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
                print(f"    📈 Processing speed: {chars_per_second:.0f} chars/sec")
        
        # Calculate overall statistics
        all_avg_times = [result['avg_time_sec'] for result in speed_results.values()]
        all_chars_per_sec = [result['chars_per_second'] for result in speed_results.values()]
        
        overall_stats = {
            'overall_avg_time_sec': statistics.mean(all_avg_times),
            'overall_min_time_sec': min(all_avg_times),
            'overall_max_time_sec': max(all_avg_times),
            'overall_avg_chars_per_sec': statistics.mean(all_chars_per_sec),
            'total_scenarios_tested': len(speed_results)
        }
        
        print(f"\n📊 Overall Performance Statistics:")
        print(f"   • Average processing time: {overall_stats['overall_avg_time_sec']*1000:.2f}ms")
        print(f"   • Fastest processing time: {overall_stats['overall_min_time_sec']*1000:.2f}ms")
        print(f"   • Slowest processing time: {overall_stats['overall_max_time_sec']*1000:.2f}ms")
        print(f"   • Average processing speed: {overall_stats['overall_avg_chars_per_sec']:.0f} chars/sec")
        
        return {
            'individual_results': speed_results,
            'overall_statistics': overall_stats
        }
    
    def run_memory_usage_profiling(self) -> Dict:
        """Profile memory usage during processing."""
        print("\n🧠 Running Memory Usage Profiling")
        print("=" * 50)
        
        import tracemalloc
        
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        scenario = get_test_scenario('crypto_analysis')  # Use representative scenario
        
        memory_profiles = []
        
        # Profile memory usage across multiple processing cycles
        for cycle in range(5):
            print(f"  🔍 Memory profiling cycle {cycle + 1}/5")
            
            # Start memory tracing
            tracemalloc.start()
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            initial_tracemalloc = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
            
            # Process content multiple times in this cycle
            for i in range(20):
                try:
                    results = analyzer.analyze_transcript(scenario['transcript'])
                except:
                    pass
            
            # Measure peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_tracemalloc = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
            
            # Force garbage collection and measure cleanup
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_tracemalloc = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
            
            tracemalloc.stop()
            
            memory_profile = {
                'cycle': cycle + 1,
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': peak_memory - initial_memory,
                'memory_cleanup_mb': peak_memory - final_memory,
                'initial_tracemalloc_mb': initial_tracemalloc,
                'peak_tracemalloc_mb': peak_tracemalloc,
                'final_tracemalloc_mb': final_tracemalloc
            }
            
            memory_profiles.append(memory_profile)
            
            print(f"    📊 Initial: {initial_memory:.1f}MB → Peak: {peak_memory:.1f}MB → Final: {final_memory:.1f}MB")
        
        # Calculate memory statistics
        avg_memory_increase = statistics.mean([p['memory_increase_mb'] for p in memory_profiles])
        avg_memory_cleanup = statistics.mean([p['memory_cleanup_mb'] for p in memory_profiles])
        max_memory_usage = max([p['peak_memory_mb'] for p in memory_profiles])
        
        memory_stats = {
            'profiles': memory_profiles,
            'avg_memory_increase_mb': avg_memory_increase,
            'avg_memory_cleanup_mb': avg_memory_cleanup,
            'max_memory_usage_mb': max_memory_usage,
            'memory_efficiency_score': max(0, 100 - avg_memory_increase * 2)  # Custom efficiency score
        }
        
        print(f"\n📊 Memory Usage Statistics:")
        print(f"   • Average memory increase: {avg_memory_increase:.1f}MB")
        print(f"   • Average memory cleanup: {avg_memory_cleanup:.1f}MB")
        print(f"   • Maximum memory usage: {max_memory_usage:.1f}MB")
        print(f"   • Memory efficiency score: {memory_stats['memory_efficiency_score']:.1f}/100")
        
        # Generate memory optimization recommendations
        if avg_memory_increase > 10:
            self.optimization_recommendations.append({
                'category': 'memory',
                'issue': 'High memory increase during processing',
                'recommendation': 'Consider implementing memory pooling or object recycling',
                'impact': 'medium'
            })
        
        if avg_memory_cleanup < avg_memory_increase * 0.8:
            self.optimization_recommendations.append({
                'category': 'memory',
                'issue': 'Poor memory cleanup efficiency',
                'recommendation': 'Add explicit garbage collection at processing boundaries',
                'impact': 'medium'
            })
        
        return memory_stats
    
    def run_scalability_analysis(self) -> Dict:
        """Analyze system scalability under various loads."""
        print("\n📈 Running Scalability Analysis")
        print("=" * 50)
        
        scenario = get_test_scenario('crypto_analysis')
        scalability_results = {}
        
        # Test different concurrent load levels
        load_levels = [1, 2, 5, 10, 20]
        
        for concurrent_requests in load_levels:
            print(f"  🔄 Testing {concurrent_requests} concurrent requests")
            
            def process_request():
                analyzer = TopicAnalyzer(use_enhanced_audio=True)
                start_time = time.perf_counter()
                try:
                    results = analyzer.analyze_transcript(scenario['transcript'])
                    end_time = time.perf_counter()
                    return end_time - start_time, 'error' not in results
                except Exception as e:
                    end_time = time.perf_counter()
                    return end_time - start_time, False
            
            # Run concurrent requests
            with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                start_time = time.perf_counter()
                futures = [executor.submit(process_request) for _ in range(concurrent_requests)]
                results = [future.result() for future in futures]
                end_time = time.perf_counter()
            
            # Analyze results
            processing_times = [result[0] for result in results]
            success_count = sum(1 for result in results if result[1])
            
            total_wall_time = end_time - start_time
            avg_processing_time = statistics.mean(processing_times)
            throughput = concurrent_requests / total_wall_time
            success_rate = (success_count / concurrent_requests) * 100
            
            # Calculate efficiency metrics
            theoretical_max_throughput = concurrent_requests / avg_processing_time
            efficiency = (throughput / theoretical_max_throughput) * 100 if theoretical_max_throughput > 0 else 0
            
            scalability_results[concurrent_requests] = {
                'concurrent_requests': concurrent_requests,
                'total_wall_time_sec': total_wall_time,
                'avg_processing_time_sec': avg_processing_time,
                'throughput_req_per_sec': throughput,
                'success_rate_percent': success_rate,
                'efficiency_percent': efficiency,
                'successful_requests': success_count
            }
            
            print(f"    📊 Throughput: {throughput:.2f} req/sec, Success: {success_rate:.1f}%, Efficiency: {efficiency:.1f}%")
        
        # Analyze scalability trends
        throughputs = [result['throughput_req_per_sec'] for result in scalability_results.values()]
        efficiencies = [result['efficiency_percent'] for result in scalability_results.values()]
        
        # Calculate scalability score based on throughput growth and efficiency maintenance
        max_throughput = max(throughputs)
        min_efficiency = min(efficiencies)
        scalability_score = (max_throughput * min_efficiency) / 100
        
        scalability_stats = {
            'load_test_results': scalability_results,
            'max_throughput_req_per_sec': max_throughput,
            'min_efficiency_percent': min_efficiency,
            'scalability_score': scalability_score,
            'optimal_concurrent_requests': max(scalability_results.keys(), 
                                             key=lambda k: scalability_results[k]['throughput_req_per_sec'])
        }
        
        print(f"\n📊 Scalability Statistics:")
        print(f"   • Maximum throughput: {max_throughput:.2f} req/sec")
        print(f"   • Minimum efficiency: {min_efficiency:.1f}%")
        print(f"   • Scalability score: {scalability_score:.2f}")
        print(f"   • Optimal concurrent requests: {scalability_stats['optimal_concurrent_requests']}")
        
        # Generate scalability optimization recommendations
        if min_efficiency < 70:
            self.optimization_recommendations.append({
                'category': 'scalability',
                'issue': 'Efficiency degradation under high load',
                'recommendation': 'Implement connection pooling and optimize resource sharing',
                'impact': 'high'
            })
        
        if max_throughput < 10:
            self.optimization_recommendations.append({
                'category': 'scalability',
                'issue': 'Low maximum throughput',
                'recommendation': 'Profile bottlenecks and optimize critical path performance',
                'impact': 'high'
            })
        
        return scalability_stats
    
    def run_content_size_performance_analysis(self) -> Dict:
        """Analyze performance impact of different content sizes."""
        print("\n📏 Running Content Size Performance Analysis")
        print("=" * 50)
        
        base_scenario = get_test_scenario('crypto_analysis')
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        
        # Create content of different sizes
        size_multipliers = [0.5, 1, 2, 3, 5]
        size_results = {}
        
        for multiplier in size_multipliers:
            # Create content of different sizes
            if multiplier < 1:
                # Truncate content
                lines = base_scenario['transcript'].split('\n')
                truncated_lines = lines[:int(len(lines) * multiplier)]
                test_content = '\n'.join(truncated_lines)
            else:
                # Repeat content
                test_content = base_scenario['transcript'] * int(multiplier)
            
            content_size = len(test_content)
            
            print(f"  📊 Testing content size: {content_size:,} characters ({multiplier}x)")
            
            # Run multiple iterations
            times = []
            for i in range(5):
                start_time = time.perf_counter()
                try:
                    results = analyzer.analyze_transcript(test_content)
                    end_time = time.perf_counter()
                    
                    if 'error' not in results:
                        times.append(end_time - start_time)
                except:
                    pass
            
            if times:
                avg_time = statistics.mean(times)
                processing_rate = content_size / avg_time if avg_time > 0 else 0
                
                size_results[multiplier] = {
                    'size_multiplier': multiplier,
                    'content_size_chars': content_size,
                    'avg_processing_time_sec': avg_time,
                    'processing_rate_chars_per_sec': processing_rate,
                    'iterations': len(times)
                }
                
                print(f"    ✅ Processing time: {avg_time*1000:.1f}ms, Rate: {processing_rate:.0f} chars/sec")
        
        # Analyze scaling behavior
        if len(size_results) >= 3:
            # Check if processing time scales linearly with content size
            sizes = [result['content_size_chars'] for result in size_results.values()]
            times = [result['avg_processing_time_sec'] for result in size_results.values()]
            
            # Calculate correlation coefficient (simplified)
            mean_size = statistics.mean(sizes)
            mean_time = statistics.mean(times)
            
            numerator = sum((s - mean_size) * (t - mean_time) for s, t in zip(sizes, times))
            denominator = (sum((s - mean_size)**2 for s in sizes) * sum((t - mean_time)**2 for t in times))**0.5
            
            correlation = numerator / denominator if denominator > 0 else 0
            
            # Determine scaling behavior
            if correlation > 0.8:
                scaling_behavior = "Linear scaling (expected)"
            elif correlation > 0.5:
                scaling_behavior = "Moderate scaling"
            else:
                scaling_behavior = "Sub-linear scaling (good optimization)"
        else:
            correlation = 0
            scaling_behavior = "Insufficient data"
        
        size_stats = {
            'size_test_results': size_results,
            'scaling_correlation': correlation,
            'scaling_behavior': scaling_behavior,
            'size_range_tested': f"{min(result['content_size_chars'] for result in size_results.values()):,} - {max(result['content_size_chars'] for result in size_results.values()):,} chars"
        }
        
        print(f"\n📊 Content Size Analysis:")
        print(f"   • Size range tested: {size_stats['size_range_tested']}")
        print(f"   • Scaling correlation: {correlation:.3f}")
        print(f"   • Scaling behavior: {scaling_behavior}")
        
        # Generate size-based optimization recommendations
        if correlation > 0.9:
            self.optimization_recommendations.append({
                'category': 'performance',
                'issue': 'Processing time increases significantly with content size',
                'recommendation': 'Implement content chunking or streaming processing',
                'impact': 'medium'
            })
        
        return size_stats
    
    def run_regression_testing(self, baseline_results: Dict = None) -> Dict:
        """Run performance regression testing against baseline."""
        print("\n🔍 Running Performance Regression Testing")
        print("=" * 50)
        
        # If no baseline provided, use current performance as baseline
        if baseline_results is None:
            print("  📊 No baseline provided, establishing current performance as baseline")
            baseline_results = self._establish_baseline()
        
        # Run current performance tests
        current_results = self._establish_baseline()
        
        # Compare results
        regression_analysis = {}
        
        for test_type in ['processing_speed', 'memory_usage', 'throughput']:
            if test_type in baseline_results and test_type in current_results:
                baseline_value = baseline_results[test_type]
                current_value = current_results[test_type]
                
                # Calculate performance change
                if baseline_value > 0:
                    change_percent = ((current_value - baseline_value) / baseline_value) * 100
                else:
                    change_percent = 0
                
                regression_analysis[test_type] = {
                    'baseline_value': baseline_value,
                    'current_value': current_value,
                    'change_percent': change_percent,
                    'regression_detected': change_percent < -10  # More than 10% slower is regression
                }
        
        # Determine overall regression status
        regressions_detected = sum(1 for analysis in regression_analysis.values() if analysis['regression_detected'])
        overall_regression = regressions_detected > 0
        
        print(f"  📊 Regression Analysis Results:")
        for test_type, analysis in regression_analysis.items():
            status = "🔴 REGRESSION" if analysis['regression_detected'] else "✅ OK"
            print(f"    • {test_type}: {analysis['change_percent']:+.1f}% {status}")
        
        print(f"\n  {'❌ REGRESSIONS DETECTED' if overall_regression else '✅ NO REGRESSIONS DETECTED'}")
        
        return {
            'regression_analysis': regression_analysis,
            'overall_regression': overall_regression,
            'regressions_detected': regressions_detected
        }
    
    def _establish_baseline(self) -> Dict:
        """Establish baseline performance metrics."""
        scenario = get_test_scenario('crypto_analysis')
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        
        # Quick baseline measurements
        times = []
        for _ in range(5):
            start_time = time.perf_counter()
            try:
                analyzer.analyze_transcript(scenario['transcript'])
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except:
                pass
        
        avg_processing_speed = statistics.mean(times) if times else 0
        
        # Quick memory measurement
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        for _ in range(10):
            try:
                analyzer.analyze_transcript(scenario['transcript'])
            except:
                pass
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_usage = peak_memory - initial_memory
        
        # Quick throughput measurement
        start_time = time.perf_counter()
        successful = 0
        for _ in range(20):
            try:
                results = analyzer.analyze_transcript(scenario['transcript'])
                if 'error' not in results:
                    successful += 1
            except:
                pass
        
        total_time = time.perf_counter() - start_time
        throughput = successful / total_time if total_time > 0 else 0
        
        return {
            'processing_speed': avg_processing_speed,
            'memory_usage': memory_usage,
            'throughput': throughput
        }
    
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report."""
        report = f"""
🔧 PERFORMANCE OPTIMIZATION REPORT
{'=' * 50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OPTIMIZATION RECOMMENDATIONS:
"""
        
        if not self.optimization_recommendations:
            report += "✅ No optimization recommendations - performance is optimal!\n"
        else:
            # Group recommendations by category
            categories = {}
            for rec in self.optimization_recommendations:
                cat = rec['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(rec)
            
            for category, recommendations in categories.items():
                report += f"\n🔍 {category.upper()} OPTIMIZATIONS:\n"
                for rec in recommendations:
                    impact_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    report += f"  {impact_icon.get(rec['impact'], '⚪')} {rec['issue']}\n"
                    report += f"    → {rec['recommendation']}\n"
        
        report += f"\n✨ PERFORMANCE SUMMARY:\n"
        report += f"• Total recommendations: {len(self.optimization_recommendations)}\n"
        report += f"• High impact items: {sum(1 for r in self.optimization_recommendations if r['impact'] == 'high')}\n"
        report += f"• Medium impact items: {sum(1 for r in self.optimization_recommendations if r['impact'] == 'medium')}\n"
        report += f"• Low impact items: {sum(1 for r in self.optimization_recommendations if r['impact'] == 'low')}\n"
        
        return report
    
    def run_comprehensive_benchmark_suite(self) -> Dict:
        """Run the complete performance benchmark suite."""
        print("⚡ STARTING COMPREHENSIVE PERFORMANCE BENCHMARK SUITE")
        print("=" * 70)
        
        benchmark_start = time.perf_counter()
        
        # Run all benchmark suites
        results = {}
        
        try:
            results['processing_speed'] = self.run_processing_speed_benchmarks()
        except Exception as e:
            print(f"❌ Processing speed benchmark failed: {str(e)}")
            results['processing_speed'] = {'error': str(e)}
        
        try:
            results['memory_usage'] = self.run_memory_usage_profiling()
        except Exception as e:
            print(f"❌ Memory usage profiling failed: {str(e)}")
            results['memory_usage'] = {'error': str(e)}
        
        try:
            results['scalability'] = self.run_scalability_analysis()
        except Exception as e:
            print(f"❌ Scalability analysis failed: {str(e)}")
            results['scalability'] = {'error': str(e)}
        
        try:
            results['content_size'] = self.run_content_size_performance_analysis()
        except Exception as e:
            print(f"❌ Content size analysis failed: {str(e)}")
            results['content_size'] = {'error': str(e)}
        
        try:
            results['regression'] = self.run_regression_testing()
        except Exception as e:
            print(f"❌ Regression testing failed: {str(e)}")
            results['regression'] = {'error': str(e)}
        
        benchmark_duration = time.perf_counter() - benchmark_start
        
        # Generate summary
        print("\n" + "=" * 70)
        print("🏁 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 70)
        
        print(f"Total benchmark duration: {benchmark_duration:.2f} seconds\n")
        
        # Print summary of each benchmark
        for benchmark_name, benchmark_results in results.items():
            if 'error' not in benchmark_results:
                print(f"✅ {benchmark_name.replace('_', ' ').title()}: COMPLETED")
            else:
                print(f"❌ {benchmark_name.replace('_', ' ').title()}: FAILED - {benchmark_results['error']}")
        
        # Generate and display optimization report
        optimization_report = self.generate_optimization_report()
        print(optimization_report)
        
        # Overall assessment
        successful_benchmarks = sum(1 for r in results.values() if 'error' not in r)
        total_benchmarks = len(results)
        success_rate = (successful_benchmarks / total_benchmarks) * 100
        
        print(f"\n🎯 BENCHMARK SUITE SUMMARY:")
        print(f"• Successful benchmarks: {successful_benchmarks}/{total_benchmarks} ({success_rate:.1f}%)")
        print(f"• Performance issues identified: {len(self.optimization_recommendations)}")
        print(f"• Overall performance: {'EXCELLENT' if success_rate >= 90 and len(self.optimization_recommendations) <= 2 else 'GOOD' if success_rate >= 80 else 'NEEDS IMPROVEMENT'}")
        
        results['benchmark_summary'] = {
            'total_duration': benchmark_duration,
            'successful_benchmarks': successful_benchmarks,
            'total_benchmarks': total_benchmarks,
            'success_rate': success_rate,
            'optimization_recommendations_count': len(self.optimization_recommendations)
        }
        
        return results


def main():
    """Run the comprehensive performance benchmark suite."""
    benchmark = PerformanceBenchmark()
    return benchmark.run_comprehensive_benchmark_suite()


if __name__ == "__main__":
    results = main()
    
    # Save results to file for future regression testing
    with open('performance_baseline.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Benchmark results saved to 'performance_baseline.json'")
    print("🚀 Performance benchmarking complete!")