#!/usr/bin/env python3
"""
Phase 5: Testing & Validation - Production Test Suite

Story 5.1: Production environment testing with comprehensive validation scenarios.

This suite tests the complete system under production-like conditions:
- Large-scale content processing
- Real-world edge cases and variations
- Performance under load
- Error recovery and graceful degradation
- Production workflow integration
"""

import sys
import os
import json
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topic_analyzer import TopicAnalyzer
from tests.test_data.sample_transcripts import get_test_scenario, get_all_scenarios


class ProductionTestResults:
    """Collects and analyzes production test results."""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = []
        self.error_cases = []
        self.start_time = None
        self.end_time = None
        
    def add_result(self, test_name: str, success: bool, duration: float, 
                   metadata: Dict = None, error: str = None):
        """Add a test result."""
        result = {
            'test_name': test_name,
            'success': success,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'error': error
        }
        self.test_results.append(result)
        
        if not success and error:
            self.error_cases.append({
                'test_name': test_name,
                'error': error,
                'timestamp': result['timestamp']
            })
    
    def add_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Add a performance metric."""
        self.performance_metrics.append({
            'metric': metric_name,
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_success_rate(self) -> float:
        """Calculate overall success rate."""
        if not self.test_results:
            return 0.0
        successful = sum(1 for r in self.test_results if r['success'])
        return (successful / len(self.test_results)) * 100
    
    def get_average_processing_time(self) -> float:
        """Calculate average processing time."""
        if not self.test_results:
            return 0.0
        durations = [r['duration'] for r in self.test_results if r['success']]
        return sum(durations) / len(durations) if durations else 0.0
    
    def get_error_patterns(self) -> Dict[str, int]:
        """Analyze error patterns."""
        error_counts = {}
        for error_case in self.error_cases:
            error_type = error_case['error'].split(':')[0] if ':' in error_case['error'] else error_case['error']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        total_time = (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0
        
        report = f"""
🧪 PRODUCTION TEST SUITE REPORT
{'=' * 50}
Test Period: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'N/A'} - {self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else 'N/A'}
Total Duration: {total_time:.2f} seconds

📊 OVERALL METRICS:
• Tests Run: {len(self.test_results)}
• Success Rate: {self.get_success_rate():.1f}%
• Average Processing Time: {self.get_average_processing_time():.3f}s
• Errors Encountered: {len(self.error_cases)}

📈 PERFORMANCE METRICS:
"""
        for metric in self.performance_metrics:
            report += f"• {metric['metric']}: {metric['value']:.3f} {metric['unit']}\n"
        
        if self.error_cases:
            report += f"\n❌ ERROR ANALYSIS:\n"
            error_patterns = self.get_error_patterns()
            for error_type, count in error_patterns.items():
                report += f"• {error_type}: {count} occurrences\n"
        
        report += f"\n✅ PRODUCTION READINESS: {'PASS' if self.get_success_rate() >= 95 else 'NEEDS IMPROVEMENT'}\n"
        
        return report


class ProductionTestSuite:
    """Comprehensive production testing suite."""
    
    def __init__(self):
        self.results = ProductionTestResults()
        self.analyzer = TopicAnalyzer(use_enhanced_audio=True)
        
    def test_basic_functionality_suite(self) -> bool:
        """Test basic functionality across all scenarios."""
        print("🔧 Running Basic Functionality Test Suite...")
        
        scenarios = get_all_scenarios()
        success_count = 0
        
        for scenario_name, scenario in scenarios.items():
            start_time = time.time()
            
            try:
                results = self.analyzer.analyze_transcript(scenario['transcript'])
                duration = time.time() - start_time
                
                if 'error' in results:
                    self.results.add_result(
                        f"basic_func_{scenario_name}", False, duration,
                        error=results['error']
                    )
                else:
                    # Validate result structure
                    validation_success = self._validate_result_structure(results)
                    
                    self.results.add_result(
                        f"basic_func_{scenario_name}", validation_success, duration,
                        metadata={
                            'duration': results['optimized_segment']['duration'],
                            'completeness_score': results['optimized_segment']['narrative_intelligence']['completeness_score']
                        }
                    )
                    
                    if validation_success:
                        success_count += 1
                        
            except Exception as e:
                duration = time.time() - start_time
                self.results.add_result(
                    f"basic_func_{scenario_name}", False, duration,
                    error=str(e)
                )
        
        success_rate = (success_count / len(scenarios)) * 100
        print(f"✅ Basic Functionality: {success_count}/{len(scenarios)} passed ({success_rate:.1f}%)")
        
        return success_rate >= 95
    
    def test_concurrent_processing(self, num_threads: int = 5, num_requests: int = 20) -> bool:
        """Test concurrent processing capabilities."""
        print(f"🔄 Running Concurrent Processing Test ({num_threads} threads, {num_requests} requests)...")
        
        scenarios = list(get_all_scenarios().items())
        test_requests = []
        
        # Create test requests by cycling through scenarios
        for i in range(num_requests):
            scenario_name, scenario = scenarios[i % len(scenarios)]
            test_requests.append((f"concurrent_{i}_{scenario_name}", scenario['transcript']))
        
        successful_concurrent = 0
        failed_concurrent = 0
        
        def process_request(request_id, transcript):
            """Process a single request."""
            start_time = time.time()
            try:
                analyzer = TopicAnalyzer(use_enhanced_audio=True)  # Thread-safe instance
                results = analyzer.analyze_transcript(transcript)
                duration = time.time() - start_time
                
                if 'error' in results:
                    return request_id, False, duration, results['error']
                else:
                    validation_success = self._validate_result_structure(results)
                    return request_id, validation_success, duration, None
                    
            except Exception as e:
                duration = time.time() - start_time
                return request_id, False, duration, str(e)
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_request = {
                executor.submit(process_request, req_id, transcript): req_id 
                for req_id, transcript in test_requests
            }
            
            for future in as_completed(future_to_request):
                request_id, success, duration, error = future.result()
                
                self.results.add_result(
                    request_id, success, duration,
                    error=error
                )
                
                if success:
                    successful_concurrent += 1
                else:
                    failed_concurrent += 1
        
        concurrent_success_rate = (successful_concurrent / num_requests) * 100
        avg_concurrent_time = self.results.get_average_processing_time()
        
        print(f"✅ Concurrent Processing: {successful_concurrent}/{num_requests} passed ({concurrent_success_rate:.1f}%)")
        print(f"📊 Average concurrent processing time: {avg_concurrent_time:.3f}s")
        
        self.results.add_performance_metric("concurrent_success_rate", concurrent_success_rate, "%")
        self.results.add_performance_metric("avg_concurrent_time", avg_concurrent_time, "s")
        
        return concurrent_success_rate >= 90
    
    def test_stress_conditions(self) -> bool:
        """Test system under stress conditions."""
        print("💪 Running Stress Condition Tests...")
        
        stress_tests = [
            self._test_large_content_handling,
            self._test_malformed_input_handling,
            self._test_memory_usage_under_load,
            self._test_rapid_sequential_processing
        ]
        
        stress_results = []
        
        for stress_test in stress_tests:
            try:
                result = stress_test()
                stress_results.append(result)
            except Exception as e:
                print(f"❌ Stress test failed: {stress_test.__name__}: {str(e)}")
                stress_results.append(False)
        
        stress_success_rate = (sum(stress_results) / len(stress_results)) * 100
        print(f"✅ Stress Testing: {sum(stress_results)}/{len(stress_results)} passed ({stress_success_rate:.1f}%)")
        
        return stress_success_rate >= 75  # Lower threshold for stress tests
    
    def test_production_workflow_integration(self) -> bool:
        """Test integration with production workflow components."""
        print("🔗 Running Production Workflow Integration Tests...")
        
        integration_tests = [
            self._test_reelsfy_integration,
            self._test_error_propagation,
            self._test_output_format_consistency,
            self._test_metadata_completeness
        ]
        
        integration_results = []
        
        for integration_test in integration_tests:
            try:
                result = integration_test()
                integration_results.append(result)
            except Exception as e:
                print(f"❌ Integration test failed: {integration_test.__name__}: {str(e)}")
                integration_results.append(False)
        
        integration_success_rate = (sum(integration_results) / len(integration_results)) * 100
        print(f"✅ Workflow Integration: {sum(integration_results)}/{len(integration_results)} passed ({integration_success_rate:.1f}%)")
        
        return integration_success_rate >= 85
    
    def _validate_result_structure(self, results: Dict) -> bool:
        """Validate the structure of analysis results."""
        required_keys = [
            'concluding_topic', 'supporting_segments', 'optimized_segment',
            'audio_analysis'
        ]
        
        for key in required_keys:
            if key not in results:
                return False
        
        # Check optimized segment structure
        optimized = results['optimized_segment']
        required_optimized_keys = ['start_time', 'end_time', 'duration', 'narrative_intelligence']
        
        for key in required_optimized_keys:
            if key not in optimized:
                return False
        
        # Check narrative intelligence metadata
        narrative_data = optimized['narrative_intelligence']
        required_narrative_keys = ['completeness_score', 'narrative_arc_quality', 'phase4_enhanced']
        
        for key in required_narrative_keys:
            if key not in narrative_data:
                return False
        
        # Validate score ranges
        if not (0.0 <= narrative_data['completeness_score'] <= 1.0):
            return False
        
        if not (0.0 <= narrative_data['narrative_arc_quality'] <= 1.0):
            return False
        
        return True
    
    def _test_large_content_handling(self) -> bool:
        """Test handling of large content."""
        print("  📏 Testing large content handling...")
        
        # Create a large transcript by repeating a scenario multiple times
        base_scenario = get_test_scenario('crypto_analysis')
        large_transcript = base_scenario['transcript'] * 3  # Triple the content
        
        start_time = time.time()
        try:
            results = self.analyzer.analyze_transcript(large_transcript)
            duration = time.time() - start_time
            
            success = 'error' not in results and duration < 10.0  # Should complete within 10 seconds
            
            self.results.add_result(
                "stress_large_content", success, duration,
                metadata={'content_size': len(large_transcript)},
                error=results.get('error') if 'error' in results else None
            )
            
            if success:
                print(f"    ✅ Large content processed in {duration:.2f}s")
            else:
                print(f"    ❌ Large content failed or too slow ({duration:.2f}s)")
                
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.add_result("stress_large_content", False, duration, error=str(e))
            print(f"    ❌ Large content exception: {str(e)}")
            return False
    
    def _test_malformed_input_handling(self) -> bool:
        """Test handling of malformed inputs."""
        print("  🔧 Testing malformed input handling...")
        
        malformed_inputs = [
            "",  # Empty
            "Not an SRT format at all",  # Invalid format
            "1\n00:00:00,000 --> 00:00:05,000\n",  # Incomplete SRT
            "1\n00:00:00,000 --> 00:00:05,000\n\n2\n00:00:05,000 --> \nIncomplete",  # Malformed timestamps
        ]
        
        malformed_success = 0
        
        for i, malformed_input in enumerate(malformed_inputs):
            start_time = time.time()
            try:
                results = self.analyzer.analyze_transcript(malformed_input)
                duration = time.time() - start_time
                
                # Should gracefully handle malformed input (either error or empty result)
                graceful_handling = 'error' in results or not results.get('concluding_topic')
                
                self.results.add_result(
                    f"stress_malformed_{i}", graceful_handling, duration,
                    metadata={'input_type': 'malformed'},
                    error=results.get('error') if 'error' in results else None
                )
                
                if graceful_handling:
                    malformed_success += 1
                    
            except Exception as e:
                duration = time.time() - start_time
                # Exception is acceptable for malformed input
                self.results.add_result(f"stress_malformed_{i}", True, duration, error=str(e))
                malformed_success += 1
        
        success_rate = (malformed_success / len(malformed_inputs)) * 100
        print(f"    ✅ Malformed input handling: {malformed_success}/{len(malformed_inputs)} ({success_rate:.1f}%)")
        
        return success_rate >= 80
    
    def _test_memory_usage_under_load(self) -> bool:
        """Test memory usage under load."""
        print("  🧠 Testing memory usage under load...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple scenarios in sequence
        scenarios = list(get_all_scenarios().values())
        
        for i in range(10):  # Process 10 iterations
            for scenario in scenarios:
                try:
                    self.analyzer.analyze_transcript(scenario['transcript'])
                except:
                    pass  # Ignore errors for memory test
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        memory_reasonable = memory_increase < 100
        
        self.results.add_performance_metric("initial_memory_mb", initial_memory, "MB")
        self.results.add_performance_metric("final_memory_mb", final_memory, "MB")
        self.results.add_performance_metric("memory_increase_mb", memory_increase, "MB")
        
        print(f"    📊 Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        if memory_reasonable:
            print("    ✅ Memory usage is reasonable")
        else:
            print("    ⚠️  High memory usage detected")
        
        return memory_reasonable
    
    def _test_rapid_sequential_processing(self) -> bool:
        """Test rapid sequential processing."""
        print("  ⚡ Testing rapid sequential processing...")
        
        scenario = get_test_scenario('crypto_analysis')
        num_rapid_requests = 50
        
        start_time = time.time()
        successful_rapid = 0
        
        for i in range(num_rapid_requests):
            try:
                results = self.analyzer.analyze_transcript(scenario['transcript'])
                if 'error' not in results:
                    successful_rapid += 1
            except:
                pass
        
        total_duration = time.time() - start_time
        avg_time_per_request = total_duration / num_rapid_requests
        requests_per_second = num_rapid_requests / total_duration
        
        self.results.add_performance_metric("rapid_requests_per_second", requests_per_second, "req/s")
        self.results.add_performance_metric("rapid_avg_time_per_request", avg_time_per_request, "s")
        
        rapid_success_rate = (successful_rapid / num_rapid_requests) * 100
        
        print(f"    📊 Rapid processing: {successful_rapid}/{num_rapid_requests} ({rapid_success_rate:.1f}%)")
        print(f"    ⚡ Processing speed: {requests_per_second:.1f} req/s")
        
        return rapid_success_rate >= 85 and requests_per_second >= 5
    
    def _test_reelsfy_integration(self) -> bool:
        """Test integration with reelsfy.py workflow."""
        print("  🎬 Testing reelsfy.py integration...")
        
        # Test that our enhanced results are compatible with reelsfy workflow
        scenario = get_test_scenario('crypto_analysis')
        
        try:
            results = self.analyzer.analyze_transcript(scenario['transcript'])
            
            # Check that results have the expected structure for reelsfy integration
            required_for_reelsfy = [
                'optimized_segment',
                'concluding_topic',
                'supporting_segments'
            ]
            
            integration_compatible = all(key in results for key in required_for_reelsfy)
            
            # Check that optimized_segment has timing information
            if integration_compatible and 'optimized_segment' in results:
                segment = results['optimized_segment']
                has_timing = 'start_time' in segment and 'end_time' in segment
                integration_compatible = integration_compatible and has_timing
            
            print(f"    {'✅' if integration_compatible else '❌'} Reelsfy integration compatibility")
            
            return integration_compatible
            
        except Exception as e:
            print(f"    ❌ Reelsfy integration test failed: {str(e)}")
            return False
    
    def _test_error_propagation(self) -> bool:
        """Test proper error propagation."""
        print("  ⚠️  Testing error propagation...")
        
        # Test with invalid inputs to ensure errors are properly handled
        invalid_inputs = ["", None, 123]  # Non-string inputs
        
        error_handling_success = 0
        
        for invalid_input in invalid_inputs:
            try:
                results = self.analyzer.analyze_transcript(invalid_input)
                # Should either return error or raise exception
                if 'error' in results:
                    error_handling_success += 1
                    print(f"    ✅ Proper error returned for invalid input")
                else:
                    print(f"    ❌ No error handling for invalid input")
            except Exception as e:
                # Exception is acceptable error handling
                error_handling_success += 1
                print(f"    ✅ Exception properly raised: {type(e).__name__}")
        
        error_success_rate = (error_handling_success / len(invalid_inputs)) * 100
        
        return error_success_rate >= 100  # All invalid inputs should be handled
    
    def _test_output_format_consistency(self) -> bool:
        """Test output format consistency."""
        print("  📋 Testing output format consistency...")
        
        scenarios = get_all_scenarios()
        format_consistent = True
        
        results_structures = []
        
        for scenario_name, scenario in scenarios.items():
            try:
                results = self.analyzer.analyze_transcript(scenario['transcript'])
                if 'error' not in results:
                    # Extract structure (keys and their types)
                    structure = self._extract_structure(results)
                    results_structures.append(structure)
            except:
                pass
        
        # Check that all successful results have consistent structure
        if results_structures:
            reference_structure = results_structures[0]
            
            for structure in results_structures[1:]:
                if structure != reference_structure:
                    format_consistent = False
                    break
        
        print(f"    {'✅' if format_consistent else '❌'} Output format consistency")
        
        return format_consistent
    
    def _test_metadata_completeness(self) -> bool:
        """Test metadata completeness."""
        print("  📊 Testing metadata completeness...")
        
        scenario = get_test_scenario('crypto_analysis')
        
        try:
            results = self.analyzer.analyze_transcript(scenario['transcript'])
            
            if 'error' in results:
                return False
            
            # Check for essential metadata
            metadata_complete = True
            
            # Check narrative intelligence metadata
            if 'narrative_intelligence' not in results['optimized_segment']:
                metadata_complete = False
            else:
                narrative_data = results['optimized_segment']['narrative_intelligence']
                required_narrative_fields = [
                    'completeness_score',
                    'narrative_arc_quality',
                    'context_coverage',
                    'story_roles_present',
                    'phase4_enhanced'
                ]
                
                for field in required_narrative_fields:
                    if field not in narrative_data:
                        metadata_complete = False
                        break
            
            print(f"    {'✅' if metadata_complete else '❌'} Metadata completeness")
            
            return metadata_complete
            
        except Exception as e:
            print(f"    ❌ Metadata test failed: {str(e)}")
            return False
    
    def _extract_structure(self, obj, level=0):
        """Extract the structure of a nested object."""
        if level > 3:  # Prevent infinite recursion
            return type(obj).__name__
            
        if isinstance(obj, dict):
            return {k: self._extract_structure(v, level+1) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._extract_structure(obj[0], level+1)] if obj else []
        else:
            return type(obj).__name__
    
    def run_full_production_test_suite(self) -> bool:
        """Run the complete production test suite."""
        print("🧪 STARTING PRODUCTION TEST SUITE")
        print("=" * 60)
        
        self.results.start_time = datetime.now()
        
        # Run all test suites
        test_suites = [
            ("Basic Functionality", self.test_basic_functionality_suite),
            ("Concurrent Processing", self.test_concurrent_processing),
            ("Stress Conditions", self.test_stress_conditions),
            ("Workflow Integration", self.test_production_workflow_integration)
        ]
        
        suite_results = []
        
        for suite_name, suite_function in test_suites:
            print(f"\n🔍 {suite_name} Test Suite")
            print("-" * 40)
            
            suite_start = time.time()
            try:
                suite_result = suite_function()
                suite_duration = time.time() - suite_start
                
                suite_results.append(suite_result)
                
                print(f"{'✅ PASSED' if suite_result else '❌ FAILED'} - {suite_name} ({suite_duration:.2f}s)")
                
            except Exception as e:
                suite_duration = time.time() - suite_start
                suite_results.append(False)
                print(f"❌ FAILED - {suite_name} ({suite_duration:.2f}s): {str(e)}")
        
        self.results.end_time = datetime.now()
        
        # Calculate overall results
        overall_success_rate = (sum(suite_results) / len(suite_results)) * 100
        
        # Generate final report
        print("\n" + "=" * 60)
        print("🏁 PRODUCTION TEST SUITE RESULTS")
        print("=" * 60)
        
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"Test Suites Passed: {sum(suite_results)}/{len(suite_results)}")
        
        # Print detailed report
        print(self.results.generate_report())
        
        production_ready = overall_success_rate >= 85
        
        if production_ready:
            print("🎉 SYSTEM IS PRODUCTION READY!")
        else:
            print("⚠️  SYSTEM NEEDS IMPROVEMENTS BEFORE PRODUCTION")
        
        return production_ready


def main():
    """Run the production test suite."""
    test_suite = ProductionTestSuite()
    return test_suite.run_full_production_test_suite()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)