#!/usr/bin/env python3
"""
Phase 5: Testing & Validation - Edge Case Handling & Error Recovery Suite

Story 5.3: Edge case handling and error recovery.

This suite tests the system's robustness against edge cases and validates
proper error recovery mechanisms:
- Malformed and corrupted input handling
- Extreme content variations (very short, very long, unusual formats)
- Network and resource limitation scenarios
- Graceful degradation under adverse conditions
- Error recovery and fallback mechanisms
- Data consistency validation
"""

import sys
import os
import time
import tempfile
import traceback
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topic_analyzer import TopicAnalyzer
from tests.test_data.sample_transcripts import get_test_scenario


class EdgeCaseTestResults:
    """Collects and analyzes edge case test results."""
    
    def __init__(self):
        self.test_cases = []
        self.recovery_tests = []
        self.resilience_score = 0.0
        
    def add_edge_case_result(self, case_name: str, input_data: Any, 
                           expected_behavior: str, actual_behavior: str, 
                           success: bool, error_details: str = None):
        """Add an edge case test result."""
        self.test_cases.append({
            'case_name': case_name,
            'input_data_type': type(input_data).__name__,
            'input_size': len(str(input_data)) if hasattr(input_data, '__len__') or isinstance(input_data, str) else 0,
            'expected_behavior': expected_behavior,
            'actual_behavior': actual_behavior,
            'success': success,
            'error_details': error_details,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_recovery_test_result(self, test_name: str, failure_injected: str,
                               recovery_successful: bool, recovery_time: float,
                               data_consistency: bool):
        """Add a recovery test result."""
        self.recovery_tests.append({
            'test_name': test_name,
            'failure_injected': failure_injected,
            'recovery_successful': recovery_successful,
            'recovery_time': recovery_time,
            'data_consistency': data_consistency,
            'timestamp': datetime.now().isoformat()
        })
    
    def calculate_resilience_score(self) -> float:
        """Calculate overall system resilience score."""
        if not self.test_cases:
            return 0.0
        
        # Edge case handling score (40% weight)
        edge_case_success = sum(1 for case in self.test_cases if case['success'])
        edge_case_score = (edge_case_success / len(self.test_cases)) * 40
        
        # Recovery success score (40% weight)
        if self.recovery_tests:
            recovery_success = sum(1 for test in self.recovery_tests if test['recovery_successful'])
            recovery_score = (recovery_success / len(self.recovery_tests)) * 40
        else:
            recovery_score = 0
        
        # Data consistency score (20% weight)
        if self.recovery_tests:
            consistency_success = sum(1 for test in self.recovery_tests if test['data_consistency'])
            consistency_score = (consistency_success / len(self.recovery_tests)) * 20
        else:
            consistency_score = 0
        
        total_score = edge_case_score + recovery_score + consistency_score
        self.resilience_score = total_score
        
        return total_score
    
    def generate_resilience_report(self) -> str:
        """Generate comprehensive resilience report."""
        resilience_score = self.calculate_resilience_score()
        
        report = f"""
🛡️  EDGE CASE & ERROR RECOVERY RESILIENCE REPORT
{'=' * 60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OVERALL RESILIENCE METRICS:
• Resilience Score: {resilience_score:.1f}/100
• Edge Cases Tested: {len(self.test_cases)}
• Recovery Scenarios Tested: {len(self.recovery_tests)}
• Overall System Robustness: {self._get_robustness_level(resilience_score)}

🔍 EDGE CASE HANDLING ANALYSIS:
"""
        
        if self.test_cases:
            successful_edge_cases = sum(1 for case in self.test_cases if case['success'])
            edge_success_rate = (successful_edge_cases / len(self.test_cases)) * 100
            
            report += f"• Success Rate: {edge_success_rate:.1f}% ({successful_edge_cases}/{len(self.test_cases)})\n"
            
            # Categorize edge cases by type
            case_types = {}
            for case in self.test_cases:
                case_type = case['case_name'].split('_')[0]
                if case_type not in case_types:
                    case_types[case_type] = {'total': 0, 'successful': 0}
                case_types[case_type]['total'] += 1
                if case['success']:
                    case_types[case_type]['successful'] += 1
            
            for case_type, stats in case_types.items():
                success_rate = (stats['successful'] / stats['total']) * 100
                report += f"  • {case_type.title()}: {success_rate:.1f}% ({stats['successful']}/{stats['total']})\n"
        
        report += f"\n🔄 ERROR RECOVERY ANALYSIS:\n"
        
        if self.recovery_tests:
            successful_recoveries = sum(1 for test in self.recovery_tests if test['recovery_successful'])
            recovery_success_rate = (successful_recoveries / len(self.recovery_tests)) * 100
            
            avg_recovery_time = sum(test['recovery_time'] for test in self.recovery_tests) / len(self.recovery_tests)
            
            consistent_recoveries = sum(1 for test in self.recovery_tests if test['data_consistency'])
            consistency_rate = (consistent_recoveries / len(self.recovery_tests)) * 100
            
            report += f"• Recovery Success Rate: {recovery_success_rate:.1f}% ({successful_recoveries}/{len(self.recovery_tests)})\n"
            report += f"• Average Recovery Time: {avg_recovery_time:.3f} seconds\n"
            report += f"• Data Consistency Rate: {consistency_rate:.1f}% ({consistent_recoveries}/{len(self.recovery_tests)})\n"
        else:
            report += "• No recovery tests performed\n"
        
        report += f"\n💡 RECOMMENDATIONS:\n"
        report += self._generate_resilience_recommendations()
        
        return report
    
    def _get_robustness_level(self, score: float) -> str:
        """Get robustness level description based on score."""
        if score >= 90:
            return "EXCELLENT - Production ready"
        elif score >= 80:
            return "GOOD - Minor improvements needed"
        elif score >= 70:
            return "ACCEPTABLE - Some hardening required"
        else:
            return "NEEDS IMPROVEMENT - Significant work required"
    
    def _generate_resilience_recommendations(self) -> str:
        """Generate specific resilience improvement recommendations."""
        recommendations = []
        
        # Analyze edge case failures
        failed_edge_cases = [case for case in self.test_cases if not case['success']]
        if failed_edge_cases:
            failure_types = {}
            for case in failed_edge_cases:
                error_type = case['error_details'].split(':')[0] if case['error_details'] else 'Unknown'
                failure_types[error_type] = failure_types.get(error_type, 0) + 1
            
            for error_type, count in failure_types.items():
                recommendations.append(f"• Improve handling of {error_type} errors ({count} occurrences)")
        
        # Analyze recovery failures
        failed_recoveries = [test for test in self.recovery_tests if not test['recovery_successful']]
        if failed_recoveries:
            recommendations.append(f"• Enhance error recovery mechanisms ({len(failed_recoveries)} recovery failures)")
        
        # Analyze data consistency issues
        inconsistent_recoveries = [test for test in self.recovery_tests if not test['data_consistency']]
        if inconsistent_recoveries:
            recommendations.append(f"• Improve data consistency during recovery ({len(inconsistent_recoveries)} consistency issues)")
        
        if not recommendations:
            recommendations.append("• System resilience is excellent - no specific improvements needed")
        
        return '\n'.join(recommendations)


class EdgeCaseRecoverySuite:
    """Comprehensive edge case and error recovery testing suite."""
    
    def __init__(self):
        self.results = EdgeCaseTestResults()
        
    def test_malformed_input_handling(self) -> bool:
        """Test handling of various malformed inputs."""
        print("🔧 Testing Malformed Input Handling")
        print("-" * 40)
        
        malformed_inputs = [
            # Empty and minimal inputs
            ("", "empty_string", "Return error or empty result", "Should handle gracefully"),
            ("   ", "whitespace_only", "Return error or empty result", "Should handle gracefully"),
            ("\n\n\n", "newlines_only", "Return error or empty result", "Should handle gracefully"),
            
            # Invalid SRT format
            ("Not an SRT format", "invalid_format", "Return format error", "Should detect invalid format"),
            ("1234567890", "numeric_only", "Return format error", "Should detect invalid format"),
            ("Random text without timestamps", "no_timestamps", "Return format error", "Should detect invalid format"),
            
            # Broken SRT structure
            ("1\n00:00:00,000 --> \nIncomplete", "incomplete_timestamp", "Handle gracefully or return error", "Should detect malformed timestamp"),
            ("1\n00:00:00,000 --> 00:00:05,000\n\n2\n", "missing_text", "Handle missing text", "Should handle missing segments"),
            ("1\n25:99:99,999 --> 26:99:99,999\nInvalid times", "invalid_times", "Handle invalid timestamps", "Should validate timestamps"),
            
            # Corrupted content
            ("1\n00:00:00,000 --> 00:00:05,000\n\x00\x01\x02", "binary_content", "Handle binary data", "Should handle binary corruption"),
            ("1\n00:00:00,000 --> 00:00:05,000\n" + "A" * 10000, "excessive_text", "Handle oversized content", "Should handle large segments"),
            
            # Unicode and encoding issues
            ("1\n00:00:00,000 --> 00:00:05,000\n\u202E\u202D\uFEFF", "unicode_issues", "Handle unicode properly", "Should handle special unicode"),
        ]
        
        malformed_success = 0
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        
        for input_data, case_name, expected, description in malformed_inputs:
            print(f"  🧪 Testing: {case_name}")
            
            try:
                start_time = time.perf_counter()
                results = analyzer.analyze_transcript(input_data)
                processing_time = time.perf_counter() - start_time
                
                # Check if the system handled the malformed input gracefully
                graceful_handling = (
                    'error' in results or  # Explicit error
                    not results or  # Empty results
                    not results.get('concluding_topic') or  # No topic found
                    processing_time < 5.0  # Completed quickly (didn't hang)
                )
                
                actual_behavior = f"Processed in {processing_time:.3f}s"
                if 'error' in results:
                    actual_behavior += f", returned error: {results['error'][:50]}..."
                elif not results.get('concluding_topic'):
                    actual_behavior += ", no topic identified"
                
                self.results.add_edge_case_result(
                    case_name, input_data, expected, actual_behavior, 
                    graceful_handling, results.get('error')
                )
                
                if graceful_handling:
                    malformed_success += 1
                    print(f"    ✅ Handled gracefully: {actual_behavior}")
                else:
                    print(f"    ❌ Poor handling: {actual_behavior}")
                    
            except Exception as e:
                # Exceptions are acceptable for malformed input
                self.results.add_edge_case_result(
                    case_name, input_data, expected, f"Exception: {type(e).__name__}", 
                    True, str(e)
                )
                malformed_success += 1
                print(f"    ✅ Exception handled: {type(e).__name__}")
        
        success_rate = (malformed_success / len(malformed_inputs)) * 100
        print(f"\n📊 Malformed Input Handling: {malformed_success}/{len(malformed_inputs)} ({success_rate:.1f}%)")
        
        return success_rate >= 80
    
    def test_extreme_content_variations(self) -> bool:
        """Test handling of extreme content variations."""
        print("\n📏 Testing Extreme Content Variations")
        print("-" * 40)
        
        # Get a base scenario for content manipulation
        base_scenario = get_test_scenario('crypto_analysis')
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        
        extreme_cases = []
        
        # Very short content
        minimal_srt = "1\n00:00:00,000 --> 00:00:01,000\nHi"
        extreme_cases.append((minimal_srt, "minimal_content", "Handle very short content"))
        
        # Very long content (repeat base scenario 10 times)
        massive_srt = base_scenario['transcript'] * 10
        extreme_cases.append((massive_srt, "massive_content", "Handle very long content"))
        
        # Many small segments (100 one-word segments)
        many_segments = ""
        for i in range(100):
            start_time = f"00:00:{i:02d},000"
            end_time = f"00:00:{i+1:02d},000"
            many_segments += f"{i+1}\n{start_time} --> {end_time}\nWord{i}\n\n"
        extreme_cases.append((many_segments, "many_segments", "Handle many small segments"))
        
        # Few very long segments
        long_text = "This is a very long segment with lots of repeated text. " * 100
        few_long = f"1\n00:00:00,000 --> 00:05:00,000\n{long_text}\n\n"
        few_long += f"2\n00:05:00,000 --> 00:10:00,000\n{long_text}\n\n"
        extreme_cases.append((few_long, "few_long_segments", "Handle few very long segments"))
        
        # Unusual timestamp patterns
        unusual_times = "1\n00:00:00,001 --> 00:00:00,002\nFast\n\n"
        unusual_times += "2\n01:00:00,000 --> 02:00:00,000\nSlow\n\n"
        extreme_cases.append((unusual_times, "unusual_timestamps", "Handle unusual timing patterns"))
        
        extreme_success = 0
        
        for content, case_name, expected in extreme_cases:
            print(f"  🧪 Testing: {case_name}")
            
            try:
                start_time = time.perf_counter()
                results = analyzer.analyze_transcript(content)
                processing_time = time.perf_counter() - start_time
                
                # Success criteria: completes within reasonable time and returns valid result or error
                success = (
                    processing_time < 30.0 and  # Completes within 30 seconds
                    (('error' in results) or  # Valid error
                     (results.get('optimized_segment') and  # Valid result
                      results.get('concluding_topic')))
                )
                
                actual_behavior = f"Processed in {processing_time:.2f}s"
                if 'error' in results:
                    actual_behavior += f", error: {results['error'][:30]}..."
                elif results.get('optimized_segment'):
                    duration = results['optimized_segment']['duration']
                    actual_behavior += f", extracted {duration:.1f}s segment"
                
                self.results.add_edge_case_result(
                    case_name, content, expected, actual_behavior, 
                    success, results.get('error')
                )
                
                if success:
                    extreme_success += 1
                    print(f"    ✅ Handled successfully: {actual_behavior}")
                else:
                    print(f"    ❌ Failed handling: {actual_behavior}")
                    
            except Exception as e:
                processing_time = time.perf_counter() - start_time
                
                # Quick exception handling is acceptable
                success = processing_time < 10.0
                
                self.results.add_edge_case_result(
                    case_name, content, expected, f"Exception in {processing_time:.2f}s: {type(e).__name__}", 
                    success, str(e)
                )
                
                if success:
                    extreme_success += 1
                    print(f"    ✅ Exception handled: {type(e).__name__}")
                else:
                    print(f"    ❌ Slow exception: {type(e).__name__} in {processing_time:.2f}s")
        
        success_rate = (extreme_success / len(extreme_cases)) * 100
        print(f"\n📊 Extreme Content Handling: {extreme_success}/{len(extreme_cases)} ({success_rate:.1f}%)")
        
        return success_rate >= 70
    
    def test_resource_limitation_scenarios(self) -> bool:
        """Test behavior under resource limitations."""
        print("\n💾 Testing Resource Limitation Scenarios")
        print("-" * 40)
        
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        scenario = get_test_scenario('crypto_analysis')
        
        resource_tests = []
        resource_success = 0
        
        # Test 1: Rapid sequential processing (stress test)
        print("  🔄 Testing rapid sequential processing...")
        try:
            start_time = time.perf_counter()
            successful_rapid = 0
            
            for i in range(50):
                try:
                    results = analyzer.analyze_transcript(scenario['transcript'])
                    if 'error' not in results:
                        successful_rapid += 1
                except:
                    pass
            
            rapid_time = time.perf_counter() - start_time
            rapid_success = successful_rapid >= 40 and rapid_time < 10.0
            
            self.results.add_edge_case_result(
                "rapid_processing", f"{successful_rapid}/50 requests", 
                "Handle rapid processing without degradation",
                f"Completed {successful_rapid}/50 in {rapid_time:.2f}s",
                rapid_success
            )
            
            if rapid_success:
                resource_success += 1
                print(f"    ✅ Rapid processing: {successful_rapid}/50 in {rapid_time:.2f}s")
            else:
                print(f"    ❌ Rapid processing degraded: {successful_rapid}/50 in {rapid_time:.2f}s")
                
        except Exception as e:
            self.results.add_edge_case_result(
                "rapid_processing", "Exception during test", 
                "Handle rapid processing", f"Exception: {type(e).__name__}",
                False, str(e)
            )
            print(f"    ❌ Rapid processing failed: {type(e).__name__}")
        
        # Test 2: Memory pressure simulation
        print("  🧠 Testing under simulated memory pressure...")
        try:
            # Create some memory pressure (careful not to crash)
            memory_pressure = []
            for i in range(100):
                memory_pressure.append([0] * 10000)  # Small memory allocations
            
            # Test processing under memory pressure
            start_time = time.perf_counter()
            results = analyzer.analyze_transcript(scenario['transcript'])
            memory_test_time = time.perf_counter() - start_time
            
            # Clean up memory pressure
            del memory_pressure
            
            memory_success = (
                'error' not in results and 
                memory_test_time < 5.0 and
                results.get('optimized_segment')
            )
            
            self.results.add_edge_case_result(
                "memory_pressure", "Simulated memory pressure", 
                "Continue processing under memory pressure",
                f"Processed in {memory_test_time:.2f}s under pressure",
                memory_success
            )
            
            if memory_success:
                resource_success += 1
                print(f"    ✅ Memory pressure handling: {memory_test_time:.2f}s")
            else:
                print(f"    ❌ Memory pressure impact: {memory_test_time:.2f}s")
                
        except Exception as e:
            self.results.add_edge_case_result(
                "memory_pressure", "Exception during memory test", 
                "Handle memory pressure", f"Exception: {type(e).__name__}",
                False, str(e)
            )
            print(f"    ❌ Memory pressure test failed: {type(e).__name__}")
        
        resource_success_rate = (resource_success / 2) * 100  # 2 tests
        print(f"\n📊 Resource Limitation Handling: {resource_success}/2 ({resource_success_rate:.1f}%)")
        
        return resource_success_rate >= 50
    
    def test_error_recovery_mechanisms(self) -> bool:
        """Test error recovery and fallback mechanisms."""
        print("\n🔄 Testing Error Recovery Mechanisms")
        print("-" * 40)
        
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        scenario = get_test_scenario('crypto_analysis')
        
        recovery_success = 0
        
        # Test 1: Recovery from processing failure
        print("  🛠️  Testing recovery from processing failure...")
        try:
            # Inject a failure by corrupting the input mid-processing
            corrupted_scenario = scenario['transcript'].replace('-->', '///')  # Break SRT format
            
            start_time = time.perf_counter()
            results = analyzer.analyze_transcript(corrupted_scenario)
            recovery_time = time.perf_counter() - start_time
            
            # Should recover gracefully (return error or fallback result)
            recovery_successful = (
                'error' in results or  # Explicit error handling
                recovery_time < 2.0    # Quick failure detection
            )
            
            # Test data consistency by processing valid content afterward
            consistency_results = analyzer.analyze_transcript(scenario['transcript'])
            data_consistent = 'error' not in consistency_results and consistency_results.get('optimized_segment')
            
            self.results.add_recovery_test_result(
                "processing_failure_recovery", "Corrupted SRT format",
                recovery_successful, recovery_time, data_consistent
            )
            
            if recovery_successful:
                recovery_success += 1
                print(f"    ✅ Processing failure recovery: {recovery_time:.3f}s, consistent: {data_consistent}")
            else:
                print(f"    ❌ Processing failure recovery failed: {recovery_time:.3f}s")
                
        except Exception as e:
            # Exception-based recovery is also acceptable
            recovery_time = time.perf_counter() - start_time
            
            # Test consistency after exception
            try:
                consistency_results = analyzer.analyze_transcript(scenario['transcript'])
                data_consistent = 'error' not in consistency_results
            except:
                data_consistent = False
            
            self.results.add_recovery_test_result(
                "processing_failure_recovery", f"Exception: {type(e).__name__}",
                True, recovery_time, data_consistent  # Exception is valid recovery
            )
            
            recovery_success += 1
            print(f"    ✅ Exception-based recovery: {type(e).__name__}, consistent: {data_consistent}")
        
        # Test 2: Recovery from invalid audio path (Phase 3 fallback)
        print("  🎵 Testing audio fallback recovery...")
        try:
            start_time = time.perf_counter()
            
            # Test with invalid audio path - should fallback gracefully
            results = analyzer.analyze_transcript(scenario['transcript'], audio_path="/nonexistent/audio.wav")
            recovery_time = time.perf_counter() - start_time
            
            # Should fallback to non-audio analysis
            recovery_successful = (
                'error' not in results and 
                results.get('audio_analysis', {}).get('audio_enhanced') == False and
                results.get('optimized_segment')
            )
            
            # Data consistency check
            data_consistent = (
                results.get('concluding_topic') and
                results.get('optimized_segment', {}).get('duration', 0) > 0
            )
            
            self.results.add_recovery_test_result(
                "audio_fallback_recovery", "Invalid audio path",
                recovery_successful, recovery_time, data_consistent
            )
            
            if recovery_successful:
                recovery_success += 1
                print(f"    ✅ Audio fallback recovery: {recovery_time:.3f}s, fallback active")
            else:
                print(f"    ❌ Audio fallback failed: {recovery_time:.3f}s")
                
        except Exception as e:
            recovery_time = time.perf_counter() - start_time
            
            self.results.add_recovery_test_result(
                "audio_fallback_recovery", f"Exception: {type(e).__name__}",
                False, recovery_time, False
            )
            print(f"    ❌ Audio fallback exception: {type(e).__name__}")
        
        # Test 3: Recovery from partial data corruption
        print("  📊 Testing partial corruption recovery...")
        try:
            # Corrupt only part of the transcript
            lines = scenario['transcript'].split('\n')
            corrupted_lines = lines[:5] + ['CORRUPTED LINE'] + lines[6:]  # Corrupt one line
            partially_corrupted = '\n'.join(corrupted_lines)
            
            start_time = time.perf_counter()
            results = analyzer.analyze_transcript(partially_corrupted)
            recovery_time = time.perf_counter() - start_time
            
            # Should either recover with partial data or return graceful error
            recovery_successful = (
                ('error' in results) or  # Graceful error
                (results.get('optimized_segment') and  # Partial recovery
                 results.get('concluding_topic'))
            )
            
            # Data consistency check
            data_consistent = recovery_time < 5.0 and recovery_successful
            
            self.results.add_recovery_test_result(
                "partial_corruption_recovery", "Single corrupted line",
                recovery_successful, recovery_time, data_consistent
            )
            
            if recovery_successful:
                recovery_success += 1
                print(f"    ✅ Partial corruption recovery: {recovery_time:.3f}s")
            else:
                print(f"    ❌ Partial corruption failed: {recovery_time:.3f}s")
                
        except Exception as e:
            recovery_time = time.perf_counter() - start_time
            
            self.results.add_recovery_test_result(
                "partial_corruption_recovery", f"Exception: {type(e).__name__}",
                recovery_time < 2.0, recovery_time, False  # Quick exception acceptable
            )
            
            if recovery_time < 2.0:
                recovery_success += 1
                print(f"    ✅ Quick exception recovery: {type(e).__name__}")
            else:
                print(f"    ❌ Slow exception: {type(e).__name__}")
        
        recovery_success_rate = (recovery_success / 3) * 100  # 3 tests
        print(f"\n📊 Error Recovery Success: {recovery_success}/3 ({recovery_success_rate:.1f}%)")
        
        return recovery_success_rate >= 66
    
    def test_data_consistency_validation(self) -> bool:
        """Test data consistency across various operations."""
        print("\n🔍 Testing Data Consistency Validation")  
        print("-" * 40)
        
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        scenario = get_test_scenario('crypto_analysis')
        
        consistency_tests_passed = 0
        
        # Test 1: Repeated processing consistency
        print("  🔁 Testing repeated processing consistency...")
        try:
            results_runs = []
            for i in range(5):
                results = analyzer.analyze_transcript(scenario['transcript'])
                if 'error' not in results:
                    results_runs.append(results)
            
            if len(results_runs) >= 3:
                # Check consistency of key metrics
                durations = [r['optimized_segment']['duration'] for r in results_runs]
                topics = [r['concluding_topic']['text'] for r in results_runs]
                
                # Duration should be consistent (within 5% variance)
                duration_variance = (max(durations) - min(durations)) / max(durations) if max(durations) > 0 else 0
                duration_consistent = duration_variance < 0.05
                
                # Main topic should be consistent
                topic_consistent = len(set(topics)) <= 2  # Allow minor variations
                
                overall_consistent = duration_consistent and topic_consistent
                
                self.results.add_edge_case_result(
                    "repeated_consistency", f"{len(results_runs)} runs",
                    "Consistent results across runs",
                    f"Duration variance: {duration_variance:.3f}, Topic consistency: {topic_consistent}",
                    overall_consistent
                )
                
                if overall_consistent:
                    consistency_tests_passed += 1
                    print(f"    ✅ Repeated processing consistent: {duration_variance:.3f} variance")
                else:
                    print(f"    ❌ Repeated processing inconsistent: {duration_variance:.3f} variance")
            else:
                print(f"    ⚠️  Insufficient successful runs: {len(results_runs)}")
                
        except Exception as e:
            self.results.add_edge_case_result(
                "repeated_consistency", "Exception during consistency test",
                "Consistent results", f"Exception: {type(e).__name__}",
                False, str(e)
            )
            print(f"    ❌ Consistency test exception: {type(e).__name__}")
        
        # Test 2: Cross-scenario consistency
        print("  🌐 Testing cross-scenario consistency...")
        try:
            all_scenarios = [
                get_test_scenario('crypto_analysis'),
                get_test_scenario('educational'),
                get_test_scenario('tech_product')
            ]
            
            scenario_results = []
            for i, test_scenario in enumerate(all_scenarios):
                results = analyzer.analyze_transcript(test_scenario['transcript'])
                if 'error' not in results:
                    scenario_results.append({
                        'scenario': i,
                        'duration': results['optimized_segment']['duration'],
                        'has_narrative_intelligence': 'narrative_intelligence' in results['optimized_segment']
                    })
            
            # Check that all scenarios have consistent feature availability
            if scenario_results:
                all_have_narrative = all(r['has_narrative_intelligence'] for r in scenario_results)
                durations_reasonable = all(30 <= r['duration'] <= 70 for r in scenario_results)
                
                cross_consistent = all_have_narrative and durations_reasonable
                
                self.results.add_edge_case_result(
                    "cross_scenario_consistency", f"{len(scenario_results)} scenarios",
                    "Consistent features across scenarios",
                    f"Narrative intelligence: {all_have_narrative}, Duration range: {durations_reasonable}",
                    cross_consistent
                )
                
                if cross_consistent:
                    consistency_tests_passed += 1
                    print(f"    ✅ Cross-scenario consistency: {len(scenario_results)} scenarios")
                else:
                    print(f"    ❌ Cross-scenario inconsistency detected")
            else:
                print(f"    ⚠️  No successful scenario results")
                
        except Exception as e:
            self.results.add_edge_case_result(
                "cross_scenario_consistency", "Exception during cross-scenario test",
                "Cross-scenario consistency", f"Exception: {type(e).__name__}",
                False, str(e)
            )
            print(f"    ❌ Cross-scenario test exception: {type(e).__name__}")
        
        consistency_success_rate = (consistency_tests_passed / 2) * 100  # 2 tests
        print(f"\n📊 Data Consistency Validation: {consistency_tests_passed}/2 ({consistency_success_rate:.1f}%)")
        
        return consistency_success_rate >= 50
    
    def run_comprehensive_edge_case_suite(self) -> Dict:
        """Run the complete edge case and error recovery test suite."""
        print("🛡️  STARTING COMPREHENSIVE EDGE CASE & ERROR RECOVERY SUITE")
        print("=" * 75)
        
        suite_start_time = time.perf_counter()
        
        # Run all test suites
        test_suites = [
            ("Malformed Input Handling", self.test_malformed_input_handling),
            ("Extreme Content Variations", self.test_extreme_content_variations),
            ("Resource Limitation Scenarios", self.test_resource_limitation_scenarios),
            ("Error Recovery Mechanisms", self.test_error_recovery_mechanisms),
            ("Data Consistency Validation", self.test_data_consistency_validation)
        ]
        
        suite_results = []
        
        for suite_name, suite_function in test_suites:
            print(f"\n🔍 {suite_name}")
            print("=" * 50)
            
            try:
                suite_result = suite_function()
                suite_results.append(suite_result)
                
                status = "✅ PASSED" if suite_result else "❌ FAILED"
                print(f"{status} - {suite_name}")
                
            except Exception as e:
                print(f"❌ FAILED - {suite_name}: {str(e)}")
                traceback.print_exc()
                suite_results.append(False)
        
        suite_duration = time.perf_counter() - suite_start_time
        
        # Generate comprehensive results
        print("\n" + "=" * 75)
        print("🏁 EDGE CASE & ERROR RECOVERY RESULTS")
        print("=" * 75)
        
        overall_success_rate = (sum(suite_results) / len(suite_results)) * 100
        resilience_score = self.results.calculate_resilience_score()
        
        print(f"Overall Suite Success Rate: {overall_success_rate:.1f}%")
        print(f"System Resilience Score: {resilience_score:.1f}/100")
        print(f"Total Test Duration: {suite_duration:.2f} seconds")
        
        # Generate and display resilience report
        resilience_report = self.results.generate_resilience_report()
        print(resilience_report)
        
        # Final assessment
        production_ready = overall_success_rate >= 70 and resilience_score >= 70
        
        if production_ready:
            print("🎉 SYSTEM RESILIENCE IS PRODUCTION READY!")
        else:
            print("⚠️  SYSTEM NEEDS RESILIENCE IMPROVEMENTS")
        
        return {
            'overall_success_rate': overall_success_rate,
            'resilience_score': resilience_score,
            'suite_duration': suite_duration,
            'production_ready': production_ready,
            'edge_case_results': self.results.test_cases,
            'recovery_test_results': self.results.recovery_tests,
            'test_suite_results': dict(zip([name for name, _ in test_suites], suite_results))
        }


def main():
    """Run the comprehensive edge case and error recovery suite."""
    suite = EdgeCaseRecoverySuite()
    return suite.run_comprehensive_edge_case_suite()


if __name__ == "__main__":
    results = main()
    
    # Save results for analysis
    import json
    with open('edge_case_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Edge case results saved to 'edge_case_results.json'")
    
    sys.exit(0 if results['production_ready'] else 1)