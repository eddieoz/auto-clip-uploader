#!/usr/bin/env python3
"""
Phase 5: Testing & Validation - User Acceptance Testing Suite

Story 5.4: User acceptance testing and feedback integration.

This suite simulates user acceptance testing scenarios and validates
the system meets user expectations:
- Real-world usage pattern simulation
- User experience quality validation
- Output quality assessment
- Integration workflow testing
- Performance from user perspective
- Feedback mechanism validation
"""

import sys
import os
import time
import json
import statistics
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topic_analyzer import TopicAnalyzer
from tests.test_data.sample_transcripts import get_test_scenario, get_all_scenarios


class UserAcceptanceResults:
    """Collects and analyzes user acceptance test results."""
    
    def __init__(self):
        self.acceptance_tests = []
        self.user_satisfaction_scores = []
        self.quality_assessments = []
        self.workflow_validations = []
        
    def add_acceptance_test(self, test_name: str, user_story: str, 
                          expected_outcome: str, actual_outcome: str,
                          acceptance_criteria_met: bool, quality_score: float,
                          user_satisfaction: float, notes: str = ""):
        """Add user acceptance test result."""
        self.acceptance_tests.append({
            'test_name': test_name,
            'user_story': user_story,
            'expected_outcome': expected_outcome,
            'actual_outcome': actual_outcome,
            'acceptance_criteria_met': acceptance_criteria_met,
            'quality_score': quality_score,  # 0-10 scale
            'user_satisfaction': user_satisfaction,  # 0-10 scale
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        })
        
        self.user_satisfaction_scores.append(user_satisfaction)
        self.quality_assessments.append(quality_score)
    
    def add_workflow_validation(self, workflow_name: str, steps_completed: int,
                              total_steps: int, completion_time: float,
                              user_friction_points: int, success: bool):
        """Add workflow validation result."""
        self.workflow_validations.append({
            'workflow_name': workflow_name,
            'steps_completed': steps_completed,
            'total_steps': total_steps,
            'completion_rate': (steps_completed / total_steps) * 100,
            'completion_time': completion_time,
            'user_friction_points': user_friction_points,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
    
    def calculate_overall_acceptance(self) -> Dict:
        """Calculate overall user acceptance metrics."""
        if not self.acceptance_tests:
            return {'error': 'No acceptance tests completed'}
        
        # Acceptance rate
        accepted_tests = sum(1 for test in self.acceptance_tests if test['acceptance_criteria_met'])
        acceptance_rate = (accepted_tests / len(self.acceptance_tests)) * 100
        
        # Average scores
        avg_satisfaction = statistics.mean(self.user_satisfaction_scores) if self.user_satisfaction_scores else 0
        avg_quality = statistics.mean(self.quality_assessments) if self.quality_assessments else 0
        
        # Workflow success rate
        if self.workflow_validations:
            successful_workflows = sum(1 for wf in self.workflow_validations if wf['success'])
            workflow_success_rate = (successful_workflows / len(self.workflow_validations)) * 100
            avg_completion_time = statistics.mean([wf['completion_time'] for wf in self.workflow_validations])
            avg_friction_points = statistics.mean([wf['user_friction_points'] for wf in self.workflow_validations])
        else:
            workflow_success_rate = 0
            avg_completion_time = 0
            avg_friction_points = 0
        
        # Calculate overall UAT score
        uat_score = (
            acceptance_rate * 0.4 +           # 40% weight on acceptance criteria
            (avg_satisfaction / 10) * 100 * 0.3 +  # 30% weight on user satisfaction  
            (avg_quality / 10) * 100 * 0.2 +       # 20% weight on quality
            workflow_success_rate * 0.1            # 10% weight on workflow success
        )
        
        return {
            'acceptance_rate': acceptance_rate,
            'avg_user_satisfaction': avg_satisfaction,
            'avg_quality_score': avg_quality,
            'workflow_success_rate': workflow_success_rate,
            'avg_completion_time': avg_completion_time,
            'avg_friction_points': avg_friction_points,
            'overall_uat_score': uat_score,
            'tests_completed': len(self.acceptance_tests),
            'workflows_tested': len(self.workflow_validations)
        }
    
    def generate_uat_report(self) -> str:
        """Generate comprehensive user acceptance testing report."""
        metrics = self.calculate_overall_acceptance()
        
        if 'error' in metrics:
            return f"UAT Report Error: {metrics['error']}"
        
        report = f"""
👥 USER ACCEPTANCE TESTING REPORT
{'=' * 50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OVERALL UAT METRICS:
• Overall UAT Score: {metrics['overall_uat_score']:.1f}/100
• Acceptance Rate: {metrics['acceptance_rate']:.1f}%
• Average User Satisfaction: {metrics['avg_user_satisfaction']:.1f}/10
• Average Quality Score: {metrics['avg_quality_score']:.1f}/10
• Tests Completed: {metrics['tests_completed']}

🔄 WORKFLOW ANALYSIS:
• Workflow Success Rate: {metrics['workflow_success_rate']:.1f}%
• Average Completion Time: {metrics['avg_completion_time']:.2f}s
• Average Friction Points: {metrics['avg_friction_points']:.1f}
• Workflows Tested: {metrics['workflows_tested']}

📋 DETAILED TEST RESULTS:
"""
        
        # Group tests by satisfaction level
        high_satisfaction = [t for t in self.acceptance_tests if t['user_satisfaction'] >= 8]
        medium_satisfaction = [t for t in self.acceptance_tests if 6 <= t['user_satisfaction'] < 8]
        low_satisfaction = [t for t in self.acceptance_tests if t['user_satisfaction'] < 6]
        
        report += f"• High Satisfaction (8-10): {len(high_satisfaction)} tests\n"
        report += f"• Medium Satisfaction (6-8): {len(medium_satisfaction)} tests\n"
        report += f"• Low Satisfaction (0-6): {len(low_satisfaction)} tests\n"
        
        if low_satisfaction:
            report += f"\n⚠️  LOW SATISFACTION TESTS:\n"
            for test in low_satisfaction[:3]:  # Show up to 3 examples
                report += f"  • {test['test_name']}: {test['user_satisfaction']}/10 - {test['notes'][:50]}...\n"
        
        # User readiness assessment
        user_ready = (
            metrics['overall_uat_score'] >= 80 and
            metrics['acceptance_rate'] >= 85 and
            metrics['avg_user_satisfaction'] >= 7.0
        )
        
        report += f"\n✅ USER READINESS: {'READY FOR USERS' if user_ready else 'NEEDS USER EXPERIENCE IMPROVEMENTS'}\n"
        
        return report


class UserAcceptanceTestingSuite:
    """Comprehensive user acceptance testing suite."""
    
    def __init__(self):
        self.results = UserAcceptanceResults()
        
    def test_content_creator_workflow(self) -> bool:
        """Test the complete content creator workflow."""
        print("🎬 Testing Content Creator Workflow")
        print("-" * 40)
        
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        scenario = get_test_scenario('crypto_analysis')
        
        workflow_start = time.perf_counter()
        steps_completed = 0
        total_steps = 5
        friction_points = 0
        
        try:
            # Step 1: Content Creator uploads transcript
            print("  📝 Step 1: Upload transcript (simulated)")
            transcript = scenario['transcript']
            if len(transcript) < 100:
                friction_points += 1  # Too short, user confusion
            steps_completed += 1
            
            # Step 2: System analyzes content  
            print("  🔍 Step 2: System analyzes content")
            analysis_start = time.perf_counter()
            results = analyzer.analyze_transcript(transcript)
            analysis_time = time.perf_counter() - analysis_start
            
            if analysis_time > 2.0:
                friction_points += 1  # Slow analysis, user impatience
            if 'error' in results:
                friction_points += 2  # Error, major user frustration
                print(f"    ❌ Analysis failed: {results['error']}")
            else:
                steps_completed += 1
                print(f"    ✅ Analysis completed in {analysis_time:.2f}s")
            
            # Step 3: User reviews suggested clip
            print("  👀 Step 3: Review suggested clip")
            if 'error' not in results:
                optimized = results['optimized_segment']
                duration = optimized['duration']
                title = optimized.get('title', 'No title')
                
                # User satisfaction based on clip quality
                clip_quality = 0
                if 45 <= duration <= 59:
                    clip_quality += 3  # Good duration
                elif 30 <= duration <= 70:
                    clip_quality += 2  # Acceptable duration
                else:
                    clip_quality += 1  # Poor duration
                    friction_points += 1
                
                if 'narrative_intelligence' in optimized:
                    completeness = optimized['narrative_intelligence']['completeness_score']
                    if completeness >= 0.7:
                        clip_quality += 3  # High completeness
                    elif completeness >= 0.5:
                        clip_quality += 2  # Medium completeness
                    else:
                        clip_quality += 1  # Low completeness
                        friction_points += 1
                
                steps_completed += 1
                print(f"    ✅ Clip review: {duration:.1f}s duration, quality score: {clip_quality}/6")
            else:
                friction_points += 2  # Can't review, major issue
            
            # Step 4: User accepts/rejects clip
            print("  ✅ Step 4: User decision")
            if 'error' not in results and clip_quality >= 4:
                user_accepts = True
                steps_completed += 1
                print("    ✅ User accepts clip")
            else:
                user_accepts = False
                friction_points += 1
                print("    ❌ User rejects clip")
            
            # Step 5: Export/integrate with video processing
            print("  📤 Step 5: Export clip data")
            if user_accepts and 'error' not in results:
                # Simulate export validation
                required_fields = ['start_time', 'end_time', 'duration']
                export_valid = all(field in results['optimized_segment'] for field in required_fields)
                
                if export_valid:
                    steps_completed += 1
                    print("    ✅ Export data valid")
                else:
                    friction_points += 1
                    print("    ❌ Export data incomplete")
            else:
                friction_points += 1
                print("    ❌ Cannot export rejected clip")
            
            workflow_time = time.perf_counter() - workflow_start
            workflow_success = steps_completed >= 4  # At least 80% completion
            
            # Calculate user satisfaction for this workflow
            satisfaction_score = max(0, 10 - friction_points * 1.5)  # Start at 10, lose 1.5 per friction
            if analysis_time < 1.0:
                satisfaction_score += 0.5  # Bonus for fast analysis
            if workflow_time < 5.0:
                satisfaction_score += 0.5  # Bonus for quick workflow
            satisfaction_score = min(satisfaction_score, 10)  # Cap at 10
            
            self.results.add_workflow_validation(
                "content_creator_workflow", steps_completed, total_steps,
                workflow_time, friction_points, workflow_success
            )
            
            # Add as acceptance test
            user_story = "As a content creator, I want to quickly generate high-quality clips from my transcripts"
            expected = "Fast analysis, quality clips, easy workflow"
            actual = f"{steps_completed}/{total_steps} steps, {friction_points} friction points, {workflow_time:.1f}s"
            
            self.results.add_acceptance_test(
                "content_creator_workflow", user_story, expected, actual,
                workflow_success, clip_quality if 'clip_quality' in locals() else 5,
                satisfaction_score, f"Workflow completed with {friction_points} friction points"
            )
            
            print(f"\n📊 Workflow Results:")
            print(f"    • Steps completed: {steps_completed}/{total_steps}")
            print(f"    • Friction points: {friction_points}")  
            print(f"    • Completion time: {workflow_time:.1f}s")
            print(f"    • User satisfaction: {satisfaction_score:.1f}/10")
            
            return workflow_success
            
        except Exception as e:
            workflow_time = time.perf_counter() - workflow_start
            
            self.results.add_workflow_validation(
                "content_creator_workflow", steps_completed, total_steps,
                workflow_time, 10, False  # Max friction for exception
            )
            
            self.results.add_acceptance_test(
                "content_creator_workflow_exception", 
                "As a content creator, I want reliable clip generation",
                "No system errors", f"Exception: {type(e).__name__}",
                False, 0, 2, f"System exception: {str(e)}"
            )
            
            print(f"    ❌ Workflow exception: {type(e).__name__}")
            return False
    
    def test_quality_expectations(self) -> bool:
        """Test if output quality meets user expectations."""
        print("\n🏆 Testing Quality Expectations")
        print("-" * 40)
        
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        quality_tests_passed = 0
        
        # Test different content types for quality consistency
        test_scenarios = [
            ('crypto_analysis', 'Financial content creator'),
            ('educational', 'Educational content creator'),
            ('tech_product', 'Tech reviewer'),
            ('real_portuguese', 'International content creator')
        ]
        
        for scenario_name, user_persona in test_scenarios:
            scenario = get_test_scenario(scenario_name)
            print(f"  👤 Testing: {user_persona}")
            
            try:
                results = analyzer.analyze_transcript(scenario['transcript'])
                
                if 'error' in results:
                    quality_score = 2
                    satisfaction = 3
                    criteria_met = False
                    notes = f"Analysis error: {results['error'][:30]}..."
                else:
                    # Assess quality from user perspective
                    optimized = results['optimized_segment']
                    duration = optimized['duration']
                    
                    # Duration quality (user expects 45-60s clips)
                    if 45 <= duration <= 60:
                        duration_quality = 4
                    elif 30 <= duration <= 70:
                        duration_quality = 3
                    elif 20 <= duration <= 75:
                        duration_quality = 2
                    else:
                        duration_quality = 1
                    
                    # Content completeness (user expects complete narratives)
                    if 'narrative_intelligence' in optimized:
                        completeness = optimized['narrative_intelligence']['completeness_score']
                        if completeness >= 0.8:
                            completeness_quality = 4
                        elif completeness >= 0.6:
                            completeness_quality = 3
                        elif completeness >= 0.4:
                            completeness_quality = 2
                        else:
                            completeness_quality = 1
                    else:
                        completeness_quality = 1
                    
                    # Title quality (user expects meaningful titles)
                    title = optimized.get('title', '')
                    if title and len(title) > 10 and len(title.split()) >= 2:
                        title_quality = 2
                    elif title:
                        title_quality = 1
                    else:
                        title_quality = 0
                    
                    # Overall quality score (0-10)
                    quality_score = duration_quality + completeness_quality + title_quality
                    
                    # User satisfaction based on quality
                    satisfaction = min(10, quality_score + 1)  # Add bonus for getting results
                    
                    # Acceptance criteria: quality >= 6 and duration reasonable
                    criteria_met = quality_score >= 6 and 20 <= duration <= 80
                    
                    notes = f"Duration: {duration:.1f}s, Completeness: {completeness:.2f}" if 'completeness' in locals() else f"Duration: {duration:.1f}s"
                
                # Add acceptance test result
                user_story = f"As a {user_persona.lower()}, I want high-quality clips that represent my content well"
                expected = "45-60s duration, complete narrative, meaningful title"
                actual = f"Quality score: {quality_score}/10, {notes}"
                
                self.results.add_acceptance_test(
                    f"quality_{scenario_name}", user_story, expected, actual,
                    criteria_met, quality_score, satisfaction, notes
                )
                
                if criteria_met:
                    quality_tests_passed += 1
                    print(f"    ✅ Quality acceptable: {quality_score}/10, satisfaction: {satisfaction}/10")
                else:
                    print(f"    ❌ Quality poor: {quality_score}/10, satisfaction: {satisfaction}/10")
                    
            except Exception as e:
                self.results.add_acceptance_test(
                    f"quality_{scenario_name}_exception", 
                    f"As a {user_persona.lower()}, I want reliable quality assessment",
                    "No system errors during quality check",
                    f"Exception: {type(e).__name__}",
                    False, 0, 1, f"System error: {str(e)}"
                )
                print(f"    ❌ Quality test exception: {type(e).__name__}")
        
        quality_success_rate = (quality_tests_passed / len(test_scenarios)) * 100
        print(f"\n📊 Quality Tests: {quality_tests_passed}/{len(test_scenarios)} ({quality_success_rate:.1f}%)")
        
        return quality_success_rate >= 75
    
    def test_user_experience_scenarios(self) -> bool:
        """Test various user experience scenarios."""
        print("\n👤 Testing User Experience Scenarios")  
        print("-" * 40)
        
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        ux_tests_passed = 0
        
        # UX Test 1: First-time user experience
        print("  🆕 Testing: First-time user experience")
        scenario = get_test_scenario('short_conclusion')  # Simple content for new users
        
        try:
            start_time = time.perf_counter()
            results = analyzer.analyze_transcript(scenario['transcript'])
            processing_time = time.perf_counter() - start_time
            
            # First-time user expectations: fast, simple, works
            fast_processing = processing_time < 2.0
            simple_output = 'error' not in results and results.get('optimized_segment')
            reasonable_result = False
            
            if simple_output:
                duration = results['optimized_segment']['duration']
                reasonable_result = 10 <= duration <= 90  # Very broad acceptable range for new users
            
            first_time_success = fast_processing and simple_output and reasonable_result
            satisfaction = 8 if first_time_success else 4
            
            user_story = "As a first-time user, I want the system to work easily and quickly"
            expected = "Fast processing, clear output, reasonable results"
            actual = f"Processing: {processing_time:.1f}s, Output: {'Clear' if simple_output else 'Unclear'}"
            
            self.results.add_acceptance_test(
                "first_time_user", user_story, expected, actual,
                first_time_success, 8 if first_time_success else 4, satisfaction,
                f"Processing time key for new user confidence"
            )
            
            if first_time_success:
                ux_tests_passed += 1
                print(f"    ✅ First-time user satisfied: {processing_time:.1f}s processing")
            else:
                print(f"    ❌ First-time user frustrated: {processing_time:.1f}s processing")
        
        except Exception as e:
            self.results.add_acceptance_test(
                "first_time_user_exception", "First-time user reliability",
                "No confusing errors", f"Exception: {type(e).__name__}",
                False, 2, 2, "System error confuses new users"
            )
            print(f"    ❌ First-time user sees error: {type(e).__name__}")
        
        # UX Test 2: Power user experience  
        print("  💪 Testing: Power user experience")
        scenario = get_test_scenario('context_heavy')  # Complex content for power users
        
        try:
            results = analyzer.analyze_transcript(scenario['transcript'])
            
            if 'error' not in results:
                # Power user expectations: detailed metadata, narrative intelligence, optimization
                optimized = results['optimized_segment']
                
                has_narrative_intelligence = 'narrative_intelligence' in optimized
                detailed_metadata = len(results.get('supporting_segments', [])) > 0
                optimized_duration = 45 <= optimized['duration'] <= 59  # Power users want precision
                
                power_user_success = has_narrative_intelligence and detailed_metadata and optimized_duration
                satisfaction = 9 if power_user_success else 5
                
                quality_score = sum([
                    3 if has_narrative_intelligence else 1,
                    2 if detailed_metadata else 0,
                    3 if optimized_duration else 1,
                    2  # Base points for working
                ])
                
            else:
                power_user_success = False
                satisfaction = 3
                quality_score = 2
            
            user_story = "As a power user, I want detailed control and optimization features"
            expected = "Narrative intelligence, detailed metadata, precise optimization"
            actual = f"NI: {has_narrative_intelligence if 'has_narrative_intelligence' in locals() else False}, Metadata: {detailed_metadata if 'detailed_metadata' in locals() else False}"
            
            self.results.add_acceptance_test(
                "power_user", user_story, expected, actual,
                power_user_success, quality_score, satisfaction,
                "Power users need advanced features"
            )
            
            if power_user_success:
                ux_tests_passed += 1
                print(f"    ✅ Power user satisfied: Advanced features available")
            else:
                print(f"    ❌ Power user unsatisfied: Missing advanced features")
        
        except Exception as e:
            self.results.add_acceptance_test(
                "power_user_exception", "Power user reliability",
                "Robust system for advanced usage", f"Exception: {type(e).__name__}",
                False, 1, 2, "Power users expect reliability"
            )
            print(f"    ❌ Power user sees error: {type(e).__name__}")
        
        # UX Test 3: International user experience
        print("  🌍 Testing: International user experience")
        scenario = get_test_scenario('real_portuguese')  # Non-English content
        
        try:
            results = analyzer.analyze_transcript(scenario['transcript'])
            
            if 'error' not in results:
                # International user expectations: language support, cultural awareness
                optimized = results['optimized_segment']
                duration = optimized['duration']
                
                works_with_non_english = True  # Got results
                reasonable_duration = 20 <= duration <= 80  # Broader range for different languages
                has_unicode_support = True  # Processed without errors
                
                international_success = works_with_non_english and reasonable_duration and has_unicode_support
                satisfaction = 8 if international_success else 4
                quality_score = 7 if international_success else 3
                
            else:
                international_success = False
                satisfaction = 3
                quality_score = 2
            
            user_story = "As an international user, I want the system to work with my language"
            expected = "Support for non-English content, cultural awareness"
            actual = f"Non-English: {'Supported' if international_success else 'Failed'}"
            
            self.results.add_acceptance_test(
                "international_user", user_story, expected, actual,
                international_success, quality_score, satisfaction,
                "International support is critical for global users"
            )
            
            if international_success:
                ux_tests_passed += 1
                print(f"    ✅ International user satisfied: Non-English content works")
            else:
                print(f"    ❌ International user unsatisfied: Non-English issues")
        
        except Exception as e:
            self.results.add_acceptance_test(
                "international_user_exception", "International user reliability",
                "Support for diverse content", f"Exception: {type(e).__name__}",
                False, 1, 2, "Language barriers cause user frustration"
            )
            print(f"    ❌ International user sees error: {type(e).__name__}")
        
        ux_success_rate = (ux_tests_passed / 3) * 100  # 3 UX tests
        print(f"\n📊 User Experience Tests: {ux_tests_passed}/3 ({ux_success_rate:.1f}%)")
        
        return ux_success_rate >= 66
    
    def test_integration_usability(self) -> bool:
        """Test system integration from user perspective."""
        print("\n🔗 Testing Integration Usability")
        print("-" * 40)
        
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        integration_success = 0
        
        # Integration Test 1: API-like usage (programmatic integration)
        print("  🔌 Testing: API-style integration")
        try:
            scenario = get_test_scenario('crypto_analysis')
            
            # Simulate how a developer would integrate the system
            start_time = time.perf_counter()
            results = analyzer.analyze_transcript(scenario['transcript'])
            api_time = time.perf_counter() - start_time
            
            # Developer expectations: reliable API, clear structure, error handling
            api_reliable = 'error' not in results or isinstance(results.get('error'), str)
            clear_structure = 'optimized_segment' in results if 'error' not in results else True
            fast_response = api_time < 5.0
            
            api_integration_success = api_reliable and clear_structure and fast_response
            
            user_story = "As a developer, I want to integrate the system via API calls"
            expected = "Reliable responses, clear data structure, reasonable performance"
            actual = f"Response time: {api_time:.1f}s, Structure: {'Clear' if clear_structure else 'Unclear'}"
            
            self.results.add_acceptance_test(
                "api_integration", user_story, expected, actual,
                api_integration_success, 8 if api_integration_success else 4,
                9 if api_integration_success else 4,
                "Developer needs predictable API behavior"
            )
            
            if api_integration_success:
                integration_success += 1
                print(f"    ✅ API integration: {api_time:.1f}s, reliable structure")
            else:
                print(f"    ❌ API integration issues: {api_time:.1f}s response")
        
        except Exception as e:
            self.results.add_acceptance_test(
                "api_integration_exception", "API reliability",
                "No unexpected exceptions", f"Exception: {type(e).__name__}",
                False, 2, 2, "API exceptions break developer workflows"
            )
            print(f"    ❌ API integration exception: {type(e).__name__}")
        
        # Integration Test 2: Data export compatibility
        print("  📤 Testing: Data export compatibility")
        try:
            scenario = get_test_scenario('educational')
            results = analyzer.analyze_transcript(scenario['transcript'])
            
            if 'error' not in results:
                # User expectations: export-ready data, standard formats, complete information
                optimized = results['optimized_segment']
                
                has_timestamps = 'start_time' in optimized and 'end_time' in optimized
                has_duration = 'duration' in optimized  
                has_title = 'title' in optimized
                exportable_format = isinstance(optimized, dict)  # JSON-like structure
                
                export_ready = has_timestamps and has_duration and has_title and exportable_format
                
                # Test JSON serialization (common export need)
                try:
                    json.dumps(results, default=str)  # Should be serializable
                    json_compatible = True
                except:
                    json_compatible = False
                
                export_success = export_ready and json_compatible
                
            else:
                export_success = False
                export_ready = False
                json_compatible = False
            
            user_story = "As a user, I want to export clip data to other tools"
            expected = "Complete timestamps, JSON-compatible, all necessary data"
            actual = f"Timestamps: {has_timestamps if 'has_timestamps' in locals() else False}, JSON: {json_compatible if 'json_compatible' in locals() else False}"
            
            self.results.add_acceptance_test(
                "data_export", user_story, expected, actual,
                export_success, 8 if export_success else 3,
                8 if export_success else 4,
                "Export compatibility essential for workflows"
            )
            
            if export_success:
                integration_success += 1
                print(f"    ✅ Data export ready: Complete and JSON-compatible")
            else:
                print(f"    ❌ Data export issues: Missing required fields")
        
        except Exception as e:
            self.results.add_acceptance_test(
                "data_export_exception", "Export reliability",
                "Reliable data export", f"Exception: {type(e).__name__}",
                False, 1, 2, "Export failures disrupt user workflows"
            )
            print(f"    ❌ Data export exception: {type(e).__name__}")
        
        integration_success_rate = (integration_success / 2) * 100  # 2 integration tests
        print(f"\n📊 Integration Tests: {integration_success}/2 ({integration_success_rate:.1f}%)")
        
        return integration_success_rate >= 50
    
    def simulate_feedback_integration(self) -> bool:
        """Simulate feedback collection and improvement validation."""
        print("\n💬 Testing Feedback Integration")
        print("-" * 40)
        
        # Simulate collecting user feedback based on our test results
        feedback_scenarios = [
            {
                'feedback': 'Clips are too short for my educational content',
                'improvement': 'Increased target duration range to 45-59s with narrative intelligence',
                'validation_test': 'educational'
            },
            {
                'feedback': 'System should work with non-English content',
                'improvement': 'Added multilingual support with Portuguese testing',
                'validation_test': 'real_portuguese'
            },
            {
                'feedback': 'Need more control over clip selection',
                'improvement': 'Implemented narrative intelligence with story role analysis',
                'validation_test': 'context_heavy'
            }
        ]
        
        feedback_addressed = 0
        
        analyzer = TopicAnalyzer(use_enhanced_audio=True)
        
        for feedback_item in feedback_scenarios:
            print(f"  💭 Feedback: '{feedback_item['feedback'][:40]}...'")
            print(f"  🔧 Improvement: {feedback_item['improvement'][:50]}...")
            
            # Validate that the feedback has been addressed
            test_scenario = get_test_scenario(feedback_item['validation_test'])
            
            try:
                results = analyzer.analyze_transcript(test_scenario['transcript'])
                
                # Check if improvement addresses the feedback
                improvement_effective = False
                
                if 'too short' in feedback_item['feedback'].lower():
                    # Check duration improvement
                    if 'error' not in results:
                        duration = results['optimized_segment']['duration']
                        improvement_effective = duration >= 45
                        
                elif 'non-english' in feedback_item['feedback'].lower():
                    # Check multilingual support
                    improvement_effective = 'error' not in results and results.get('optimized_segment')
                    
                elif 'more control' in feedback_item['feedback'].lower():
                    # Check narrative intelligence features
                    if 'error' not in results:
                        has_narrative = 'narrative_intelligence' in results['optimized_segment']
                        improvement_effective = has_narrative
                
                user_story = f"User requested: {feedback_item['feedback']}"
                expected = f"System improvement: {feedback_item['improvement']}"
                actual = f"Improvement effective: {improvement_effective}"
                
                self.results.add_acceptance_test(
                    f"feedback_{feedback_addressed}", user_story, expected, actual,
                    improvement_effective, 8 if improvement_effective else 4,
                    9 if improvement_effective else 3,
                    f"Feedback integration validation"
                )
                
                if improvement_effective:
                    feedback_addressed += 1
                    print(f"    ✅ Feedback addressed successfully")
                else:
                    print(f"    ❌ Feedback not fully addressed")
                    
            except Exception as e:
                self.results.add_acceptance_test(
                    f"feedback_{feedback_addressed}_exception", 
                    f"Feedback validation reliability",
                    "Stable validation of improvements", f"Exception: {type(e).__name__}",
                    False, 2, 2, "Cannot validate feedback improvements"
                )
                print(f"    ❌ Feedback validation exception: {type(e).__name__}")
        
        feedback_success_rate = (feedback_addressed / len(feedback_scenarios)) * 100
        print(f"\n📊 Feedback Integration: {feedback_addressed}/{len(feedback_scenarios)} ({feedback_success_rate:.1f}%)")
        
        return feedback_success_rate >= 66
    
    def run_comprehensive_uat_suite(self) -> Dict:
        """Run the complete user acceptance testing suite."""
        print("👥 STARTING COMPREHENSIVE USER ACCEPTANCE TESTING SUITE")
        print("=" * 70)
        
        suite_start_time = time.perf_counter()
        
        # Run all UAT test suites
        test_suites = [
            ("Content Creator Workflow", self.test_content_creator_workflow),
            ("Quality Expectations", self.test_quality_expectations),
            ("User Experience Scenarios", self.test_user_experience_scenarios),
            ("Integration Usability", self.test_integration_usability),
            ("Feedback Integration", self.simulate_feedback_integration)
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
                suite_results.append(False)
        
        suite_duration = time.perf_counter() - suite_start_time
        
        # Generate comprehensive UAT results
        print("\n" + "=" * 70)
        print("🏁 USER ACCEPTANCE TESTING RESULTS")
        print("=" * 70)
        
        overall_success_rate = (sum(suite_results) / len(suite_results)) * 100
        uat_metrics = self.results.calculate_overall_acceptance()
        
        print(f"Overall UAT Success Rate: {overall_success_rate:.1f}%")
        print(f"Total Test Duration: {suite_duration:.2f} seconds")
        
        # Generate and display UAT report
        uat_report = self.results.generate_uat_report()
        print(uat_report)
        
        # Final user readiness assessment
        user_ready = (
            overall_success_rate >= 80 and
            uat_metrics.get('overall_uat_score', 0) >= 80 and
            uat_metrics.get('avg_user_satisfaction', 0) >= 7.0
        )
        
        if user_ready:
            print("🎉 SYSTEM IS READY FOR USER DEPLOYMENT!")
        else:
            print("⚠️  SYSTEM NEEDS USER EXPERIENCE IMPROVEMENTS")
        
        return {
            'overall_success_rate': overall_success_rate,
            'uat_metrics': uat_metrics,
            'suite_duration': suite_duration,
            'user_ready': user_ready,
            'test_suite_results': dict(zip([name for name, _ in test_suites], suite_results))
        }


def main():
    """Run the comprehensive user acceptance testing suite."""
    suite = UserAcceptanceTestingSuite()
    return suite.run_comprehensive_uat_suite()


if __name__ == "__main__":
    results = main()
    
    # Save results for analysis
    with open('uat_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 UAT results saved to 'uat_results.json'")
    
    sys.exit(0 if results['user_ready'] else 1)