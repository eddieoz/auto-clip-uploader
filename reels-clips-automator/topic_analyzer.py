"""
Topic Transition Detection Module

This module provides functionality to analyze transcripts and identify topic transitions,
with special focus on identifying concluding topics from 90-second content segments.

Designed for live stream content where the final topic is the primary extraction target.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class TopicSegment:
    """Represents a topic segment within the transcript."""
    start_time: float
    end_time: float
    start_time_srt: str
    end_time_srt: str
    text: str
    topic_keywords: List[str]
    topic_confidence: float
    is_concluding: bool = False
    supporting_segments: List[int] = None

    def __post_init__(self):
        if self.supporting_segments is None:
            self.supporting_segments = []

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'start_time_srt': self.start_time_srt,
            'end_time_srt': self.end_time_srt,
            'text': self.text,
            'topic_keywords': self.topic_keywords,
            'topic_confidence': self.topic_confidence,
            'is_concluding': self.is_concluding,
            'supporting_segments': self.supporting_segments,
            'duration': self.duration
        }


class TopicAnalyzer:
    """
    Analyzes transcripts to identify topic transitions and concluding topics.
    
    This class is specifically designed for 90-second live stream content where:
    - Multiple topics may be discussed
    - The final/concluding topic is the primary extraction target
    - Supporting context from earlier segments enhances the final topic
    
    Phase 3 Enhancement: Integrated with enhanced audio analysis for better
    narrative flow detection and natural boundary identification.
    """

    def __init__(self, use_enhanced_audio=True):
        """
        Initialize the topic analyzer with configuration options.
        
        Args:
            use_enhanced_audio: Whether to use enhanced audio analysis for
                               better narrative flow detection
        """
        self.use_enhanced_audio = use_enhanced_audio
        # Common topic transition indicators
        self.transition_phrases = [
            "now", "so", "but", "however", "anyway", "moving on", "speaking of",
            "on the other hand", "meanwhile", "actually", "by the way", "also",
            "furthermore", "in addition", "let me talk about", "another thing",
            "what about", "regarding", "concerning", "as for"
        ]
        
        # Concluding topic indicators
        self.conclusion_phrases = [
            "finally", "in conclusion", "to summarize", "bottom line", "the point is",
            "what this means", "so basically", "to wrap up", "in the end",
            "the key takeaway", "most importantly", "here's the thing"
        ]
        
        # Topic coherence indicators
        self.coherence_keywords = [
            "because", "therefore", "thus", "hence", "as a result", "consequently",
            "due to", "leads to", "causes", "results in", "means that"
        ]

    def parse_srt_segments(self, transcript_content: str) -> List[TopicSegment]:
        """
        Parse SRT transcript content into TopicSegment objects.
        
        Args:
            transcript_content: Raw SRT format transcript
            
        Returns:
            List of TopicSegment objects
        """
        segments = []
        current_segment = {}
        
        for line in transcript_content.split("\n"):
            line = line.strip()
            if not line:
                if current_segment and current_segment.get('text') and current_segment.get('start') is not None:
                    # Convert to TopicSegment
                    segment = TopicSegment(
                        start_time=current_segment['start'],
                        end_time=current_segment['end'],
                        start_time_srt=current_segment['start_time'],
                        end_time_srt=current_segment['end_time'],
                        text=current_segment['text'],
                        topic_keywords=[],  # Will be filled by analyze_topics
                        topic_confidence=0.0  # Will be calculated by analyze_topics
                    )
                    segments.append(segment)
                current_segment = {}
                continue

            if "-->" in line:
                # Parse timestamp line
                start_time, end_time = line.split(" --> ")
                # Convert SRT timestamp to seconds
                start_seconds = self._srt_to_seconds(start_time)
                end_seconds = self._srt_to_seconds(end_time)
                
                current_segment["start_time"] = start_time
                current_segment["end_time"] = end_time
                current_segment["start"] = start_seconds
                current_segment["end"] = end_seconds
            elif line and not line.isdigit() and not current_segment.get("text"):
                # This is the text line
                current_segment["text"] = line

        # Add the last segment if exists
        if current_segment and current_segment.get('text') and current_segment.get('start') is not None:
            segment = TopicSegment(
                start_time=current_segment['start'],
                end_time=current_segment['end'],
                start_time_srt=current_segment['start_time'],
                end_time_srt=current_segment['end_time'],
                text=current_segment['text'],
                topic_keywords=[],
                topic_confidence=0.0
            )
            segments.append(segment)

        return segments

    def _srt_to_seconds(self, srt_time: str) -> float:
        """Convert SRT timestamp to seconds."""
        time_parts, ms = srt_time.split(",")
        h, m, s = map(int, time_parts.split(":"))
        total_seconds = h * 3600 + m * 60 + s + int(ms) / 1000
        return total_seconds

    def _extract_keywords(self, text: str, min_word_length: int = 3) -> List[str]:
        """
        Extract meaningful keywords from text.
        
        Args:
            text: Input text to analyze
            min_word_length: Minimum word length to consider
            
        Returns:
            List of keywords
        """
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose',
            'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'
        }

        # Extract words, convert to lowercase, remove punctuation
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if len(word) >= min_word_length and word not in stop_words
        ]
        
        # Count frequency and return most common
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:10]]  # Top 10 keywords

    def analyze_topic_transitions(self, segments: List[TopicSegment]) -> List[TopicSegment]:
        """
        Analyze segments to identify topic transitions and extract keywords.
        
        Args:
            segments: List of TopicSegment objects
            
        Returns:
            Updated segments with topic analysis
        """
        for i, segment in enumerate(segments):
            # Extract keywords for this segment
            segment.topic_keywords = self._extract_keywords(segment.text)
            
            # Calculate topic confidence based on keyword density and coherence
            segment.topic_confidence = self._calculate_topic_confidence(segment)
            
            # Check for transition indicators
            has_transition = any(
                phrase in segment.text.lower() 
                for phrase in self.transition_phrases
            )
            
            # Check for conclusion indicators
            has_conclusion = any(
                phrase in segment.text.lower() 
                for phrase in self.conclusion_phrases
            )
            
            # Boost confidence for segments with clear indicators
            if has_conclusion:
                segment.topic_confidence += 0.2
            elif has_transition:
                segment.topic_confidence += 0.1

        return segments

    def _calculate_topic_confidence(self, segment: TopicSegment) -> float:
        """
        Calculate confidence score for a topic segment.
        
        Args:
            segment: TopicSegment to analyze
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_score = 0.5
        text = segment.text.lower()
        
        # Boost for keyword density
        keyword_density = len(segment.topic_keywords) / max(len(text.split()), 1)
        keyword_bonus = min(keyword_density * 2, 0.3)
        
        # Boost for coherence indicators
        coherence_bonus = sum(0.05 for phrase in self.coherence_keywords if phrase in text)
        coherence_bonus = min(coherence_bonus, 0.2)
        
        # Boost for longer segments (more content)
        duration_bonus = min(segment.duration / 30.0 * 0.1, 0.1)
        
        total_confidence = base_score + keyword_bonus + coherence_bonus + duration_bonus
        return min(total_confidence, 1.0)

    def identify_concluding_topic(self, segments: List[TopicSegment]) -> Optional[TopicSegment]:
        """
        Identify the concluding topic from analyzed segments.
        
        For 90-second content, this focuses on the final topic discussed,
        giving preference to segments that appear later in the timeline.
        
        Args:
            segments: List of analyzed TopicSegment objects
            
        Returns:
            TopicSegment representing the concluding topic, or None if not found
        """
        if not segments:
            return None

        # Score segments with recency bias (later segments score higher)
        scored_segments = []
        total_duration = segments[-1].end_time if segments else 90.0
        
        for segment in segments:
            # Base score from topic confidence
            score = segment.topic_confidence
            
            # Recency bonus (segments in final 30 seconds get higher score)
            time_position = segment.start_time / total_duration
            if time_position > 0.67:  # Final third
                score += 0.3
            elif time_position > 0.33:  # Middle third
                score += 0.1
                
            # Duration bonus (longer segments are more substantial)
            duration_bonus = min(segment.duration / 20.0, 0.2)
            score += duration_bonus
            
            # Conclusion phrase bonus
            has_conclusion = any(
                phrase in segment.text.lower() 
                for phrase in self.conclusion_phrases
            )
            if has_conclusion:
                score += 0.2
                
            scored_segments.append((segment, score))

        # Sort by score and return the highest scoring segment
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        concluding_segment = scored_segments[0][0]
        concluding_segment.is_concluding = True
        
        return concluding_segment

    def find_supporting_context(
        self, 
        concluding_segment: TopicSegment, 
        all_segments: List[TopicSegment],
        max_supporting_duration: float = 30.0
    ) -> List[int]:
        """
        PHASE 4: Enhanced supporting context tracing with narrative intelligence.
        
        Story 1.2: Supporting context tracing enhancement
        - Smarter cross-segment relationship analysis
        - Narrative arc completion validation
        - Context-aware selection improvements
        
        Args:
            concluding_segment: The identified concluding topic segment
            all_segments: All available segments
            max_supporting_duration: Maximum duration of supporting context
            
        Returns:
            List of segment indices that provide supporting context
        """
        supporting_indices = []
        supporting_duration = 0.0
        
        # Look for segments before the concluding segment
        concluding_index = all_segments.index(concluding_segment)
        
        # Phase 4: Enhanced narrative arc analysis
        narrative_segments = self._trace_narrative_arc(all_segments, concluding_segment, concluding_index)
        
        # Phase 4: Context-aware selection with cross-segment relationships
        selected_segments = self._select_context_aware_segments(
            narrative_segments, concluding_segment, max_supporting_duration
        )
        
        # Build supporting indices from selected segments
        for segment_data in selected_segments:
            segment_index = segment_data['index']
            supporting_indices.append(segment_index)
            supporting_duration += segment_data['segment'].duration

        return sorted(supporting_indices)  # Return in chronological order

    def _trace_narrative_arc(
        self,
        all_segments: List[TopicSegment],
        concluding_segment: TopicSegment,
        concluding_index: int
    ) -> List[Dict]:
        """
        PHASE 4 Story 1.2: Trace the narrative arc leading to the concluding topic.
        
        Enhanced cross-segment relationship analysis to identify narrative flow.
        """
        narrative_segments = []
        
        # Analyze segments in reverse chronological order
        for i in range(concluding_index - 1, -1, -1):
            segment = all_segments[i]
            
            # Calculate enhanced relevance with narrative intelligence
            relevance_data = self._calculate_narrative_relevance(
                segment, concluding_segment, all_segments, i, concluding_index
            )
            
            narrative_segments.append({
                'index': i,
                'segment': segment,
                'relevance_score': relevance_data['total_score'],
                'narrative_role': relevance_data['narrative_role'],
                'connection_strength': relevance_data['connection_strength'],
                'arc_position': relevance_data['arc_position']
            })
        
        # Sort by narrative importance (relevance + arc position)
        narrative_segments.sort(
            key=lambda x: x['relevance_score'] + x['arc_position'], 
            reverse=True
        )
        
        return narrative_segments
    
    def _calculate_narrative_relevance(
        self,
        segment: TopicSegment,
        concluding_segment: TopicSegment,
        all_segments: List[TopicSegment],
        segment_index: int,
        concluding_index: int
    ) -> Dict:
        """
        PHASE 4: Enhanced relevance calculation with narrative intelligence.
        
        Story 1.2: Cross-segment relationship analysis
        Story 2.2: Narrative completeness validation
        """
        # Original relevance factors
        segment_keywords = set(segment.topic_keywords)
        concluding_keywords = set(concluding_segment.topic_keywords)
        
        if not concluding_keywords:
            keyword_score = 0.1
        else:
            keyword_overlap = len(segment_keywords & concluding_keywords)
            keyword_score = keyword_overlap / len(concluding_keywords)
        
        # Enhanced proximity scoring with narrative context
        time_gap = concluding_segment.start_time - segment.end_time
        proximity_score = max(0, 1.0 - (time_gap / 45.0))  # Extended range for narrative flow
        
        # Coherence indicator bonus
        has_coherence = any(
            phrase in segment.text.lower() 
            for phrase in self.coherence_keywords
        )
        coherence_bonus = 0.2 if has_coherence else 0
        
        # PHASE 4: Enhanced narrative analysis
        narrative_role = self._identify_narrative_role(segment, concluding_segment)
        connection_strength = self._analyze_cross_segment_connections(
            segment, concluding_segment, all_segments, segment_index, concluding_index
        )
        arc_position = self._calculate_arc_position(segment_index, concluding_index)
        
        # PHASE 4: Narrative completeness factors
        completeness_score = self._validate_narrative_completeness(
            segment, concluding_segment, narrative_role
        )
        
        # Enhanced total relevance with narrative intelligence
        total_relevance = (
            keyword_score * 0.25 +           # Reduced keyword weight
            proximity_score * 0.20 +        # Reduced proximity weight
            coherence_bonus +                # Coherence bonus
            connection_strength * 0.30 +    # NEW: Cross-segment connections
            completeness_score * 0.25       # NEW: Narrative completeness
        )
        
        return {
            'total_score': min(total_relevance, 1.0),
            'narrative_role': narrative_role,
            'connection_strength': connection_strength,
            'arc_position': arc_position,
            'completeness_score': completeness_score
        }
    
    def _identify_narrative_role(self, segment: TopicSegment, concluding_segment: TopicSegment) -> str:
        """
        PHASE 4 Story 2.2: Identify the narrative role of a segment.
        """
        segment_text = segment.text.lower()
        
        # Setup/Introduction indicators
        setup_phrases = ['let me explain', 'first', 'to begin', 'initially', 'starting with']
        if any(phrase in segment_text for phrase in setup_phrases):
            return 'setup'
        
        # Development/Elaboration indicators  
        development_phrases = ['furthermore', 'additionally', 'also', 'moreover', 'next']
        if any(phrase in segment_text for phrase in development_phrases):
            return 'development'
        
        # Bridge/Transition indicators
        bridge_phrases = ['however', 'but', 'on the other hand', 'meanwhile', 'then']
        if any(phrase in segment_text for phrase in bridge_phrases):
            return 'bridge'
        
        # Evidence/Support indicators
        evidence_phrases = ['for example', 'such as', 'specifically', 'in fact', 'actually']
        if any(phrase in segment_text for phrase in evidence_phrases):
            return 'evidence'
        
        return 'context'  # Default role
    
    def _analyze_cross_segment_connections(
        self,
        segment: TopicSegment,
        concluding_segment: TopicSegment,
        all_segments: List[TopicSegment],
        segment_index: int,
        concluding_index: int
    ) -> float:
        """
        PHASE 4 Story 1.2: Analyze connections between segments in the narrative arc.
        """
        connection_score = 0.0
        
        # Direct connection to concluding segment
        direct_connection = self._measure_semantic_connection(segment, concluding_segment)
        connection_score += direct_connection * 0.5
        
        # Connections to adjacent segments (narrative flow)
        adjacent_connections = 0.0
        adjacent_count = 0
        
        # Check connection with next segment
        if segment_index + 1 < len(all_segments):
            next_segment = all_segments[segment_index + 1]
            adjacent_connections += self._measure_semantic_connection(segment, next_segment)
            adjacent_count += 1
        
        # Check connection with previous segment  
        if segment_index > 0:
            prev_segment = all_segments[segment_index - 1]
            adjacent_connections += self._measure_semantic_connection(prev_segment, segment)
            adjacent_count += 1
        
        if adjacent_count > 0:
            connection_score += (adjacent_connections / adjacent_count) * 0.3
        
        # Chain connection (how well segment fits in the narrative chain)
        chain_strength = self._measure_narrative_chain_strength(
            all_segments, segment_index, concluding_index
        )
        connection_score += chain_strength * 0.2
        
        return min(connection_score, 1.0)
    
    def _measure_semantic_connection(self, segment1: TopicSegment, segment2: TopicSegment) -> float:
        """
        Measure semantic connection between two segments.
        """
        # Simple keyword-based semantic connection
        keywords1 = set(segment1.topic_keywords)
        keywords2 = set(segment2.topic_keywords)
        
        if not keywords1 or not keywords2:
            return 0.1
        
        # Jaccard similarity
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        if union == 0:
            return 0.1
            
        return intersection / union
    
    def _measure_narrative_chain_strength(
        self, all_segments: List[TopicSegment], start_index: int, end_index: int
    ) -> float:
        """
        Measure how well segments form a narrative chain.
        """
        if start_index >= end_index or end_index - start_index < 2:
            return 0.0
        
        chain_segments = all_segments[start_index:end_index + 1]
        total_connections = 0.0
        connection_count = 0
        
        # Measure connections between consecutive segments in the chain
        for i in range(len(chain_segments) - 1):
            connection = self._measure_semantic_connection(chain_segments[i], chain_segments[i + 1])
            total_connections += connection
            connection_count += 1
        
        if connection_count == 0:
            return 0.0
            
        return total_connections / connection_count
    
    def _calculate_arc_position(self, segment_index: int, concluding_index: int) -> float:
        """
        Calculate the position of a segment in the narrative arc.
        """
        if concluding_index == 0:
            return 0.0
            
        # Position in the arc (0 = beginning, 1 = just before concluding)
        position = (concluding_index - segment_index) / concluding_index
        
        # Bonus for segments closer to the conclusion
        return 1.0 - position
    
    def _validate_narrative_completeness(
        self, segment: TopicSegment, concluding_segment: TopicSegment, narrative_role: str
    ) -> float:
        """
        PHASE 4 Story 2.2: Validate how much this segment contributes to narrative completeness.
        """
        completeness_score = 0.0
        
        # Role-based completeness scoring
        role_scores = {
            'setup': 0.8,        # High value for setting up the conclusion
            'evidence': 0.7,     # High value for supporting evidence
            'development': 0.6,  # Good value for developing the topic
            'bridge': 0.5,       # Moderate value for transitions
            'context': 0.3       # Basic value for general context
        }
        
        completeness_score += role_scores.get(narrative_role, 0.3)
        
        # Information density bonus
        if len(segment.topic_keywords) >= 3:
            completeness_score += 0.2
        
        # Length appropriateness (not too short, not too long)
        if 5.0 <= segment.duration <= 20.0:
            completeness_score += 0.1
        
        return min(completeness_score, 1.0)
    
    def _select_context_aware_segments(
        self, narrative_segments: List[Dict], concluding_segment: TopicSegment, max_duration: float
    ) -> List[Dict]:
        """
        PHASE 4 Story 4.2: Context-aware selection of supporting segments.
        
        Select segments that create the most complete narrative arc.
        """
        selected_segments = []
        total_duration = 0.0
        
        # Ensure we include different narrative roles for completeness
        role_requirements = {
            'setup': False,
            'evidence': False,
            'development': False
        }
        
        # First pass: Select high-scoring segments that fulfill narrative roles
        for segment_data in narrative_segments:
            segment = segment_data['segment']
            narrative_role = segment_data['narrative_role']
            
            # Check if we have budget and this adds narrative value
            if total_duration + segment.duration <= max_duration:
                should_include = False
                
                # Include if high relevance score
                if segment_data['relevance_score'] > 0.4:
                    should_include = True
                
                # Include if fills a needed narrative role
                elif narrative_role in role_requirements and not role_requirements[narrative_role]:
                    if segment_data['relevance_score'] > 0.2:
                        should_include = True
                        role_requirements[narrative_role] = True
                
                if should_include:
                    selected_segments.append(segment_data)
                    total_duration += segment.duration
        
        # Second pass: Fill remaining duration with best available segments
        for segment_data in narrative_segments:
            if segment_data in selected_segments:
                continue
                
            segment = segment_data['segment']
            if total_duration + segment.duration <= max_duration:
                if segment_data['relevance_score'] > 0.25:  # Lower threshold for fill
                    selected_segments.append(segment_data)
                    total_duration += segment.duration
        
        return selected_segments
    
    def _calculate_relevance(
        self, 
        segment: TopicSegment, 
        concluding_segment: TopicSegment
    ) -> float:
        """
        Legacy relevance calculation for backward compatibility.
        Now uses enhanced narrative intelligence.
        """
        relevance_data = self._calculate_narrative_relevance(
            segment, concluding_segment, [segment, concluding_segment], 0, 1
        )
        return relevance_data['total_score']

    def create_optimized_segment(
        self, 
        concluding_segment: TopicSegment,
        supporting_segments: List[TopicSegment],
        target_duration_range: Tuple[float, float] = (45.0, 59.0)
    ) -> Dict:
        """
        PHASE 4: Enhanced segment optimization with narrative completeness validation.
        
        Story 2.2: Narrative completeness validation
        Story 4.2: Context-aware selection improvements
        
        Args:
            concluding_segment: The main concluding topic
            supporting_segments: Supporting context segments
            target_duration_range: Desired duration range (min, max) in seconds
            
        Returns:
            Dictionary with optimized segment information and narrative intelligence
        """
        min_duration, max_duration = target_duration_range
        
        # PHASE 4: Pre-validate narrative completeness
        completeness_report = self._validate_segment_completeness(
            concluding_segment, supporting_segments
        )
        
        # Start with the concluding segment
        all_segments = supporting_segments + [concluding_segment]
        all_segments.sort(key=lambda x: x.start_time)
        
        # Find optimal start and end times
        start_time = all_segments[0].start_time
        end_time = all_segments[-1].end_time
        current_duration = end_time - start_time
        
        # If too short, try to expand with narrative intelligence
        if current_duration < min_duration:
            # PHASE 4: Smart expansion based on narrative needs
            expanded_result = self._expand_segment_intelligently(
                all_segments, min_duration, completeness_report
            )
            start_time = expanded_result['start_time']
            end_time = expanded_result['end_time']
            current_duration = end_time - start_time
            
        # If too long, use narrative-aware trimming
        if current_duration > max_duration:
            # PHASE 4: Narrative-aware trimming instead of simple relevance
            trimmed_result = self._trim_segment_narratively(
                all_segments, concluding_segment, max_duration, completeness_report
            )
            start_time = trimmed_result['start_time']
            end_time = trimmed_result['end_time']
            current_duration = end_time - start_time
            supporting_segments = trimmed_result['supporting_segments']

        # PHASE 4: Final narrative completeness validation
        final_completeness = self._validate_final_segment_completeness(
            concluding_segment, supporting_segments, current_duration
        )
        
        # Create the final optimized segment with enhanced metadata
        return {
            'start_time': self._seconds_to_srt(start_time),
            'end_time': self._seconds_to_srt(end_time),
            'duration': current_duration,
            'title': self._generate_narrative_title(concluding_segment, supporting_segments),
            'concluding_topic': concluding_segment.to_dict(),
            'supporting_segments': [seg.to_dict() for seg in supporting_segments],
            'optimization_notes': {
                'target_range': target_duration_range,
                'achieved_duration': current_duration,
                'includes_conclusion': True,
                'supporting_context_count': len(supporting_segments)
            },
            # PHASE 4: Enhanced narrative intelligence metadata
            'narrative_intelligence': {
                'completeness_score': final_completeness['overall_score'],
                'narrative_arc_quality': final_completeness['arc_quality'],
                'context_coverage': final_completeness['context_coverage'],
                'story_roles_present': final_completeness['roles_present'],
                'phase4_enhanced': True
            }
        }

    def _seconds_to_srt(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs % 1) * 1000)
        secs = int(secs)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _validate_segment_completeness(
        self, concluding_segment: TopicSegment, supporting_segments: List[TopicSegment]
    ) -> Dict:
        """
        PHASE 4 Story 2.2: Validate narrative completeness of the segment selection.
        """
        # Analyze narrative roles present
        roles_present = set()
        total_info_density = 0.0
        
        for segment in supporting_segments:
            role = self._identify_narrative_role(segment, concluding_segment)
            roles_present.add(role)
            total_info_density += len(segment.topic_keywords)
        
        # Add concluding segment role
        roles_present.add('conclusion')
        total_info_density += len(concluding_segment.topic_keywords)
        
        # Calculate completeness metrics
        ideal_roles = {'setup', 'evidence', 'development', 'conclusion'}
        role_coverage = len(roles_present & ideal_roles) / len(ideal_roles)
        
        avg_info_density = total_info_density / (len(supporting_segments) + 1)
        
        return {
            'roles_present': list(roles_present),
            'role_coverage': role_coverage,
            'info_density': avg_info_density,
            'needs_setup': 'setup' not in roles_present,
            'needs_evidence': 'evidence' not in roles_present,
            'segment_count': len(supporting_segments) + 1
        }
    
    def _expand_segment_intelligently(
        self, all_segments: List[TopicSegment], min_duration: float, completeness_report: Dict
    ) -> Dict:
        """
        PHASE 4: Intelligently expand segment based on narrative needs.
        """
        start_time = all_segments[0].start_time
        end_time = all_segments[-1].end_time
        current_duration = end_time - start_time
        needed_duration = min_duration - current_duration
        
        # Smart expansion: expand more towards the beginning if we need setup
        if completeness_report['needs_setup']:
            # Expand 70% backwards for context, 30% forwards
            backward_expansion = needed_duration * 0.7
            forward_expansion = needed_duration * 0.3
        else:
            # Balanced expansion
            backward_expansion = needed_duration * 0.5
            forward_expansion = needed_duration * 0.5
        
        new_start = max(0, start_time - backward_expansion)
        new_end = end_time + forward_expansion
        
        return {
            'start_time': new_start,
            'end_time': new_end,
            'expansion_strategy': 'narrative_aware'
        }
    
    def _trim_segment_narratively(
        self, all_segments: List[TopicSegment], concluding_segment: TopicSegment, 
        max_duration: float, completeness_report: Dict
    ) -> Dict:
        """
        PHASE 4: Narratively-aware segment trimming that preserves story completeness.
        """
        # Always keep the concluding segment
        essential_segments = [concluding_segment]
        essential_duration = concluding_segment.duration
        
        # Rank supporting segments by narrative importance
        supporting_segments = [seg for seg in all_segments if seg != concluding_segment]
        
        if supporting_segments:
            # Score segments by narrative necessity
            scored_segments = []
            for segment in supporting_segments:
                narrative_role = self._identify_narrative_role(segment, concluding_segment)
                
                # Priority scoring based on narrative completeness needs
                priority_score = 0.5  # Base score
                
                if narrative_role == 'setup' and completeness_report['needs_setup']:
                    priority_score += 0.4
                elif narrative_role == 'evidence' and completeness_report['needs_evidence']:
                    priority_score += 0.3
                elif narrative_role == 'development':
                    priority_score += 0.2
                
                # Add relevance score
                relevance = self._calculate_relevance(segment, concluding_segment)
                priority_score += relevance * 0.3
                
                scored_segments.append((segment, priority_score))
            
            # Sort by priority and add segments that fit
            scored_segments.sort(key=lambda x: x[1], reverse=True)
            
            for segment, score in scored_segments:
                if essential_duration + segment.duration <= max_duration:
                    essential_segments.append(segment)
                    essential_duration += segment.duration
        
        # Sort by chronological order
        essential_segments.sort(key=lambda x: x.start_time)
        supporting_segments = [seg for seg in essential_segments if seg != concluding_segment]
        
        return {
            'start_time': essential_segments[0].start_time,
            'end_time': essential_segments[-1].end_time,
            'supporting_segments': supporting_segments,
            'trim_strategy': 'narrative_priority'
        }
    
    def _validate_final_segment_completeness(
        self, concluding_segment: TopicSegment, supporting_segments: List[TopicSegment], duration: float
    ) -> Dict:
        """
        PHASE 4 Story 2.2: Final validation of segment narrative completeness.
        """
        # Re-analyze final segment composition
        final_roles = set()
        for segment in supporting_segments:
            role = self._identify_narrative_role(segment, concluding_segment)
            final_roles.add(role)
        final_roles.add('conclusion')
        
        # Calculate final scores
        ideal_roles = {'setup', 'evidence', 'development', 'conclusion'}
        arc_quality = len(final_roles & ideal_roles) / len(ideal_roles)
        
        context_coverage = len(supporting_segments) / max(3, len(supporting_segments))  # Normalize
        context_coverage = min(context_coverage, 1.0)
        
        # Overall completeness score
        duration_score = 1.0 if 45 <= duration <= 59 else 0.8
        overall_score = (arc_quality * 0.4 + context_coverage * 0.3 + duration_score * 0.3)
        
        return {
            'overall_score': overall_score,
            'arc_quality': arc_quality,
            'context_coverage': context_coverage,
            'roles_present': list(final_roles),
            'duration_appropriateness': duration_score
        }
    
    def _generate_narrative_title(
        self, 
        concluding_segment: TopicSegment, 
        supporting_segments: List[TopicSegment]
    ) -> str:
        """
        PHASE 4: Enhanced title generation with narrative intelligence.
        
        Generate a title that reflects the complete narrative arc.
        """
        # Analyze the narrative flow for title generation
        if not concluding_segment.topic_keywords:
            return "Content Segment"
        
        primary_keywords = concluding_segment.topic_keywords[:2]
        title_parts = [" ".join(primary_keywords).title()]
        
        # Add narrative context based on supporting segments
        if supporting_segments:
            # Find the most informative supporting context
            context_keywords = []
            narrative_indicators = []
            
            for segment in supporting_segments:
                role = self._identify_narrative_role(segment, concluding_segment)
                
                if role == 'setup':
                    context_keywords.extend(segment.topic_keywords[:1])
                    narrative_indicators.append('Introduction')
                elif role == 'evidence':
                    context_keywords.extend(segment.topic_keywords[:1])
                    narrative_indicators.append('Analysis')
                elif role == 'development':
                    context_keywords.extend(segment.topic_keywords[:1])
            
            # Create narrative-aware title
            unique_context = [kw for kw in context_keywords if kw not in primary_keywords][:2]
            
            if unique_context:
                title_parts.append(" ".join(unique_context).title())
            elif narrative_indicators:
                title_parts.append(narrative_indicators[0])
        
        return " — ".join(title_parts)
    
    def _generate_title(
        self, 
        concluding_segment: TopicSegment, 
        supporting_segments: List[TopicSegment]
    ) -> str:
        """
        Legacy title generation - now uses narrative intelligence.
        """
        return self._generate_narrative_title(concluding_segment, supporting_segments)

    def analyze_with_enhanced_audio(
        self, 
        transcript_content: str, 
        audio_path: str = None
    ) -> Dict:
        """
        Enhanced analysis pipeline with audio integration for Phase 3.
        
        Args:
            transcript_content: Raw SRT format transcript
            audio_path: Path to audio file for enhanced analysis
            
        Returns:
            Complete analysis results with audio enhancements
        """
        # First do standard topic analysis (without audio to avoid recursion)
        base_results = self._analyze_transcript_only(transcript_content)
        
        if 'error' in base_results or not audio_path:
            return base_results
        
        try:
            # Try to use enhanced audio analysis
            from enhanced_audio_analyzer import EnhancedAudioAnalyzer
            
            print("🎵 Integrating enhanced audio analysis...")
            audio_analyzer = EnhancedAudioAnalyzer(audio_path)
            
            if not audio_analyzer.check_validity():
                print("⚠️  Audio analysis not available, using topic-only results")
                return base_results
            
            # Convert segments for audio analysis
            audio_segments = []
            for seg_dict in base_results['segments']:
                audio_segments.append({
                    'start': seg_dict['start_time'],
                    'end': seg_dict['end_time'],
                    'start_time': seg_dict['start_time_srt'],
                    'end_time': seg_dict['end_time_srt'],
                    'text': seg_dict['text']
                })
            
            # Get narrative-focused segments
            narrative_segments = audio_analyzer.find_narrative_moments(
                audio_segments, min_duration=10.0, max_duration=60.0
            )
            
            if narrative_segments:
                print(f"✅ Enhanced audio analysis found {len(narrative_segments)} narrative segments")
                
                # Get optimal boundaries
                optimal_boundaries = audio_analyzer.get_optimal_boundaries(
                    audio_segments, target_duration_range=(45.0, 59.0)
                )
                
                # Enhance the optimized segment with audio insights
                if optimal_boundaries:
                    base_results['optimized_segment'].update({
                        'audio_enhanced': True,
                        'natural_boundaries': optimal_boundaries.get('natural_boundaries', []),
                        'boundary_adjustment': optimal_boundaries.get('adjustment', 'none'),
                        'narrative_quality': narrative_segments[0].get('narrative_quality', 'medium'),
                        'audio_score': narrative_segments[0].get('score', 0.0)
                    })
                    
                    # Update duration if boundaries were adjusted
                    if optimal_boundaries.get('adjustment') != 'none':
                        new_start = optimal_boundaries['start']
                        new_end = optimal_boundaries['end']
                        base_results['optimized_segment']['start_time'] = self._seconds_to_srt(new_start)
                        base_results['optimized_segment']['end_time'] = self._seconds_to_srt(new_end)
                        base_results['optimized_segment']['duration'] = new_end - new_start
                        print(f"🎯 Audio-optimized duration: {base_results['optimized_segment']['duration']:.1f}s")
                
                # Add audio analysis summary
                base_results['audio_analysis'] = {
                    'narrative_segments_found': len(narrative_segments),
                    'best_segment_score': narrative_segments[0]['score'],
                    'narrative_quality': narrative_segments[0]['narrative_quality'],
                    'natural_boundaries_detected': len(optimal_boundaries.get('natural_boundaries', [])),
                    'audio_enhanced': True
                }
            else:
                print("⚠️  No suitable narrative segments found in audio analysis")
                base_results['audio_analysis'] = {'audio_enhanced': False, 'reason': 'no_suitable_segments'}
                
        except ImportError:
            print("⚠️  Enhanced audio analyzer not available, using topic-only results")
            base_results['audio_analysis'] = {'audio_enhanced': False, 'reason': 'module_unavailable'}
        except Exception as e:
            print(f"⚠️  Enhanced audio analysis failed: {str(e)}")
            base_results['audio_analysis'] = {'audio_enhanced': False, 'reason': f'error: {str(e)}'}
        
        return base_results

    def _analyze_transcript_only(self, transcript_content: str) -> Dict:
        """
        Core transcript analysis without audio integration.
        """
        # Parse segments
        segments = self.parse_srt_segments(transcript_content)
        
        if not segments:
            return {
                'error': 'No segments found in transcript',
                'segments': [],
                'concluding_topic': None,
                'optimized_segment': None
            }

        # Analyze topics
        segments = self.analyze_topic_transitions(segments)
        
        # Identify concluding topic
        concluding_topic = self.identify_concluding_topic(segments)
        
        if not concluding_topic:
            return {
                'error': 'No concluding topic identified',
                'segments': [seg.to_dict() for seg in segments],
                'concluding_topic': None,
                'optimized_segment': None
            }

        # Find supporting context
        supporting_indices = self.find_supporting_context(concluding_topic, segments)
        supporting_segments = [segments[i] for i in supporting_indices]
        
        # Update concluding topic with supporting segment references
        concluding_topic.supporting_segments = supporting_indices
        
        # Create optimized segment
        optimized_segment = self.create_optimized_segment(
            concluding_topic, 
            supporting_segments
        )

        return {
            'segments': [seg.to_dict() for seg in segments],
            'concluding_topic': concluding_topic.to_dict(),
            'supporting_segments': [seg.to_dict() for seg in supporting_segments],
            'optimized_segment': optimized_segment,
            'analysis_summary': {
                'total_segments': len(segments),
                'concluding_segment_index': segments.index(concluding_topic),
                'supporting_segment_count': len(supporting_segments),
                'total_content_duration': segments[-1].end_time if segments else 0,
                'optimized_duration': optimized_segment['duration']
            }
        }

    def analyze_transcript(self, transcript_content: str, audio_path: str = None) -> Dict:
        """
        Complete analysis pipeline for transcript content.
        
        Args:
            transcript_content: Raw SRT format transcript
            audio_path: Optional audio path for enhanced analysis
            
        Returns:
            Complete analysis results
        """
        # Use enhanced analysis if audio is available and enabled
        if self.use_enhanced_audio and audio_path:
            return self.analyze_with_enhanced_audio(transcript_content, audio_path)
        
        # Standard topic-only analysis
        base_results = self._analyze_transcript_only(transcript_content)
        
        # Add audio analysis status
        if 'error' not in base_results:
            base_results['audio_analysis'] = {'audio_enhanced': False, 'reason': 'not_available'}
        
        return base_results


# Testing functionality
if __name__ == "__main__":
    # Simple test with sample SRT content
    sample_srt = """1
00:00:00,000 --> 00:00:05,000
So we've been talking about crypto fundamentals today

2
00:00:05,000 --> 00:00:12,000
Market analysis shows some interesting trends in Bitcoin

3
00:00:12,000 --> 00:00:18,000
Technical indicators are pointing to a potential breakout

4
00:00:18,000 --> 00:00:25,000
But here's the key thing - looking at the volume patterns

5
00:00:25,000 --> 00:00:32,000
And the support levels we discussed earlier

6
00:00:32,000 --> 00:00:40,000
My prediction is we'll see Bitcoin hit $75,000 by end of year

7
00:00:40,000 --> 00:00:45,000
That's based on all the analysis we just covered"""

    analyzer = TopicAnalyzer()
    results = analyzer.analyze_transcript(sample_srt)
    
    print("=== Topic Analysis Results ===")
    print(json.dumps(results, indent=2))