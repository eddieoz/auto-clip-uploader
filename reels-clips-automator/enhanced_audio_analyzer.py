"""
Enhanced Audio Analyzer for Topic-Focused Content Processing

This module extends the original AudioAnalyzer with improvements for Phase 3:
- Story T.2: Updated audio analysis scoring weights for narrative flow
- Story 3.1: Prioritize content flow over virality characteristics  
- Story T.4: Natural boundary detection for better segment cuts

Focus: Content completeness and narrative flow over viral characteristics.
"""

import librosa
import numpy as np
from typing import List, Dict, Tuple, Optional
import scipy.signal
from dataclasses import dataclass

# Import the original AudioAnalyzer as base
from audio_analyzer import AudioAnalyzer as BaseAudioAnalyzer


@dataclass
class NarrativeFeatures:
    """Extended features for narrative flow analysis."""
    energy_consistency: float  # How consistent energy levels are
    speech_pace_stability: float  # Stability of speech rate
    natural_pauses: List[float]  # Detected natural pause points
    content_density: float  # Information density indicator
    flow_quality: float  # Overall narrative flow quality
    boundary_confidence: float  # Confidence in segment boundaries


class EnhancedAudioAnalyzer(BaseAudioAnalyzer):
    """
    Enhanced audio analyzer prioritizing content flow over viral characteristics.
    
    Key improvements:
    - Narrative flow scoring over energy spikes
    - Natural boundary detection
    - Content completeness metrics
    - Stable pacing over rapid delivery
    """

    def __init__(self, audio_path, sr=22050):
        """Initialize with enhanced analysis capabilities."""
        super().__init__(audio_path, sr)
        
        # Enhanced thresholds for content-focused analysis
        self.min_pause_duration = 0.3  # Minimum pause to consider as boundary
        self.energy_consistency_window = 3.0  # Window for consistency analysis
        self.speech_stability_threshold = 0.2  # Threshold for stable speech
        
        # Narrative flow weights (prioritize consistency over peaks)
        self.narrative_weights = {
            'energy_consistency': 0.25,      # Consistent energy > spiky energy
            'speech_stability': 0.20,        # Stable pace > rapid delivery
            'content_density': 0.25,         # Information density
            'natural_flow': 0.20,            # Natural speech patterns
            'boundary_quality': 0.10         # Clean segment boundaries
        }

    def analyze_energy_consistency(self, start_idx: int, end_idx: int) -> float:
        """
        Analyze energy consistency instead of just peak energy.
        Consistent energy indicates steady, informative content.
        """
        try:
            if start_idx >= end_idx or start_idx < 0 or end_idx > len(self.y):
                return 0.01
                
            segment = self.y[start_idx:end_idx]
            if len(segment) == 0:
                return 0.01
            
            # Calculate energy in windows
            window_size = int(self.energy_consistency_window * self.sr)
            windows = []
            
            for i in range(0, len(segment) - window_size, window_size // 2):
                window = segment[i:i + window_size]
                energy = np.mean(librosa.feature.rms(y=window)[0])
                windows.append(energy)
            
            if len(windows) < 2:
                # For short segments, use standard energy
                return np.mean(librosa.feature.rms(y=segment)[0])
            
            # Calculate consistency (lower variance = higher consistency)
            energy_variance = np.var(windows)
            mean_energy = np.mean(windows)
            
            # Normalize: high consistency + adequate energy = good score
            if mean_energy > 0:
                consistency_score = 1.0 / (1.0 + energy_variance / mean_energy)
                return consistency_score * mean_energy
            
            return 0.01
            
        except Exception as e:
            print(f"Error in analyze_energy_consistency: {e}")
            return 0.01

    def analyze_speech_stability(self, start_idx: int, end_idx: int) -> float:
        """
        Analyze speech rate stability instead of just rapid delivery.
        Stable speech indicates clear, understandable content.
        """
        try:
            if start_idx >= end_idx or start_idx < 0 or end_idx > len(self.y):
                return 0.01
                
            segment = self.y[start_idx:end_idx]
            if len(segment) == 0:
                return 0.01
            
            # Detect onsets in overlapping windows
            window_duration = 2.0  # 2-second windows
            window_size = int(window_duration * self.sr)
            
            if len(segment) < window_size:
                # For short segments, use standard rate
                onsets = librosa.onset.onset_detect(y=segment, sr=self.sr)
                return len(onsets) / (len(segment) / self.sr)
            
            rates = []
            for i in range(0, len(segment) - window_size, window_size // 2):
                window = segment[i:i + window_size]
                onsets = librosa.onset.onset_detect(y=window, sr=self.sr)
                rate = len(onsets) / window_duration
                rates.append(rate)
            
            if len(rates) < 2:
                return rates[0] if rates else 0.01
            
            # Calculate stability (lower variance in speech rate = better)
            mean_rate = np.mean(rates)
            rate_variance = np.var(rates)
            
            # Stable speech rate with good pace scores highest
            stability_score = 1.0 / (1.0 + rate_variance)
            
            # Prefer moderate speech rates (not too fast, not too slow)
            optimal_rate = 3.0  # onsets per second
            rate_quality = 1.0 - min(1.0, abs(mean_rate - optimal_rate) / optimal_rate)
            
            return stability_score * rate_quality * mean_rate
            
        except Exception as e:
            print(f"Error in analyze_speech_stability: {e}")
            return 0.01

    def detect_natural_boundaries(self, start_idx: int, end_idx: int) -> List[float]:
        """
        Detect natural pause points for better segment boundaries.
        """
        try:
            if start_idx >= end_idx or start_idx < 0 or end_idx > len(self.y):
                return []
                
            segment = self.y[start_idx:end_idx]
            if len(segment) == 0:
                return []
            
            # Calculate energy envelope
            hop_length = 512
            energy = librosa.feature.rms(y=segment, hop_length=hop_length)[0]
            
            # Find low-energy regions (potential pauses)
            energy_threshold = np.mean(energy) * 0.3  # 30% of mean energy
            low_energy_regions = energy < energy_threshold
            
            # Find continuous low-energy regions
            boundaries = []
            in_pause = False
            pause_start = 0
            
            for i, is_low in enumerate(low_energy_regions):
                time_pos = i * hop_length / self.sr
                
                if is_low and not in_pause:
                    # Start of potential pause
                    pause_start = time_pos
                    in_pause = True
                elif not is_low and in_pause:
                    # End of pause
                    pause_duration = time_pos - pause_start
                    if pause_duration >= self.min_pause_duration:
                        # This is a significant pause
                        boundaries.append(pause_start + pause_duration / 2)
                    in_pause = False
            
            return boundaries
            
        except Exception as e:
            print(f"Error in detect_natural_boundaries: {e}")
            return []

    def analyze_content_density(self, start_idx: int, end_idx: int) -> float:
        """
        Analyze content density - how much information is packed in the segment.
        Higher density with good flow indicates valuable content.
        """
        try:
            if start_idx >= end_idx or start_idx < 0 or end_idx > len(self.y):
                return 0.01
                
            segment = self.y[start_idx:end_idx]
            if len(segment) == 0:
                return 0.01
            
            # Analyze spectral features that indicate information density
            # More spectral complexity often indicates richer content
            spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=self.sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=self.sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=self.sr)[0]
            
            # Calculate information metrics
            centroid_variance = np.var(spectral_centroids)  # Spectral complexity
            rolloff_mean = np.mean(spectral_rolloff)        # Frequency richness
            contrast_mean = np.mean(spectral_contrast)      # Timbral variation
            
            # Combine metrics for content density score
            # Normalize each component
            density_score = (
                min(1.0, centroid_variance / 1000000) * 0.4 +  # Complexity
                min(1.0, rolloff_mean / self.sr) * 0.3 +        # Richness
                min(1.0, abs(contrast_mean) / 10) * 0.3         # Variation
            )
            
            return density_score
            
        except Exception as e:
            print(f"Error in analyze_content_density: {e}")
            return 0.01

    def analyze_narrative_flow(self, start_idx: int, end_idx: int) -> float:
        """
        Analyze overall narrative flow quality.
        Good flow has consistent energy, stable pace, and natural progression.
        """
        try:
            if start_idx >= end_idx or start_idx < 0 or end_idx > len(self.y):
                return 0.01
                
            segment = self.y[start_idx:end_idx]
            duration = (end_idx - start_idx) / self.sr
            
            if len(segment) == 0 or duration < 1.0:
                return 0.01
            
            # Analyze flow components
            energy_consistency = self.analyze_energy_consistency(start_idx, end_idx)
            speech_stability = self.analyze_speech_stability(start_idx, end_idx)
            content_density = self.analyze_content_density(start_idx, end_idx)
            
            # Natural boundaries analysis
            boundaries = self.detect_natural_boundaries(start_idx, end_idx)
            boundary_quality = min(1.0, len(boundaries) / (duration / 10))  # ~1 pause per 10s is good
            
            # Combine for overall flow score
            flow_score = (
                energy_consistency * self.narrative_weights['energy_consistency'] +
                speech_stability * self.narrative_weights['speech_stability'] +
                content_density * self.narrative_weights['content_density'] +
                boundary_quality * self.narrative_weights['boundary_quality']
            )
            
            # Natural flow bonus for smooth transitions
            if boundaries:
                # Even distribution of pauses indicates good pacing
                pause_intervals = []
                prev_boundary = 0
                for boundary in boundaries:
                    pause_intervals.append(boundary - prev_boundary)
                    prev_boundary = boundary
                
                if len(pause_intervals) > 1:
                    interval_consistency = 1.0 - (np.var(pause_intervals) / np.mean(pause_intervals))
                    flow_score *= (1.0 + interval_consistency * self.narrative_weights['natural_flow'])
            
            return min(flow_score, 1.0)
            
        except Exception as e:
            print(f"Error in analyze_narrative_flow: {e}")
            return 0.01

    def analyze_enhanced_segment(self, start_time: float, end_time: float) -> Dict:
        """
        Enhanced segment analysis focusing on narrative flow.
        """
        try:
            start_idx = int(start_time * self.sr)
            end_idx = int(end_time * self.sr)
            
            # Get original features for compatibility
            original_features = super().analyze_segment(start_time, end_time)
            
            # Add enhanced narrative features
            enhanced_features = {
                'energy_consistency': self.analyze_energy_consistency(start_idx, end_idx),
                'speech_stability': self.analyze_speech_stability(start_idx, end_idx),
                'content_density': self.analyze_content_density(start_idx, end_idx),
                'narrative_flow': self.analyze_narrative_flow(start_idx, end_idx),
                'natural_boundaries': self.detect_natural_boundaries(start_idx, end_idx),
                'boundary_count': len(self.detect_natural_boundaries(start_idx, end_idx))
            }
            
            # Combine original and enhanced features
            all_features = {**original_features, **enhanced_features}
            return all_features
            
        except Exception as e:
            print(f"Error in analyze_enhanced_segment: {e}")
            # Fallback to original analysis
            return super().analyze_segment(start_time, end_time)

    def find_narrative_moments(self, segments, min_duration=10.0, max_duration=60.0):
        """
        Find segments optimized for narrative flow instead of viral characteristics.
        
        This replaces find_viral_moments with a content-focused approach.
        """
        if not segments:
            print("No segments provided for narrative moment detection")
            return []

        # Filter out invalid segments before processing
        original_segment_count = len(segments)
        segments = [s for s in segments if isinstance(s, dict) and 'start' in s and 'end' in s]
        
        if len(segments) < original_segment_count:
            print(f"Warning: Skipped {original_segment_count - len(segments)} invalid segments.")

        if not segments:
            print("No valid segments remaining for narrative analysis.")
            return []
        
        print(f"Processing {len(segments)} segments for narrative optimization")
        
        # First pass - collect enhanced features
        all_features = []
        max_values = {
            'energy_consistency': 0.01,
            'speech_stability': 0.01,
            'content_density': 0.01,
            'narrative_flow': 0.01,
            'boundary_quality': 0.01
        }
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        # Process segments for narrative analysis
        for i, segment in enumerate(sorted_segments):
            start_time = segment['start']
            end_time = segment['end']
            
            # Ensure no overlaps
            if i < len(sorted_segments) - 1:
                next_start = sorted_segments[i + 1]['start']
                if end_time > next_start:
                    end_time = next_start
            
            duration = end_time - start_time
            
            if duration < min_duration or duration > max_duration:
                continue
            
            # Get enhanced features
            enhanced_features = self.analyze_enhanced_segment(start_time, end_time)
            
            # Update max values for normalization
            for key in max_values:
                if key in enhanced_features:
                    max_values[key] = max(max_values[key], enhanced_features[key])
            
            all_features.append({
                'segment': {
                    **segment,
                    'end': end_time
                },
                'features': enhanced_features,
                'duration': duration
            })
        
        print(f"Collected enhanced features for {len(all_features)} qualified segments")
        
        # Second pass - score segments with narrative focus
        narrative_segments = []
        for item in all_features:
            segment = item['segment']
            features = item['features']
            duration = item['duration']
            
            # Normalize enhanced features
            normalized = {}
            for key, max_val in max_values.items():
                if key in features and max_val > 0:
                    normalized[key] = features[key] / max_val
                else:
                    normalized[key] = 0
            
            # Calculate narrative-focused score
            score = (
                normalized.get('energy_consistency', 0) * 0.25 +      # Consistent > spiky
                normalized.get('speech_stability', 0) * 0.20 +        # Stable > rapid
                normalized.get('content_density', 0) * 0.25 +         # Information rich
                normalized.get('narrative_flow', 0) * 0.20 +          # Good flow
                (1.0 if features.get('boundary_count', 0) > 0 else 0.5) * 0.10  # Natural boundaries
            )
            
            # Duration optimization (prefer segments closer to 50 seconds for complete narratives)
            ideal_duration = 50.0
            duration_factor = 1.0 - min(1.0, abs(duration - ideal_duration) / ideal_duration)
            score = score * (1.0 + duration_factor * 0.3)
            
            # Completeness bonus for segments with good narrative flow
            if normalized.get('narrative_flow', 0) > 0.7:
                score *= 1.15  # 15% bonus for excellent narrative flow
            
            narrative_segments.append({
                'segment': segment,
                'score': score,
                'features': features,
                'normalized_features': normalized,
                'narrative_quality': 'high' if score > 0.7 else 'medium' if score > 0.4 else 'low'
            })
        
        # Sort by narrative score
        narrative_segments.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"Narrative analysis complete. Top segment score: {narrative_segments[0]['score']:.2f}" 
              if narrative_segments else "No segments qualified")
        
        return narrative_segments

    def get_optimal_boundaries(self, segments, target_duration_range=(45.0, 59.0)):
        """
        Find optimal segment boundaries using natural pause detection.
        """
        if not segments:
            return None
        
        # Find the best narrative segment
        narrative_segments = self.find_narrative_moments(segments)
        if not narrative_segments:
            return None
        
        best_segment = narrative_segments[0]['segment']
        
        # Get natural boundaries within this segment
        start_idx = int(best_segment['start'] * self.sr)
        end_idx = int(best_segment['end'] * self.sr)
        natural_boundaries = self.detect_natural_boundaries(start_idx, end_idx)
        
        # Adjust boundaries for optimal duration
        target_min, target_max = target_duration_range
        current_duration = best_segment['end'] - best_segment['start']
        
        if len(natural_boundaries) > 0 and (current_duration < target_min or current_duration > target_max):
            # Try to adjust using natural boundaries
            segment_start = best_segment['start']
            
            if current_duration < target_min:
                # Extend to nearest natural boundary
                needed_extension = target_min - current_duration
                for boundary in natural_boundaries:
                    if boundary > needed_extension / 2:
                        return {
                            'start': max(0, segment_start - needed_extension / 2),
                            'end': best_segment['end'] + needed_extension / 2,
                            'natural_boundaries': natural_boundaries,
                            'adjustment': 'extended'
                        }
            
            elif current_duration > target_max:
                # Trim to natural boundary
                excess_duration = current_duration - target_max
                for boundary in reversed(natural_boundaries):
                    if boundary < current_duration - excess_duration / 2:
                        return {
                            'start': segment_start + excess_duration / 2,
                            'end': boundary + segment_start,
                            'natural_boundaries': natural_boundaries,
                            'adjustment': 'trimmed'
                        }
        
        return {
            'start': best_segment['start'],
            'end': best_segment['end'],
            'natural_boundaries': natural_boundaries,
            'adjustment': 'none'
        }


# Testing functionality
if __name__ == "__main__":
    import os
    import sys
    
    def test_enhanced_audio_analyzer():
        """Test the enhanced audio analyzer."""
        # Create test audio file
        print("Testing Enhanced Audio Analyzer...")
        
        try:
            # Test with a sample audio file if available
            test_audio = "/tmp/test_audio.wav"
            
            # Create simple test audio
            import subprocess
            cmd = "ffmpeg -y -f lavfi -i sine=frequency=440:duration=10 -ac 1 -ar 22050 /tmp/test_audio.wav"
            subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL)
            
            if os.path.exists(test_audio):
                analyzer = EnhancedAudioAnalyzer(test_audio)
                print("Enhanced AudioAnalyzer initialized successfully")
                
                # Test enhanced features
                features = analyzer.analyze_enhanced_segment(0, 5)
                print("Enhanced features:")
                for key, value in features.items():
                    if isinstance(value, list):
                        print(f"  {key}: {len(value)} items")
                    else:
                        print(f"  {key}: {value:.3f}")
                
                # Test segments
                test_segments = [
                    {'start': 0, 'end': 3, 'text': 'Test segment 1'},
                    {'start': 3, 'end': 6, 'text': 'Test segment 2'},
                    {'start': 6, 'end': 10, 'text': 'Test segment 3'}
                ]
                
                narrative_segments = analyzer.find_narrative_moments(test_segments, min_duration=2, max_duration=8)
                print(f"\nFound {len(narrative_segments)} narrative segments")
                
                if narrative_segments:
                    best = narrative_segments[0]
                    print(f"Best segment: {best['segment']['text']}")
                    print(f"Score: {best['score']:.3f}")
                    print(f"Quality: {best['narrative_quality']}")
                
                print("\n✅ Enhanced Audio Analyzer test successful!")
                
            else:
                print("❌ Could not create test audio file")
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
    test_enhanced_audio_analyzer()
