import librosa
import numpy as np


class AudioAnalyzer:
    def __init__(self, audio_path, sr=22050):
        try:
            self.y, self.sr = librosa.load(audio_path, sr=sr)
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)
            self.is_valid = True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            self.is_valid = False
            self.y = np.zeros(1000)  # Small empty array
            self.sr = sr
            self.duration = 0

    def check_validity(self):
        return self.is_valid and len(self.y) > 0
    
    def analyze_energy(self, start_idx, end_idx):
        try:
            if start_idx >= end_idx or start_idx < 0 or end_idx > len(self.y):
                return 0.01
            segment = self.y[start_idx:end_idx]
            if len(segment) == 0:
                return 0.01
            energy = np.mean(librosa.feature.rms(y=segment)[0])
            return energy
        except Exception as e:
            print(f"Error in analyze_energy: {e}")
            return 0.01
    
    def analyze_speech_rate(self, start_idx, end_idx):
        try:
            if start_idx >= end_idx or start_idx < 0 or end_idx > len(self.y):
                return 0.01
            segment = self.y[start_idx:end_idx]
            if len(segment) == 0:
                return 0.01
            onsets = librosa.onset.onset_detect(y=segment, sr=self.sr)
            speech_rate = len(onsets) / (len(segment) / self.sr)
            return speech_rate
        except Exception as e:
            print(f"Error in analyze_speech_rate: {e}")
            return 0.01
    
    def analyze_emotional_content(self, start_idx, end_idx):
        try:
            if start_idx >= end_idx or start_idx < 0 or end_idx > len(self.y):
                return 0.01
            segment = self.y[start_idx:end_idx]
            if len(segment) == 0:
                return 0.01
            spectral_contrast = np.mean(
                librosa.feature.spectral_contrast(y=segment, sr=self.sr)[0]
            )
            return abs(spectral_contrast)
        except Exception as e:
            print(f"Error in analyze_emotional_content: {e}")
            return 0.01
    
    def analyze_onset_strength(self, start_idx, end_idx):
        try:
            if start_idx >= end_idx or start_idx < 0 or end_idx > len(self.y):
                return 0.01
            segment = self.y[start_idx:end_idx]
            if len(segment) == 0:
                return 0.01
            onset_env = librosa.onset.onset_strength(y=segment, sr=self.sr)
            return np.mean(onset_env)
        except Exception as e:
            print(f"Error in analyze_onset_strength: {e}")
            return 0.01
    
    def analyze_segment(self, start_time, end_time):
        try:
            start_idx = int(start_time * self.sr)
            end_idx = int(end_time * self.sr)
            
            features = {
                'energy': self.analyze_energy(start_idx, end_idx),
                'speech_rate': self.analyze_speech_rate(start_idx, end_idx),
                'emotional_content': self.analyze_emotional_content(start_idx, end_idx),
                'onset_strength': self.analyze_onset_strength(start_idx, end_idx)
            }
            return features
        except Exception as e:
            print(f"Error in analyze_segment: {e}")
            return {
                'energy': 0.01,
                'speech_rate': 0.01,
                'emotional_content': 0.01,
                'onset_strength': 0.01
            }
    
    def find_viral_moments(self, segments, min_duration=10.0, max_duration=60.0):
        if not segments:
            print("No segments provided for viral moment detection")
            return []
        
        print(f"Processing {len(segments)} segments for viral moments")
        
        # First pass - collect features and find max values for normalization
        all_features = []
        max_values = {
            'energy': 0.01,
            'speech_rate': 0.01,
            'emotional_content': 0.01,
            'onset_strength': 0.01
        }
        
        # Sort segments by start time to ensure proper ordering
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        # Process segments ensuring no overlaps
        for i, segment in enumerate(sorted_segments):
            start_time = segment['start']
            end_time = segment['end']
            
            # If this isn't the last segment, ensure it ends where the next one starts
            if i < len(sorted_segments) - 1:
                next_start = sorted_segments[i + 1]['start']
                if end_time > next_start:
                    end_time = next_start
            
            duration = end_time - start_time
            
            if duration < min_duration or duration > max_duration:
                continue
                
            features = self.analyze_segment(start_time, end_time)
            
            # Update max values
            for key, value in features.items():
                max_values[key] = max(max_values[key], value)
            
            all_features.append({
                'segment': {
                    **segment,
                    'end': end_time  # Update the segment with adjusted end time
                },
                'features': features,
                'duration': duration
            })
        
        print(f"Collected features for {len(all_features)} qualified segments")
        
        # Second pass - score segments and find viral moments
        viral_segments = []
        for item in all_features:
            segment = item['segment']
            features = item['features']
            duration = item['duration']
            
            # Normalize features
            normalized = {}
            for key, value in features.items():
                if max_values[key] > 0:
                    normalized[key] = value / max_values[key]
                else:
                    normalized[key] = 0
            
            # Calculate score with adjusted weights
            score = (
                normalized['energy'] * 0.3 +
                normalized['speech_rate'] * 0.2 +
                normalized['emotional_content'] * 0.3 +
                normalized['onset_strength'] * 0.2
            )
            
            # Duration bonus (prefer segments closer to 45 seconds)
            ideal_duration = 45.0
            duration_factor = 1.0 - min(
                1.0, abs(duration - ideal_duration) / ideal_duration
            )
            score = score * (1.0 + duration_factor * 0.2)
            
            viral_segments.append({
                'segment': segment,
                'score': score,
                'features': features,
                'normalized_features': normalized
            })
        
        # Sort by score
        viral_segments.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top segments
        return viral_segments

# Add testing capabilities
if __name__ == "__main__":
    import sys
    import subprocess
    import tempfile
    
    def test_audio_analyzer(input_file=None):
        """Test the AudioAnalyzer with a given file or by creating a test file."""
        if input_file and os.path.exists(input_file):
            audio_path = input_file
            print(f"Using provided audio file: {audio_path}")
        else:
            # Create a test WAV file using ffmpeg in the current directory
            audio_path = "./test_audio.wav"
            print(f"Creating test audio file at: {audio_path}")
            
            # Generate a test tone
            command = f"ffmpeg -y -f lavfi -i sine=frequency=440:duration=5 -ac 1 -ar 22050 {audio_path}"
            subprocess.call(command, shell=True)
            
            if not os.path.exists(audio_path):
                print("Failed to create test audio file")
                return False
        
        try:
            # Create test segments
            test_segments = [
                {'start_time': '00:00:00,000', 'end_time': '00:00:01,000', 'start': 0, 'end': 1, 'text': 'Test 1'},
                {'start_time': '00:00:01,000', 'end_time': '00:00:03,000', 'start': 1, 'end': 3, 'text': 'Test 2'},
                {'start_time': '00:00:03,000', 'end_time': '00:00:05,000', 'start': 3, 'end': 5, 'text': 'Test 3'}
            ]
            
            # Initialize analyzer
            print(f"Testing AudioAnalyzer with file: {audio_path}")
            analyzer = AudioAnalyzer(audio_path)
            print("AudioAnalyzer initialized successfully")
            
            # Test features extraction
            print("Testing feature extraction...")
            features = analyzer.analyze_segment(0, 1)
            print(f"Features extracted: {features}")
            
            # Test viral moment detection
            print("Testing viral moment detection...")
            viral_segments = analyzer.find_viral_moments(test_segments, min_duration=0.5, max_duration=10)
            print(f"Found {len(viral_segments)} viral segments")
            
            if viral_segments:
                print("Audio analysis test successful!")
                return True
            else:
                print("Warning: No viral segments found")
                return False
                
        except Exception as e:
            import traceback
            print(f"Error testing AudioAnalyzer: {str(e)}")
            traceback.print_exc()
            return False

    # Run test if called directly
    if len(sys.argv) > 1:
        test_audio_analyzer(sys.argv[1])
    else:
        test_audio_analyzer() 