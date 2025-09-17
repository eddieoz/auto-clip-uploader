#!/usr/bin/env python3
"""
Simplified BDD tests for WhisperTranscriber without actual whisper import
"""

import sys
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add the reels-clips-automator directory to path
sys.path.insert(0, 'reels-clips-automator')

def run_bdd_tests():
    """Run all BDD tests for WhisperTranscriber with proper mocking."""

    print("=== Whisper Transcriber BDD Tests (TDD Green Phase) ===\n")

    failed_tests = 0
    total_tests = 0

    # Mock whisper completely before importing
    fake_whisper = MagicMock()

    with patch.dict(sys.modules, {'whisper': fake_whisper}):
        # Now import the transcriber
        try:
            from transcribers.whisper_transcriber import WhisperTranscriber
            print("✓ WhisperTranscriber import successful")
        except Exception as e:
            print(f"✗ WhisperTranscriber import failed: {e}")
            return False

        # Test 1: Backwards compatibility identical transcription
        total_tests += 1
        print(f"\nTest {total_tests}: Backwards compatibility - identical transcription")
        try:
            mock_whisper_result = {
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": " This is the first segment."},
                    {"start": 2.5, "end": 5.0, "text": " This is the second segment."}
                ]
            }

            mock_model = Mock()
            mock_model.transcribe.return_value = mock_whisper_result
            fake_whisper.load_model.return_value = mock_model

            transcriber = WhisperTranscriber()

            with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
                result = transcriber.transcribe(temp_audio.name)

            expected = [
                {"start": 0.0, "end": 2.5, "text": "This is the first segment."},
                {"start": 2.5, "end": 5.0, "text": "This is the second segment."}
            ]

            assert result == expected, f"Expected {expected}, got {result}"
            print("✓ Backwards compatibility test passed")
        except Exception as e:
            print(f"✗ Backwards compatibility test failed: {e}")
            failed_tests += 1

        # Test 2: Model configuration preserved
        total_tests += 1
        print(f"\nTest {total_tests}: Model configuration preserved")
        try:
            with patch.dict(os.environ, {'WHISPER_MODEL': 'medium'}):
                mock_model = Mock()
                mock_model.transcribe.return_value = {"segments": []}
                fake_whisper.load_model.return_value = mock_model

                transcriber = WhisperTranscriber()

                with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
                    transcriber.transcribe(temp_audio.name)

                fake_whisper.load_model.assert_called_with('medium')
                print("✓ Model configuration test passed")
        except Exception as e:
            print(f"✗ Model configuration test failed: {e}")
            failed_tests += 1

        # Test 3: Default model configuration
        total_tests += 1
        print(f"\nTest {total_tests}: Default model configuration")
        try:
            with patch.dict(os.environ, {}, clear=True):
                mock_model = Mock()
                mock_model.transcribe.return_value = {"segments": []}
                fake_whisper.load_model.return_value = mock_model

                transcriber = WhisperTranscriber()

                with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
                    transcriber.transcribe(temp_audio.name)

                fake_whisper.load_model.assert_called_with('small')
                print("✓ Default model test passed")
        except Exception as e:
            print(f"✗ Default model test failed: {e}")
            failed_tests += 1

        # Test 4: Availability check
        total_tests += 1
        print(f"\nTest {total_tests}: Availability check")
        try:
            transcriber = WhisperTranscriber()
            assert transcriber.is_available() is True
            print("✓ Availability test passed")
        except Exception as e:
            print(f"✗ Availability test failed: {e}")
            failed_tests += 1

        # Test 5: Transcriber name
        total_tests += 1
        print(f"\nTest {total_tests}: Transcriber name property")
        try:
            with patch.dict(os.environ, {'WHISPER_MODEL': 'large'}):
                transcriber = WhisperTranscriber()
                assert transcriber.name == "Whisper (large)"
                print("✓ Transcriber name test passed")
        except Exception as e:
            print(f"✗ Transcriber name test failed: {e}")
            failed_tests += 1

        # Test 6: Lazy loading
        total_tests += 1
        print(f"\nTest {total_tests}: Model lazy loading")
        try:
            transcriber = WhisperTranscriber()
            # Model should not be loaded yet
            assert transcriber._model is None

            mock_model = Mock()
            mock_model.transcribe.return_value = {"segments": []}
            fake_whisper.load_model.return_value = mock_model

            with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
                transcriber.transcribe(temp_audio.name)

            # Now model should be loaded
            assert transcriber._model is not None
            print("✓ Lazy loading test passed")
        except Exception as e:
            print(f"✗ Lazy loading test failed: {e}")
            failed_tests += 1

        # Test 7: Model reuse
        total_tests += 1
        print(f"\nTest {total_tests}: Model reuse across transcriptions")
        try:
            # Reset the mock for this test
            fake_whisper.reset_mock()

            mock_model = Mock()
            mock_model.transcribe.return_value = {"segments": []}
            fake_whisper.load_model.return_value = mock_model

            transcriber = WhisperTranscriber()

            with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
                transcriber.transcribe(temp_audio.name)
                transcriber.transcribe(temp_audio.name)

            # Model should only be loaded once
            call_count = fake_whisper.load_model.call_count
            assert call_count == 1, f"Expected 1 call, got {call_count}"
            print("✓ Model reuse test passed")
        except Exception as e:
            print(f"✗ Model reuse test failed: {e}")
            failed_tests += 1

    # Summary
    passed_tests = total_tests - failed_tests
    print(f"\n=== Test Results ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")

    if failed_tests == 0:
        print("🎉 All BDD tests PASSED! WhisperTranscriber implementation complete.")
        return True
    else:
        print("❌ Some tests FAILED! Need to fix WhisperTranscriber implementation.")
        return False

if __name__ == "__main__":
    success = run_bdd_tests()
    sys.exit(0 if success else 1)