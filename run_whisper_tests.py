#!/usr/bin/env python3
"""
Custom test runner for Whisper transcriber BDD tests
"""

import sys
import os
import tempfile
from unittest.mock import Mock, patch

# Add the reels-clips-automator directory to path
sys.path.insert(0, 'reels-clips-automator')

def run_bdd_tests():
    """Run all BDD tests for WhisperTranscriber."""

    print("=== Whisper Transcriber BDD Tests (TDD Red Phase) ===\n")

    failed_tests = 0
    total_tests = 0

    # Import the transcriber (this will fail if not implemented correctly)
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

        import sys
        fake_whisper = Mock()
        fake_whisper.load_model = Mock()

        with patch.dict(sys.modules, {'whisper': fake_whisper}):
            mock_load = fake_whisper.load_model
            mock_model = Mock()
            mock_model.transcribe.return_value = mock_whisper_result
            mock_load.return_value = mock_model

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
            import sys
        fake_whisper = Mock()
        fake_whisper.load_model = Mock()

        with patch.dict(sys.modules, {'whisper': fake_whisper}):
            mock_load = fake_whisper.load_model
                mock_model = Mock()
                mock_load.return_value = mock_model
                mock_model.transcribe.return_value = {"segments": []}

                transcriber = WhisperTranscriber()

                with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
                    transcriber.transcribe(temp_audio.name)

                mock_load.assert_called_with('medium')
                print("✓ Model configuration test passed")
    except Exception as e:
        print(f"✗ Model configuration test failed: {e}")
        failed_tests += 1

    # Test 3: Default model configuration
    total_tests += 1
    print(f"\nTest {total_tests}: Default model configuration")
    try:
        with patch.dict(os.environ, {}, clear=True):
            import sys
        fake_whisper = Mock()
        fake_whisper.load_model = Mock()

        with patch.dict(sys.modules, {'whisper': fake_whisper}):
            mock_load = fake_whisper.load_model
                mock_model = Mock()
                mock_load.return_value = mock_model
                mock_model.transcribe.return_value = {"segments": []}

                transcriber = WhisperTranscriber()

                with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
                    transcriber.transcribe(temp_audio.name)

                mock_load.assert_called_with('small')
                print("✓ Default model test passed")
    except Exception as e:
        print(f"✗ Default model test failed: {e}")
        failed_tests += 1

    # Test 4: Availability check
    total_tests += 1
    print(f"\nTest {total_tests}: Availability check")
    try:
        # Mock the whisper module completely
        import sys
        fake_whisper = Mock()
        fake_whisper.load_model = Mock()
        with patch.dict(sys.modules, {'whisper': fake_whisper}):
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
        import sys
        fake_whisper = Mock()
        fake_whisper.load_model = Mock()

        with patch.dict(sys.modules, {'whisper': fake_whisper}):
            mock_load = fake_whisper.load_model
            transcriber = WhisperTranscriber()
            mock_load.assert_not_called()

            mock_model = Mock()
            mock_model.transcribe.return_value = {"segments": []}
            mock_load.return_value = mock_model

            with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
                transcriber.transcribe(temp_audio.name)

            mock_load.assert_called_once()
            print("✓ Lazy loading test passed")
    except Exception as e:
        print(f"✗ Lazy loading test failed: {e}")
        failed_tests += 1

    # Summary
    passed_tests = total_tests - failed_tests
    print(f"\n=== Test Results ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")

    if failed_tests == 0:
        print("🎉 All tests PASSED! Moving to TDD Green phase.")
        return True
    else:
        print("❌ Some tests FAILED! Need to implement/fix WhisperTranscriber.")
        return False

if __name__ == "__main__":
    success = run_bdd_tests()
    sys.exit(0 if success else 1)