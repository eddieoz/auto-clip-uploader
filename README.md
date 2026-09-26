# Auto Clip Uploader

An automated video processing system that monitors for new video files and converts them into social media-ready vertical clips with AI-powered viral segment detection.

Built with Python 3.11, CUDA acceleration, and PyTorch for high-performance video processing. The core video processing engine is powered by [Izabella](https://github.com/eddieoz/reels-clips-automator).

## Features

- 📁 **Automated File Monitoring** - Watches configured folders for new video files
- 🎯 **AI-Powered Viral Detection** - Topic-aware segment selection: an LLM pass trims the previous topic's tail and picks the hook-first moment; heuristic topic analysis stays as the fallback
- 🎬 **Video Processing** - Converts to 9:16 vertical with layout-aware framing for screencasts (circular face-cam PiP badge or motion-following zoom) and face tracking for talking heads
- 🗣️ **Audio Mastering** - loudnorm (−14 LUFS) + gentle compression on every final output
- ✂️ **Jump Cuts** - Intra-segment dead air removed with caption timing remapping
- 📝 **Karaoke Captions** - Per-word highlighted ASS subtitles with customizable font and color
- 🪝 **Cold-Open Hook Teaser** - The hook sentence is cut, cropped, and burned before the full clip
- 🎨 **Polish** - Mild visual grade, optional LUT, end-card CTA, optional music bed
- 🖼️ **Thumbnail Creation** - Best-quality smile frame with hook-text preference
- 📱 **Multi-Platform Publishing** - Drip publishing to Twitter, Instagram, YouTube, and more via Postiz
- 🔍 **Face Enhancement** - Optional RealESRGAN face upscaling

## Quick Start

### Prerequisites

**System Requirements:**
- Python 3.11
- NVIDIA GPU with CUDA support (recommended for video processing)
- PyTorch with CUDA support
- Postiz for post scheduling and publishing

```bash
# System dependencies
sudo apt-get install -y libnppicc11 libnppig11 libnppidei11 libnppif11 ffmpeg

# Install PyTorch with CUDA (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Python dependencies
pip install -r requirements.txt
pip install -r reels-clips-automator/requirements.txt
```

### Configuration

1. Copy environment templates:
```bash
cp .env.example .env
cp reels-clips-automator/.env.example reels-clips-automator/.env
```

2. Configure your API keys and settings:
```bash
# Edit .env - Add your Postiz API key and channel IDs
# Edit reels-clips-automator/.env - Add your OpenAI API key
```

3. Set up your video monitoring folder in `.env`:
```bash
VIDEO_FOLDER="input/"
```

### Running

**Start monitoring (recommended):**
```bash
python monitor.py
```

**Process single video:**
```bash
cd reels-clips-automator
python reelsfy.py --file /path/to/video.mp4 --upscale --enhance --thumb
```

## Configuration

### Main Environment (.env)
- `POSTIZ_API_KEY` - Your Postiz API key
- `POSTIZ_ENDPOINT` - Your Postiz instance URL
- `VIDEO_FOLDER` - Directory to monitor for new videos
- Platform-specific channel IDs and publishing preferences

### Video Processing (reels-clips-automator/.env)
- `OPENAI_API_KEY` - For AI-powered segment selection and metadata generation
- `CHANNEL_NAME` - Your channel identifier
- `STT_MODEL` / `WHISPER_MODEL` / `FASTER_WHISPER_*` - Transcription engine
- `TOPIC_BOUNDARY_LLM` - LLM topic-start pass (default on; `false` = legacy heuristics)
- `JUMP_CUTS_ENABLED` / `JUMP_CUT_PAUSE_THRESHOLD` - Dead-air removal
- `CAPTION_FONT_FAMILY` / `CAPTION_HIGHLIGHT_COLOR` - Karaoke caption styling
- `FRAMING_MODE` - Screencast framing: `auto`, `pip`, `screen_zoom`, `closeup`
- `VIDEO_GRADE` / `VIDEO_LUT` - Visual grade and optional .cube LUT
- `END_CARD_ENABLED` / `END_CARD_TEXT` / `END_CARD_COLOR` - End-card CTA
- `MUSIC_BED` - Path to an audio file looped under the speech (empty = off)

See `reels-clips-automator/.env.example` for all video-editing options with
defaults, and `docs/VIDEO_EDITING_IMPROVEMENT_PLAN.md` +
`docs/VIRAL_SELECTION_IMPROVEMENT_PLAN.md` for the design behind them.

### OBS Replay Buffer (capture length)

Clips are built from an **OBS replay buffer of 90 seconds**. This is the single
most important capture setting for clip quality:

- A clip can be at most **59 seconds** (the YouTube hard cap), so the buffer must
  hold at least that much — plus margin — for the segment selector to find a
  complete, engaging topic.
- The final topic *concludes at the end* of the capture (recency anchor). The
  buffer's **start bounds how much topic context is available**: too short and the
  clip is cut off mid-topic, leaving it feeling incomplete. 90 s gives the selector
  room to extend a clip's start earlier when the LLM says the topic began before the
  heuristic window.
- If clips feel truncated or the LLM keeps returning the minimum duration, raise the
  OBS replay buffer (or ensure OBS actually captures the full 90 s) before tuning
  anything else.

### Postiz Setup
Add to your Postiz docker-compose.yml backend service:
```yaml
environment:
  - UPLOAD_MAX_FILE_SIZE=100MB
  - BODY_PARSER_LIMIT=100MB
  - EXPRESS_MAX_FILE_SIZE=104857600
```

## Architecture

- **Root Level**: File monitoring and social media publishing
- **reels-clips-automator/**: Core video processing pipeline
  - Audio transcription and analysis
  - AI-powered viral moment detection
  - Video cropping and enhancement
  - Subtitle generation and overlay

## Output

Processed videos are saved to `output/` with:
- Vertical format video files (.mp4)
- Subtitle files (.srt)
- JSON metadata with AI-generated titles and descriptions

## Testing

```bash
# Run all tests
pytest

# Test specific components
python -m pytest tests/test_monitor.py
python -m pytest tests/test_audio_analysis.py
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For issues and feature requests, please use the GitHub issue tracker.