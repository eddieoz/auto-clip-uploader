# Auto Clip Uploader

An automated video processing system that monitors for new video files and converts them into social media-ready vertical clips with AI-powered viral segment detection.

Built with Python 3.11, CUDA acceleration, and PyTorch for high-performance video processing. The core video processing engine is powered by [reels-clips-automator](https://github.com/eddieoz/reels-clips-automator).

## Features

- 📁 **Automated File Monitoring** - Watches configured folders for new video files
- 🎯 **AI-Powered Viral Detection** - Uses AI to identify engaging segments
- 🎬 **Video Processing** - Converts to 9:16 vertical format with cropping and enhancement
- 📝 **Automatic Transcription** - Whisper-based audio transcription
- 🎨 **Subtitle Generation** - Overlay subtitles with customizable styling
- 🖼️ **Thumbnail Creation** - Generates eye-catching thumbnails
- 📱 **Multi-Platform Publishing** - Direct posting to Twitter, Instagram, YouTube, and much more via Postiz
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
- `OPENAI_API_KEY` - For AI-powered viral segment detection
- `CHANNEL_NAME` - Your channel identifier

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