# YouTube Clips Automator

This tool automatically creates viral short-form videos from longer YouTube videos. It analyzes both the video content and audio to identify the most engaging segments.

## Features

- Downloads videos from YouTube using video ID or local file
- Transcribes video content using Whisper
- Analyzes audio features to identify engaging moments:
  - Energy/Volume levels
  - Speech rate variations
  - Emotional content analysis
  - Sudden changes detection
- Generates viral segments based on both text and audio analysis
- Creates vertical format videos optimized for social media
- Generates thumbnails with face detection
- Supports face upscaling and enhancement
- Adds subtitles to the final videos

## Requirements

- Python 3.8+
- FFmpeg with CUDA support
- OpenAI API key (for viral segment detection)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/youtube-clips-automator.git
cd youtube-clips-automator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a .env file with your OpenAI API key:
```bash
cp .env.sample .env
# Edit .env and add your OpenAI API key
```

## Usage

Basic usage with a YouTube video:
```bash
python reelsfy.py -v VIDEO_ID
```

Using a local video file:
```bash
python reelsfy.py -f path/to/video.mp4
```

Additional options:
- `--upscale`: Enable face upscaling for small faces
- `--enhance`: Enable face enhancement (upscaling + GFPGAN)
- `--thumb`: Generate thumbnails for the clips

## Audio Analysis Features

The tool now includes advanced audio analysis to better identify viral moments:

1. **Energy Analysis**: Detects segments with higher energy levels that might indicate excitement or emphasis
2. **Speech Rate Analysis**: Identifies variations in speech rate that could indicate important or engaging content
3. **Emotional Content Analysis**: Uses MFCC features to detect emotional intensity in speech
4. **Onset Detection**: Identifies sudden changes in audio that might indicate important moments

These audio features are combined with the text analysis to provide a more comprehensive understanding of potential viral moments.

## Output

The tool creates:
- Transcribed segments in SRT format
- Vertical format videos optimized for social media
- Thumbnails (if enabled)
- Metadata and descriptions for each segment

## License

MIT License - see LICENSE file for details 