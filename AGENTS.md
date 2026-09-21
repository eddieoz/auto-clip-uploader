# Agent Guidelines for Auto Clip Uploader

This file provides guidance for AI agents working on this repository.

## Project Overview

Auto Clip Uploader is an automated video processing system that monitors for new video files and converts them into social media-ready vertical clips.

## Key Components

- `monitor.py` - File system watcher that triggers processing
- `reels-clips-automator/` - Core video processing logic
  - `reelsfy.py` - Main processing script
  - `social_media_publisher/` - Postiz API integration for social media publishing

## Development Principles

1. **Keep it simple** - Prefer straightforward solutions over complex abstractions
2. **Test first** - Write tests before implementing new features
3. **Preserve existing behavior** - Don't break existing workflows
4. **Clear commits** - Write descriptive commit messages
5. **Follow existing style** - Match the code style already present in the repository

## Common Tasks

### Adding New Features
1. Write tests first in the `tests/` directory
2. Implement the feature following existing patterns
3. Run tests to ensure nothing is broken
4. Update documentation if needed

### Debugging
1. Check the logs in the output directories
2. Run existing tests to see if anything is broken
3. Use the test scripts in `tests/` to reproduce issues
4. Make minimal, focused changes

### Social Media Publishing (Postiz)
- Edited videos are appended to a disk-backed publish queue (`publish_queue.json`, gitignored) and drip-published one at a time on a configurable interval
- Configuration is done via environment variables in `.env` (`PUBLISH_INTERVAL_MINUTES`, `PUBLISH_MAX_RETRIES`, `DASHBOARD_INTERVAL_SECONDS`)
- See `social_media_publisher/publish_queue.py` for the queue/scheduler and `social_media_publisher/publisher.py` for the Postiz posting workflow

### Video Editing Pipeline (reelsfy)
- Segment selection keeps the OBS replay-buffer model: the topic CONCLUDES near the end of the capture (recency anchor). An LLM pass (`TOPIC_BOUNDARY_LLM`, default on) trims the previous topic's tail from the clip start and proposes hook candidates; every LLM/audio failure degrades to the heuristic behavior.
- Screencast framing (`FRAMING_MODE`: auto/pip/screen_zoom/closeup) alternates PiP badge ↔ screen-zoom every 5 seconds; the PiP badge is EMA-stabilized and only re-crops on real face movement.
- All editing features are env-gated with in-code defaults (`JUMP_CUTS_ENABLED`, `CAPTION_FONT_FAMILY`, `CAPTION_HIGHLIGHT_COLOR`, `VIDEO_GRADE`, `VIDEO_LUT`, `END_CARD_ENABLED`, `END_CARD_TEXT`, `END_CARD_COLOR`, `MUSIC_BED`) — see `reels-clips-automator/.env.example`.
- Design docs: `docs/VIDEO_EDITING_IMPROVEMENT_PLAN.md` (rendering) and `docs/VIRAL_SELECTION_IMPROVEMENT_PLAN.md` (segment selection).
- Regression tests live in `tests/` (gitignored — force-add new test files); run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/`.