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
- The publisher uses a rate limiter to prevent API overuse
- Configuration is done via environment variables in `.env`
- See `social_media_publisher/publisher.py` for the rate limiting implementation