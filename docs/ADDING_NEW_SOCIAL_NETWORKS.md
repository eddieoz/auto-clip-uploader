# Adding New Social Networks

This guide explains how to add support for a new social media platform to the Auto Clip Uploader system.

## Overview

The system uses a modular approach where each social network is configured through multiple components:
- Environment configuration
- Platform configuration management
- Content formatting with platform-specific limits
- API client integration
- Validation

## Step-by-Step Guide

### Step 1: Environment Configuration

Add the new platform to `.env.example` and your local `.env` file:

```bash
# Add channel ID configuration
POSTIZ_NEWPLATFORM_CHANNEL_ID="your-newplatform-channel-id"

# Add publishing toggle (supports scheduling)
PUBLISH_TO_NEWPLATFORM="true, now + 2 hours"
# or simple boolean for immediate posting
PUBLISH_TO_NEWPLATFORM=false
```

### Step 2: Update Configuration Management

Edit `social_media_publisher/config.py`:

**Add to channel_ids dictionary** (around line 32):
```python
# Platform channel IDs
self.channel_ids = {
    "twitter": os.getenv("POSTIZ_TWITTER_CHANNEL_ID"),
    "instagram": os.getenv("POSTIZ_INSTAGRAM_CHANNEL_ID"),
    # ... existing platforms
    "newplatform": os.getenv("POSTIZ_NEWPLATFORM_CHANNEL_ID"),  # Add this line
}
```

**Add to platforms list** (around line 77):
```python
platforms = ["twitter", "instagram", "youtube", "tiktok", "nostr", "bsky", "mastodon", "newplatform"]
```

### Step 3: Add Content Formatting Support

Edit `social_media_publisher/utils/content_formatter.py`:

**Add platform limits** (around line 25):
```python
PLATFORM_LIMITS = {
    "twitter": PlatformLimits(max_length=280, hashtag_limit=3),
    # ... existing platforms
    "newplatform": PlatformLimits(max_length=1000, hashtag_limit=15),  # Adjust limits as needed
}
```

**Add formatting case** (around line 73):
```python
elif platform == "newplatform":
    return self._format_newplatform(limits, segment_zero)
```

**Create formatting method** (add after existing format methods):
```python
def _format_newplatform(self, limits: PlatformLimits, segment_zero) -> str:
    """Format content for NewPlatform"""
    components = []
    
    title = segment_zero.title if segment_zero else self.metadata.title
    if title and title != f"Segment {segment_zero.index if segment_zero else 0}":
        components.append(title)
    
    if segment_zero and segment_zero.description:
        components.append(segment_zero.description)
    elif self.metadata.description:
        components.append(self.metadata.description)
    
    # Combine components (adjust formatting as needed for the platform)
    content = "\n\n".join(components)
    
    # Add hashtags first to calculate exact length
    hashtags = self._select_hashtags(limits.hashtag_limit)
    hashtag_text = f"\n\n{' '.join(hashtags)}" if hashtags else ""
    
    # Ensure we respect the character limit
    if len(content) + len(hashtag_text) > limits.max_length:
        max_content_length = limits.max_length - len(hashtag_text) - 3  # Reserve for "..."
        content = content[:max_content_length].rsplit(' ', 1)[0] + "..."
    
    # Add hashtags
    content += hashtag_text
    
    # Final safety check
    if len(content) > limits.max_length:
        if hashtags:
            content_without_hashtags = content.replace(hashtag_text, "").strip()
            if len(content_without_hashtags) <= limits.max_length:
                content = content_without_hashtags
            else:
                content = content_without_hashtags[:limits.max_length-3] + "..."
        else:
            content = content[:limits.max_length-3] + "..."
    
    return content
```

### Step 4: Update API Client Platform Settings

Edit `social_media_publisher/postiz/client.py`:

**Add platform case** (around line 527):
```python
elif platform == "newplatform":
    # NewPlatform settings
    return {
        **base_settings,
        # Add platform-specific settings here
        "custom_setting": "value",  # Example: adjust based on platform requirements
    }
```

### Step 5: Update Validation Script

Edit `validate_social_media_config.py`:

**Add to required fields** (around line 94):
```python
required_fields = [
    "POSTIZ_API_KEY",
    "POSTIZ_TWITTER_CHANNEL_ID",
    # ... existing fields
    "POSTIZ_NEWPLATFORM_CHANNEL_ID",  # Add this line
]
```

## Platform-Specific Considerations

### Character Limits Research

Before implementing, research the platform's limits:
- Post character limit
- Hashtag limits
- Media requirements
- Special formatting needs

Example research process:
```bash
# Use web search to find current limits
# Check platform's official API documentation
# Look at other social media management tools for reference
```

### Content Formatting Guidelines

Different platforms have different content styles:

- **Twitter/X**: Concise, hashtag-focused
- **Instagram**: Visual-first, emoji-heavy, many hashtags
- **LinkedIn**: Professional tone, fewer hashtags
- **TikTok**: Casual, trend-focused hashtags
- **YouTube**: Descriptive, call-to-action focused
- **Mastodon**: Community-focused, moderate hashtags

### API Settings

Some platforms require specific settings:

- **TikTok**: Requires privacy levels, duet settings, etc.
- **Instagram**: Minimal settings work best
- **YouTube**: Needs visibility settings (public/private)
- **Twitter**: Basic settings sufficient

## Testing Your Implementation

1. **Update your .env file**:
   ```bash
   POSTIZ_NEWPLATFORM_CHANNEL_ID="your-actual-channel-id"
   PUBLISH_TO_NEWPLATFORM=true
   ```

2. **Run validation**:
   ```bash
   python validate_social_media_config.py
   ```

3. **Test with a sample video**:
   ```bash
   python monitor.py  # Or your preferred testing method
   ```

4. **Check character limits**:
   - Monitor the logs for character count warnings
   - Verify posts don't get truncated unexpectedly
   - Test with various content lengths

## Example: Adding Discord Support

Here's a complete example for adding Discord:

```python
# .env
POSTIZ_DISCORD_CHANNEL_ID="your-discord-channel-id"
PUBLISH_TO_DISCORD="true, now"

# config.py - add to channel_ids
"discord": os.getenv("POSTIZ_DISCORD_CHANNEL_ID"),

# config.py - add to platforms
platforms = [..., "discord"]

# content_formatter.py - add limits
"discord": PlatformLimits(max_length=2000, hashtag_limit=5),

# content_formatter.py - add format method
def _format_discord(self, limits: PlatformLimits, segment_zero) -> str:
    # Discord-specific formatting
    pass

# client.py - add platform settings
elif platform == "discord":
    return {**base_settings, "discord_specific": "setting"}

# validate_social_media_config.py
required_fields = [..., "POSTIZ_DISCORD_CHANNEL_ID"]
```

## Troubleshooting

### Common Issues

1. **Character limit exceeded**: Check your formatting method's length calculations
2. **Validation fails**: Ensure all required fields are added to validation script
3. **API errors**: Verify platform-specific settings in the client
4. **Content not formatted properly**: Review the `_format_newplatform` method

### Debugging Tips

- Enable debug logging in the API client
- Test with various content lengths
- Check Postiz dashboard for error messages
- Verify channel IDs are correct format for the platform

## Best Practices

1. **Follow existing patterns**: Use the same structure as existing platforms
2. **Research thoroughly**: Understand the platform's requirements before implementing
3. **Test extensively**: Try various content types and lengths
4. **Document platform quirks**: Add comments for any special requirements
5. **Keep it simple**: Start with minimal settings and add complexity as needed

## Getting Help

- Check existing platform implementations for reference
- Review Postiz documentation for platform-specific requirements
- Test with Postiz directly to understand expected formats
- Look at the platform's official API documentation

---

*This tutorial was created based on the successful implementation of Mastodon support in the Auto Clip Uploader system.*