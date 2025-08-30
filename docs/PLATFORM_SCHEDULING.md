# Platform-Specific Scheduling Configuration

This document describes the enhanced platform configuration system that supports different posting times for each social media platform.

## Overview

The new configuration format allows you to specify both platform enablement and posting times in a single environment variable using a comma-separated format. This prevents notification spam by allowing you to stagger posts across different platforms.

## Configuration Format

### New Comma-Separated Format

```bash
PUBLISH_TO_PLATFORM="enabled, posting_time"
```

**Examples:**
```bash
PUBLISH_TO_TWITTER="true, now"                    # Immediate posting
PUBLISH_TO_INSTAGRAM="true, now + 1 hour"         # Post 1 hour from now
PUBLISH_TO_YOUTUBE="true, now + 2 hours"          # Post 2 hours from now
PUBLISH_TO_TIKTOK="false"                         # Disabled (time ignored)
```

### Old Boolean Format (Backward Compatible)

```bash
PUBLISH_TO_TWITTER=true
PUBLISH_TO_INSTAGRAM=false
```

When using the old format, all enabled platforms use the global `POSTIZ_POSTING_TIME` setting.

## Supported Time Formats

| Format | Description | Example |
|--------|-------------|---------|
| `now` | Immediate posting | `"true, now"` |
| `now + N minutes` | Delay by N minutes | `"true, now + 30 minutes"` |
| `now + N mins` | Delay by N minutes (short form) | `"true, now + 15 mins"` |
| `now + N hours` | Delay by N hours | `"true, now + 2 hours"` |
| `now + N hrs` | Delay by N hours (short form) | `"true, now + 1 hrs"` |

**Note:** Extra spaces are automatically handled (`" true , now + 1 hour "` works fine).

## Posting Behavior

### Bulk Posting (Same Time)

When all enabled platforms have the same posting time, the system uses **bulk posting**:

```bash
PUBLISH_TO_TWITTER="true, now"
PUBLISH_TO_INSTAGRAM="true, now" 
PUBLISH_TO_YOUTUBE="true, now"
```

→ Creates a single multi-platform post for all three platforms simultaneously.

### Individual Scheduling (Different Times)

When platforms have different posting times, the system uses **individual scheduling**:

```bash
PUBLISH_TO_TWITTER="true, now"
PUBLISH_TO_INSTAGRAM="true, now + 1 hour"
PUBLISH_TO_YOUTUBE="true, now + 2 hours"
```

→ Creates separate posts: Twitter immediately, Instagram in 1 hour, YouTube in 2 hours.

### Mixed Grouping (Optimized)

Platforms with the same time are grouped together for efficiency:

```bash
PUBLISH_TO_TWITTER="true, now"
PUBLISH_TO_INSTAGRAM="true, now"          # Same as Twitter
PUBLISH_TO_YOUTUBE="true, now + 2 hours"
```

→ Creates 2 posts: Twitter+Instagram immediately, YouTube in 2 hours.

## Configuration Examples

### Example 1: Staggered Social Media Strategy
```bash
# Immediate posting to Twitter for breaking news
PUBLISH_TO_TWITTER="true, now"

# Instagram 30 minutes later for better engagement timing
PUBLISH_TO_INSTAGRAM="true, now + 30 minutes"

# YouTube 2 hours later for long-form content
PUBLISH_TO_YOUTUBE="true, now + 2 hours"

# TikTok disabled for this content type
PUBLISH_TO_TIKTOK="false"

# Global fallback (not used since all have specific times)
POSTIZ_POSTING_TIME="now + 1 hour"
```

### Example 2: Bulk Posting with Delay
```bash
# All platforms post together after 1 hour
PUBLISH_TO_TWITTER="true, now + 1 hour"
PUBLISH_TO_INSTAGRAM="true, now + 1 hour" 
PUBLISH_TO_YOUTUBE="true, now + 1 hour"
PUBLISH_TO_TIKTOK="true, now + 1 hour"

POSTIZ_POSTING_TIME="now"  # Not used
```

### Example 3: Mixed Old and New Formats
```bash
# New format with specific time
PUBLISH_TO_TWITTER="true, now"

# Old format - uses global fallback
PUBLISH_TO_INSTAGRAM=true

# New format with different time  
PUBLISH_TO_YOUTUBE="true, now + 3 hours"

# Old format - disabled
PUBLISH_TO_TIKTOK=false

# Global fallback for old format platforms
POSTIZ_POSTING_TIME="now + 30 minutes"
```

→ Twitter posts immediately, Instagram posts in 30 minutes (global fallback), YouTube posts in 3 hours.

## Error Handling

### Invalid Time Formats

Invalid time formats automatically fall back to the global `POSTIZ_POSTING_TIME`:

```bash
PUBLISH_TO_TWITTER="true, invalid_format"  # Falls back to global time
PUBLISH_TO_INSTAGRAM="true, now +, broken" # Falls back to global time
POSTIZ_POSTING_TIME="now + 15 minutes"     # Fallback time
```

**Warning logged:**
```
⚠️  Invalid time format for TWITTER: 'invalid_format' - falling back to global POSTIZ_POSTING_TIME
```

### Empty or Malformed Configurations

```bash
PUBLISH_TO_TWITTER=""           # Disables platform
PUBLISH_TO_INSTAGRAM="   "      # Disables platform (whitespace only)
PUBLISH_TO_YOUTUBE="true,,"     # Falls back to global time with warning
```

### Platform Failures

Individual platform failures don't stop other platforms:

- If Twitter fails but Instagram succeeds, Instagram still posts
- Detailed error logging for troubleshooting
- Graceful degradation ensures maximum content distribution

## Migration Guide

### From Old Boolean Format

**Before:**
```bash
PUBLISH_TO_TWITTER=true
PUBLISH_TO_INSTAGRAM=true
PUBLISH_TO_YOUTUBE=false
POSTIZ_POSTING_TIME="now + 30 minutes"
```

**After (same behavior):**
```bash
PUBLISH_TO_TWITTER="true, now + 30 minutes"
PUBLISH_TO_INSTAGRAM="true, now + 30 minutes"
PUBLISH_TO_YOUTUBE=false  # or "false, any_time"
```

**After (improved with staggering):**
```bash
PUBLISH_TO_TWITTER="true, now"
PUBLISH_TO_INSTAGRAM="true, now + 1 hour" 
PUBLISH_TO_YOUTUBE=false
```

### Gradual Migration

You can migrate platforms one at a time:

```bash
PUBLISH_TO_TWITTER="true, now"           # New format
PUBLISH_TO_INSTAGRAM=true                # Old format
PUBLISH_TO_YOUTUBE=true                  # Old format
POSTIZ_POSTING_TIME="now + 45 minutes"  # Used by Instagram and YouTube
```

## Monitoring and Troubleshooting

### Log Messages

The system provides detailed logging:

```
🕒 Platforms have different posting times - using individual scheduling
🕒 Platform scheduling groups:
   now: ['twitter']
   now + 1 hour: ['instagram'] 
   now + 2 hours: ['youtube']
📅 Scheduling 1 platforms for: now
✅ Successfully scheduled 1 platforms for now
```

### Configuration Summary

Use the configuration validation script to check your setup:

```python
from social_media_publisher.config import PostizConfig

config = PostizConfig()
summary = config.get_configuration_summary()
print(f"Platform posting times: {summary['platform_posting_times']}")
print(f"Platforms using global fallback: {summary['platforms_using_global_fallback']}")
```

### Common Issues

1. **All platforms failing**: Check API key and channel IDs
2. **Invalid time format**: Check for typos in time expressions
3. **Unexpected bulk posting**: Verify platforms have different times
4. **Rate limiting**: Individual scheduling includes automatic delays

## API Rate Limiting

The system automatically handles rate limiting:

- **Bulk posting**: Single API call for same-time platforms
- **Individual scheduling**: Automatic 2-second delays between groups
- **Exponential backoff**: Automatic retry on temporary failures
- **Rate limit respect**: Honors server-suggested retry delays

## Performance Considerations

### Bulk vs Individual Posting

| Scenario | API Calls | Rate Limit Impact | Recommended For |
|----------|-----------|------------------|-----------------|
| All same time | 1 call | Minimal | Simple campaigns |
| 2 time groups | 2 calls | Low | Balanced strategy |
| 4 different times | 4 calls | Moderate | Complex scheduling |

### Best Practices

1. **Group similar times** when possible to minimize API calls
2. **Use meaningful delays** (30+ minutes) to spread engagement
3. **Test with mock mode** first: `POSTIZ_MOCK_MODE="true"`
4. **Monitor logs** for scheduling confirmation and errors

## Testing

### Mock Mode Testing

Enable mock mode to test without actual posting:

```bash
POSTIZ_MOCK_MODE="true"
PUBLISH_TO_TWITTER="true, now"
PUBLISH_TO_INSTAGRAM="true, now + 1 hour"
```

Generates logs like:
```
🎭 MOCK MODE: Creating immediate multi-platform post for 1 channels
🎭 MOCK MODE: Scheduling multi-platform post for 1 channels at 2024-01-15T15:30:00
```

### Validation Scripts

Run the test scripts to validate your configuration:

```bash
python test_config_manual.py           # Basic configuration tests
python test_full_implementation.py     # Comprehensive end-to-end tests
```

## Backwards Compatibility

✅ **Full backward compatibility** - all existing configurations continue to work without changes.

The new feature is **additive only** - it adds functionality without breaking existing setups.