"""
Configuration management for Postiz API integration
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, List
from datetime import datetime, timedelta


class PostizConfig:
    """Configuration manager for Postiz API settings"""
    
    def __init__(self):
        # Load environment variables from root .env file
        self._load_env_file()
        
        # Postiz API settings
        self.api_key = os.getenv("POSTIZ_API_KEY")
        self.endpoint = os.getenv("POSTIZ_ENDPOINT", "https://postiz.eddieoz.com/api")
        
        # Platform channel IDs
        self.channel_ids = {
            "twitter": os.getenv("POSTIZ_TWITTER_CHANNEL_ID"),
            "instagram": os.getenv("POSTIZ_INSTAGRAM_CHANNEL_ID"),
            "youtube": os.getenv("POSTIZ_YOUTUBE_CHANNEL_ID"),
            "tiktok": os.getenv("POSTIZ_TIKTOK_CHANNEL_ID"),
            "nostr": os.getenv("POSTIZ_NOSTR_CHANNEL_ID"),
            "bsky": os.getenv("POSTIZ_BSKY_CHANNEL_ID"),
            "mastodon": os.getenv("POSTIZ_MASTODON_CHANNEL_ID"),
        }
        
        # Platform enable/disable flags and posting times
        self.platform_configs = self._parse_platform_configs()
        
        # Backward compatibility - extract enabled status
        self.enabled_platforms = {
            platform: config["enabled"] for platform, config in self.platform_configs.items()
        }
        
        # Posting time configuration
        self.posting_time = os.getenv("POSTIZ_POSTING_TIME", "now")
        
        # Mock mode for testing/development
        self.mock_mode = os.getenv("POSTIZ_MOCK_MODE", "false").lower() == "true"
        
        # Validate configuration
        self._validate_config()
    
    def _load_env_file(self):
        """Load environment variables from root .env file"""
        # Find root .env file (go up from current directory)
        current_dir = Path(__file__).parent
        root_dir = current_dir.parent  # Go up to auto-clip-uploader directory
        env_path = root_dir / ".env"
        
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded configuration from: {env_path}")
        else:
            print(f"Warning: .env file not found at {env_path}")
    
    def _parse_platform_configs(self) -> Dict[str, Dict[str, any]]:
        """
        Parse platform configurations supporting both old boolean format and new comma-separated format
        
        Supported formats:
        - "true" / "false" (backward compatibility) -> uses global POSTIZ_POSTING_TIME
        - "true, now" -> enabled with immediate posting
        - "true, now + 2 hours" -> enabled with specific posting time
        - "false, ..." -> disabled (posting time ignored)
        
        Returns:
            Dict[platform_name, {"enabled": bool, "posting_time": str}]
        """
        platforms = ["twitter", "instagram", "youtube", "tiktok", "nostr", "bsky", "mastodon"]
        configs = {}
        
        for platform in platforms:
            env_var = f"PUBLISH_TO_{platform.upper()}"
            raw_value = os.getenv(env_var, "true")
            
            config = self._parse_single_platform_config(platform, raw_value)
            configs[platform] = config
            
        return configs
    
    def _parse_single_platform_config(self, platform: str, raw_value: str) -> Dict[str, any]:
        """
        Parse a single platform configuration string
        
        Args:
            platform: Platform name (e.g., "twitter")
            raw_value: Raw environment variable value
            
        Returns:
            Dict with "enabled" and "posting_time" keys
        """
        # Handle empty or whitespace-only values
        if not raw_value or not raw_value.strip():
            return {"enabled": False, "posting_time": None}
        
        # Clean the value
        raw_value = raw_value.strip()
        
        # Check if it's a comma-separated format
        if "," in raw_value:
            parts = [part.strip() for part in raw_value.split(",", 1)]
            
            # Must have exactly 2 parts for valid comma-separated format
            if len(parts) == 2:
                enabled_str, posting_time = parts
                
                # Parse enabled status
                enabled = enabled_str.lower() == "true"
                
                # If disabled, ignore posting time
                if not enabled:
                    return {"enabled": False, "posting_time": None}
                
                # Validate posting time format
                if self._validate_time_format(posting_time):
                    return {"enabled": True, "posting_time": posting_time}
                else:
                    # Invalid time format - fallback to global setting with warning
                    print(f"⚠️  Invalid time format for {platform.upper()}: '{posting_time}' - falling back to global POSTIZ_POSTING_TIME")
                    return {"enabled": True, "posting_time": None}  # None means use global
            else:
                # Malformed comma-separated format
                print(f"⚠️  Malformed configuration for {platform.upper()}: '{raw_value}' - falling back to global POSTIZ_POSTING_TIME")
                return {"enabled": True, "posting_time": None}
        else:
            # Backward compatibility - simple boolean format
            enabled = raw_value.lower() == "true"
            return {"enabled": enabled, "posting_time": None}  # None means use global
    
    def _validate_time_format(self, time_str: str) -> bool:
        """
        Validate posting time format
        
        Args:
            time_str: Time string to validate
            
        Returns:
            True if valid, False if invalid
        """
        if not time_str or not time_str.strip():
            return False
            
        time_str = time_str.lower().strip()
        
        # Check for "now" format
        if time_str == "now":
            return True
        
        # Check for "now + N minutes/hours" format
        pattern = r"^now\s*\+\s*\d+\s*(minutes?|hours?|mins?|hrs?)$"
        return bool(re.match(pattern, time_str))
    
    def _validate_config(self):
        """Validate required configuration"""
        if not self.api_key:
            raise ValueError("POSTIZ_API_KEY is required in .env file")
        
        # Check if at least one platform is enabled and has channel ID
        enabled_with_channel = []
        for platform, enabled in self.enabled_platforms.items():
            if enabled and self.channel_ids.get(platform):
                enabled_with_channel.append(platform)
        
        if not enabled_with_channel:
            print("Warning: No platforms are enabled with valid channel IDs")
        else:
            print(f"Enabled platforms: {', '.join(enabled_with_channel)}")
    
    def get_enabled_channels(self) -> Dict[str, str]:
        """Get channel IDs for enabled platforms"""
        enabled_channels = {}
        for platform, enabled in self.enabled_platforms.items():
            if enabled and self.channel_ids.get(platform):
                enabled_channels[platform] = self.channel_ids[platform]
        return enabled_channels
    
    def get_platform_posting_time(self, platform: str) -> str:
        """
        Get posting time for a specific platform
        
        Args:
            platform: Platform name (e.g., "twitter", "instagram")
            
        Returns:
            Platform-specific posting time or global fallback
        """
        if platform not in self.platform_configs:
            return self.posting_time
        
        platform_config = self.platform_configs[platform]
        
        # Return platform-specific time if configured, otherwise global fallback
        return platform_config["posting_time"] or self.posting_time
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return bool(self.api_key and self.get_enabled_channels())
    
    def validate_api_connection(self) -> Dict[str, str]:
        """
        Validate API connectivity with Postiz endpoint
        
        Returns:
            Dict with validation results
        """
        if not self.api_key:
            return {
                "status": "error",
                "message": "No API key configured"
            }
        
        try:
            from .postiz.client import PostizClient
            
            client = PostizClient(self.api_key, self.endpoint)
            return client.validate_connection()
            
        except ImportError:
            return {
                "status": "error", 
                "message": "Postiz client not available"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection validation failed: {str(e)}"
            }
    
    def get_configuration_summary(self) -> Dict[str, any]:
        """
        Get comprehensive configuration summary for debugging
        
        Returns:
            Dict with configuration details
        """
        enabled_channels = self.get_enabled_channels()
        
        # Build platform-specific posting time summary
        platform_posting_times = {}
        for platform in self.platform_configs:
            if self.enabled_platforms.get(platform, False):
                platform_posting_times[platform] = self.get_platform_posting_time(platform)
        
        return {
            "api_key_present": bool(self.api_key),
            "api_key_preview": f"{self.api_key[:8]}..." if self.api_key else None,
            "endpoint": self.endpoint,
            "enabled_platforms": [platform for platform, enabled in self.enabled_platforms.items() if enabled],
            "configured_channels": list(enabled_channels.keys()),
            "channel_count": len(enabled_channels),
            "global_posting_time": self.posting_time,
            "platform_posting_times": platform_posting_times,
            "platforms_using_global_fallback": [
                platform for platform, config in self.platform_configs.items() 
                if self.enabled_platforms.get(platform, False) and not config["posting_time"]
            ],
            "is_valid": self.is_valid()
        }
    
    def parse_posting_time(self, platform: Optional[str] = None) -> Dict[str, any]:
        """
        Parse posting time configuration and calculate posting type and delay
        
        Args:
            platform: Specific platform name to get posting time for. If None, uses global time.
        
        Supported formats:
        - "now" -> immediate posting
        - "now + 5 minutes" -> 5 minutes delay
        - "now + 30 minutes" -> 30 minutes delay
        - "now + 2 hours" -> 2 hours delay
        
        Returns:
            Dict with posting type and calculated datetime
        """
        # Get the appropriate posting time (platform-specific or global)
        if platform:
            config_time = self.get_platform_posting_time(platform)
        else:
            config_time = self.posting_time
            
        return self._parse_time_string(config_time, platform)
    
    def _parse_time_string(self, config_time: str, platform: Optional[str] = None) -> Dict[str, any]:
        """
        Parse a time string and return posting configuration
        
        Args:
            config_time: Time string to parse
            platform: Platform name for error messages
        
        Returns:
            Dict with posting type and calculated datetime
        """
        config_time = config_time.lower().strip()
        
        if config_time == "now":
            return {
                "type": "now",
                "delay_minutes": 0,
                "scheduled_time": None,
                "description": "Immediate posting"
            }
        
        # Parse "now + N minutes/hours" format
        pattern = r"now\s*\+\s*(\d+)\s*(minutes?|hours?|mins?|hrs?)"
        match = re.match(pattern, config_time)
        
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            
            # Convert to minutes
            if unit.startswith('hour') or unit.startswith('hr'):
                delay_minutes = amount * 60
            else:  # minutes
                delay_minutes = amount
            
            # Calculate scheduled time
            scheduled_time = datetime.now() + timedelta(minutes=delay_minutes)
            
            return {
                "type": "date",
                "delay_minutes": delay_minutes,
                "scheduled_time": scheduled_time,
                "description": f"Scheduled for {delay_minutes} minutes from now ({scheduled_time.strftime('%Y-%m-%d %H:%M:%S')})"
            }
        
        # If format is invalid, default to "now"
        if platform:
            print(f"⚠️  Invalid posting time format for {platform.upper()}: '{config_time}' - defaulting to 'now'")
        else:
            print(f"⚠️  Invalid POSTIZ_POSTING_TIME format: '{config_time}' - defaulting to 'now'")
        print(f"   Supported formats: 'now', 'now + 5 minutes', 'now + 2 hours'")
        
        return {
            "type": "now",
            "delay_minutes": 0,
            "scheduled_time": None,
            "description": "Immediate posting (invalid format fallback)"
        }
    
    def get_posting_type_for_postiz(self, platform: Optional[str] = None) -> str:
        """
        Get the posting type string for Postiz API
        
        Args:
            platform: Specific platform to get posting type for
        
        Returns:
            "now" for immediate posting or "date" for scheduled posting
        """
        parsed_time = self.parse_posting_time(platform)
        return parsed_time["type"]
    
    def get_scheduled_datetime_iso(self, platform: Optional[str] = None) -> Optional[str]:
        """
        Get scheduled datetime in ISO format for Postiz API
        
        Args:
            platform: Specific platform to get scheduled time for
        
        Returns:
            ISO datetime string for scheduled posts, None for immediate posts
        """
        parsed_time = self.parse_posting_time(platform)
        if parsed_time["scheduled_time"]:
            return parsed_time["scheduled_time"].isoformat()
        return None
    
    def get_platform_scheduling_info(self, platform: str) -> Dict[str, any]:
        """
        Get comprehensive scheduling information for a specific platform
        
        Args:
            platform: Platform name (e.g., "twitter", "instagram")
        
        Returns:
            Dict with scheduling details for the platform
        """
        if not self.enabled_platforms.get(platform, False):
            return {
                "enabled": False,
                "posting_time": None,
                "parsed_time": None,
                "posting_type": None,
                "scheduled_iso": None
            }
        
        posting_time = self.get_platform_posting_time(platform)
        parsed_time = self.parse_posting_time(platform)
        
        return {
            "enabled": True,
            "posting_time": posting_time,
            "parsed_time": parsed_time,
            "posting_type": parsed_time["type"],
            "scheduled_iso": self.get_scheduled_datetime_iso(platform)
        }
    
    def platforms_have_different_times(self) -> bool:
        """
        Check if enabled platforms have different posting times
        
        Returns:
            True if platforms have different posting times, False if all use the same time
        """
        enabled_platforms = [platform for platform, enabled in self.enabled_platforms.items() if enabled]
        
        if len(enabled_platforms) <= 1:
            return False
        
        # Get posting times for all enabled platforms
        posting_times = set()
        for platform in enabled_platforms:
            posting_time = self.get_platform_posting_time(platform)
            posting_times.add(posting_time)
        
        # If all platforms have the same posting time, return False
        return len(posting_times) > 1
    
    def group_platforms_by_posting_time(self) -> Dict[str, List[str]]:
        """
        Group enabled platforms by their posting times
        
        Returns:
            Dict mapping posting times to lists of platform names
        """
        groups = {}
        
        for platform, enabled in self.enabled_platforms.items():
            if enabled:
                posting_time = self.get_platform_posting_time(platform)
                
                if posting_time not in groups:
                    groups[posting_time] = []
                
                groups[posting_time].append(platform)
        
        return groups
    
    def validate_posting_time_config(self) -> bool:
        """
        Validate the posting time configuration format
        
        Returns:
            True if valid, False if invalid
        """
        parsed_time = self.parse_posting_time()
        return not parsed_time["description"].endswith("(invalid format fallback)")