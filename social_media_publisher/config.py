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
        }
        
        # Platform enable/disable flags
        self.enabled_platforms = {
            "twitter": os.getenv("PUBLISH_TO_TWITTER", "true").lower() == "true",
            "instagram": os.getenv("PUBLISH_TO_INSTAGRAM", "true").lower() == "true",
            "youtube": os.getenv("PUBLISH_TO_YOUTUBE", "true").lower() == "true",
            "tiktok": os.getenv("PUBLISH_TO_TIKTOK", "true").lower() == "true",
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
        
        return {
            "api_key_present": bool(self.api_key),
            "api_key_preview": f"{self.api_key[:8]}..." if self.api_key else None,
            "endpoint": self.endpoint,
            "enabled_platforms": [platform for platform, enabled in self.enabled_platforms.items() if enabled],
            "configured_channels": list(enabled_channels.keys()),
            "channel_count": len(enabled_channels),
            "posting_time_config": self.posting_time,
            "is_valid": self.is_valid()
        }
    
    def parse_posting_time(self) -> Dict[str, any]:
        """
        Parse posting time configuration and calculate posting type and delay
        
        Supported formats:
        - "now" -> immediate posting
        - "now + 5 minutes" -> 5 minutes delay
        - "now + 30 minutes" -> 30 minutes delay
        - "now + 2 hours" -> 2 hours delay
        
        Returns:
            Dict with posting type and calculated datetime
        """
        config_time = self.posting_time.lower().strip()
        
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
        print(f"⚠️  Invalid POSTIZ_POSTING_TIME format: '{self.posting_time}' - defaulting to 'now'")
        print(f"   Supported formats: 'now', 'now + 5 minutes', 'now + 2 hours'")
        
        return {
            "type": "now",
            "delay_minutes": 0,
            "scheduled_time": None,
            "description": "Immediate posting (invalid format fallback)"
        }
    
    def get_posting_type_for_postiz(self) -> str:
        """
        Get the posting type string for Postiz API
        
        Returns:
            "now" for immediate posting or "date" for scheduled posting
        """
        parsed_time = self.parse_posting_time()
        return parsed_time["type"]
    
    def get_scheduled_datetime_iso(self) -> Optional[str]:
        """
        Get scheduled datetime in ISO format for Postiz API
        
        Returns:
            ISO datetime string for scheduled posts, None for immediate posts
        """
        parsed_time = self.parse_posting_time()
        if parsed_time["scheduled_time"]:
            return parsed_time["scheduled_time"].isoformat()
        return None
    
    def validate_posting_time_config(self) -> bool:
        """
        Validate the posting time configuration format
        
        Returns:
            True if valid, False if invalid
        """
        parsed_time = self.parse_posting_time()
        return not parsed_time["description"].endswith("(invalid format fallback)")