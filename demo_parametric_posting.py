#!/usr/bin/env python3
"""
Demo script for parametric posting time configuration
Shows how the .env POSTIZ_POSTING_TIME setting works
"""

import os
from pathlib import Path
from datetime import datetime


def demo_posting_time_config():
    """Demonstrate parametric posting time configuration"""
    print("🕐 Parametric Posting Time Configuration Demo")
    print("=" * 50)
    
    # Test different configurations
    test_configs = [
        "now",
        "now + 5 minutes", 
        "now + 30 minutes",
        "now + 2 hours",
        "now + 1 hour",
        "invalid format"  # This will show fallback behavior
    ]
    
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Import after sys path setup
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from social_media_publisher.config import PostizConfig
    
    for config_value in test_configs:
        print(f"📝 Testing: POSTIZ_POSTING_TIME=\"{config_value}\"")
        
        # Set environment variable
        os.environ["POSTIZ_POSTING_TIME"] = config_value
        os.environ["POSTIZ_API_KEY"] = "demo_key"
        
        # Create config and parse (with mock env loading to avoid file dependencies)
        from unittest.mock import patch
        
        with patch.object(PostizConfig, '_load_env_file'):
            config = PostizConfig()
            
            # Parse and display results
            parsed_time = config.parse_posting_time()
            posting_type = config.get_posting_type_for_postiz()
            scheduled_datetime = config.get_scheduled_datetime_iso()
            is_valid = config.validate_posting_time_config()
            
            print(f"   Type: {posting_type}")
            print(f"   Valid: {is_valid}")
            print(f"   Description: {parsed_time['description']}")
            if scheduled_datetime:
                print(f"   Scheduled: {scheduled_datetime}")
            print(f"   Delay: {parsed_time['delay_minutes']} minutes")
            print()
    
    print("🎯 Usage in .env file:")
    print("""
# Immediate posting (default)
POSTIZ_POSTING_TIME="now"

# Schedule 15 minutes from now  
POSTIZ_POSTING_TIME="now + 15 minutes"

# Schedule 2 hours from now
POSTIZ_POSTING_TIME="now + 2 hours"

# Also supports: mins, hrs, minute, hour variations
POSTIZ_POSTING_TIME="now + 30 mins"
POSTIZ_POSTING_TIME="now + 1 hour"
""")
    
    print("📋 Benefits:")
    print("   • Perfect for testing - avoid immediate publishing")
    print("   • Schedule posts at optimal times")  
    print("   • Review content before it goes live")
    print("   • Coordinate multi-platform releases")
    print("   • Fallback to immediate posting for invalid formats")


if __name__ == "__main__":
    demo_posting_time_config()