#!/usr/bin/env python3
"""
Configuration validation utility for social media publisher

Run this script to validate your Postiz API configuration:
python validate_social_media_config.py
"""

import sys
from pathlib import Path


def main():
    """Validate social media publisher configuration"""
    print("🔧 Social Media Publisher Configuration Validator")
    print("=" * 55)
    
    try:
        # Import the configuration
        from social_media_publisher.config import PostizConfig
        
        print("📋 Loading configuration...")
        config = PostizConfig()
        
        # Get configuration summary
        summary = config.get_configuration_summary()
        
        print("\n📊 Configuration Summary:")
        print(f"   API Key Present: {'✅' if summary['api_key_present'] else '❌'}")
        if summary['api_key_preview']:
            print(f"   API Key Preview: {summary['api_key_preview']}")
        print(f"   Endpoint: {summary['endpoint']}")
        print(f"   Enabled Platforms: {summary['enabled_platforms']}")
        print(f"   Configured Channels: {summary['configured_channels']}")
        print(f"   Channel Count: {summary['channel_count']}")
        print(f"   Basic Validation: {'✅ Valid' if summary['is_valid'] else '❌ Invalid'}")
        
        if not summary['is_valid']:
            print("\n❌ Configuration Issues:")
            if not summary['api_key_present']:
                print("   - POSTIZ_API_KEY is missing")
            if summary['channel_count'] == 0:
                print("   - No enabled platforms with channel IDs")
            return False
        
        # Test API connectivity
        print("\n🌐 Testing API Connectivity...")
        connection_result = config.validate_api_connection()
        
        if connection_result["status"] == "success":
            print("✅ API Connection: SUCCESS")
            print(f"   Response Code: {connection_result.get('response_code', 'N/A')}")
        elif connection_result["status"] == "rate_limited":
            print("⚠️ API Connection: RATE LIMITED")
            print(f"   Message: {connection_result['message']}")
        else:
            print("❌ API Connection: FAILED")
            print(f"   Error: {connection_result['message']}")
            return False
        
        print("\n✅ Configuration validation completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Test with a sample video using the publisher")
        print("   2. Monitor logs for any publishing issues")
        print("   3. Check Postiz dashboard for successful posts")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure social_media_publisher module is properly installed")
        return False
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        return False


def check_env_file():
    """Check if .env file exists and has required fields"""
    print("\n📝 Checking .env file...")
    
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found in current directory")
        print("   Create .env file with required Postiz configuration")
        return False
    
    print("✅ .env file found")
    
    # Read .env and check for required fields
    with open(env_path, 'r') as f:
        env_content = f.read()
    
    required_fields = [
        "POSTIZ_API_KEY",
        "POSTIZ_TWITTER_CHANNEL_ID",
        "POSTIZ_INSTAGRAM_CHANNEL_ID", 
        "POSTIZ_YOUTUBE_CHANNEL_ID",
        "POSTIZ_TIKTOK_CHANNEL_ID"
    ]
    
    present_fields = []
    missing_fields = []
    
    for field in required_fields:
        if field in env_content and f"{field}=" in env_content:
            present_fields.append(field)
        else:
            missing_fields.append(field)
    
    print(f"   Present fields: {len(present_fields)}/{len(required_fields)}")
    for field in present_fields:
        print(f"   ✅ {field}")
    
    if missing_fields:
        print(f"   Missing fields: {len(missing_fields)}")
        for field in missing_fields:
            print(f"   ❌ {field}")
    
    return len(missing_fields) == 0


if __name__ == "__main__":
    print("Starting configuration validation...\n")
    
    # Check .env file first
    env_ok = check_env_file()
    
    if env_ok:
        # Run main validation
        success = main()
        sys.exit(0 if success else 1)
    else:
        print("\n❌ Please fix .env file configuration before proceeding")
        sys.exit(1)