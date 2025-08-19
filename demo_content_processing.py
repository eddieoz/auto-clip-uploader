#!/usr/bin/env python3
"""
Demo script for enhanced content processing
Shows how the content extractor and formatter work with segment 0
"""

import tempfile
import shutil
from pathlib import Path


def create_sample_content():
    """Create sample content.txt file for demonstration"""
    return """Title: Revolutionary AI Breakthrough Changes Everything

Description: Scientists have made an incredible discovery that could revolutionize how we think about artificial intelligence and its applications in everyday life.

Segments:
0: Mind-blowing opening revelation (0:00-0:20)
1: Technical deep dive into the research (0:20-0:45) 
2: Real-world impact and future implications (0:45-1:00)

Using segment 0 for maximum viral potential on social media platforms.

#AI #breakthrough #technology #innovation #science #viral #trending #research #future"""


def demo_content_processing():
    """Demonstrate the enhanced content processing system"""
    print("🎬 Social Media Content Processing Demo")
    print("=" * 50)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    video_dir = temp_dir / "demo_video"
    video_dir.mkdir(parents=True)
    
    try:
        # Create sample files
        (video_dir / "final_video.mp4").touch()
        (video_dir / "content.txt").write_text(create_sample_content())
        
        print("📄 Sample content.txt created:")
        print("-" * 30)
        print(create_sample_content()[:300] + "...")
        
        # Import content processing modules
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        from social_media_publisher.utils.content_extractor import ContentExtractor
        from social_media_publisher.utils.content_formatter import SocialMediaFormatter
        
        # Extract metadata
        print("\n🔍 Extracting Metadata...")
        extractor = ContentExtractor(video_dir)
        metadata = extractor.extract_metadata()
        
        print(f"✅ Title: {metadata.title}")
        print(f"✅ Description: {metadata.description[:100]}...")
        print(f"✅ Segments found: {len(metadata.segments)}")
        print(f"✅ Hashtags: {metadata.hashtags}")
        
        # Show segment 0 information
        print(f"\n🎯 Segment 0 (as specified in requirements):")
        segment_zero = metadata.get_segment_zero()
        if segment_zero:
            print(f"   Index: {segment_zero.index}")
            print(f"   Title: {segment_zero.title}")
            print(f"   Description: {segment_zero.description}")
            print(f"   Time: {segment_zero.start_time} → {segment_zero.end_time}")
        
        # Format for different platforms
        print(f"\n📱 Platform-Specific Formatting:")
        formatter = SocialMediaFormatter(metadata)
        
        platforms = ["twitter", "instagram", "youtube", "tiktok"]
        
        for platform in platforms:
            formatted = formatter.format_for_platform(platform)
            print(f"\n📺 {platform.upper()}:")
            print(f"   Length: {len(formatted)} characters")
            print(f"   Content: {formatted[:100]}...")
            
            if platform == "twitter":
                print(f"   ✅ Under 280 char limit: {len(formatted) <= 280}")
            elif platform == "instagram":
                print(f"   ✅ Has emojis: {'🎬' in formatted or '✨' in formatted}")
            elif platform == "youtube":
                print(f"   ✅ Has call to action: {'SUBSCRIBE' in formatted.upper()}")
            elif platform == "tiktok":
                print(f"   ✅ Has TikTok hashtags: {'#fyp' in formatted}")
        
        # Show unified content
        print(f"\n🔄 Unified Content (works across all platforms):")
        unified = formatter.get_unified_content(platforms)
        print(f"   Length: {len(unified)} characters")
        print(f"   Content: {unified}")
        
        print(f"\n✅ Content processing demo completed successfully!")
        print(f"\n💡 Key Features Demonstrated:")
        print(f"   • Structured content.txt parsing")
        print(f"   • Segment 0 extraction (as specified)")
        print(f"   • Platform-specific formatting")
        print(f"   • Hashtag optimization")
        print(f"   • Content length management")
        print(f"   • Time range parsing")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_fallback_handling():
    """Demo fallback handling when content.txt is missing"""
    print(f"\n🔧 Fallback Handling Demo")
    print("=" * 30)
    
    # Create temporary directory without content.txt
    temp_dir = Path(tempfile.mkdtemp()) 
    video_dir = temp_dir / "fallback_video"
    video_dir.mkdir(parents=True)
    
    try:
        # Only create video file, no content.txt
        (video_dir / "final_video.mp4").touch()
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        from social_media_publisher.utils.content_extractor import ContentExtractor
        
        extractor = ContentExtractor(video_dir)
        metadata = extractor.extract_metadata()
        
        print(f"✅ Fallback title: {metadata.title}")
        print(f"✅ Fallback description: {metadata.description}")
        print(f"✅ Default hashtags: {metadata.hashtags}")
        print(f"✅ Default segment 0 created: {metadata.get_segment_zero() is not None}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    demo_content_processing()
    demo_fallback_handling()