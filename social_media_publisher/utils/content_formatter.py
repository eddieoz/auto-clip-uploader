"""
Content formatter for social media platforms
"""

import re
from typing import Dict, List
from dataclasses import dataclass

from .content_extractor import VideoMetadata


@dataclass
class PlatformLimits:
    """Character limits and requirements for social media platforms"""
    max_length: int
    hashtag_limit: int
    supports_emojis: bool = True
    line_breaks_allowed: bool = True


class SocialMediaFormatter:
    """Formats content for different social media platforms"""
    
    # Platform-specific limits
    PLATFORM_LIMITS = {
        "twitter": PlatformLimits(max_length=280, hashtag_limit=3),
        "instagram": PlatformLimits(max_length=2200, hashtag_limit=30), 
        "youtube": PlatformLimits(max_length=5000, hashtag_limit=15),
        "tiktok": PlatformLimits(max_length=2200, hashtag_limit=20),
        "linkedin": PlatformLimits(max_length=3000, hashtag_limit=10),
        "facebook": PlatformLimits(max_length=63206, hashtag_limit=20)
    }
    
    def __init__(self, metadata: VideoMetadata):
        """
        Initialize formatter with video metadata
        
        Args:
            metadata: VideoMetadata object with video information
        """
        self.metadata = metadata
    
    def format_for_platform(self, platform: str) -> str:
        """
        Format content for specific social media platform
        
        Args:
            platform: Platform name (twitter, instagram, youtube, tiktok, etc.)
            
        Returns:
            Formatted content string optimized for the platform
        """
        platform = platform.lower()
        limits = self.PLATFORM_LIMITS.get(platform, self.PLATFORM_LIMITS["twitter"])
        
        # Use segment 0 content as specified in requirements
        segment_zero = self.metadata.get_segment_zero()
        
        # Build content components
        if platform == "twitter":
            return self._format_twitter(limits, segment_zero)
        elif platform == "instagram":
            return self._format_instagram(limits, segment_zero)
        elif platform == "youtube":
            return self._format_youtube(limits, segment_zero)
        elif platform == "tiktok":
            return self._format_tiktok(limits, segment_zero)
        else:
            return self._format_generic(limits, segment_zero)
    
    def _format_twitter(self, limits: PlatformLimits, segment_zero) -> str:
        """Format content for Twitter/X"""
        components = []
        
        # Use title or segment title
        title = segment_zero.title if segment_zero else self.metadata.title
        if title and title != f"Segment {segment_zero.index if segment_zero else 0}":
            components.append(title)
        
        # Add segment description if different from title
        if segment_zero and segment_zero.description:
            if not title or segment_zero.description != title:
                components.append(segment_zero.description)
        elif self.metadata.description:
            components.append(self.metadata.description)
        
        # Combine and truncate
        content = " - ".join(components)
        
        # Reserve space for hashtags
        hashtag_space = self._calculate_hashtag_space(limits.hashtag_limit)
        max_content_length = limits.max_length - hashtag_space - 5  # Buffer for spacing
        
        if len(content) > max_content_length:
            content = content[:max_content_length].rsplit(' ', 1)[0] + "..."
        
        # Add hashtags
        hashtags = self._select_hashtags(limits.hashtag_limit)
        if hashtags:
            content += f"\n\n{' '.join(hashtags)}"
        
        return content
    
    def _format_instagram(self, limits: PlatformLimits, segment_zero) -> str:
        """Format content for Instagram"""
        components = []
        
        # Instagram allows longer content, so be more descriptive
        title = segment_zero.title if segment_zero else self.metadata.title
        if title and title != f"Segment {segment_zero.index if segment_zero else 0}":
            components.append(f"🎬 {title}")
        
        # Add description with emojis
        if segment_zero and segment_zero.description:
            components.append(f"✨ {segment_zero.description}")
        elif self.metadata.description:
            components.append(f"✨ {self.metadata.description}")
        
        # Add call to action
        components.append("💫 Siga para mais conteúdos como esse!")
        
        content = "\n\n".join(components)
        
        # Add hashtags (Instagram loves hashtags)
        hashtags = self._select_hashtags(limits.hashtag_limit)
        if hashtags:
            # content += f"\n\n{' '.join(hashtags)}"
            content += "\n"
            
        # Add trending hashtags specific to Instagram
        # instagram_hashtags = ["#reels", "#viral", "#explore", "#fyp"]
        instagram_hashtags = ["#MorningCrypto"]
        remaining_space = limits.max_length - len(content) - 10
        
        for tag in instagram_hashtags:
            if tag not in content and len(content + " " + tag) < remaining_space:
                content += f" {tag}"
        
        return content[:limits.max_length]
    
    def _format_youtube(self, limits: PlatformLimits, segment_zero) -> str:
        """Format content for YouTube Shorts"""
        components = []
        
        # YouTube title/description format
        title = segment_zero.title if segment_zero else self.metadata.title
        if title and title != f"Segment {segment_zero.index if segment_zero else 0}":
            components.append(title.upper())  # YouTube likes bold titles
        
        if segment_zero and segment_zero.description:
            components.append(segment_zero.description)
        elif self.metadata.description:
            components.append(self.metadata.description)
        
        # Add YouTube-specific call to action
        components.append("👍 LIKE & SUBSCRIBE for more content!")
        components.append("🔔 Turn on notifications!")
        
        content = "\n\n".join(components)
        
        # Add hashtags
        hashtags = self._select_hashtags(limits.hashtag_limit)
        # youtube_hashtags = ["#shorts", "#viral", "#trending"]
        youtube_hashtags = ["#MorningCrypto"]
        
        # Combine hashtags
        all_hashtags = hashtags + [tag for tag in youtube_hashtags if tag not in hashtags]
        all_hashtags = all_hashtags[:limits.hashtag_limit]
        
        if all_hashtags:
            content += f"\n\n{' '.join(all_hashtags)}"
        
        return content[:limits.max_length]
    
    def _format_tiktok(self, limits: PlatformLimits, segment_zero) -> str:
        """Format content for TikTok"""
        components = []
        
        # TikTok prefers casual, engaging tone
        title = segment_zero.title if segment_zero else self.metadata.title
        if title and title != f"Segment {segment_zero.index if segment_zero else 0}":
            # Make title more engaging for TikTok
            components.append(f"✨ {title} ✨")
        
        if segment_zero and segment_zero.description:
            components.append(segment_zero.description)
        elif self.metadata.description:
            components.append(self.metadata.description)
        
        content = "\n\n".join(components)
        
        # Add TikTok-specific hashtags
        hashtags = self._select_hashtags(limits.hashtag_limit)
        # tiktok_hashtags = ["#fyp", "#viral", "#foryou", "#trending"]
        tiktok_hashtags = ["#MorningCrypto"]
        
        # Prioritize TikTok hashtags
        final_hashtags = []
        for tag in tiktok_hashtags:
            if tag not in content:
                final_hashtags.append(tag)
        
        # Add original hashtags that aren't duplicates
        for tag in hashtags:
            if tag not in final_hashtags and len(final_hashtags) < limits.hashtag_limit:
                final_hashtags.append(tag)
        
        if final_hashtags:
            content += f"\n\n{' '.join(final_hashtags)}"
        
        return content[:limits.max_length]
    
    def _format_generic(self, limits: PlatformLimits, segment_zero) -> str:
        """Generic format for other platforms"""
        components = []
        
        title = segment_zero.title if segment_zero else self.metadata.title
        if title and title != f"Segment {segment_zero.index if segment_zero else 0}":
            components.append(title)
        
        if segment_zero and segment_zero.description:
            components.append(segment_zero.description)
        elif self.metadata.description:
            components.append(self.metadata.description)
        
        content = " - ".join(components)
        
        # Reserve space for hashtags and truncate if needed
        hashtag_space = self._calculate_hashtag_space(limits.hashtag_limit)
        max_content_length = limits.max_length - hashtag_space - 5
        
        if len(content) > max_content_length:
            content = content[:max_content_length].rsplit(' ', 1)[0] + "..."
        
        # Add hashtags
        hashtags = self._select_hashtags(limits.hashtag_limit)
        if hashtags:
            content += f"\n\n{' '.join(hashtags)}"
        
        return content
    
    def _select_hashtags(self, limit: int) -> List[str]:
        """
        Select most relevant hashtags up to the platform limit
        
        Args:
            limit: Maximum number of hashtags allowed
            
        Returns:
            List of selected hashtags
        """
        # Start with hashtags from content
        hashtags = self.metadata.hashtags.copy()
        
        # Add default hashtags if not present
        # default_tags = ["#viral", "#shorts", "#content"]
        default_tags = ["#MorningCrypto"]
        
        for tag in default_tags:
            if tag not in hashtags:
                hashtags.append(tag)
        
        # Prioritize hashtags (viral content first)
        priority_order = ["MorningCrypto", "#viral", "#trending", "#fyp", "#shorts", "#content", "#video"]
        
        prioritized = []
        remaining = []
        
        for tag in hashtags:
            if tag in priority_order:
                prioritized.append(tag)
            else:
                remaining.append(tag)
        
        # Sort prioritized by priority order
        prioritized.sort(key=lambda x: priority_order.index(x) if x in priority_order else 999)
        
        # Combine and limit
        final_hashtags = prioritized + remaining
        return final_hashtags[:limit]
    
    def _calculate_hashtag_space(self, hashtag_limit: int) -> int:
        """
        Calculate approximate space needed for hashtags
        
        Args:
            hashtag_limit: Maximum number of hashtags
            
        Returns:
            Estimated character count for hashtags
        """
        # Estimate average hashtag length + spaces
        avg_hashtag_length = 8  # Including # and typical word length
        return hashtag_limit * (avg_hashtag_length + 1) + 5  # Buffer for formatting
    
    def get_platform_optimized_content(self, enabled_platforms: List[str]) -> Dict[str, str]:
        """
        Get content optimized for all enabled platforms
        
        Args:
            enabled_platforms: List of platform names
            
        Returns:
            Dictionary mapping platform names to formatted content
        """
        formatted_content = {}
        
        for platform in enabled_platforms:
            formatted_content[platform] = self.format_for_platform(platform)
        
        return formatted_content
    
    def get_unified_content(self, enabled_platforms: List[str]) -> str:
        """
        Get unified content that works across all enabled platforms
        Uses most restrictive limits to ensure compatibility
        
        Args:
            enabled_platforms: List of platform names
            
        Returns:
            Unified content string
        """
        # Find most restrictive limits
        min_length = min(
            self.PLATFORM_LIMITS.get(p, self.PLATFORM_LIMITS["twitter"]).max_length 
            for p in enabled_platforms
        )
        min_hashtags = min(
            self.PLATFORM_LIMITS.get(p, self.PLATFORM_LIMITS["twitter"]).hashtag_limit 
            for p in enabled_platforms
        )
        
        # Create unified limits
        unified_limits = PlatformLimits(max_length=min_length, hashtag_limit=min_hashtags)
        
        # Use generic formatting with unified limits
        segment_zero = self.metadata.get_segment_zero()
        return self._format_generic(unified_limits, segment_zero)