"""
Main Postiz Publisher class for social media publishing
"""

import threading
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timezone

from .config import PostizConfig
from .utils.logger import PublishingLogger


# ponytail: module-level rate limiter. One gate, all publisher threads pass through.
# Counts successful channel posts against POSTIZ_MAX_POSTS_PER_BATCH; when the batch
# fills, callers block on the condition until POSTIZ_COOLDOWN_MINUTES elapses and the
# counter resets. Single global lock is sufficient for this monitor's throughput.
class _BatchRateLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._count = 0
        self._limit = 6
        self._cooldown = 3600.0  # seconds
        self._window_start = time.monotonic()

    def configure(self, limit: int, cooldown_seconds: float):
        # Reconfigure is allowed mid-run; safe under the lock.
        with self._lock:
            self._limit = max(1, limit)
            self._cooldown = max(0.0, cooldown_seconds)

    def acquire(self, permits: int):
        """Block until `permits` slots are available in the current batch window."""
        with self._cond:
            while True:
                # Reset window when the cooldown since the first post of the batch elapses.
                if time.monotonic() - self._window_start >= self._cooldown and self._count > 0:
                    self._count = 0
                    self._window_start = time.monotonic()
                    self._cond.notify_all()
                if self._count + permits <= self._limit:
                    self._count += permits
                    return
                wait = self._cooldown - (time.monotonic() - self._window_start)
                if wait > 0:
                    print(f"⏳ Rate limit: batch full ({self._count}/{self._limit}), "
                          f"cooling down {wait:.0f}s before next batch...")
                    self._cond.wait(timeout=wait)
                else:
                    self._cond.wait(timeout=1.0)


_rate_limiter = _BatchRateLimiter()


class PostizPublisher:
    """
    Main publisher class that integrates with monitor.py workflow
    to publish videos to multiple social media platforms via Postiz API
    """
    
    def __init__(self, output_directory: str, source_link: Optional[str] = None):
        """
        Initialize publisher with video output directory
        
        Args:
            output_directory: Path to the video output directory containing
                            final_video.mp4, content.txt, etc.
            source_link: Optional URL of the original video/source
        """
        self.output_dir = Path(output_directory)
        self.video_name = self.output_dir.name
        self.source_link = source_link
        
        # Initialize configuration and logging
        try:
            self.config = PostizConfig()
            self.logger = PublishingLogger(self.video_name)
            
            # Validate basic configuration
            if not self.config.is_valid():
                raise ValueError("Invalid Postiz configuration")
            
            # Log configuration summary
            config_summary = self.config.get_configuration_summary()
            self.logger.info(f"Publisher initialized for video: {self.video_name}")
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info(f"API endpoint: {config_summary['endpoint']}")
            self.logger.info(f"Enabled platforms: {config_summary['enabled_platforms']}")
            self.logger.info(f"Configured channels: {config_summary['channel_count']}")
            
            # Initialize Postiz client
            from .postiz.client import PostizClient
            self.client = PostizClient(self.config.api_key, self.config.endpoint, mock_mode=self.config.mock_mode)
            
            # Log mock mode status
            if self.config.mock_mode:
                self.logger.info("🎭 MOCK MODE ENABLED - No real API calls will be made")
            
            # Configure the shared batch rate limiter from env (idempotent across instances)
            _rate_limiter.configure(
                self.config.max_posts_per_batch,
                self.config.cooldown_minutes * 60.0,
            )
            self.logger.info(
                f"Rate limiting: max {self.config.max_posts_per_batch} posts per batch, "
                f"{self.config.cooldown_minutes}-minute cooldown"
            )
        except Exception as e:
            print(f"❌ Failed to initialize PostizPublisher: {e}")
            raise
    
    def publish_async(self) -> None:
        """
        Start publishing in background thread (non-blocking)
        Returns immediately to allow monitor.py to continue processing
        """
        self.logger.info("Starting asynchronous publishing workflow")
        
        # Start publishing in daemon thread
        publishing_thread = threading.Thread(
            target=self._publish_workflow,
            name=f"postiz-publisher-{self.video_name}",
            daemon=True
        )
        publishing_thread.start()
        
        self.logger.info("Publishing thread started, returning control to monitor")
    
    def _publish_workflow(self) -> Dict[str, str]:
        """
        Complete Postiz publishing workflow: validate → upload → post → log results
        This runs in background thread
        """
        start_time = time.time()
        self.logger.info("=== Starting Postiz Publishing Workflow ===")
        
        try:
            # Step 1: Validate API connectivity
            self.logger.info("Step 1: Validating API connectivity")
            connection_result = self.config.validate_api_connection()
            if connection_result["status"] != "success":
                if connection_result["status"] == "rate_limited":
                    self.logger.warning(f"Rate limited: {connection_result['message']}")
                    return {"error": "rate_limited", "message": connection_result["message"]}
                else:
                    self.logger.error(f"API validation failed: {connection_result['message']}")
                    return {"error": "api_validation", "message": connection_result["message"]}
            
            self.logger.info("✅ API connectivity validated")
            
            # Step 2: Validate and prepare content
            self.logger.info("Step 2: Validating video files and extracting content")
            video_path, platform_content, metadata = self._prepare_content()
            
            # Step 3: Upload video to Postiz
            self.logger.info("Step 3: Uploading video to Postiz")
            try:
                file_info = self.client.upload_video(str(video_path))
                self.logger.info(f"✅ Video uploaded successfully: {file_info.get('id', 'unknown')}")
            except Exception as e:
                self.logger.error(f"Video upload failed: {str(e)}")
                return {"error": "upload_failed", "message": str(e)}
            
            # Step 4: Create multi-platform post with platform-specific scheduling
            self.logger.info("Step 4: Creating multi-platform post with unified scheduling")
            enabled_channels = self.config.get_enabled_channels()
            
            # Create platform mapping for platform-specific settings (channel_id -> platform_name)
            platform_mapping = {channel_id: platform for platform, channel_id in enabled_channels.items()}
            
            self.logger.info(f"Target platforms: {list(enabled_channels.keys())}")
            self.logger.info(f"Channel IDs: {list(enabled_channels.values())}")
            
            # Create schedule mapping for per-channel scheduling
            self.logger.info("📅 Generating schedule mapping for all channels...")
            schedule_mapping = {}
            for platform, channel_id in enabled_channels.items():
                posting_type = self.config.get_posting_type_for_postiz(platform)
                scheduled_datetime = self.config.get_scheduled_datetime_iso(platform)
                
                schedule_mapping[channel_id] = {
                    "type": posting_type,
                    "date": scheduled_datetime,
                    "platform": platform
                }
                
                # Log scheduling info
                if posting_type == "date":
                    self.logger.info(f"   {platform} ({channel_id}): Scheduled for {scheduled_datetime}")
                else:
                    self.logger.info(f"   {platform} ({channel_id}): Immediate posting")

            # Always use bulk scheduling with the schedule mapping
            # Acquire rate-limit permits for every channel in this post before the API call.
            channel_count_for_post = len(enabled_channels)
            self.logger.info(
                f"🔒 Acquiring rate-limit permits for {channel_count_for_post} post(s)..."
            )
            _rate_limiter.acquire(channel_count_for_post)
            self.logger.info(f"🔒 Permits acquired — posting now")
            post_results = self._handle_bulk_platform_scheduling(
                file_info, platform_content, enabled_channels, platform_mapping, metadata, schedule_mapping
            )
            
            if post_results["success"]:
                success_count = post_results.get("success_count", len(enabled_channels))
                total_count = post_results.get("total_channels", len(enabled_channels))
                
                self.logger.info(f"✅ Post creation completed: {success_count}/{total_count} successful")
                
                if post_results.get("bulk_creation", False):
                    self.logger.info("   Used bulk creation for all platforms")
                else:
                    self.logger.info("   Used individual platform fallback")
                
                # Log successful posts
                if "successful_posts" in post_results:
                    for post in post_results["successful_posts"]:
                        if isinstance(post, dict) and "channel_id" in post:
                            self.logger.info(f"   ✅ Posted to channel: {post['channel_id']}")
                        else:
                            self.logger.info(f"   ✅ Posted to channel: {post}")
                
                # Log failed platforms
                if post_results.get("failed_platforms"):
                    self.logger.warning(f"⚠️  {len(post_results['failed_platforms'])} platforms failed:")
                    for failure in post_results["failed_platforms"]:
                        self.logger.warning(f"   ❌ {failure['channel_id']}: {failure['error']}")
            
            else:
                self.logger.error("❌ All platform posting attempts failed")
                return {"error": "all_posts_failed", "details": post_results}
            
            # Step 5: Log results
            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ Publishing completed successfully in {elapsed_time:.2f} seconds")
            
            # Format results for logging
            success_result = {
                "success": True, 
                "platforms": list(enabled_channels.keys()),
                "file_id": file_info.get("id"),
                "post_results": post_results
            }
            self._log_results(success_result)
            
            return success_result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"❌ Publishing failed after {elapsed_time:.2f} seconds: {str(e)}")
            self._log_error(str(e))
            return {"error": str(e)}
    
    def _handle_bulk_platform_scheduling(self, file_info, platform_content, enabled_channels, platform_mapping, metadata, schedule_mapping) -> Dict[str, any]:
        """
        Handle bulk platform scheduling using unified schedule mapping
        
        Args:
            file_info: File information from upload
            platform_content: Platform-specific content
            enabled_channels: Dict of enabled platforms and their channel IDs
            platform_mapping: Channel ID to platform mapping
            metadata: Video metadata
            schedule_mapping: Dict mapping channel_id -> {type, date, platform}
            
        Returns:
            Dict with posting results
        """
        channel_ids = list(enabled_channels.values())
        
        # Use a default global configuration for the main request wrapper (fallback)
        # The client will use the schedule_mapping for per-post details
        first_platform = list(enabled_channels.keys())[0]
        posting_type = self.config.get_posting_type_for_postiz(first_platform)
        scheduled_datetime = self.config.get_scheduled_datetime_iso(first_platform)
        
        try:
            post_results = self.client.create_post_with_fallback(
                file_info, platform_content, channel_ids, posting_type, scheduled_datetime,
                metadata=metadata,
                platform_mapping=platform_mapping,
                schedule_mapping=schedule_mapping
            )
            return post_results
        except Exception as e:
            self.logger.error(f"Bulk posting failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _prepare_content(self) -> tuple[Path, str, object]:
        """
        Validate video files exist and extract content for publishing
        
        Returns:
            tuple: (video_path, formatted_content, metadata)
        """
        # Look for video file
        video_path = self.output_dir / "final_video.mp4"
        if not video_path.exists():
            # Try alternative names
            video_files = list(self.output_dir.glob("*.mp4"))
            if video_files:
                video_path = video_files[0]
                self.logger.info(f"Using video file: {video_path.name}")
            else:
                raise FileNotFoundError(f"No video file found in {self.output_dir}")
        
        # Extract metadata using enhanced content extractor
        from .utils.content_extractor import ContentExtractor
        from .utils.content_formatter import SocialMediaFormatter
        
        extractor = ContentExtractor(self.output_dir)
        metadata = extractor.extract_metadata()
        
        # Add source link to metadata if available
        if self.source_link:
            metadata.source_link = self.source_link
            self.logger.info(f"Added source link to metadata: {self.source_link}")
        
        # Log extracted metadata
        self.logger.info(f"Extracted metadata:")
        self.logger.info(f"  Title: {metadata.title}")
        self.logger.info(f"  Description: {metadata.description[:100]}...")
        self.logger.info(f"  Segments: {len(metadata.segments)}")
        self.logger.info(f"  Hashtags: {metadata.hashtags}")
        
        # Get segment 0 information as specified in requirements
        segment_zero = metadata.get_segment_zero()
        if segment_zero:
            self.logger.info(f"Using segment 0: {segment_zero.title} - {segment_zero.description[:50]}...")
        
        # Format content for social media platforms
        formatter = SocialMediaFormatter(metadata)
        enabled_platforms = list(self.config.get_enabled_channels().keys())
        
        # Get platform-optimized content instead of unified content
        # This allows each platform to use its full character limits
        platform_content = formatter.get_platform_optimized_content(enabled_platforms)
        
        # For logging purposes, show the first platform's content
        first_platform = list(enabled_platforms)[0] if enabled_platforms else "twitter"
        formatted_content = platform_content.get(first_platform, "")
        
        self.logger.info(f"Video file: {video_path}")
        self.logger.info(f"Formatted content length: {len(formatted_content)} characters")
        self.logger.info(f"Content preview: {formatted_content[:100]}...")
        
        return video_path, platform_content, metadata
    
    def _log_results(self, results: Dict[str, str]):
        """Log successful publishing results"""
        self.logger.info("=== Publishing Results ===")
        if results.get("success"):
            platforms = results.get("platforms", [])
            self.logger.info(f"Successfully published to: {', '.join(platforms)}")
        else:
            self.logger.error(f"Publishing failed: {results}")
    
    def _log_error(self, error_message: str):
        """Log error information"""
        self.logger.error("=== Publishing Error ===")
        self.logger.error(f"Error: {error_message}")
        self.logger.error(f"Video: {self.video_name}")
        self.logger.error(f"Directory: {self.output_dir}")


# Backward compatibility alias
SocialMediaPublisher = PostizPublisher