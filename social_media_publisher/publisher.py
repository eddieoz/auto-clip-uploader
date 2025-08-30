"""
Main Postiz Publisher class for social media publishing
"""

import threading
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from .config import PostizConfig
from .utils.logger import PublishingLogger


class PostizPublisher:
    """
    Main publisher class that integrates with monitor.py workflow
    to publish videos to multiple social media platforms via Postiz API
    """
    
    def __init__(self, output_directory: str):
        """
        Initialize publisher with video output directory
        
        Args:
            output_directory: Path to the video output directory containing
                            final_video.mp4, content.txt, etc.
        """
        self.output_dir = Path(output_directory)
        self.video_name = self.output_dir.name
        
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
            self.logger.info("Step 4: Creating multi-platform post with platform-specific scheduling")
            enabled_channels = self.config.get_enabled_channels()
            
            # Create platform mapping for platform-specific settings (channel_id -> platform_name)
            platform_mapping = {channel_id: platform for platform, channel_id in enabled_channels.items()}
            
            self.logger.info(f"Target platforms: {list(enabled_channels.keys())}")
            self.logger.info(f"Channel IDs: {list(enabled_channels.values())}")
            self.logger.info(f"Platform mapping: {platform_mapping}")
            
            # Check if platforms have different posting times
            if self.config.platforms_have_different_times():
                self.logger.info("🕒 Platforms have different posting times - using individual scheduling")
                post_results = self._handle_individual_platform_scheduling(
                    file_info, platform_content, enabled_channels, platform_mapping, metadata
                )
            else:
                self.logger.info("🕒 All platforms have same posting time - using bulk posting")
                post_results = self._handle_bulk_platform_scheduling(
                    file_info, platform_content, enabled_channels, platform_mapping, metadata
                )
            
            if post_results["success"]:
                success_count = post_results.get("success_count", len(enabled_channels))
                total_count = post_results.get("total_channels", len(enabled_channels))
                
                self.logger.info(f"✅ Post creation completed: {success_count}/{total_count} successful")
                
                if post_results.get("bulk_creation", False):
                    self.logger.info("   Used bulk creation for all platforms")
                elif post_results.get("individual_creation", False):
                    self.logger.info("   Used individual platform scheduling")
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
    
    def _handle_bulk_platform_scheduling(self, file_info, platform_content, enabled_channels, platform_mapping, metadata) -> Dict[str, any]:
        """
        Handle bulk platform scheduling when all platforms have the same posting time
        
        Args:
            file_info: File information from upload
            platform_content: Platform-specific content
            enabled_channels: Dict of enabled platforms and their channel IDs
            platform_mapping: Channel ID to platform mapping
            metadata: Video metadata
            
        Returns:
            Dict with posting results
        """
        channel_ids = list(enabled_channels.values())
        
        # Since all platforms have the same time, use any platform to get the posting configuration
        first_platform = list(enabled_channels.keys())[0]
        posting_time_info = self.config.parse_posting_time(first_platform)
        posting_type = self.config.get_posting_type_for_postiz(first_platform)
        scheduled_datetime = self.config.get_scheduled_datetime_iso(first_platform)
        
        # Debug: Log posting time processing
        self.logger.info(f"🕒 Using {first_platform} posting configuration for all platforms")
        self.logger.info(f"🕒 Posting time: '{self.config.get_platform_posting_time(first_platform)}'")
        self.logger.info(f"🕒 Parsed posting type: '{posting_type}'")
        self.logger.info(f"Posting schedule: {posting_time_info['description']}")
        if posting_type == "date":
            self.logger.info(f"Scheduled for: {scheduled_datetime}")
        
        try:
            post_results = self.client.create_post_with_fallback(
                file_info, platform_content, channel_ids, posting_type, scheduled_datetime,
                metadata=metadata,
                platform_mapping=platform_mapping
            )
            return post_results
        except Exception as e:
            self.logger.error(f"Bulk posting failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _handle_individual_platform_scheduling(self, file_info, platform_content, enabled_channels, platform_mapping, metadata) -> Dict[str, any]:
        """
        Handle individual platform scheduling when platforms have different posting times
        
        Args:
            file_info: File information from upload
            platform_content: Platform-specific content
            enabled_channels: Dict of enabled platforms and their channel IDs
            platform_mapping: Channel ID to platform mapping
            metadata: Video metadata
            
        Returns:
            Dict with posting results
        """
        successful_posts = []
        failed_platforms = []
        
        # Group platforms by their posting times for efficient scheduling
        time_groups = self.config.group_platforms_by_posting_time()
        
        self.logger.info(f"🕒 Platform scheduling groups:")
        for posting_time, platforms in time_groups.items():
            self.logger.info(f"   {posting_time}: {platforms}")
        
        # Process each time group
        for posting_time, platforms in time_groups.items():
            try:
                # Get posting configuration for this time
                first_platform = platforms[0]  # Use first platform in group for configuration
                posting_time_info = self.config.parse_posting_time(first_platform)
                posting_type = self.config.get_posting_type_for_postiz(first_platform)
                scheduled_datetime = self.config.get_scheduled_datetime_iso(first_platform)
                
                # Get channel IDs for this time group
                group_channel_ids = [enabled_channels[platform] for platform in platforms if platform in enabled_channels]
                group_platform_mapping = {enabled_channels[platform]: platform for platform in platforms if platform in enabled_channels}
                
                self.logger.info(f"📅 Scheduling {len(platforms)} platforms for: {posting_time}")
                self.logger.info(f"   Platforms: {platforms}")
                self.logger.info(f"   Posting type: '{posting_type}'")
                self.logger.info(f"   Description: {posting_time_info['description']}")
                if posting_type == "date":
                    self.logger.info(f"   Scheduled for: {scheduled_datetime}")
                
                # Create post for this time group
                try:
                    group_results = self.client.create_post_with_fallback(
                        file_info, platform_content, group_channel_ids, posting_type, scheduled_datetime,
                        metadata=metadata,
                        platform_mapping=group_platform_mapping
                    )
                    
                    if group_results["success"]:
                        # Add successful platforms
                        if "successful_posts" in group_results:
                            successful_posts.extend(group_results["successful_posts"])
                        else:
                            # Fallback: assume all platforms in group succeeded
                            for platform in platforms:
                                if platform in enabled_channels:
                                    successful_posts.append({
                                        "channel_id": enabled_channels[platform],
                                        "platform": platform,
                                        "posting_time": posting_time
                                    })
                        
                        # Add any failed platforms from this group
                        if group_results.get("failed_platforms"):
                            failed_platforms.extend(group_results["failed_platforms"])
                        
                        self.logger.info(f"✅ Successfully scheduled {len(platforms)} platforms for {posting_time}")
                    else:
                        # Entire group failed
                        for platform in platforms:
                            if platform in enabled_channels:
                                failed_platforms.append({
                                    "channel_id": enabled_channels[platform],
                                    "platform": platform,
                                    "error": group_results.get("error", "Unknown error")
                                })
                        self.logger.error(f"❌ Failed to schedule platforms for {posting_time}: {group_results.get('error', 'Unknown error')}")
                    
                except Exception as e:
                    # Group failed with exception
                    for platform in platforms:
                        if platform in enabled_channels:
                            failed_platforms.append({
                                "channel_id": enabled_channels[platform],
                                "platform": platform,
                                "error": str(e)
                            })
                    self.logger.error(f"❌ Exception while scheduling platforms for {posting_time}: {str(e)}")
                
                # Add delay between time groups to respect rate limits
                if len(time_groups) > 1:
                    import time
                    self.logger.info("⏳ Waiting 2 seconds before next time group...")
                    time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"❌ Failed to process time group {posting_time}: {str(e)}")
                # Mark all platforms in this group as failed
                for platform in platforms:
                    if platform in enabled_channels:
                        failed_platforms.append({
                            "channel_id": enabled_channels[platform],
                            "platform": platform,
                            "error": f"Time group processing failed: {str(e)}"
                        })
        
        # Compile results
        total_platforms = len(enabled_channels)
        success_count = len(successful_posts)
        failure_count = len(failed_platforms)
        
        return {
            "success": success_count > 0,
            "individual_creation": True,
            "successful_posts": successful_posts,
            "failed_platforms": failed_platforms,
            "total_channels": total_platforms,
            "success_count": success_count,
            "failure_count": failure_count,
            "time_groups": len(time_groups)
        }
    
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