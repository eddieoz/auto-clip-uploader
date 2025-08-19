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
            video_path, content, metadata = self._prepare_content()
            
            # Step 3: Upload video to Postiz
            self.logger.info("Step 3: Uploading video to Postiz")
            try:
                file_info = self.client.upload_video(str(video_path))
                self.logger.info(f"✅ Video uploaded successfully: {file_info.get('id', 'unknown')}")
            except Exception as e:
                self.logger.error(f"Video upload failed: {str(e)}")
                return {"error": "upload_failed", "message": str(e)}
            
            # Step 4: Create multi-platform post
            self.logger.info("Step 4: Creating multi-platform post")
            enabled_channels = self.config.get_enabled_channels()
            channel_ids = list(enabled_channels.values())
            
            self.logger.info(f"Target platforms: {list(enabled_channels.keys())}")
            self.logger.info(f"Channel IDs: {channel_ids}")
            
            # Get posting time configuration
            posting_time_info = self.config.parse_posting_time()
            posting_type = self.config.get_posting_type_for_postiz()
            scheduled_datetime = self.config.get_scheduled_datetime_iso()
            
            self.logger.info(f"Posting schedule: {posting_time_info['description']}")
            if posting_type == "date":
                self.logger.info(f"Scheduled for: {scheduled_datetime}")
            
            try:
                post_results = self.client.create_post_with_fallback(
                    file_info, content, channel_ids, posting_type, scheduled_datetime,
                    metadata=metadata  # Pass metadata for platform-specific handling
                )
                
                if post_results["success"]:
                    success_count = post_results.get("success_count", len(channel_ids))
                    total_count = post_results.get("total_channels", len(channel_ids))
                    
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
                    
            except Exception as e:
                self.logger.error(f"Post creation failed: {str(e)}")
                return {"error": "post_failed", "message": str(e)}
            
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
        
        # Use unified content that works across all platforms
        formatted_content = formatter.get_unified_content(enabled_platforms)
        
        self.logger.info(f"Video file: {video_path}")
        self.logger.info(f"Formatted content length: {len(formatted_content)} characters")
        self.logger.info(f"Content preview: {formatted_content[:100]}...")
        
        return video_path, formatted_content, metadata
    
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