"""
Video file validator for social media publishing
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Video file information"""
    file_path: str
    file_size: int
    duration: float
    width: int
    height: int
    frame_rate: float
    format_name: str
    codec: str
    bitrate: int
    aspect_ratio: str
    
    def __post_init__(self):
        """Calculate additional properties after initialization"""
        if self.width and self.height:
            ratio = self.width / self.height
            if abs(ratio - 16/9) < 0.1:
                self.aspect_ratio = "16:9"
            elif abs(ratio - 9/16) < 0.1:
                self.aspect_ratio = "9:16"
            elif abs(ratio - 4/3) < 0.1:
                self.aspect_ratio = "4:3"
            elif abs(ratio - 1) < 0.1:
                self.aspect_ratio = "1:1"
            else:
                self.aspect_ratio = f"{self.width}:{self.height}"


@dataclass
class ValidationResult:
    """Video validation result"""
    is_valid: bool
    video_info: Optional[VideoInfo]
    errors: List[str]
    warnings: List[str]
    
    def add_error(self, message: str):
        """Add validation error"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add validation warning"""
        self.warnings.append(message)


class VideoValidator:
    """Validates video files for social media platform requirements"""
    
    # Platform requirements
    PLATFORM_REQUIREMENTS = {
        "twitter": {
            "max_duration": 140,  # seconds
            "max_size_mb": 512,
            "supported_formats": ["mp4", "mov"],
            "min_resolution": (32, 32),
            "max_resolution": (1920, 1200),
            "aspect_ratios": ["16:9", "9:16", "1:1", "4:3"]
        },
        "instagram": {
            "max_duration": 90,
            "max_size_mb": 1024,
            "supported_formats": ["mp4", "mov"],
            "min_resolution": (600, 315),
            "max_resolution": (1080, 1920),
            "aspect_ratios": ["9:16", "1:1", "4:5"]
        },
        "youtube": {
            "max_duration": 60,  # For shorts
            "max_size_mb": 2048,
            "supported_formats": ["mp4", "mov", "avi", "wmv", "flv", "webm"],
            "min_resolution": (640, 360),
            "max_resolution": (1920, 1080),
            "aspect_ratios": ["16:9", "9:16"]
        },
        "tiktok": {
            "max_duration": 60,
            "max_size_mb": 287,
            "supported_formats": ["mp4", "mov"],
            "min_resolution": (540, 960),
            "max_resolution": (1080, 1920),
            "aspect_ratios": ["9:16"]
        }
    }
    
    def validate_video(self, video_path: str, target_platforms: List[str] = None) -> ValidationResult:
        """
        Validate video file for social media publishing
        
        Args:
            video_path: Path to video file
            target_platforms: List of target platforms to validate against
            
        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(is_valid=True, video_info=None, errors=[], warnings=[])
        
        # Check file existence
        if not Path(video_path).exists():
            result.add_error(f"Video file not found: {video_path}")
            return result
        
        try:
            # Get video information
            video_info = self._get_video_info(video_path)
            result.video_info = video_info
            
            # Validate basic requirements
            self._validate_basic_requirements(video_info, result)
            
            # Validate platform-specific requirements
            if target_platforms:
                self._validate_platform_requirements(video_info, target_platforms, result)
            
            # Additional quality checks
            self._validate_quality_requirements(video_info, result)
            
        except Exception as e:
            result.add_error(f"Failed to analyze video: {str(e)}")
        
        return result
    
    def _get_video_info(self, video_path: str) -> VideoInfo:
        """
        Extract video information using ffprobe
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoInfo object with video details
        """
        import json
        
        try:
            # Use ffprobe to get video information
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break
            
            if not video_stream:
                raise ValueError("No video stream found in file")
            
            # Extract information
            format_info = data.get("format", {})
            file_size = int(format_info.get("size", 0))
            duration = float(format_info.get("duration", 0))
            bitrate = int(format_info.get("bit_rate", 0))
            
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            
            # Parse frame rate
            frame_rate_str = video_stream.get("r_frame_rate", "0/1")
            if "/" in frame_rate_str:
                num, den = frame_rate_str.split("/")
                frame_rate = float(num) / float(den) if float(den) != 0 else 0
            else:
                frame_rate = float(frame_rate_str)
            
            codec = video_stream.get("codec_name", "unknown")
            format_name = format_info.get("format_name", "unknown")
            
            return VideoInfo(
                file_path=video_path,
                file_size=file_size,
                duration=duration,
                width=width,
                height=height,
                frame_rate=frame_rate,
                format_name=format_name,
                codec=codec,
                bitrate=bitrate,
                aspect_ratio=""  # Will be calculated in __post_init__
            )
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Video analysis timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse video information: {e}")
        except Exception as e:
            raise RuntimeError(f"Video analysis error: {e}")
    
    def _validate_basic_requirements(self, video_info: VideoInfo, result: ValidationResult):
        """Validate basic video requirements"""
        
        # Check file size (basic limit: 100MB for safety)
        max_size_bytes = 1000 * 1024 * 1024  # 100MB
        if video_info.file_size > max_size_bytes:
            result.add_error(f"File too large: {video_info.file_size / 1024 / 1024:.1f}MB (max 1000MB)")
        
        # Check duration (basic limit: 5 minutes)
        if video_info.duration > 300:  # 5 minutes
            result.add_error(f"Video too long: {video_info.duration:.1f}s (max 300s)")
        elif video_info.duration < 1:
            result.add_error(f"Video too short: {video_info.duration:.1f}s (min 1s)")
        
        # Check resolution
        if video_info.width < 32 or video_info.height < 32:
            result.add_error(f"Resolution too low: {video_info.width}x{video_info.height}")
        
        # Check for zero duration/dimensions
        if video_info.duration <= 0:
            result.add_error("Invalid video duration")
        
        if video_info.width <= 0 or video_info.height <= 0:
            result.add_error("Invalid video dimensions")
    
    def _validate_platform_requirements(self, video_info: VideoInfo, platforms: List[str], result: ValidationResult):
        """Validate against platform-specific requirements"""
        
        for platform in platforms:
            platform_lower = platform.lower()
            if platform_lower not in self.PLATFORM_REQUIREMENTS:
                result.add_warning(f"Unknown platform requirements: {platform}")
                continue
            
            req = self.PLATFORM_REQUIREMENTS[platform_lower]
            
            # Duration check
            if video_info.duration > req["max_duration"]:
                result.add_error(f"{platform}: Video too long ({video_info.duration:.1f}s, max {req['max_duration']}s)")
            
            # File size check
            max_bytes = req["max_size_mb"] * 1024 * 1024
            if video_info.file_size > max_bytes:
                result.add_error(f"{platform}: File too large ({video_info.file_size / 1024 / 1024:.1f}MB, max {req['max_size_mb']}MB)")
            
            # Resolution check
            min_w, min_h = req["min_resolution"]
            max_w, max_h = req["max_resolution"]
            
            if video_info.width < min_w or video_info.height < min_h:
                result.add_error(f"{platform}: Resolution too low ({video_info.width}x{video_info.height}, min {min_w}x{min_h})")
            
            if video_info.width > max_w or video_info.height > max_h:
                result.add_error(f"{platform}: Resolution too high ({video_info.width}x{video_info.height}, max {max_w}x{max_h})")
            
            # Aspect ratio check
            if video_info.aspect_ratio not in req["aspect_ratios"]:
                result.add_warning(f"{platform}: Non-optimal aspect ratio ({video_info.aspect_ratio})")
            
            # Format check
            file_ext = Path(video_info.file_path).suffix.lower().lstrip('.')
            if file_ext not in req["supported_formats"]:
                result.add_error(f"{platform}: Unsupported format (.{file_ext})")
    
    def _validate_quality_requirements(self, video_info: VideoInfo, result: ValidationResult):
        """Validate video quality requirements"""
        
        # Frame rate checks
        if video_info.frame_rate < 15:
            result.add_warning(f"Low frame rate: {video_info.frame_rate:.1f} FPS (recommended: 24-60 FPS)")
        elif video_info.frame_rate > 120:
            result.add_warning(f"Very high frame rate: {video_info.frame_rate:.1f} FPS")
        
        # Bitrate checks (if available)
        if video_info.bitrate > 0:
            # Calculate bitrate per pixel
            pixels = video_info.width * video_info.height
            bitrate_per_pixel = video_info.bitrate / pixels if pixels > 0 else 0
            
            if bitrate_per_pixel < 0.1:
                result.add_warning("Low bitrate may affect video quality")
            elif bitrate_per_pixel > 5:
                result.add_warning("High bitrate may cause upload issues")
        
        # Common codec check
        preferred_codecs = ["h264", "hevc", "vp9", "av1"]
        if video_info.codec.lower() not in preferred_codecs:
            result.add_warning(f"Non-standard codec: {video_info.codec} (recommended: H.264)")
    
    def get_optimization_suggestions(self, video_info: VideoInfo, target_platforms: List[str]) -> List[str]:
        """
        Get optimization suggestions for better platform compatibility
        
        Args:
            video_info: Video information
            target_platforms: Target platforms
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Find most restrictive requirements across platforms
        min_duration = min(self.PLATFORM_REQUIREMENTS[p.lower()]["max_duration"] 
                          for p in target_platforms 
                          if p.lower() in self.PLATFORM_REQUIREMENTS)
        
        min_size = min(self.PLATFORM_REQUIREMENTS[p.lower()]["max_size_mb"] 
                      for p in target_platforms 
                      if p.lower() in self.PLATFORM_REQUIREMENTS)
        
        # Duration suggestions
        if video_info.duration > min_duration:
            suggestions.append(f"Trim video to under {min_duration}s for all platforms")
        
        # Size suggestions
        size_mb = video_info.file_size / 1024 / 1024
        if size_mb > min_size:
            suggestions.append(f"Reduce file size to under {min_size}MB")
            suggestions.append("Consider lowering bitrate or resolution")
        
        # Aspect ratio suggestions
        if video_info.aspect_ratio not in ["9:16", "16:9"]:
            suggestions.append("Consider 9:16 aspect ratio for mobile platforms")
        
        # Quality suggestions
        if video_info.frame_rate < 24:
            suggestions.append("Increase frame rate to 24+ FPS for smoother playback")
        
        return suggestions