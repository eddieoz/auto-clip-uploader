"""
Postiz API Client for social media publishing
"""

import requests
import time
from typing import Dict, Optional, List
from pathlib import Path


class PostizAPIError(Exception):
    """Custom exception for Postiz API errors"""
    
    def __init__(self, message: str, recoverable: bool = False):
        super().__init__(message)
        self.recoverable = recoverable  # Whether the error might be temporary and worth retrying


class PostizRateLimitError(PostizAPIError):
    """Exception for rate limit exceeded"""
    
    def __init__(self, message: str):
        super().__init__(message, recoverable=True)  # Rate limits are always recoverable


class PostizClient:
    """
    Client for interacting with Postiz API
    Handles authentication, rate limiting, and API calls
    """
    
    def __init__(self, api_key: str, endpoint: str, timeout: int = 30, mock_mode: bool = False):
        """
        Initialize Postiz API client
        
        Args:
            api_key: Postiz API key for authentication
            endpoint: Postiz API endpoint URL
            timeout: Request timeout in seconds
            mock_mode: Enable mock mode for testing (bypasses actual API calls)
        """
        self.api_key = api_key
        self.endpoint = endpoint.rstrip('/')  # Remove trailing slash
        self.timeout = timeout
        self.mock_mode = mock_mode
        
        # Setup session with authentication
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": self.api_key,  # Direct API key (not Bearer format)
            "User-Agent": "AutoClipUploader/1.0"
            # Note: NOT setting Content-Type here to allow proper multipart handling
        })
        
        # Enhanced rate limiting tracking
        self.last_request_time = 0
        self.min_request_interval = 1  # 30 requests per hour = 120 seconds between requests
        self.request_history = []  # Track request timestamps for more accurate rate limiting
        self.max_requests_per_hour = 500
        self.retry_after = None  # Server-suggested retry delay
    
    def _check_rate_limit(self):
        """Enhanced rate limiting to respect 30 requests per hour limit"""
        current_time = time.time()
        
        # Check server-suggested retry delay first
        if self.retry_after and current_time < self.retry_after:
            wait_time = self.retry_after - current_time
            raise PostizRateLimitError(f"Rate limit: server requested wait {wait_time:.1f} seconds")
        
        # Clean old requests from history (older than 1 hour)
        hour_ago = current_time - 3600
        self.request_history = [t for t in self.request_history if t > hour_ago]
        
        # Check if we've hit the hourly limit
        if len(self.request_history) >= self.max_requests_per_hour:
            oldest_request = min(self.request_history)
            wait_time = 3600 - (current_time - oldest_request)
            raise PostizRateLimitError(
                f"Rate limit: {self.max_requests_per_hour} requests per hour exceeded. "
                f"Wait {wait_time:.0f} seconds for oldest request to expire."
            )
        
        # Check minimum interval between requests (backup safety)
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last_request
            raise PostizRateLimitError(f"Rate limit: minimum {self.min_request_interval}s between requests")
        
        # Record this request
        self.request_history.append(current_time)
        self.last_request_time = current_time
    
    def _make_request(self, method: str, path: str, timeout: Optional[int] = None, **kwargs) -> requests.Response:
        """
        Make HTTP request with error handling and rate limiting
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., '/upload')
            **kwargs: Additional request arguments
            
        Returns:
            Response object
            
        Raises:
            PostizAPIError: For API errors
            PostizRateLimitError: For rate limit issues
        """
        # Check rate limiting
        self._check_rate_limit()
        
        url = f"{self.endpoint}{path}"
        
        try:
            print(f"🌐 {method} {url} - Making API request")
            
            # Debug: Log request details for upload requests
            if path == "/public/v1/upload":
                print(f"🔍 Upload request debug:")
                print(f"   URL: {url}")
                print(f"   Headers (excluding file data): {[k for k in kwargs.get('headers', {}).keys()]}")
                print(f"   Files: {[k for k in kwargs.get('files', {}).keys()]}")
                if 'files' in kwargs and 'file' in kwargs['files']:
                    file_info = kwargs['files']['file']
                    print(f"   File name: {file_info[0] if isinstance(file_info, tuple) else 'unknown'}")
                    print(f"   MIME type: {file_info[2] if isinstance(file_info, tuple) and len(file_info) > 2 else 'unknown'}")
            
            response = self.session.request(
                method=method,
                url=url,
                timeout=timeout or self.timeout,
                **kwargs
            )
            
            # Handle rate limiting response with server retry-after header
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    self.retry_after = time.time() + int(retry_after)
                    print(f"⏰ Rate limited - server requested {retry_after}s delay")
                
                error_msg = f"API rate limit exceeded (429)"
                if retry_after:
                    error_msg += f" - retry after {retry_after}s"
                raise PostizRateLimitError(error_msg)
            
            # Handle different categories of HTTP errors with detailed logging
            if response.status_code >= 500:
                # Server errors (5xx) - likely temporary, good for retry
                error_msg = f"Postiz server error {response.status_code}: {response.text[:200]}"
                print(f"🚨 Server error {response.status_code} - may be temporary")
                raise PostizAPIError(error_msg, recoverable=True)
                
            elif response.status_code >= 400:
                # Client errors (4xx) - usually permanent, log details
                error_msg = f"Postiz client error {response.status_code}: {response.text[:200]}"
                
                # Special handling for common client errors
                if response.status_code == 401:
                    print(f"🔑 Authentication failed - check API key")
                elif response.status_code == 403:
                    print(f"🚫 Access forbidden - check permissions")
                elif response.status_code == 404:
                    print(f"🔍 Endpoint not found: {path}")
                elif response.status_code == 413:
                    print(f"📦 Payload too large - file size exceeded backend limit")
                    print(f"💡 Postiz backend needs configuration update:")
                    print(f"   Add to docker environment: UPLOAD_MAX_FILE_SIZE=100MB")
                    print(f"   Or modify Express.js body parser limits in backend code")
                else:
                    print(f"❌ Client error {response.status_code}")
                
                raise PostizAPIError(error_msg, recoverable=False)
            
            # Success - log and return
            print(f"✅ {method} {path} - Success ({response.status_code})")
            return response
            
        except requests.exceptions.Timeout:
            raise PostizAPIError(f"Request timeout after {self.timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise PostizAPIError(f"Connection error to {url}")
        except requests.exceptions.RequestException as e:
            print(f"🔌 Network error: {str(e)}")
            raise PostizAPIError(f"Request failed: {str(e)}", recoverable=True)
    
    def _make_request_with_retry(self, method: str, path: str, max_retries: int = 3, 
                                base_delay: float = 1.0, **kwargs) -> requests.Response:
        """
        Make HTTP request with exponential backoff retry logic
        
        Args:
            method: HTTP method
            path: API path
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            **kwargs: Additional request arguments
            
        Returns:
            Response object
            
        Raises:
            PostizAPIError: When all retries are exhausted
            PostizRateLimitError: When rate limited (no retry attempted)
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return self._make_request(method, path, **kwargs)
                
            except PostizRateLimitError:
                # Don't retry rate limit errors - they need to wait
                print(f"⏰ Rate limit hit on attempt {attempt + 1} - not retrying")
                raise
                
            except PostizAPIError as e:
                last_exception = e
                
                # Don't retry non-recoverable errors
                if not e.recoverable:
                    print(f"❌ Non-recoverable error on attempt {attempt + 1} - not retrying: {e}")
                    raise
                
                # Calculate backoff delay for recoverable errors
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s, 8s...
                    print(f"🔄 Recoverable error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                    print(f"⏳ Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"💥 All {max_retries + 1} attempts failed")
                    break
        
        # All retries exhausted
        raise last_exception or PostizAPIError("Request failed after all retry attempts")
    
    def validate_connection(self) -> Dict[str, str]:
        """
        Validate API connectivity and authentication with retry logic
        
        Returns:
            Dict with connection status and details
        """
        try:
            # Try a lightweight request to validate connection
            # Use /public/v1/integrations endpoint (confirmed in Postiz logs)
            print("🔍 Validating API connection...")
            response = self._make_request_with_retry("GET", "/public/v1/integrations", max_retries=2)
            
            print(f"✅ Connection validation successful")
            return {
                "status": "success", 
                "message": "Connection successful",
                "endpoint": self.endpoint,
                "response_code": response.status_code,
                "rate_limit_remaining": self.max_requests_per_hour - len(self.request_history)
            }
            
        except PostizRateLimitError as e:
            print(f"⏰ Connection validation rate limited: {e}")
            return {
                "status": "rate_limited",
                "message": str(e),
                "endpoint": self.endpoint,
                "rate_limit_remaining": 0
            }
        except PostizAPIError as e:
            print(f"❌ Connection validation failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "endpoint": self.endpoint,
                "recoverable": getattr(e, 'recoverable', False)
            }
    
    def upload_video(self, video_path: str, validate: bool = True, max_retries: int = 1) -> Dict[str, str]:
        """Mock-aware video upload method"""
        if self.mock_mode:
            return self._mock_upload_video(video_path)
        
        return self._real_upload_video(video_path, validate, max_retries)
    
    def _mock_upload_video(self, video_path: str) -> Dict[str, str]:
        """Mock video upload for testing"""
        video_file_path = Path(video_path)
        if not video_file_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        file_size_mb = video_file_path.stat().st_size / (1024 * 1024)
        mock_id = f"mock_file_{hash(video_path) % 10000}"
        
        print(f"🎭 MOCK MODE: Simulating upload of {video_file_path.name} ({file_size_mb:.1f}MB)")
        print(f"✅ Mock upload successful: file_id={mock_id}")
        
        return {
            "id": mock_id,
            "path": f"/mock/uploads/{video_file_path.name}"
        }
    
    def _real_upload_video(self, video_path: str, validate: bool = True, max_retries: int = 1) -> Dict[str, str]:
        """
        Upload video file to Postiz with validation and retry logic
        
        Args:
            video_path: Path to video file
            validate: Whether to validate video before upload
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dict with file ID and path from Postiz response
            
        Raises:
            PostizAPIError: For upload failures
            FileNotFoundError: If video file doesn't exist
            ValueError: If video validation fails
        """
        video_file_path = Path(video_path)
        if not video_file_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check file size before attempting upload
        file_size_bytes = video_file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Common API file size limits to warn about
        if file_size_mb > 1000:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB exceeds typical API limits (50MB)")
        elif file_size_mb > 50:
            print(f"⚠️  Large file: {file_size_mb:.1f}MB - may hit API limits")
        elif file_size_mb > 10:
            print(f"📁 Medium file: {file_size_mb:.1f}MB")
        
        # Validate video file if requested
        if validate:
            from ..utils.validator import VideoValidator
            
            validator = VideoValidator()
            validation_result = validator.validate_video(video_path)
            
            if not validation_result.is_valid:
                error_msg = f"Video validation failed: {'; '.join(validation_result.errors)}"
                raise ValueError(error_msg)
            
            if validation_result.warnings:
                print(f"Video validation warnings: {'; '.join(validation_result.warnings)}")
        
        # Attempt upload with retry logic
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return self._attempt_upload(video_file_path, attempt)
                
            except PostizRateLimitError:
                # Don't retry rate limit errors
                raise
            except PostizAPIError as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"Upload attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"Upload failed after {max_retries + 1} attempts")
                    break
            except Exception as e:
                # Don't retry other exceptions
                raise PostizAPIError(f"Video upload failed: {str(e)}")
        
        # If we get here, all retries failed
        raise last_exception or PostizAPIError("Upload failed after all retry attempts")
    
    def _attempt_upload(self, video_file_path: Path, attempt: int) -> Dict[str, str]:
        """
        Single upload attempt
        
        Args:
            video_file_path: Path to video file
            attempt: Attempt number (for logging)
            
        Returns:
            Upload result dictionary
            
        Raises:
            PostizAPIError: For upload failures
        """
        try:
            # Log upload attempt with detailed file info
            file_size_bytes = video_file_path.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            print(f"Upload attempt {attempt + 1}: {video_file_path.name}")
            print(f"  File size: {file_size_mb:.1f}MB ({file_size_bytes:,} bytes)")
            
            # Check if file size might be too large for API
            if file_size_mb > 25:  # Many APIs have 25MB limit
                print(f"  ⚠️  File size ({file_size_mb:.1f}MB) may exceed API limits")
            
            # For multipart uploads, we don't set Content-Type header
            # (requests will handle it automatically)
            headers = dict(self.session.headers)
            
            # Determine MIME type based on file extension
            file_extension = video_file_path.suffix.lower()
            mime_type = self._get_video_mime_type(file_extension)
            print(f"  File extension: {file_extension}")
            print(f"  MIME type: {mime_type}")
            print(f"  Should trigger video validation (1GB limit): {mime_type.startswith('video/')}")
            
            with open(video_file_path, 'rb') as video_file:
                files = {'file': (video_file_path.name, video_file, mime_type)}
                
                # Make upload request with longer timeout for large files
                upload_timeout = max(120, file_size_mb * 10)  # 10s per MB, min 2min
                
                response = self._make_request(
                    "POST", 
                    "/public/v1/upload",
                    files=files,
                    headers=headers,
                    timeout=upload_timeout
                )
            
            # Debug response before parsing
            print(f"📋 Upload response status: {response.status_code}")
            print(f"📋 Upload response headers: {dict(response.headers)}")
            print(f"📋 Upload response text: '{response.text[:200]}...'")
            
            try:
                upload_result = response.json()
            except ValueError as e:
                raise PostizAPIError(f"Invalid JSON response from upload: '{response.text[:200]}...' - {str(e)}")
            
            # Validate response structure
            if 'id' not in upload_result or 'path' not in upload_result:
                raise PostizAPIError(f"Invalid upload response format - missing id or path. Got: {upload_result}")
            
            print(f"✅ Upload successful: file_id={upload_result['id']}")
            return upload_result
            
        except PostizAPIError:
            raise
        except Exception as e:
            raise PostizAPIError(f"Upload attempt failed: {str(e)}")
    
    def _get_video_mime_type(self, file_extension: str) -> str:
        """
        Get appropriate MIME type for video file
        
        Args:
            file_extension: File extension (with dot)
            
        Returns:
            MIME type string
        """
        mime_types = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.flv': 'video/x-flv',
            '.wmv': 'video/x-ms-wmv',
            '.m4v': 'video/x-m4v'
        }
        
        return mime_types.get(file_extension, 'video/mp4')  # Default to mp4
    
    def _get_platform_settings(self, platform: Optional[str]) -> Dict[str, any]:
        """
        Get platform-specific settings for post creation
        
        Args:
            platform: Platform name (e.g., 'tiktok', 'instagram', 'youtube', 'twitter')
            
        Returns:
            Dict with platform-specific settings
        """
        # Base settings that work for most platforms
        base_settings = {
            "post_type": "post",  # Required: "post" or "story"
            "type": "public",     # Required for YouTube: "public", "private", "unlisted"
        }
        
        # Platform-specific settings
        if platform == "tiktok":
            # TikTok has many specific requirements
            return {
                **base_settings,
                "privacy_level": "PUBLIC_TO_EVERYONE",  # TikTok specific values
                "duet": True,         # TikTok: allow duets
                "comment": True,      # TikTok: allow comments
                "stitch": True,       # TikTok: allow stitching
                "autoAddMusic": "no", # TikTok: "yes" or "no"
                "brand_content_toggle": False,     # TikTok: branded content
                "brand_organic_toggle": False,     # TikTok: organic content
                "disclosure_enabled": False,       # TikTok: disclosure settings
                "content_posting_method": "UPLOAD" # TikTok: "DIRECT_POST" or "UPLOAD"
            }
        elif platform == "instagram":
            # Instagram uses minimal settings - avoid TikTok-specific parameters
            # Instagram doesn't need title or many other settings
            return {
                "post_type": "post",  # Only essential settings for Instagram
                # Remove "type": "public" as Instagram may not expect it
                # Avoid any additional settings that could cause rejection
            }
        elif platform == "youtube":
            # YouTube has its own requirements
            return {
                **base_settings,
                "type": "public",  # YouTube: "public", "private", "unlisted"
                # YouTube may need additional settings in the future
            }
        elif platform == "twitter":
            # Twitter/X has minimal requirements
            return {
                **base_settings,
                # Twitter typically only needs basic settings
            }
        else:
            # Unknown/generic platform - use minimal safe settings
            return base_settings
    
    def create_post(self, file_info: Dict[str, str], content, channel_ids: List[str], 
                   posting_type: str = "now", scheduled_datetime: Optional[str] = None, 
                   metadata=None, platform_mapping: Optional[Dict[str, str]] = None) -> Dict[str, any]:
        """
        Create multi-platform post with uploaded video (mock-aware)
        
        Args:
            file_info: File information from upload (id, path)
            content: Post content/caption
            channel_ids: List of channel IDs to post to
            posting_type: "now" for immediate posting or "date" for scheduled posting
            scheduled_datetime: ISO datetime string for scheduled posts
            platform_mapping: Optional dict mapping channel_id -> platform_name for platform-specific settings
            
        Returns:
            Dict with post creation results including URLs and IDs
            
        Raises:
            PostizAPIError: For post creation failures
        """
        if self.mock_mode:
            return self._mock_create_post(file_info, content, channel_ids, posting_type, scheduled_datetime)
        
        return self._real_create_post(file_info, content, channel_ids, posting_type, scheduled_datetime, metadata, platform_mapping)
    
    def _mock_create_post(self, file_info: Dict[str, str], content, channel_ids: List[str], 
                         posting_type: str, scheduled_datetime: Optional[str]) -> Dict[str, any]:
        """Mock post creation for testing"""
        if posting_type == "now":
            print(f"🎭 MOCK MODE: Creating immediate multi-platform post for {len(channel_ids)} channels")
        else:
            print(f"🎭 MOCK MODE: Scheduling multi-platform post for {len(channel_ids)} channels at {scheduled_datetime}")
        
        # Simulate successful post creation
        mock_posts = []
        platforms = ["twitter", "instagram", "youtube", "tiktok"]
        
        for i, channel_id in enumerate(channel_ids):
            platform = platforms[i % len(platforms)]  # Cycle through platforms
            mock_post = {
                "id": f"mock_post_{hash(channel_id) % 10000}",
                "platform": platform,
                "url": f"https://{platform}.com/mock/post/{hash(content) % 10000}",
                "status": "scheduled" if posting_type == "date" else "published"
            }
            mock_posts.append(mock_post)
            print(f"  ✅ Mock post created for {platform}: {mock_post['url']}")
        
        return {
            "success": True,
            "posts": mock_posts,
            "message": "Mock post creation successful"
        }
    
    def _real_create_post(self, file_info: Dict[str, str], content, channel_ids: List[str], 
                         posting_type: str, scheduled_datetime: Optional[str], metadata=None, 
                         platform_mapping: Optional[Dict[str, str]] = None) -> Dict[str, any]:
        """Real post creation implementation"""
        if posting_type == "now":
            print(f"Creating immediate multi-platform post for {len(channel_ids)} channels")
        else:
            print(f"Scheduling multi-platform post for {len(channel_ids)} channels at {scheduled_datetime}")
        
        # Handle both string content (backward compatibility) and dict content (platform-specific)
        if isinstance(content, dict):
            platform_content = content
        else:
            # Backward compatibility: use same content for all platforms
            platform_content = {platform: content for platform in (platform_mapping.values() if platform_mapping else [])}
            # Add fallback content for unmapped channels
            platform_content["default"] = content
        
        # Build platform posts for each channel with platform-specific settings
        posts = []
        for channel_id in channel_ids:
            # Get platform name for this channel (fallback to generic if not provided)
            platform = platform_mapping.get(channel_id) if platform_mapping else None
            
            # Get platform-specific content
            post_content = platform_content.get(platform, platform_content.get("default", list(platform_content.values())[0] if platform_content else ""))
            
            # Get platform-specific settings
            settings = self._get_platform_settings(platform)
            
            # Add platform-specific settings
            # Import PLATFORM_LIMITS to respect platform-specific character limits
            from ..utils.content_formatter import SocialMediaFormatter
            
            # Get platform-specific limits
            platform_limits = SocialMediaFormatter.PLATFORM_LIMITS.get(platform, SocialMediaFormatter.PLATFORM_LIMITS["twitter"])
            
            # For title extraction, use platform-appropriate limits
            if platform in ["twitter"]:
                # Twitter needs shorter title due to character limits
                title_limit = 47
            elif platform in ["youtube"]:
                # YouTube can handle longer titles
                title_limit = 100
            else:
                # For Instagram, TikTok, etc., use more generous limits since they don't strictly need titles
                title_limit = min(200, platform_limits.max_length // 4)
            
            # Extract title based on platform limits
            if len(post_content) > title_limit:
                title = post_content[:title_limit-3] + "..."
            else:
                title = post_content
                
            # Use metadata for YouTube title if available
            if metadata and hasattr(metadata, 'get_youtube_title'):
                youtube_title = metadata.get_youtube_title()
                # Clean title - remove emojis and newlines for title
                import re
                clean_title = re.sub(r'[^\w\s-]', '', youtube_title).strip()
            else:
                # Fallback: use content for title
                import re
                clean_title = re.sub(r'[^\w\s-]', '', title).strip()
                
            if not clean_title or len(clean_title) < 2:
                clean_title = "Auto Generated Video"
            
            # Only add title for platforms that need it (YouTube, not Instagram)
            if platform in ["youtube", "twitter"] or platform is None:
                settings["title"] = clean_title[:100]  # Max 100 chars
            
            post_data = {
                "integration": {"id": channel_id},
                "value": [{
                    "content": post_content,
                    "image": [{
                        "id": file_info["id"],
                        "path": file_info["path"]
                    }]
                }],
                "settings": settings
            }
            
            # Debug: Log the request payload for Instagram to identify differences
            if platform == "instagram":
                print(f"📱 Instagram API payload:")
                print(f"   Channel ID: {channel_id}")
                print(f"   Channel ID type: {type(channel_id)}")
                print(f"   Channel ID length: {len(str(channel_id))}")
                print(f"   Settings: {settings}")
                print(f"   File info: {file_info}")
                
                # Validate channel ID format
                if not channel_id or not str(channel_id).strip():
                    print(f"⚠️  WARNING: Invalid Instagram channel ID: '{channel_id}'")
                elif len(str(channel_id).strip()) < 10:
                    print(f"⚠️  WARNING: Instagram channel ID seems too short: '{channel_id}'")
                
            posts.append(post_data)
        
        # Fix posting type - API expects "schedule" not "date"
        api_posting_type = "schedule" if posting_type == "date" else posting_type
        
        # Debug: Log the posting type transformation
        print(f"🕒 Posting type transformation:")
        print(f"   Input posting_type: '{posting_type}'")
        print(f"   Final api_posting_type: '{api_posting_type}'")
        
        # Build payload with all required fields
        payload = {
            "type": api_posting_type,
            "shortLink": False,  # Required boolean field
            "tags": [],  # Required array field
            "posts": posts
        }
        
        # Add date field - required by Postiz API for all posts
        if posting_type == "date" and scheduled_datetime:
            payload["date"] = scheduled_datetime
        else:
            # For immediate posts, use current datetime
            from datetime import datetime
            payload["date"] = datetime.now().isoformat()
        
        try:
            print(f"Sending post creation request with payload: {len(posts)} posts")
            # Add Content-Type header specifically for JSON requests
            json_headers = {"Content-Type": "application/json"}
            response = self._make_request("POST", "/public/v1/posts", json=payload, headers=json_headers)
            result = response.json()
            
            print(f"✅ Multi-platform post created successfully")
            
            # Handle case where result might be a list or dict
            if isinstance(result, list):
                print(f"   Created {len(result)} platform posts")
                return {
                    "success": True,
                    "posts": result,
                    "message": f"Created {len(result)} posts successfully"
                }
            elif isinstance(result, dict) and "posts" in result:
                print(f"   Created {len(result['posts'])} platform posts")
                return result
            else:
                return {
                    "success": True,
                    "posts": result if isinstance(result, list) else [result],
                    "message": "Post creation successful"
                }
            
        except PostizAPIError as e:
            # Enhanced error handling for Instagram-specific issues
            error_msg = str(e)
            if "instagram" in error_msg.lower() or "GraphMethodException" in error_msg:
                print(f"🔍 Instagram API Error Details:")
                print(f"   Error: {error_msg}")
                if platform_mapping:
                    instagram_channels = [cid for cid, plat in platform_mapping.items() if plat == "instagram"]
                    print(f"   Instagram channel IDs: {instagram_channels}")
                print(f"   Payload that failed: {payload}")
            raise
        except Exception as e:
            raise PostizAPIError(f"Post creation failed: {str(e)}")
    
    def create_post_with_fallback(self, file_info: Dict[str, str], content, channel_ids: List[str],
                                 posting_type: str = "now", scheduled_datetime: Optional[str] = None, 
                                 metadata=None, platform_mapping: Optional[Dict[str, str]] = None) -> Dict[str, any]:
        """
        Create multi-platform post with individual platform fallback handling
        
        Args:
            file_info: File information from upload (id, path)  
            content: Post content/caption
            channel_ids: List of channel IDs to post to
            posting_type: "now" for immediate posting or "date" for scheduled posting
            scheduled_datetime: ISO datetime string for scheduled posts
            platform_mapping: Optional dict mapping channel_id -> platform_name for platform-specific settings
            
        Returns:
            Dict with results including successful and failed platforms
        """
        successful_posts = []
        failed_platforms = []
        
        # Try bulk creation first
        try:
            result = self.create_post(file_info, content, channel_ids, posting_type, scheduled_datetime, metadata, platform_mapping)
            return {
                "success": True,
                "bulk_creation": True,
                "result": result,
                "successful_posts": channel_ids,
                "failed_platforms": [],
                "total_channels": len(channel_ids),
                "success_count": len(channel_ids),
                "failure_count": 0
            }
        except PostizAPIError as e:
            print(f"⚠️  Bulk post creation failed: {e}")
            print("🔄 Attempting individual platform posting...")
        
        # Fallback: Try each platform individually with rate limiting
        import time
        for i, channel_id in enumerate(channel_ids):
            try:
                # Add delay between individual requests to respect rate limits
                if i > 0:
                    print(f"⏳ Waiting 2 seconds before next platform...")
                    time.sleep(2)
                
                single_result = self.create_post(file_info, content, [channel_id], posting_type, scheduled_datetime, metadata, platform_mapping)
                successful_posts.append({
                    "channel_id": channel_id,
                    "result": single_result
                })
                print(f"✅ Successfully posted to channel: {channel_id}")
                
            except PostizAPIError as e:
                failed_platforms.append({
                    "channel_id": channel_id,
                    "error": str(e)
                })
                print(f"❌ Failed to post to channel {channel_id}: {e}")
        
        return {
            "success": len(successful_posts) > 0,
            "bulk_creation": False,
            "successful_posts": successful_posts,
            "failed_platforms": failed_platforms,
            "total_channels": len(channel_ids),
            "success_count": len(successful_posts),
            "failure_count": len(failed_platforms)
        }
    
    def get_channels(self) -> List[Dict[str, str]]:
        """
        Get available channels/integrations
        
        Returns:
            List of available channels
        """
        try:
            # For GET requests, we don't need Content-Type header
            response = self._make_request("GET", "/public/v1/integrations")
            return response.json()
        except PostizAPIError:
            raise
        except Exception as e:
            raise PostizAPIError(f"Failed to get channels: {str(e)}")