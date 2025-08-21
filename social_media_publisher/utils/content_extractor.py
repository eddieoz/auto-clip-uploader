"""
Content extractor for processing video metadata and segment information
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VideoSegment:
    """Data class for video segment information"""
    index: int
    start_time: str
    end_time: str
    title: str
    description: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VideoSegment':
        """Create VideoSegment from dictionary"""
        return cls(
            index=data.get('index', 0),
            start_time=data.get('start_time', '00:00:00,000'),
            end_time=data.get('end_time', '00:00:00,000'),
            title=data.get('title', ''),
            description=data.get('description', '')
        )


@dataclass
class VideoMetadata:
    """Data class for complete video metadata"""
    title: str
    description: str
    segments: List[VideoSegment]
    hashtags: List[str]
    duration: Optional[float] = None
    
    def get_segment_zero(self) -> Optional[VideoSegment]:
        """Get segment 0 (first segment) as specified in requirements"""
        for segment in self.segments:
            if segment.index == 0:
                return segment
        return self.segments[0] if self.segments else None
    
    def get_primary_content(self) -> str:
        """Get primary content for social media using segment 0 as specified"""
        segment_zero = self.get_segment_zero()
        
        if segment_zero and segment_zero.description:
            # Use segment 0 description as primary content (as per requirements)
            return segment_zero.description
        elif self.description:
            # Fallback to main description
            return self.description
        elif self.title:
            # Last fallback to title
            return self.title
        else:
            return f"New video content"
    
    def get_youtube_title(self) -> str:
        """Get title for YouTube (from segment 0 or metadata)"""
        segment_zero = self.get_segment_zero()
        
        if segment_zero and segment_zero.title:
            return segment_zero.title
        elif self.title:
            return self.title
        else:
            return "Auto Generated Video"


class ContentExtractor:
    """Extracts and processes video content metadata"""
    
    def __init__(self, output_directory: Path):
        """
        Initialize content extractor for a video output directory
        
        Args:
            output_directory: Path to video output directory
        """
        self.output_dir = output_directory
        self.video_name = output_directory.name
    
    def extract_metadata(self) -> VideoMetadata:
        """
        Extract complete metadata from content.txt and other files
        
        Returns:
            VideoMetadata object with all extracted information
        """
        content_file = self.output_dir / "content.txt"
        
        if not content_file.exists():
            # Create fallback metadata from video name
            return self._create_fallback_metadata()
        
        try:
            content_text = content_file.read_text(encoding='utf-8')
            
            # First try to parse as JSON (new format)
            try:
                import json
                content_data = json.loads(content_text)
                return self._parse_json_content(content_data)
            except json.JSONDecodeError:
                # Fall back to text-based parsing (old format)
                print("Content.txt is not JSON, using text parsing...")
                return self._parse_content_file(content_text)
                
        except Exception as e:
            print(f"Warning: Failed to parse content.txt: {e}")
            return self._create_fallback_metadata()
    
    def _parse_json_content(self, content_data: dict) -> VideoMetadata:
        """
        Parse JSON content.txt file (new format)
        
        Args:
            content_data: Parsed JSON data from content.txt
            
        Returns:
            VideoMetadata object with proper unicode handling
        """
        # Extract metadata section
        metadata = content_data.get("metadata", {})
        title = self._decode_unicode(metadata.get("title", ""))
        description = self._decode_unicode(metadata.get("description", ""))
        
        # Extract segments
        segments = []
        segments_data = content_data.get("segments", [])
        
        for i, segment_data in enumerate(segments_data):
            segment = VideoSegment(
                index=i,
                start_time=segment_data.get("start_time", "00:00:00,000"),
                end_time=segment_data.get("end_time", "00:01:00,000"),
                title=self._decode_unicode(segment_data.get("title", f"Segment {i}")),
                description=self._decode_unicode(segment_data.get("description", ""))
            )
            segments.append(segment)
        
        # Extract hashtags from description and segments
        hashtags = []
        all_text = f"{title} {description}"
        for segment in segments:
            all_text += f" {segment.title} {segment.description}"
        
        hashtags = self._extract_hashtags(all_text)
        
        # If no title, use video name
        if not title:
            title = self.video_name.replace('_', ' ').title()
        
        return VideoMetadata(
            title=title,
            description=description,
            segments=segments,
            hashtags=hashtags
        )
    
    def _decode_unicode(self, text: str) -> str:
        """
        Decode unicode escape sequences to proper characters
        
        Args:
            text: Text with unicode escapes like \\ud83c\\udf10
            
        Returns:
            Properly decoded text with emojis and special characters
        """
        if not text:
            return ""
        
        try:
            # Handle unicode escape sequences
            # Convert \\ud83c\\udf10 format to proper unicode
            import re
            
            def decode_unicode_match(match):
                unicode_str = match.group(0)
                try:
                    # Convert \\uXXXX to actual unicode character
                    return unicode_str.encode().decode('unicode-escape')
                except:
                    return unicode_str
            
            # Find and replace all \\uXXXX patterns
            unicode_pattern = r'\\u[0-9a-fA-F]{4}'
            decoded_text = re.sub(unicode_pattern, decode_unicode_match, text)
            
            return decoded_text
            
        except Exception as e:
            print(f"Warning: Unicode decoding failed for '{text[:50]}...': {e}")
            return text
    
    def _parse_content_file(self, content_text: str) -> VideoMetadata:
        """
        Parse content.txt file to extract structured metadata
        
        Args:
            content_text: Raw content from content.txt
            
        Returns:
            VideoMetadata object
        """
        # Initialize default values
        title = ""
        description = ""
        segments = []
        hashtags = []
        
        # Extract title (look for "Title:" pattern)
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', content_text, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
        
        # Extract description (look for "Description:" pattern)  
        desc_match = re.search(r'Description:\s*(.+?)(?:\n\n|\nSegments?:|\nUsing|\n#|$)', content_text, re.IGNORECASE | re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()
        
        # Extract hashtags
        hashtags = self._extract_hashtags(content_text)
        
        # Extract segments information
        segments = self._extract_segments(content_text)
        
        # If no explicit title found, use first segment title or video name
        if not title and segments:
            title = segments[0].title or self.video_name.replace('_', ' ').title()
        elif not title:
            title = self.video_name.replace('_', ' ').title()
        
        # If no description found, use segment 0 description
        if not description and segments:
            segment_zero = next((s for s in segments if s.index == 0), segments[0])
            description = segment_zero.description
        
        return VideoMetadata(
            title=title,
            description=description,
            segments=segments,
            hashtags=hashtags
        )
    
    def _extract_segments(self, content_text: str) -> List[VideoSegment]:
        """
        Extract segment information from content text
        
        Args:
            content_text: Raw content text
            
        Returns:
            List of VideoSegment objects
        """
        segments = []
        
        # Try to parse JSON segment data first
        json_segments = self._extract_json_segments(content_text)
        if json_segments:
            return json_segments
        
        # Fall back to pattern-based extraction
        segment_patterns = [
            # Pattern: "0: Description (time range)"
            r'(\d+):\s*([^(]+?)(?:\s*\(([^)]+)\))?(?:\n|$)',
            # Pattern: "Segment 0: Description"
            r'Segment\s+(\d+):\s*(.+?)(?:\n|$)',
            # Pattern: "0 - Description"
            r'(\d+)\s*-\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in segment_patterns:
            matches = re.findall(pattern, content_text, re.IGNORECASE | re.MULTILINE)
            if matches:
                for match in matches:
                    index = int(match[0])
                    description = match[1].strip()
                    time_range = match[2] if len(match) > 2 else ""
                    
                    # Parse time range if available
                    start_time, end_time = self._parse_time_range(time_range)
                    
                    segments.append(VideoSegment(
                        index=index,
                        start_time=start_time,
                        end_time=end_time,
                        title=f"Segment {index}",
                        description=description
                    ))
                break
        
        # If no segments found, create a default segment 0
        if not segments:
            segments.append(VideoSegment(
                index=0,
                start_time="00:00:00,000",
                end_time="00:01:00,000", 
                title="Main Content",
                description=self._get_main_description(content_text)
            ))
        
        return segments
    
    def _extract_json_segments(self, content_text: str) -> List[VideoSegment]:
        """
        Try to extract segments from JSON data in content
        
        Args:
            content_text: Raw content text
            
        Returns:
            List of VideoSegment objects or empty list
        """
        # Look for JSON-like segment data
        json_pattern = r'\{[^{}]*"segments"[^{}]*\[[^\]]*\][^{}]*\}'
        json_match = re.search(json_pattern, content_text, re.DOTALL)
        
        if json_match:
            try:
                json_data = json.loads(json_match.group(0))
                segments = []
                
                for i, segment_data in enumerate(json_data.get("segments", [])):
                    segments.append(VideoSegment(
                        index=i,
                        start_time=segment_data.get("start_time", "00:00:00,000"),
                        end_time=segment_data.get("end_time", "00:01:00,000"),
                        title=segment_data.get("title", f"Segment {i}"),
                        description=segment_data.get("description", "")
                    ))
                
                return segments
            except json.JSONDecodeError:
                pass
        
        return []
    
    def _parse_time_range(self, time_range: str) -> Tuple[str, str]:
        """
        Parse time range string into start and end times
        
        Args:
            time_range: Time range string like "0:00-0:15" or "0:00:00,000-0:00:15,000"
            
        Returns:
            Tuple of (start_time, end_time) in SRT format
        """
        if not time_range:
            return "00:00:00,000", "00:01:00,000"
        
        # Try different time range formats
        patterns = [
            r'(\d+:\d+:\d+,\d+)-(\d+:\d+:\d+,\d+)',  # SRT format
            r'(\d+:\d+:\d+)-(\d+:\d+:\d+)',          # HH:MM:SS format
            r'(\d+:\d+)-(\d+:\d+)',                  # MM:SS format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, time_range)
            if match:
                start, end = match.groups()
                return self._normalize_time(start), self._normalize_time(end)
        
        return "00:00:00,000", "00:01:00,000"
    
    def _normalize_time(self, time_str: str) -> str:
        """
        Normalize time string to SRT format (HH:MM:SS,mmm)
        
        Args:
            time_str: Time string in various formats
            
        Returns:
            Time string in SRT format
        """
        # If already in SRT format, return as is
        if re.match(r'\d+:\d+:\d+,\d+', time_str):
            return time_str
        
        # Convert MM:SS to HH:MM:SS,000
        if re.match(r'\d+:\d+$', time_str):
            return f"00:{time_str},000"
        
        # Convert HH:MM:SS to HH:MM:SS,000
        if re.match(r'\d+:\d+:\d+$', time_str):
            return f"{time_str},000"
        
        return time_str
    
    def _extract_hashtags(self, content_text: str) -> List[str]:
        """
        Extract hashtags from content text
        
        Args:
            content_text: Raw content text
            
        Returns:
            List of hashtags (including #)
        """
        hashtag_pattern = r'#\w+'
        hashtags = re.findall(hashtag_pattern, content_text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_hashtags = []
        for tag in hashtags:
            if tag.lower() not in seen:
                seen.add(tag.lower())
                unique_hashtags.append(tag)
        
        return unique_hashtags
    
    def _get_main_description(self, content_text: str) -> str:
        """
        Extract main description from content text
        
        Args:
            content_text: Raw content text
            
        Returns:
            Main description string
        """
        # Remove hashtags and segment markers for clean description
        clean_text = re.sub(r'#\w+', '', content_text)
        clean_text = re.sub(r'Segment\s*\d+:', '', clean_text)
        clean_text = re.sub(r'\d+:', '', clean_text)
        clean_text = re.sub(r'Title:', '', clean_text)
        clean_text = re.sub(r'Description:', '', clean_text)
        
        # Get first substantial line
        lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
        if lines:
            return lines[0][:200]  # Limit length
        
        return f"Content from {self.video_name}"
    
    def _create_fallback_metadata(self) -> VideoMetadata:
        """
        Create fallback metadata when content.txt is missing or invalid
        
        Returns:
            Basic VideoMetadata object
        """
        title = self.video_name.replace('_', ' ').title()
        description = f"New video: {title}"
        
        # Create a default segment 0
        segment_zero = VideoSegment(
            index=0,
            start_time="00:00:00,000",
            end_time="00:01:00,000",
            title=title,
            description=description
        )
        
        # Add default hashtags
        # hashtags = ["#viral", "#shorts", "#content", "#video"]
        hashtags = ["#MorningCrypto"]
        
        return VideoMetadata(
            title=title,
            description=description,
            segments=[segment_zero],
            hashtags=hashtags
        )