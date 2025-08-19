"""
Publishing logger for tracking social media publishing activities
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class PublishingLogger:
    """Logger for social media publishing activities"""
    
    def __init__(self, video_name: str, log_dir: Optional[Path] = None):
        """
        Initialize logger for a specific video
        
        Args:
            video_name: Name of the video being published
            log_dir: Directory to store log files (optional)
        """
        self.video_name = video_name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(f"postiz_publisher_{video_name}")
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"publishing_{self.video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(f"[{self.video_name}] {message}")
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(f"[{self.video_name}] {message}")
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(f"[{self.video_name}] {message}")
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(f"[{self.video_name}] {message}")