"""
Social Media Publisher for Viral Video Clips via Postiz API

Automated social media publishing system that integrates with monitor.py
to publish processed video clips across multiple platforms using Postiz API.
"""

from .publisher import PostizPublisher

__all__ = ['PostizPublisher']