"""
UI Components Module
"""
from .speaker_profile import display_speaker_profile, display_speaker_statistics
from .timeline import display_timeline_card, display_timeline
from .statistics import display_statistics, display_detailed_statistics

__all__ = [
    'display_speaker_profile', 
    'display_speaker_statistics',
    'display_timeline_card', 
    'display_timeline',
    'display_statistics',
    'display_detailed_statistics'
]