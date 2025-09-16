"""Speaker detection modules."""

from .speaker_detector import SpeakerDetector
from .practical_speaker_detector import PracticalSpeakerDetector
from .enhanced_speaker_detector import EnhancedSpeakerDetector
from .improved_speaker_detector import ImprovedSpeakerDetector
# from .advanced_speaker_detector import AdvancedSpeakerDetector  # Requires speechbrain
# from .huggingface_speaker_detector import HuggingFaceSpeakerDetector  # Requires pyannote
from .optimized_speaker_detector import OptimizedSpeakerDetector

__all__ = [
    'SpeakerDetector',
    'PracticalSpeakerDetector',
    'EnhancedSpeakerDetector',
    'ImprovedSpeakerDetector',
    # 'AdvancedSpeakerDetector',  # Requires speechbrain
    # 'HuggingFaceSpeakerDetector',  # Requires pyannote
    'OptimizedSpeakerDetector',
]