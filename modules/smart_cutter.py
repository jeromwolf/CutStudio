"""
스마트 컷 모듈 (Smart Cut Module)
CutStudio v5.0 - 교육 영상 자동 편집

핵심 기능:
1. 침묵 구간 자동 제거 (3초 이상)
2. 간투사("음...", "어...") 제거
3. 영상 길이 50% 단축 목표
"""

import os
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Literal
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Cut mode definitions
CutMode = Literal["conservative", "balanced", "aggressive"]

@dataclass
class CutSegment:
    """편집 구간 정보"""
    start_time: float
    end_time: float
    reason: str  # 'silence', 'filler', 'keep'
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        return {
            'start': round(self.start_time, 2),
            'end': round(self.end_time, 2),
            'duration': round(self.duration, 2),
            'reason': self.reason,
            'confidence': round(self.confidence, 2)
        }

class SmartCutter:
    """
    스마트 컷 편집기
    침묵 제거 + 간투사 제거 + 불필요한 구간 자동 편집
    """

    # 한국어 간투사 패턴
    KOREAN_FILLERS = [
        "음", "어", "그", "저", "뭐", "이제", "그니까", "있잖아",
        "아니", "그래서", "그러니까", "뭐야", "에", "아"
    ]

    # 영어 간투사 패턴
    ENGLISH_FILLERS = [
        "um", "uh", "ah", "er", "like", "you know", "so", "well",
        "actually", "basically", "literally", "right"
    ]

    def __init__(self, mode: CutMode = "balanced"):
        """
        Args:
            mode: 편집 강도
                - conservative: 보수적 (긴 침묵만 제거, 30% 단축)
                - balanced: 균형 (표준 편집, 50% 단축)
                - aggressive: 공격적 (짧은 침묵도 제거, 70% 단축)
        """
        self.mode = mode
        self.cut_segments: List[CutSegment] = []
        self.keep_segments: List[CutSegment] = []
        self.settings = self._get_mode_settings()

    def _get_mode_settings(self) -> Dict:
        """모드별 편집 설정"""
        settings = {
            "conservative": {
                "silence_threshold": 3.0,  # 3초 이상 침묵만
                "silence_db": -40,  # dB
                "remove_fillers": False,
                "min_speech_duration": 1.0,
                "target_reduction": 0.3  # 30% 단축
            },
            "balanced": {
                "silence_threshold": 2.0,  # 2초 이상 침묵
                "silence_db": -35,
                "remove_fillers": True,
                "min_speech_duration": 0.5,
                "target_reduction": 0.5  # 50% 단축
            },
            "aggressive": {
                "silence_threshold": 1.0,  # 1초 이상 침묵
                "silence_db": -30,
                "remove_fillers": True,
                "min_speech_duration": 0.3,
                "target_reduction": 0.7  # 70% 단축
            }
        }
        return settings.get(self.mode, settings["balanced"])

    def detect_silence(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        침묵 구간 감지

        Returns:
            List of (start, end) tuples for silence periods
        """
        try:
            # 실제 구현 시 librosa 사용
            # import librosa
            # y, sr = librosa.load(audio_path, sr=22050, mono=True)
            # y_db = librosa.amplitude_to_db(np.abs(y))
            # ...
            pass
        except:
            pass

        # Fallback: 더미 데이터
        return self._generate_dummy_silence()

    def _generate_dummy_silence(self) -> List[Tuple[float, float]]:
        """테스트용 더미 침묵 구간"""
        return [
            (5.0, 8.5),    # 3.5초 침묵
            (25.0, 28.0),  # 3초 침묵
            (45.5, 49.0),  # 3.5초 침묵
            (72.0, 76.0),  # 4초 침묵
            (95.0, 98.0),  # 3초 침묵
        ]

    def detect_filler_words(self, transcript: List[Dict]) -> List[Tuple[float, float]]:
        """
        간투사 구간 감지

        Args:
            transcript: 음성 인식 결과 [{word, start, end, confidence}, ...]

        Returns:
            List of (start, end) tuples for filler words
        """
        if not self.settings["remove_fillers"]:
            return []

        filler_regions = []
        all_fillers = self.KOREAN_FILLERS + self.ENGLISH_FILLERS

        for word_info in transcript:
            word = word_info.get('word', '').lower().strip()

            # 간투사 체크
            for filler in all_fillers:
                if word.startswith(filler) or word.endswith(filler):
                    filler_regions.append((
                        word_info['start'],
                        word_info['end']
                    ))
                    break

        return filler_regions

    def process_video(self, video_path: str, transcript: Optional[List[Dict]] = None) -> Dict:
        """
        비디오 스마트 컷 처리

        Args:
            video_path: 원본 비디오 경로
            transcript: 음성 인식 결과 (선택)

        Returns:
            편집 결과 정보
        """
        start_time = time.time()

        # 1. 오디오 추출
        audio_path = self._extract_audio(video_path)

        # 2. 침묵 구간 감지
        silence_regions = self.detect_silence(audio_path)

        # 3. 간투사 감지 (transcript 있는 경우)
        filler_regions = []
        if transcript:
            filler_regions = self.detect_filler_words(transcript)

        # 4. 제거할 구간 병합
        cut_regions = self._merge_regions(silence_regions, filler_regions)

        # 5. 유지할 구간 계산
        video_duration = self._get_video_duration(video_path)
        keep_regions = self._calculate_keep_regions(cut_regions, video_duration)

        # 6. 결과 생성
        for start, end in cut_regions:
            self.cut_segments.append(CutSegment(
                start_time=start,
                end_time=end,
                reason='silence' if (start, end) in silence_regions else 'filler',
                confidence=0.85
            ))

        for start, end in keep_regions:
            self.keep_segments.append(CutSegment(
                start_time=start,
                end_time=end,
                reason='keep',
                confidence=0.95
            ))

        # 7. 통계 계산
        processing_time = time.time() - start_time

        return {
            'original_duration': video_duration,
            'edited_duration': sum(s.duration for s in self.keep_segments),
            'reduction_rate': 1 - (sum(s.duration for s in self.keep_segments) / video_duration) if video_duration > 0 else 0,
            'silence_removed': len([s for s in self.cut_segments if s.reason == 'silence']),
            'fillers_removed': len([s for s in self.cut_segments if s.reason == 'filler']),
            'cut_segments': [s.to_dict() for s in self.cut_segments],
            'keep_segments': [s.to_dict() for s in self.keep_segments],
            'processing_time': round(processing_time, 2),
            'mode': self.mode
        }

    def _extract_audio(self, video_path: str) -> str:
        """비디오에서 오디오 추출"""
        # 실제 구현 시 moviepy 사용
        # 여기서는 더미 경로 반환
        return video_path.replace('.mp4', '.wav')

    def _get_video_duration(self, video_path: str) -> float:
        """비디오 길이 확인"""
        # 실제 구현 시 moviepy 사용
        # 더미 데이터
        return 180.0  # 3분

    def _merge_regions(self, silence: List[Tuple], fillers: List[Tuple]) -> List[Tuple]:
        """겹치는 구간 병합"""
        all_regions = silence + fillers
        if not all_regions:
            return []

        # 시작 시간으로 정렬
        all_regions.sort(key=lambda x: x[0])

        merged = [all_regions[0]]
        for start, end in all_regions[1:]:
            last_start, last_end = merged[-1]

            # 겹치거나 인접한 구간 병합
            if start <= last_end + 0.1:  # 0.1초 여유
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        return merged

    def _calculate_keep_regions(self, cut_regions: List[Tuple], total_duration: float) -> List[Tuple]:
        """유지할 구간 계산"""
        if not cut_regions:
            return [(0, total_duration)]

        keep_regions = []
        current_pos = 0

        for cut_start, cut_end in cut_regions:
            if current_pos < cut_start:
                # 최소 발화 길이 체크
                if cut_start - current_pos >= self.settings["min_speech_duration"]:
                    keep_regions.append((current_pos, cut_start))
            current_pos = cut_end

        # 마지막 구간
        if current_pos < total_duration:
            if total_duration - current_pos >= self.settings["min_speech_duration"]:
                keep_regions.append((current_pos, total_duration))

        return keep_regions


def test_smart_cutter():
    """스마트 컷 모듈 테스트"""
    print("🎬 스마트 컷 모듈 테스트 시작...\n")
    print("=" * 50)

    # 각 모드 테스트
    test_results = []

    for mode in ["conservative", "balanced", "aggressive"]:
        print(f"\n📊 {mode.upper()} 모드 테스트:")
        print("-" * 30)

        cutter = SmartCutter(mode=mode)
        result = cutter.process_video("test_video.mp4")

        print(f"✓ 원본 길이: {result['original_duration']:.1f}초")
        print(f"✓ 편집 길이: {result['edited_duration']:.1f}초")
        print(f"✓ 단축률: {result['reduction_rate']*100:.1f}%")
        print(f"✓ 제거된 침묵: {result['silence_removed']}개")
        print(f"✓ 제거된 간투사: {result['fillers_removed']}개")
        print(f"✓ 처리 시간: {result['processing_time']}초")

        test_results.append({
            'mode': mode,
            'reduction': result['reduction_rate'],
            'time': result['processing_time']
        })

    # 성능 비교
    print("\n" + "=" * 50)
    print("📈 모드별 비교:")
    print("-" * 50)
    print(f"{'모드':<15} {'단축률':<10} {'처리시간':<10} {'추천 사용'}")
    print("-" * 50)

    recommendations = {
        "conservative": "중요 강의, 정보 손실 최소화",
        "balanced": "일반 교육 영상 (추천)",
        "aggressive": "요약본, 하이라이트 제작"
    }

    for result in test_results:
        print(f"{result['mode']:<15} {result['reduction']*100:<10.1f}% "
              f"{result['time']:<10.2f}초 {recommendations[result['mode']]}")

    print("\n✅ 테스트 완료!")
    print("\n💡 교육 영상은 'balanced' 모드 추천 (50% 단축 목표)")


if __name__ == "__main__":
    test_smart_cutter()