import whisper
import torch
from pathlib import Path
import tempfile
import subprocess
import os
import logging
from moviepy.editor import VideoFileClip

class SpeechRecognizer:
    def __init__(self, model_size="base"):
        """
        음성 인식기 초기화
        model_size: tiny, base, small, medium, large, large-v2, large-v3
        """
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 모델 로드
        self._load_model()
    
    def _load_model(self):
        """Whisper 모델 로드"""
        try:
            print(f"Whisper 모델 로딩 중 ({self.model_size})...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"모델 로딩 완료: {self.model_size} on {self.device}")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            # 더 작은 모델로 재시도
            if self.model_size != "tiny":
                print("더 작은 모델로 재시도...")
                self.model_size = "tiny"
                try:
                    self.model = whisper.load_model("tiny", device=self.device)
                    print("tiny 모델 로딩 성공")
                except Exception as e2:
                    print(f"tiny 모델도 실패: {e2}")
                    self.model = None
    
    def extract_audio_from_video(self, video_path, output_audio_path=None):
        """비디오에서 오디오 추출"""
        if output_audio_path is None:
            output_audio_path = tempfile.mktemp(suffix='.wav')
        
        try:
            # MoviePy 사용하여 오디오 추출
            video = VideoFileClip(video_path)
            if video.audio is None:
                print("비디오에 오디오가 없습니다.")
                return None
            
            audio = video.audio
            audio.write_audiofile(
                output_audio_path,
                verbose=False,
                logger=None,
                fps=16000  # Whisper는 16kHz를 선호
            )
            
            video.close()
            audio.close()
            
            return output_audio_path
            
        except Exception as e:
            print(f"오디오 추출 실패: {e}")
            return None
    
    def transcribe_audio(self, audio_path, language="ko"):
        """오디오 파일 음성 인식"""
        if self.model is None:
            print("모델이 로드되지 않았습니다.")
            return None
        
        try:
            print("음성 인식 중...")
            
            # Whisper 옵션 설정
            options = {
                "language": language,
                "task": "transcribe",
                "fp16": False,  # CPU에서는 fp16 비활성화
                "word_timestamps": True,  # 단어별 타임스탬프
                "verbose": False
            }
            
            # 음성 인식 실행
            result = self.model.transcribe(audio_path, **options)
            
            return result
            
        except Exception as e:
            print(f"음성 인식 실패: {e}")
            return None
    
    def transcribe_video(self, video_path, language="ko"):
        """비디오 파일 전체 음성 인식"""
        # 임시 오디오 파일 추출
        temp_audio = self.extract_audio_from_video(video_path)
        if temp_audio is None:
            return None
        
        try:
            # 음성 인식 수행
            result = self.transcribe_audio(temp_audio, language)
            return result
            
        finally:
            # 임시 파일 정리
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)
    
    def transcribe_segments(self, video_path, speaker_segments, language="ko"):
        """화자별 세그먼트 음성 인식"""
        if not speaker_segments:
            return []
        
        recognized_segments = []
        
        print(f"화자 세그먼트 음성 인식 시작: {len(speaker_segments)}개 세그먼트")
        
        # 전체 비디오 음성 인식 결과 얻기
        full_transcription = self.transcribe_video(video_path, language)
        if full_transcription is None:
            print("전체 음성 인식 실패")
            return speaker_segments
        
        print(f"전체 음성 인식 성공: {len(full_transcription.get('segments', []))}개 텍스트 세그먼트")
        
        try:
            # 각 화자 세그먼트에 텍스트 매칭
            for segment in speaker_segments:
                start_time = segment['start']
                end_time = segment['end']
                
                # 해당 시간대의 텍스트 찾기
                segment_text = self._extract_text_for_timerange(
                    full_transcription, start_time, end_time
                )
                
                # 세그먼트에 텍스트 추가
                enhanced_segment = segment.copy()
                enhanced_segment['text'] = segment_text
                enhanced_segment['has_text'] = bool(segment_text.strip())
                
                recognized_segments.append(enhanced_segment)
            
            return recognized_segments
            
        except Exception as e:
            print(f"세그먼트 음성 인식 실패: {e}")
            # 실패 시 원본 세그먼트 반환
            return speaker_segments
    
    def _extract_text_for_timerange(self, transcription_result, start_time, end_time):
        """특정 시간대의 텍스트 추출"""
        if not transcription_result or 'segments' not in transcription_result:
            return ""
        
        segment_texts = []
        
        for segment in transcription_result['segments']:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # 시간대가 겹치는 부분 찾기
            if self._time_ranges_overlap(start_time, end_time, seg_start, seg_end):
                text = segment.get('text', '').strip()
                if text:
                    segment_texts.append(text)
        
        return ' '.join(segment_texts)
    
    def _time_ranges_overlap(self, start1, end1, start2, end2):
        """두 시간 범위가 겹치는지 확인"""
        return max(start1, start2) < min(end1, end2)
    
    def generate_summary(self, recognized_segments):
        """화자별 대화 내용 요약 생성"""
        if not recognized_segments:
            return {}
        
        summary = {}
        
        try:
            # 화자별로 그룹화
            speaker_groups = {}
            for segment in recognized_segments:
                speaker_id = segment['speaker']
                if speaker_id not in speaker_groups:
                    speaker_groups[speaker_id] = []
                speaker_groups[speaker_id].append(segment)
            
            # 각 화자별 요약 생성
            for speaker_id, segments in speaker_groups.items():
                texts = [seg.get('text', '') for seg in segments if seg.get('text', '').strip()]
                combined_text = ' '.join(texts)
                
                # 기본 통계
                word_count = len(combined_text.split()) if combined_text else 0
                char_count = len(combined_text)
                
                # 주요 키워드 추출 (간단한 방식)
                keywords = self._extract_keywords(combined_text)
                
                summary[speaker_id] = {
                    'speaker': speaker_id,
                    'full_text': combined_text,
                    'word_count': word_count,
                    'char_count': char_count,
                    'keywords': keywords,
                    'has_content': bool(combined_text.strip()),
                    'segment_count': len(segments)
                }
            
            return summary
            
        except Exception as e:
            print(f"요약 생성 실패: {e}")
            return {}
    
    def _extract_keywords(self, text, max_keywords=5):
        """간단한 키워드 추출"""
        if not text:
            return []
        
        # 불용어 제거 및 단어 빈도 계산
        stopwords = {'이', '그', '저', '것', '의', '가', '을', '를', '에', '에서', '와', '과', '은', '는', '이다', '있다', '없다', '하다', '되다', '같다'}
        words = text.split()
        
        # 단어 빈도 계산
        word_freq = {}
        for word in words:
            word = word.strip('.,!?()[]{}""\'')
            if len(word) > 1 and word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도순으로 정렬하여 상위 키워드 반환
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def generate_simple_summary(self, text, max_length=100):
        """간단한 텍스트 요약 (API 키 불필요)"""
        if not text:
            return "내용 없음"
        
        # 문장 단위로 분리
        sentences = text.replace('?', '.').replace('!', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # 첫 문장과 키워드 기반 요약
        summary = sentences[0]
        keywords = self._extract_keywords(text, 3)
        
        if keywords:
            summary += f" (주요 키워드: {', '.join(keywords)})"
        
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary

class AdvancedSpeechAnalyzer:
    """고급 음성 분석 기능"""
    
    def __init__(self, speech_recognizer):
        self.recognizer = speech_recognizer
    
    def analyze_conversation_flow(self, recognized_segments):
        """대화 흐름 분석"""
        if not recognized_segments:
            return {}
        
        analysis = {
            'total_speakers': len(set(seg['speaker'] for seg in recognized_segments)),
            'total_segments': len(recognized_segments),
            'conversation_turns': [],
            'speaker_interaction': {},
            'timeline': []
        }
        
        try:
            # 시간순으로 정렬
            sorted_segments = sorted(recognized_segments, key=lambda x: x['start'])
            
            # 대화 턴 분석
            current_speaker = None
            turn_start = None
            
            for segment in sorted_segments:
                speaker = segment['speaker']
                
                if speaker != current_speaker:
                    if current_speaker is not None:
                        # 이전 턴 종료
                        analysis['conversation_turns'].append({
                            'speaker': current_speaker,
                            'start': turn_start,
                            'end': segment['start'],
                            'duration': segment['start'] - turn_start
                        })
                    
                    # 새 턴 시작
                    current_speaker = speaker
                    turn_start = segment['start']
                
                # 타임라인 추가
                analysis['timeline'].append({
                    'time': segment['start'],
                    'speaker': speaker,
                    'text': segment.get('text', '')[:50] + '...' if len(segment.get('text', '')) > 50 else segment.get('text', ''),
                    'duration': segment['duration']
                })
            
            # 마지막 턴 처리
            if current_speaker is not None and sorted_segments:
                analysis['conversation_turns'].append({
                    'speaker': current_speaker,
                    'start': turn_start,
                    'end': sorted_segments[-1]['end'],
                    'duration': sorted_segments[-1]['end'] - turn_start
                })
            
            return analysis
            
        except Exception as e:
            print(f"대화 흐름 분석 실패: {e}")
            return analysis
    
    def generate_meeting_summary(self, recognized_segments):
        """회의/대화 종합 요약"""
        if not recognized_segments:
            return ""
        
        try:
            conversation_analysis = self.analyze_conversation_flow(recognized_segments)
            speech_summary = self.recognizer.generate_summary(recognized_segments)
            
            # 종합 요약 생성
            summary_parts = []
            summary_parts.append("=== 대화 요약 ===")
            summary_parts.append(f"참여자: {conversation_analysis['total_speakers']}명")
            summary_parts.append(f"총 발화 구간: {conversation_analysis['total_segments']}개")
            summary_parts.append("")
            
            # 화자별 주요 내용
            summary_parts.append("=== 화자별 주요 내용 ===")
            for speaker_id, content in speech_summary.items():
                if content['has_content']:
                    summary_parts.append(f"{speaker_id}:")
                    summary_parts.append(f"  - 발화 횟수: {content['segment_count']}회")
                    summary_parts.append(f"  - 단어 수: {content['word_count']}개")
                    if content['keywords']:
                        summary_parts.append(f"  - 주요 키워드: {', '.join(content['keywords'])}")
                    summary_parts.append("")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            print(f"종합 요약 생성 실패: {e}")
            return "요약 생성에 실패했습니다."