import os
import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import speech_recognition as sr
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.signal import butter, lfilter, medfilt
import warnings
warnings.filterwarnings('ignore')

# PyTorch 경고 숨기기
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# pyannote는 선택적으로 import
try:
    from pyannote.audio import Pipeline
    import torch
    # 호환성 문제 해결을 위해 명시적으로 False로 설정
    PYANNOTE_AVAILABLE = False  # 일시적으로 비활성화
except ImportError:
    PYANNOTE_AVAILABLE = False
    Pipeline = None
    torch = None

class SpeakerDetector:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.pipeline = None
        if PYANNOTE_AVAILABLE and torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        
    def initialize_pipeline(self, hf_token=None):
        """화자 분리 파이프라인 초기화"""
        if not PYANNOTE_AVAILABLE:
            print("pyannote가 설치되지 않았습니다. 향상된 VAD를 사용합니다.")
            return False
            
        try:
            if hf_token and Pipeline:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
            else:
                # 간단한 대안으로 음성 구간 검출만 수행
                self.pipeline = None
            
            if self.pipeline and self.device:
                self.pipeline.to(self.device)
            return True
        except Exception as e:
            print(f"파이프라인 초기화 실패: {e}")
            return False
    
    def extract_audio(self, video_path):
        """비디오에서 오디오 추출"""
        try:
            video = VideoFileClip(video_path)
            audio_path = tempfile.mktemp(suffix=".wav")
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            return audio_path
        except Exception as e:
            print(f"오디오 추출 실패: {e}")
            return None
    
    def preprocess_audio(self, audio_segment, sr=16000):
        """오디오 전처리: 노이즈 제거, 정규화"""
        try:
            # numpy 배열로 변환
            samples = np.array(audio_segment.get_array_of_samples(), dtype=float)
            
            # 1. DC 제거 (평균값 제거)
            samples = samples - np.mean(samples)
            
            # 2. 정규화
            max_val = np.max(np.abs(samples))
            if max_val > 0:
                samples = samples / max_val * 0.95  # 0.95로 정규화 (클리핑 방지)
            
            # 3. 고역 통과 필터 (50Hz 이하 제거 - 전원 노이즈 등)
            nyquist = sr / 2
            highpass_freq = 50 / nyquist
            if highpass_freq < 1.0:
                b, a = butter(4, highpass_freq, btype='high')
                samples = lfilter(b, a, samples)
            
            # 4. 스펙트럴 서브트랙션 (간단한 노이즈 제거)
            # 처음 0.1초를 노이즈로 가정
            noise_duration = int(0.1 * sr)
            if len(samples) > noise_duration * 2:
                noise_profile = samples[:noise_duration]
                noise_power = np.mean(noise_profile ** 2)
                
                # 노이즈 제거 (간단한 방법)
                signal_power = samples ** 2
                clean_power = signal_power - noise_power
                clean_power[clean_power < 0] = 0
                samples = np.sqrt(clean_power) * np.sign(samples)
            
            # 5. 메디안 필터 (임펄스 노이즈 제거)
            if len(samples) > 5:
                samples = medfilt(samples, kernel_size=5)
            
            return samples
            
        except Exception as e:
            print(f"오디오 전처리 실패: {e}")
            return np.array(audio_segment.get_array_of_samples(), dtype=float)
    
    def detect_speakers(self, video_path, min_duration=2.0, num_speakers=2, use_simple=False):
        """화자 구간 감지"""
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return None
        
        try:
            if self.pipeline and not use_simple:
                # Pyannote를 사용한 화자 분리
                diarization = self.pipeline(audio_path)
                
                segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.duration >= min_duration:
                        segments.append({
                            'start': turn.start,
                            'end': turn.end,
                            'speaker': speaker,
                            'duration': turn.duration
                        })
            elif use_simple:
                # 간단한 VAD 사용
                segments = self.simple_vad(audio_path, min_duration, num_speakers)
            else:
                # 향상된 음성 활동 감지 (VAD) 사용
                segments = self.enhanced_speaker_detection(audio_path, min_duration, num_speakers)
            
            # 임시 파일 삭제
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return segments
            
        except Exception as e:
            print(f"화자 감지 실패: {e}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return None
    
    def extract_features(self, audio_segment, sr=16000):
        """오디오 세그먼트에서 특징 추출 - 향상된 버전"""
        try:
            # 전처리된 오디오 샘플 얻기
            samples = self.preprocess_audio(audio_segment, sr)
            
            # 1. MFCC 특징 추출 (더 많은 계수)
            mfccs = librosa.feature.mfcc(y=samples.astype(float), sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
            
            # 2. 스펙트럴 특징
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=samples.astype(float), sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=samples.astype(float), sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=samples.astype(float), sr=sr))
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=samples.astype(float), sr=sr))
            
            # 3. Zero Crossing Rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(samples.astype(float)))
            zcr_std = np.std(librosa.feature.zero_crossing_rate(samples.astype(float)))
            
            # 4. 에너지 특징
            rmse = np.mean(librosa.feature.rms(y=samples.astype(float)))
            
            # 5. 피치 추정 (향상된 방법)
            pitches, magnitudes = librosa.piptrack(y=samples.astype(float), sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                pitch_mean = np.mean(pitch_values)
                pitch_std = np.std(pitch_values)
            else:
                pitch_mean = 0
                pitch_std = 0
            
            # 6. 포먼트 특징 (간단한 추정)
            # 첫 번째 두 포먼트의 위치를 추정
            fft = np.fft.rfft(samples.astype(float) * np.hanning(len(samples)))
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(samples), 1/sr)
            
            # 피크 찾기
            peaks = []
            for i in range(1, len(magnitude)-1):
                if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                    if freqs[i] > 200 and freqs[i] < 4000:  # 음성 주파수 범위
                        peaks.append((freqs[i], magnitude[i]))
            
            peaks.sort(key=lambda x: x[1], reverse=True)
            formant1 = peaks[0][0] if len(peaks) > 0 else 0
            formant2 = peaks[1][0] if len(peaks) > 1 else 0
            
            # 모든 특징을 하나의 벡터로 결합
            features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                mfcc_delta,
                [spectral_centroid, spectral_rolloff, spectral_bandwidth, spectral_contrast,
                 zcr, zcr_std, rmse, pitch_mean, pitch_std, formant1, formant2]
            ])
            
            return features
            
        except Exception as e:
            print(f"특징 추출 실패: {e}")
            # 기본 특징 벡터 반환
            return np.zeros(71)  # 20*3 + 11 = 71 features
    
    def estimate_num_speakers(self, features):
        """실루엣 계수와 엘보우 방법을 사용하여 최적의 화자 수 추정"""
        if len(features) < 3:
            return 2
        
        max_clusters = min(6, len(features) - 1)
        scores = []
        inertias = []
        
        for n_clusters in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
                labels = kmeans.fit_predict(features)
                
                # 모든 라벨이 동일한 경우 스킵
                if len(np.unique(labels)) < n_clusters:
                    continue
                    
                score = silhouette_score(features, labels)
                inertia = kmeans.inertia_
                
                scores.append((n_clusters, score))
                inertias.append((n_clusters, inertia))
            except:
                continue
        
        if not scores:
            return 2
        
        # 1. 실루엣 점수 기반
        best_silhouette = max(scores, key=lambda x: x[1])
        
        # 2. 엘보우 방법 (inertia 감소율)
        if len(inertias) > 2:
            inertia_diffs = []
            for i in range(1, len(inertias)):
                diff = (inertias[i-1][1] - inertias[i][1]) / inertias[i-1][1]
                inertia_diffs.append((inertias[i][0], diff))
            
            # 감소율이 급격히 줄어드는 지점 찾기
            if inertia_diffs:
                threshold = 0.1  # 10% 미만 개선
                elbow_point = 2
                for n, diff in inertia_diffs:
                    if diff < threshold:
                        elbow_point = n - 1
                        break
                    elbow_point = n
        else:
            elbow_point = best_silhouette[0]
        
        # 3. 두 방법의 결과를 종합
        # 실루엣 점수가 높으면서 엘보우 포인트에 가까운 값 선택
        final_n = best_silhouette[0]
        
        # 엘보우 포인트와 실루엣 최고점이 다른 경우
        if elbow_point != best_silhouette[0]:
            # 실루엣 점수가 최고점의 90% 이상인 것 중에서 엘보우에 가까운 것 선택
            best_score = best_silhouette[1]
            candidates = [n for n, s in scores if s >= best_score * 0.85]
            
            if elbow_point in candidates:
                final_n = elbow_point
            else:
                # 엘보우와 가장 가까운 후보 선택
                final_n = min(candidates, key=lambda x: abs(x - elbow_point))
        
        print(f"실루엣 최적: {best_silhouette[0]}, 엘보우: {elbow_point}, 최종 선택: {final_n}")
        
        return final_n
    
    def enhanced_speaker_detection(self, audio_path, min_duration=2.0, num_speakers=2):
        """향상된 화자 감지 알고리즘"""
        try:
            # 오디오 로드
            audio = AudioSegment.from_wav(audio_path)
            sample_rate = audio.frame_rate
            
            # 음성 구간 감지 (Silero VAD 우선 시도)
            try:
                speech_segments = self.detect_speech_segments_silero(audio_path, min_duration)
            except:
                # 폴백으로 기본 VAD 사용
                speech_segments = self.detect_speech_segments(audio, min_duration)
            
            if not speech_segments:
                return []
            
            # 각 음성 구간에서 특징 추출
            segment_features = []
            for seg in speech_segments:
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)
                audio_segment = audio[start_ms:end_ms]
                
                # 특징 추출
                features = self.extract_features(audio_segment, sample_rate)
                segment_features.append(features)
            
            # 특징 정규화
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(segment_features)
            
            # 화자 수 결정
            if num_speakers is None:
                # 자동으로 화자 수 추정 (실루엣 계수 사용)
                n_speakers = self.estimate_num_speakers(normalized_features)
                print(f"자동으로 감지된 화자 수: {n_speakers}명")
            else:
                n_speakers = num_speakers
            
            # K-means 클러스터링으로 화자 구분
            kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
            speaker_labels = kmeans.fit_predict(normalized_features)
            
            # 각 포인트의 클러스터 중심까지의 거리 계산 (신뢰도로 사용)
            distances = kmeans.transform(normalized_features)
            min_distances = np.min(distances, axis=1)
            max_distance = np.max(min_distances) if len(min_distances) > 0 else 1.0
            
            # 거리를 신뢰도로 변환 (가까울수록 높은 신뢰도)
            confidences = 1 - (min_distances / (max_distance + 1e-6))
            
            # 결과 정리
            final_segments = []
            for i, (seg, label, conf) in enumerate(zip(speech_segments, speaker_labels, confidences)):
                final_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'speaker': f'SPEAKER_{label}',
                    'duration': seg['duration'],
                    'confidence': float(conf)  # 신뢰도 추가
                })
            
            # 같은 화자의 연속된 구간 병합
            final_segments = self.merge_close_segments(final_segments, max_gap=1.0)
            
            return final_segments
            
        except Exception as e:
            print(f"향상된 화자 감지 실패: {e}")
            import traceback
            traceback.print_exc()
            # 폴백으로 기본 VAD 사용
            return self.simple_vad(audio_path, min_duration, num_speakers)
    
    def detect_speech_segments_silero(self, audio_path, min_duration=2.0):
        """Silero VAD를 사용한 정확한 음성 구간 감지"""
        try:
            # Silero VAD 시도
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            
            (get_speech_timestamps, save_audio, read_audio, 
             VADIterator, collect_chunks) = utils
            
            # 오디오 로드 (16kHz)
            wav = read_audio(audio_path, sampling_rate=16000)
            
            # 음성 구간 감지
            speech_timestamps = get_speech_timestamps(
                wav, 
                model,
                sampling_rate=16000,
                threshold=0.5,
                min_speech_duration_ms=int(min_duration * 1000),
                min_silence_duration_ms=300
            )
            
            segments = []
            for ts in speech_timestamps:
                segments.append({
                    'start': ts['start'] / 16000,
                    'end': ts['end'] / 16000,
                    'duration': (ts['end'] - ts['start']) / 16000
                })
            
            return segments
            
        except Exception as e:
            print(f"Silero VAD 실패, 기본 VAD 사용: {e}")
            # 폴백으로 기본 VAD 사용
            audio = AudioSegment.from_wav(audio_path)
            return self.detect_speech_segments(audio, min_duration)
    
    def detect_speech_segments(self, audio, min_duration=2.0):
        """개선된 음성 구간 감지 - 더 정교한 VAD"""
        try:
            # 파라미터 설정
            chunk_size = 50  # 50ms로 더 세밀하게
            
            # 에너지와 ZCR 기반 VAD
            energy_values = []
            zcr_values = []
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if len(chunk) > 0:
                    # 에너지 계산
                    energy_values.append(chunk.dBFS)
                    
                    # Zero Crossing Rate 계산
                    samples = np.array(chunk.get_array_of_samples())
                    if len(samples) > 1:
                        zcr = np.sum(np.abs(np.diff(np.sign(samples)))) / (2 * len(samples))
                        zcr_values.append(zcr)
                    else:
                        zcr_values.append(0)
            
            # 동적 임계값 계산
            valid_energies = [e for e in energy_values if e > -float('inf')]
            if valid_energies:
                energy_threshold = np.percentile(valid_energies, 25)  # 하위 25% 이상
                energy_high = np.percentile(valid_energies, 75)  # 상위 25%
            else:
                energy_threshold = -40
                energy_high = -20
            
            # ZCR 임계값
            if zcr_values:
                zcr_threshold = np.percentile(zcr_values, 50)
            else:
                zcr_threshold = 0.02
            
            # 음성 구간 검출 (에너지와 ZCR 조합)
            segments = []
            in_speech = False
            start_idx = 0
            min_speech_chunks = int(min_duration * 1000 / chunk_size)
            speech_confidence = []
            
            for i in range(len(energy_values)):
                energy = energy_values[i]
                zcr = zcr_values[i] if i < len(zcr_values) else 0
                
                # 음성 신뢰도 계산
                if energy > energy_high:
                    confidence = 1.0  # 높은 에너지는 확실히 음성
                elif energy > energy_threshold:
                    # 중간 에너지는 ZCR도 고려
                    if zcr < zcr_threshold * 1.5:  # 낮은 ZCR은 음성일 가능성 높음
                        confidence = 0.8
                    else:
                        confidence = 0.3
                else:
                    confidence = 0.0
                
                speech_confidence.append(confidence)
            
            # 스무딩 (노이즈 제거)
            window_size = 3
            smoothed_confidence = []
            for i in range(len(speech_confidence)):
                start = max(0, i - window_size // 2)
                end = min(len(speech_confidence), i + window_size // 2 + 1)
                smoothed_confidence.append(np.mean(speech_confidence[start:end]))
            
            # 구간 검출
            for i, confidence in enumerate(smoothed_confidence):
                is_speech = confidence > 0.5
                
                if is_speech and not in_speech:
                    # 음성 시작
                    in_speech = True
                    start_idx = i
                elif not is_speech and in_speech:
                    # 음성 종료
                    duration_chunks = i - start_idx
                    if duration_chunks >= min_speech_chunks:
                        segments.append({
                            'start': start_idx * chunk_size / 1000.0,
                            'end': i * chunk_size / 1000.0,
                            'duration': duration_chunks * chunk_size / 1000.0
                        })
                    in_speech = False
            
            # 마지막 구간 처리
            if in_speech:
                duration_chunks = len(smoothed_confidence) - start_idx
                if duration_chunks >= min_speech_chunks:
                    segments.append({
                        'start': start_idx * chunk_size / 1000.0,
                        'end': len(audio) / 1000.0,
                        'duration': (len(audio) / 1000.0) - (start_idx * chunk_size / 1000.0)
                    })
            
            return segments
            
        except Exception as e:
            print(f"음성 구간 감지 실패: {e}")
            return []
    
    def simple_vad(self, audio_path, min_duration=2.0, num_speakers=2):
        """간단한 음성 활동 감지 - 폴백용"""
        try:
            audio = AudioSegment.from_wav(audio_path)
            
            # 음성 구간 감지 (간단한 에너지 기반)
            segments = []
            chunk_size = 100  # 100ms
            
            # 에너지 임계값 계산
            energy_values = []
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if len(chunk) > 0 and chunk.dBFS > -float('inf'):
                    energy_values.append(chunk.dBFS)
            
            if energy_values:
                energy_threshold = np.median(energy_values) + 3
            else:
                energy_threshold = -30
            
            current_segment = None
            speaker_id = 0
            silence_duration = 0
            max_silence = 1000  # 1초
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if len(chunk) == 0:
                    continue
                
                is_speech = chunk.dBFS > energy_threshold
                
                if is_speech:
                    if current_segment is None:
                        current_segment = {
                            'start': i / 1000.0,
                            'speaker': f'SPEAKER_{speaker_id % num_speakers}'
                        }
                        silence_duration = 0
                    else:
                        silence_duration = 0
                else:
                    if current_segment is not None:
                        silence_duration += chunk_size
                        
                        if silence_duration >= max_silence:
                            current_segment['end'] = (i - silence_duration + chunk_size) / 1000.0
                            current_segment['duration'] = current_segment['end'] - current_segment['start']
                            
                            if current_segment['duration'] >= min_duration:
                                segments.append(current_segment)
                                speaker_id += 1
                            
                            current_segment = None
                            silence_duration = 0
            
            # 마지막 구간 처리
            if current_segment is not None:
                current_segment['end'] = len(audio) / 1000.0
                current_segment['duration'] = current_segment['end'] - current_segment['start']
                if current_segment['duration'] >= min_duration:
                    segments.append(current_segment)
            
            return segments
            
        except Exception as e:
            print(f"VAD 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def transcribe_segments(self, video_path, segments):
        """각 화자 구간의 음성을 텍스트로 변환"""
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return segments
        
        try:
            audio = AudioSegment.from_wav(audio_path)
            
            for segment in segments:
                start_ms = int(segment['start'] * 1000)
                end_ms = int(segment['end'] * 1000)
                
                # 구간 오디오 추출
                segment_audio = audio[start_ms:end_ms]
                temp_path = tempfile.mktemp(suffix=".wav")
                segment_audio.export(temp_path, format="wav")
                
                # 음성 인식
                try:
                    with sr.AudioFile(temp_path) as source:
                        audio_data = self.recognizer.record(source)
                        text = self.recognizer.recognize_google(audio_data, language='ko-KR')
                        segment['text'] = text
                except:
                    segment['text'] = "[인식 불가]"
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return segments
            
        except Exception as e:
            print(f"음성 인식 실패: {e}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return segments
    
    def merge_close_segments(self, segments, max_gap=1.0):
        """가까운 화자 구간 병합 - 개선된 버전"""
        if not segments:
            return segments
        
        # 시간순 정렬
        segments = sorted(segments, key=lambda x: x['start'])
        
        # 1차 병합: 같은 화자의 가까운 구간
        merged = []
        current = segments[0].copy()
        
        for segment in segments[1:]:
            if (segment['speaker'] == current['speaker'] and 
                segment['start'] - current['end'] <= max_gap):
                # 같은 화자의 가까운 구간 병합
                current['end'] = segment['end']
                current['duration'] = current['end'] - current['start']
                if 'text' in current and 'text' in segment:
                    current['text'] = current['text'] + ' ' + segment['text']
                # 신뢰도는 평균값 사용
                if 'confidence' in current and 'confidence' in segment:
                    current['confidence'] = (current['confidence'] + segment['confidence']) / 2
            else:
                merged.append(current)
                current = segment.copy()
        
        merged.append(current)
        
        # 2차 처리: 짧은 구간 제거 또는 병합
        final_segments = []
        min_segment_duration = 0.5  # 0.5초 미만 구간 처리
        
        for i, seg in enumerate(merged):
            if seg['duration'] < min_segment_duration:
                # 짧은 구간 처리
                if i > 0 and i < len(merged) - 1:
                    # 앞뒤 구간 확인
                    prev_seg = merged[i-1]
                    next_seg = merged[i+1]
                    
                    # 앞뒤가 같은 화자면 그 화자로 할당
                    if prev_seg['speaker'] == next_seg['speaker']:
                        seg['speaker'] = prev_seg['speaker']
                        # 이전 세그먼트와 병합
                        if final_segments and final_segments[-1]['speaker'] == seg['speaker']:
                            final_segments[-1]['end'] = seg['end']
                            final_segments[-1]['duration'] = final_segments[-1]['end'] - final_segments[-1]['start']
                            continue
                
                # 너무 짧으면 제거
                if seg['duration'] < 0.3:
                    continue
            
            final_segments.append(seg)
        
        # 3차 처리: 화자 레이블 일관성 개선
        final_segments = self.smooth_speaker_labels(final_segments)
        
        return final_segments
    
    def smooth_speaker_labels(self, segments):
        """화자 레이블 스무딩"""
        if len(segments) < 3:
            return segments
        
        smoothed = segments.copy()
        
        # 슬라이딩 윈도우로 화자 레이블 스무딩
        window_size = 3
        for i in range(1, len(segments) - 1):
            # 현재 구간이 짧고 앞뒤와 다른 화자인 경우
            if segments[i]['duration'] < 2.0:
                # 윈도우 내의 화자들 카운트
                window_start = max(0, i - window_size // 2)
                window_end = min(len(segments), i + window_size // 2 + 1)
                
                speaker_counts = {}
                total_duration = 0
                
                for j in range(window_start, window_end):
                    speaker = segments[j]['speaker']
                    duration = segments[j]['duration']
                    
                    if speaker not in speaker_counts:
                        speaker_counts[speaker] = 0
                    speaker_counts[speaker] += duration
                    total_duration += duration
                
                # 가장 많은 시간을 차지하는 화자로 할당
                if total_duration > 0:
                    dominant_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
                    # 현재 화자와 다르고, 지배적인 화자가 충분히 많은 시간을 차지하면 변경
                    if (segments[i]['speaker'] != dominant_speaker and 
                        speaker_counts[dominant_speaker] / total_duration > 0.6):
                        smoothed[i]['speaker'] = dominant_speaker
        
        return smoothed