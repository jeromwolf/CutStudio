import os
import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import librosa
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

class PracticalSpeakerDetector:
    """실용적인 화자 감지기 - 속도와 정확도의 균형"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 장치: {self.device}")
        
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
    
    def extract_practical_features(self, audio_segment, sr=16000):
        """실용적인 특징 추출 - 핵심 특징만"""
        try:
            # numpy 배열로 변환
            if isinstance(audio_segment, AudioSegment):
                samples = np.array(audio_segment.get_array_of_samples(), dtype=float)
            else:
                samples = audio_segment
            
            # 정규화
            if len(samples) > 0 and np.max(np.abs(samples)) > 0:
                samples = samples / np.max(np.abs(samples))
            
            features = []
            
            # 1. MFCC (13개 계수만 사용 - 충분함)
            mfccs = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # 2. 기본 피치 특징
            f0, voiced_flag, voiced_probs = librosa.pyin(
                samples, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                pitch_features = [
                    np.mean(valid_f0),
                    np.std(valid_f0),
                    np.percentile(valid_f0, 25),
                    np.percentile(valid_f0, 75),
                    np.max(valid_f0) - np.min(valid_f0)
                ]
            else:
                pitch_features = [0, 0, 0, 0, 0]
            features.extend(pitch_features)
            
            # 3. 스펙트럴 특징 (핵심만)
            spectral_centroids = librosa.feature.spectral_centroid(y=samples, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr)[0]
            
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff)
            ])
            
            # 4. 에너지 특징
            rmse = librosa.feature.rms(y=samples)[0]
            features.extend([
                np.mean(rmse),
                np.std(rmse),
                np.max(rmse)
            ])
            
            # 5. ZCR
            zcr = librosa.feature.zero_crossing_rate(samples)[0]
            features.extend([
                np.mean(zcr),
                np.std(zcr)
            ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"특징 추출 오류: {e}")
            # 특징 수: 13*2 + 5 + 4 + 3 + 2 = 40
            return np.zeros(40, dtype=np.float32)
    
    def fast_vad(self, audio_path, min_duration=1.0):
        """빠른 음성 구간 검출"""
        try:
            # Silero VAD 사용
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
                verbose=False
            )
            
            (get_speech_timestamps, _, read_audio, _, _) = utils
            
            # 오디오 로드
            wav = read_audio(audio_path, sampling_rate=16000)
            
            # 음성 구간 감지
            speech_timestamps = get_speech_timestamps(
                wav, 
                model,
                sampling_rate=16000,
                threshold=0.5,  # 기본값 사용
                min_speech_duration_ms=500,
                min_silence_duration_ms=300
            )
            
            segments = []
            for ts in speech_timestamps:
                segment = {
                    'start': ts['start'] / 16000,
                    'end': ts['end'] / 16000,
                    'duration': (ts['end'] - ts['start']) / 16000
                }
                if segment['duration'] >= min_duration:
                    segments.append(segment)
            
            # 너무 많은 세그먼트는 병합
            if len(segments) > 50:
                segments = self.merge_close_segments(segments, max_gap=1.0)
            
            return segments
            
        except Exception as e:
            print(f"VAD 실패, 간단한 방법 사용: {e}")
            return self.energy_based_vad(audio_path, min_duration)
    
    def energy_based_vad(self, audio_path, min_duration):
        """에너지 기반 VAD (폴백)"""
        audio = AudioSegment.from_wav(audio_path)
        
        # 전체 에너지 분석
        chunk_size = 100  # 100ms
        energies = []
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) > 0:
                energies.append(chunk.dBFS)
        
        # 에너지 분포의 30 percentile을 임계값으로
        valid_energies = [e for e in energies if e > -float('inf')]
        if valid_energies:
            threshold = np.percentile(valid_energies, 30)
        else:
            threshold = audio.dBFS - 15
        
        segments = []
        in_speech = False
        start_idx = 0
        
        for i, energy in enumerate(energies):
            if energy > threshold:
                if not in_speech:
                    in_speech = True
                    start_idx = i
            else:
                if in_speech:
                    in_speech = False
                    start_time = start_idx * chunk_size / 1000.0
                    end_time = i * chunk_size / 1000.0
                    duration = end_time - start_time
                    
                    if duration >= min_duration:
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'duration': duration
                        })
        
        # 마지막 세그먼트 처리
        if in_speech:
            start_time = start_idx * chunk_size / 1000.0
            end_time = len(audio) / 1000.0
            duration = end_time - start_time
            
            if duration >= min_duration:
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                })
        
        return segments
    
    def merge_close_segments(self, segments, max_gap=0.5):
        """가까운 세그먼트 병합"""
        if not segments:
            return segments
        
        segments = sorted(segments, key=lambda x: x['start'])
        merged = [segments[0]]
        
        for seg in segments[1:]:
            if seg['start'] - merged[-1]['end'] <= max_gap:
                merged[-1]['end'] = seg['end']
                merged[-1]['duration'] = merged[-1]['end'] - merged[-1]['start']
            else:
                merged.append(seg)
        
        return merged
    
    def adaptive_kmeans(self, features, min_speakers=2, max_speakers=6):
        """적응형 K-means 클러스터링"""
        n_samples = len(features)
        if n_samples < min_speakers:
            return [0] * n_samples, 1
        
        # 특징 정규화
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        best_score = -1
        best_k = min_speakers
        best_labels = None
        
        # 다양한 k값 테스트
        for k in range(min_speakers, min(max_speakers + 1, n_samples)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(normalized_features)
                
                if len(np.unique(labels)) == k:
                    score = silhouette_score(normalized_features, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = labels.copy()
            except:
                continue
        
        print(f"최적 화자 수: {best_k}명 (실루엣 점수: {best_score:.3f})")
        
        return best_labels if best_labels is not None else [0] * n_samples, best_k
    
    def post_process_fast(self, segments, features, labels):
        """빠른 후처리"""
        if not segments or not features:
            return segments
        
        # 화자별 평균 특징 계산
        speaker_features = {}
        for i, label in enumerate(labels):
            if label not in speaker_features:
                speaker_features[label] = []
            speaker_features[label].append(features[i])
        
        for speaker in speaker_features:
            speaker_features[speaker] = np.mean(speaker_features[speaker], axis=0)
        
        # 세그먼트에 화자 할당 및 신뢰도 계산
        for i, (seg, feat, label) in enumerate(zip(segments, features, labels)):
            seg['speaker'] = f'SPEAKER_{label}'
            
            # 신뢰도 계산 (해당 화자 중심과의 거리)
            distance = cosine(feat, speaker_features[label])
            seg['confidence'] = max(0, 1 - distance)
        
        # 시간순 정렬
        segments = sorted(segments, key=lambda x: x['start'])
        
        # 같은 화자의 인접 세그먼트 병합
        merged = []
        if segments:
            current = segments[0].copy()
            
            for seg in segments[1:]:
                if (seg['speaker'] == current['speaker'] and 
                    seg['start'] - current['end'] < 2.0 and
                    current.get('confidence', 0) > 0.5 and 
                    seg.get('confidence', 0) > 0.5):
                    # 병합
                    current['end'] = seg['end']
                    current['duration'] = current['end'] - current['start']
                    current['confidence'] = (current['confidence'] + seg['confidence']) / 2
                else:
                    merged.append(current)
                    current = seg.copy()
            
            merged.append(current)
        
        return merged
    
    def detect_speakers(self, video_path, min_duration=1.0, num_speakers=None, progress_callback=None):
        """실용적인 화자 감지"""
        print("\n=== 실용적인 화자 감지 시작 ===")
        print("1/5 단계: 오디오 추출 중...")
        
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return None
        
        try:
            # 1. 음성 구간 검출
            print("2/5 단계: 음성 구간 검출 중...")
            segments = self.fast_vad(audio_path, min_duration)
            print(f"  → {len(segments)}개의 음성 구간 감지됨")
            
            if not segments:
                return []
            
            # 세그먼트가 너무 많으면 샘플링
            if len(segments) > 100:
                print(f"  → 세그먼트가 많아 100개로 샘플링")
                step = len(segments) // 100
                segments = segments[::step][:100]
            
            # 2. 특징 추출
            print(f"3/5 단계: {len(segments)}개 구간에서 특징 추출 중...")
            
            audio = AudioSegment.from_wav(audio_path)
            features = []
            valid_segments = []
            
            for i, seg in enumerate(segments):
                if progress_callback and i % 10 == 0:
                    progress_callback(20 + (40 * i / len(segments)))
                
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)
                
                if end_ms > start_ms:
                    audio_segment = audio[start_ms:end_ms]
                    feat = self.extract_practical_features(audio_segment, audio.frame_rate)
                    
                    if feat is not None and not np.all(feat == 0):
                        features.append(feat)
                        valid_segments.append(seg)
            
            if not features:
                return []
            
            print(f"  → {len(features)}개 구간에서 특징 추출 완료")
            
            # 3. 클러스터링
            print("4/5 단계: 화자 클러스터링 중...")
            features_array = np.array(features)
            
            if num_speakers is None:
                labels, detected_speakers = self.adaptive_kmeans(features_array)
            else:
                labels, _ = self.adaptive_kmeans(
                    features_array, 
                    min_speakers=num_speakers, 
                    max_speakers=num_speakers
                )
            
            # 4. 후처리
            print("5/5 단계: 후처리 중...")
            final_segments = self.post_process_fast(valid_segments, features, labels)
            
            print(f"=== 완료! 최종 {len(final_segments)}개 구간 ===\n")
            
            # 임시 파일 삭제
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return final_segments
            
        except Exception as e:
            print(f"화자 감지 실패: {e}")
            import traceback
            traceback.print_exc()
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return None