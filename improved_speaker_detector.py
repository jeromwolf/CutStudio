import os
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import numpy as np
import torch
import torchaudio
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import librosa
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

class ImprovedSpeakerDetector:
    """개선된 화자 감지기 - 안정적이고 정확한 방법 사용"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def extract_audio(self, video_path):
        """미디어 파일에서 오디오 추출"""
        try:
            # 파일 확장자 확인
            ext = video_path.lower().split('.')[-1]
            audio_path = tempfile.mktemp(suffix=".wav")
            
            if ext in ['m4a', 'mp3', 'wav', 'aac', 'flac', 'ogg', 'wma']:
                # 오디오 파일인 경우
                audio = AudioFileClip(video_path)
                audio.write_audiofile(audio_path, verbose=False, logger=None)
                audio.close()
            else:
                # 비디오 파일인 경우
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                video.close()
                
            return audio_path
        except Exception as e:
            print(f"오디오 추출 실패: {e}")
            return None
    
    def extract_advanced_features(self, audio_segment, sr=16000):
        """고급 음성 특징 추출"""
        try:
            # numpy 배열로 변환
            if isinstance(audio_segment, AudioSegment):
                samples = np.array(audio_segment.get_array_of_samples(), dtype=float)
            else:
                samples = audio_segment
            
            # 정규화
            if len(samples) > 0:
                samples = samples / (np.max(np.abs(samples)) + 1e-6)
            
            features = []
            
            # 1. MFCC (20개 계수 + 델타 + 델타델타)
            mfccs = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=20)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            features.extend([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.mean(mfcc_delta, axis=1),
                np.std(mfcc_delta, axis=1),
                np.mean(mfcc_delta2, axis=1),
                np.std(mfcc_delta2, axis=1)
            ])
            
            # 2. 스펙트럴 특징
            spectral_centroids = librosa.feature.spectral_centroid(y=samples, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=samples, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sr)[0]
            
            features.extend([
                [np.mean(spectral_centroids), np.std(spectral_centroids)],
                [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
                [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
                np.mean(spectral_contrast, axis=1),
                np.std(spectral_contrast, axis=1)
            ])
            
            # 3. 음성 품질 특징
            zcr = librosa.feature.zero_crossing_rate(samples)[0]
            rmse = librosa.feature.rms(y=samples)[0]
            
            features.extend([
                [np.mean(zcr), np.std(zcr)],
                [np.mean(rmse), np.std(rmse)]
            ])
            
            # 4. 피치 특징 (더 정확한 방법)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                samples, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # 유효한 피치 값만 사용
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
            
            features.append(pitch_features)
            
            # 5. 포먼트 특징 (LPC 분석)
            try:
                # LPC 계수로 포먼트 추정
                lpc_order = 16
                lpc = librosa.lpc(samples, order=lpc_order)
                
                # 포먼트 주파수 추정
                roots = np.roots(lpc)
                roots = [r for r in roots if np.imag(r) >= 0]
                angles = np.arctan2(np.imag(roots), np.real(roots))
                formants = sorted(angles * (sr / (2 * np.pi)))[:4]  # 첫 4개 포먼트
                
                while len(formants) < 4:
                    formants.append(0)
                    
                features.append(formants)
            except:
                features.append([0, 0, 0, 0])
            
            # 6. 템포럴 특징
            # 에너지 엔벨로프의 변화율
            energy_envelope = np.convolve(samples**2, np.ones(int(0.01*sr))/int(0.01*sr), mode='same')
            energy_diff = np.diff(energy_envelope)
            features.append([np.mean(np.abs(energy_diff)), np.std(energy_diff)])
            
            # 모든 특징을 1차원 배열로 평탄화
            flat_features = []
            for f in features:
                if isinstance(f, list):
                    flat_features.extend(f)
                elif isinstance(f, np.ndarray):
                    flat_features.extend(f.flatten())
                else:
                    flat_features.append(f)
            
            return np.array(flat_features)
            
        except Exception as e:
            print(f"특징 추출 실패: {e}")
            # 기본 특징 벡터 반환
            return np.zeros(200)  # 대략적인 특징 수
    
    def segment_audio_advanced(self, audio_path, min_duration=1.0):
        """고급 음성 구간 검출"""
        try:
            # Silero VAD 사용
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            
            (get_speech_timestamps, save_audio, read_audio, 
             VADIterator, collect_chunks) = utils
            
            # 오디오 로드
            wav = read_audio(audio_path, sampling_rate=16000)
            
            # 음성 구간 감지 (더 세밀한 파라미터)
            speech_timestamps = get_speech_timestamps(
                wav, 
                model,
                sampling_rate=16000,
                threshold=0.4,  # 더 낮은 임계값
                min_speech_duration_ms=int(min_duration * 500),  # 최소 길이 줄임
                min_silence_duration_ms=200,  # 더 짧은 무음 허용
                window_size_samples=512,
                speech_pad_ms=30
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
            print(f"고급 VAD 실패: {e}")
            # 폴백
            return self.simple_vad(audio_path, min_duration)
    
    def cluster_speakers_spectral(self, features, num_speakers=None):
        """스펙트럴 클러스터링을 사용한 화자 구분"""
        n_samples = len(features)
        
        if n_samples < 2:
            return [0] * n_samples
        
        # 유사도 행렬 계산
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                # 코사인 유사도 사용
                similarity = 1 - cosine(features[i], features[j])
                similarity_matrix[i, j] = max(0, similarity)
        
        # 대각선은 1로 설정
        np.fill_diagonal(similarity_matrix, 1)
        
        if num_speakers is None:
            # 최적 화자 수 추정
            best_score = -1
            best_n = 2
            
            for n in range(2, min(6, n_samples)):
                try:
                    clustering = SpectralClustering(
                        n_clusters=n,
                        affinity='precomputed',
                        random_state=42
                    )
                    labels = clustering.fit_predict(similarity_matrix)
                    
                    if len(np.unique(labels)) == n:
                        score = silhouette_score(features, labels)
                        if score > best_score:
                            best_score = score
                            best_n = n
                except:
                    continue
            
            num_speakers = best_n
        
        # 최종 클러스터링
        clustering = SpectralClustering(
            n_clusters=num_speakers,
            affinity='precomputed',
            random_state=42,
            n_init=10
        )
        
        return clustering.fit_predict(similarity_matrix)
    
    def detect_speakers(self, video_path, min_duration=2.0, num_speakers=None, progress_callback=None):
        """개선된 화자 감지"""
        print("\n=== 고급 화자 감지 시작 ===")
        print("1/5 단계: 오디오 추출 중...")
            
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return None
        
        try:
            # 1. 음성 구간 검출
            print("2/5 단계: 음성 구간 검출 중... (Silero VAD)")
                
            segments = self.segment_audio_advanced(audio_path, min_duration)
            print(f"  → {len(segments)}개의 음성 구간 감지됨")
            
            if not segments:
                return []
            
            # 2. 각 구간에서 특징 추출
            print(f"3/5 단계: {len(segments)}개 구간에서 특징 추출 중...")
                
            audio = AudioSegment.from_wav(audio_path)
            features = []
            valid_segments = []
            
            for i, seg in enumerate(segments):
                if i % 10 == 0:
                    print(f"  → 진행중... {i}/{len(segments)} 구간 처리")
                    
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)
                
                if end_ms > start_ms:
                    audio_segment = audio[start_ms:end_ms]
                    feat = self.extract_advanced_features(audio_segment, audio.frame_rate)
                    
                    if feat is not None and len(feat) > 0:
                        features.append(feat)
                        valid_segments.append(seg)
            
            if not features:
                return []
            
            # 3. 특징 정규화
            print("4/5 단계: 화자 클러스터링 중...")
            features_array = np.array(features)
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(features_array)
            
            # 4. 화자 클러스터링
            labels = self.cluster_speakers_spectral(normalized_features, num_speakers)
            print(f"  → {len(np.unique(labels))}명의 화자로 분류됨")
            
            # 5. 결과 정리
            final_segments = []
            for seg, label in zip(valid_segments, labels):
                seg_copy = seg.copy()
                seg_copy['speaker'] = f'SPEAKER_{label}'
                final_segments.append(seg_copy)
            
            # 6. 후처리
            print("5/5 단계: 후처리 및 병합 중...")
            final_segments = self.post_process_segments(final_segments)
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
    
    def post_process_segments(self, segments):
        """세그먼트 후처리"""
        if not segments:
            return segments
        
        # 시간순 정렬
        segments = sorted(segments, key=lambda x: x['start'])
        
        # 1. 너무 짧은 구간 제거
        segments = [s for s in segments if s['duration'] >= 0.5]
        
        # 2. 같은 화자의 인접 구간 병합
        merged = []
        if segments:
            current = segments[0]
            
            for seg in segments[1:]:
                if (seg['speaker'] == current['speaker'] and 
                    seg['start'] - current['end'] < 1.0):
                    # 병합
                    current['end'] = seg['end']
                    current['duration'] = current['end'] - current['start']
                else:
                    merged.append(current)
                    current = seg
            
            merged.append(current)
        
        return merged
    
    def simple_vad(self, audio_path, min_duration):
        """폴백용 간단한 VAD"""
        audio = AudioSegment.from_wav(audio_path)
        
        segments = []
        chunk_size = 100  # 100ms
        energy_threshold = audio.dBFS + 10
        
        current_segment = None
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            
            if chunk.dBFS > energy_threshold:
                if current_segment is None:
                    current_segment = {'start': i / 1000.0}
            else:
                if current_segment is not None:
                    current_segment['end'] = i / 1000.0
                    current_segment['duration'] = current_segment['end'] - current_segment['start']
                    
                    if current_segment['duration'] >= min_duration:
                        segments.append(current_segment)
                    
                    current_segment = None
        
        # 마지막 구간 처리
        if current_segment is not None:
            current_segment['end'] = len(audio) / 1000.0
            current_segment['duration'] = current_segment['end'] - current_segment['start']
            if current_segment['duration'] >= min_duration:
                segments.append(current_segment)
        
        return segments