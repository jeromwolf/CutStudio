import os
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import numpy as np
import torch
import torchaudio
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import librosa
import librosa.display
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

class EnhancedSpeakerDetector:
    """고도화된 특징 추출을 사용하는 향상된 화자 감지기"""
    
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
    
    def extract_enhanced_features(self, audio_segment, sr=16000):
        """고도화된 음성 특징 추출"""
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
            
            # 1. 확장된 MFCC (40개 계수 + 델타 + 델타델타)
            mfccs = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=40)  # 20 -> 40으로 확대
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # MFCC 통계량 (평균, 표준편차, 최대, 최소, 중앙값)
            mfcc_features = []
            mfcc_features.extend(np.mean(mfccs, axis=1).tolist())
            mfcc_features.extend(np.std(mfccs, axis=1).tolist())
            mfcc_features.extend(np.max(mfccs, axis=1).tolist())
            mfcc_features.extend(np.min(mfccs, axis=1).tolist())
            mfcc_features.extend(np.median(mfccs, axis=1).tolist())
            mfcc_features.extend(np.mean(mfcc_delta, axis=1).tolist())
            mfcc_features.extend(np.std(mfcc_delta, axis=1).tolist())
            mfcc_features.extend(np.mean(mfcc_delta2, axis=1).tolist())
            mfcc_features.extend(np.std(mfcc_delta2, axis=1).tolist())
            features.append(mfcc_features)
            
            # 2. 프로소디 특징 (Prosodic Features)
            # 2.1 피치 관련 특징
            f0, voiced_flag, voiced_probs = librosa.pyin(
                samples, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # 유효한 피치 값만 사용
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                # 기본 피치 통계
                pitch_features = [
                    np.mean(valid_f0),
                    np.std(valid_f0),
                    np.percentile(valid_f0, 10),
                    np.percentile(valid_f0, 25),
                    np.percentile(valid_f0, 50),
                    np.percentile(valid_f0, 75),
                    np.percentile(valid_f0, 90),
                    np.max(valid_f0) - np.min(valid_f0),  # 피치 범위
                ]
                
                # 피치 변화율 (기울기)
                pitch_diff = np.diff(valid_f0)
                pitch_features.extend([
                    np.mean(np.abs(pitch_diff)),  # 평균 변화율
                    np.std(pitch_diff),            # 변화율 표준편차
                    np.percentile(np.abs(pitch_diff), 90)  # 급격한 변화
                ])
                
                # 피치 컨투어 특징
                if len(valid_f0) > 5:
                    # 피치 기울기 (전체적인 상승/하강 경향)
                    time_indices = np.arange(len(valid_f0))
                    pitch_slope = np.polyfit(time_indices, valid_f0, 1)[0]
                    pitch_features.append(pitch_slope)
                else:
                    pitch_features.append(0)
                    
            else:
                pitch_features = [0] * 12
            
            features.append(pitch_features)
            
            # 2.2 에너지/인텐시티 특징
            rmse = librosa.feature.rms(y=samples)[0]
            energy_features = [
                np.mean(rmse),
                np.std(rmse),
                np.max(rmse),
                np.percentile(rmse, 90),
                np.mean(np.diff(rmse)),  # 에너지 변화율
            ]
            features.append(energy_features)
            
            # 2.3 리듬 특징
            tempo, beats = librosa.beat.beat_track(y=samples, sr=sr)
            onset_env = librosa.onset.onset_strength(y=samples, sr=sr)
            
            rhythm_features = [
                tempo,
                len(beats) / (len(samples) / sr) if len(samples) > 0 else 0,  # 비트 밀도
                np.mean(onset_env),
                np.std(onset_env),
                np.max(onset_env)
            ]
            features.append(rhythm_features)
            
            # 3. 장기 스펙트럴 특징 (Long-term Spectral Features)
            # 3.1 스펙트럴 통계량
            spectral_centroids = librosa.feature.spectral_centroid(y=samples, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=samples, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sr)[0]
            spectral_flatness = librosa.feature.spectral_flatness(y=samples)[0]
            
            # 각 스펙트럴 특징의 시간에 따른 변화 패턴
            spectral_features = []
            
            # Spectral Centroid 통계 + 변화 패턴
            spectral_features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.percentile(spectral_centroids, 25),
                np.percentile(spectral_centroids, 75),
                np.mean(np.diff(spectral_centroids)),  # 변화율
                np.std(np.diff(spectral_centroids))    # 변화 안정성
            ])
            
            # Spectral Rolloff
            spectral_features.extend([
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.percentile(spectral_rolloff, 90)
            ])
            
            # Spectral Bandwidth
            spectral_features.extend([
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth),
                np.max(spectral_bandwidth) - np.min(spectral_bandwidth)
            ])
            
            # Spectral Contrast (각 주파수 대역별)
            spectral_features.extend(np.mean(spectral_contrast, axis=1).tolist())
            spectral_features.extend(np.std(spectral_contrast, axis=1).tolist())
            
            # Spectral Flatness (음성의 톤 특성)
            spectral_features.extend([
                np.mean(spectral_flatness),
                np.std(spectral_flatness)
            ])
            
            features.append(spectral_features)
            
            # 3.2 Mel-frequency 스펙트럼 특징
            mel_spectrogram = librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=128)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
            
            # Mel 스펙트럼의 시간축 평균과 주파수축 평균
            mel_time_mean = np.mean(log_mel_spectrogram, axis=0)
            mel_freq_mean = np.mean(log_mel_spectrogram, axis=1)
            
            mel_features = [
                np.mean(mel_time_mean),
                np.std(mel_time_mean),
                np.mean(mel_freq_mean[:64]),  # 저주파 대역
                np.mean(mel_freq_mean[64:])   # 고주파 대역
            ]
            features.append(mel_features)
            
            # 4. 포먼트 특징 (개선된 추출)
            # Pre-emphasis filter
            pre_emphasized = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
            
            # LPC를 이용한 포먼트 추출
            try:
                lpc_order = 16
                lpc = librosa.lpc(pre_emphasized, order=lpc_order)
                
                # 포먼트 주파수 계산
                roots = np.roots(lpc)
                roots = [r for r in roots if np.imag(r) >= 0]
                angles = np.arctan2(np.imag(roots), np.real(roots))
                formants = sorted(angles * (sr / (2 * np.pi)))
                
                # 첫 5개 포먼트 추출
                formant_features = []
                for i in range(5):
                    if i < len(formants):
                        formant_features.append(formants[i])
                    else:
                        formant_features.append(0)
                
                # 포먼트 비율 (화자 특성)
                if formants[0] > 0:
                    formant_features.append(formants[1] / formants[0])  # F2/F1
                    formant_features.append(formants[2] / formants[0])  # F3/F1
                else:
                    formant_features.extend([0, 0])
                    
            except:
                formant_features = [0] * 7
            
            features.append(formant_features)
            
            # 5. Zero Crossing Rate (ZCR) 특징
            zcr = librosa.feature.zero_crossing_rate(samples)[0]
            zcr_features = [
                np.mean(zcr),
                np.std(zcr),
                np.percentile(zcr, 90),
                np.mean(np.diff(zcr))  # ZCR 변화율
            ]
            features.append(zcr_features)
            
            # 6. 음성 품질 특징
            # 6.1 Harmonics-to-Noise Ratio (HNR) 추정
            # 자기상관 함수를 이용한 간단한 HNR 추정
            autocorr = np.correlate(samples, samples, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # 첫 번째 피크 찾기 (기본 주파수)
            if len(valid_f0) > 0:
                avg_period = int(sr / np.mean(valid_f0))
                if avg_period < len(autocorr):
                    hnr_estimate = autocorr[avg_period] / (autocorr[0] + 1e-6)
                else:
                    hnr_estimate = 0
            else:
                hnr_estimate = 0
            
            features.append([hnr_estimate])
            
            # 7. 시간 도메인 특징
            # 7.1 Short-time Energy 변화
            frame_length = int(0.025 * sr)  # 25ms
            hop_length = int(0.010 * sr)    # 10ms
            
            frames = librosa.util.frame(samples, frame_length=frame_length, hop_length=hop_length)
            frame_energy = np.sum(frames**2, axis=0)
            
            energy_variation = [
                np.std(frame_energy) / (np.mean(frame_energy) + 1e-6),  # 변동 계수
                np.percentile(frame_energy, 90) / (np.percentile(frame_energy, 10) + 1e-6)  # 동적 범위
            ]
            features.append(energy_variation)
            
            # 모든 특징을 1차원 배열로 평탄화
            flat_features = []
            for f in features:
                if isinstance(f, list):
                    for item in f:
                        if isinstance(item, (np.ndarray, list)):
                            # 재귀적으로 평탄화
                            if isinstance(item, np.ndarray):
                                flat_features.extend(item.flatten().tolist())
                            else:
                                flat_features.extend(item)
                        else:
                            flat_features.append(float(item))
                elif isinstance(f, np.ndarray):
                    flat_features.extend(f.flatten().tolist())
                else:
                    flat_features.append(float(f))
            
            return np.array(flat_features, dtype=np.float32)
            
        except Exception as e:
            print(f"특징 추출 실패: {e}")
            import traceback
            traceback.print_exc()
            # 예상되는 특징 수만큼 0으로 채운 배열 반환
            return np.zeros(400, dtype=np.float32)  # 대략적인 특징 수
    
    def segment_audio_multi_stage(self, audio_path, min_duration=1.0):
        """다단계 음성 구간 검출"""
        try:
            print("다단계 VAD 시작...")
            
            # 1단계: Silero VAD
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
            
            # 음성 구간 감지 (더 민감한 설정)
            speech_timestamps = get_speech_timestamps(
                wav, 
                model,
                sampling_rate=16000,
                threshold=0.2,  # 더욱 민감하게 (0.3 -> 0.2)
                min_speech_duration_ms=500,  # 최소 0.5초 (기존 min_duration * 700)
                min_silence_duration_ms=300,  # 무음 구간 300ms
                window_size_samples=1024,  # 윈도우 크기 증가
                speech_pad_ms=100  # 패딩 증가
            )
            
            # 디버깅 정보
            print(f"Silero VAD 결과: {len(speech_timestamps)}개 구간 감지")
            
            # 2단계: 에너지 기반 필터링
            audio = AudioSegment.from_wav(audio_path)
            filtered_segments = []
            
            for ts in speech_timestamps:
                start_ms = int(ts['start'] / 16000 * 1000)  # 수정: 16 -> 16000
                end_ms = int(ts['end'] / 16000 * 1000)  # 수정: 16 -> 16000
                
                if end_ms > start_ms:
                    segment = audio[start_ms:end_ms]
                    
                    # 에너지가 충분한 구간만 선택 (기준 완화)
                    # 전체 오디오의 평균 에너지보다 30dB 낮은 것까지 허용
                    if segment.dBFS > audio.dBFS - 30:  # 20 -> 30으로 완화
                        filtered_segments.append({
                            'start': ts['start'] / 16000,
                            'end': ts['end'] / 16000,
                            'duration': (ts['end'] - ts['start']) / 16000
                        })
                    else:
                        print(f"에너지 부족으로 제외: {segment.dBFS:.1f} dBFS (기준: {audio.dBFS - 30:.1f})")
            
            print(f"검출된 음성 구간: {len(filtered_segments)}개")
            
            # 만약 음성 구간이 검출되지 않으면 더 간단한 방법 시도
            if len(filtered_segments) == 0:
                print("음성 구간이 검출되지 않음. 대체 방법 시도...")
                return self.simple_vad_enhanced(audio_path, min_duration)
                
            return filtered_segments
            
        except Exception as e:
            print(f"다단계 VAD 실패: {e}")
            return self.simple_vad(audio_path, min_duration)
    
    def adaptive_clustering(self, features, min_speakers=2, max_speakers=10):
        """적응형 클러스터링으로 최적 화자 수 결정"""
        n_samples = len(features)
        
        if n_samples < min_speakers:
            return [0] * n_samples, 1
        
        # 특징 정규화
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        best_score = -np.inf
        best_n_speakers = min_speakers
        best_labels = None
        
        scores = []
        
        # 여러 화자 수에 대해 테스트
        for n_speakers in range(min_speakers, min(max_speakers + 1, n_samples)):
            try:
                # Gaussian Mixture Model 사용
                gmm = GaussianMixture(
                    n_components=n_speakers,
                    covariance_type='diag',
                    n_init=3,
                    random_state=42
                )
                labels = gmm.fit_predict(normalized_features)
                
                # 여러 지표를 종합하여 평가
                if len(np.unique(labels)) == n_speakers:
                    # BIC (Bayesian Information Criterion) - 낮을수록 좋음
                    bic = gmm.bic(normalized_features)
                    
                    # Silhouette Score - 높을수록 좋음
                    silhouette = silhouette_score(normalized_features, labels)
                    
                    # Davies-Bouldin Score - 낮을수록 좋음
                    davies_bouldin = davies_bouldin_score(normalized_features, labels)
                    
                    # 종합 점수 계산 (정규화된 점수들의 가중 평균)
                    # BIC는 음수로 변환하여 높을수록 좋게 만듦
                    combined_score = (
                        0.4 * silhouette +  # 40%
                        0.3 * (-bic / 10000) +  # 30% (스케일 조정)
                        0.3 * (1 / (1 + davies_bouldin))  # 30%
                    )
                    
                    scores.append({
                        'n_speakers': n_speakers,
                        'score': combined_score,
                        'silhouette': silhouette,
                        'bic': bic,
                        'davies_bouldin': davies_bouldin
                    })
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_n_speakers = n_speakers
                        best_labels = labels.copy()
                        
            except:
                continue
        
        # 점수 출력 (디버깅용)
        if scores:
            print("\n클러스터링 점수:")
            for s in scores:
                print(f"화자 {s['n_speakers']}명: 종합={s['score']:.3f}, "
                      f"실루엣={s['silhouette']:.3f}, BIC={s['bic']:.1f}, "
                      f"DB={s['davies_bouldin']:.3f}")
        
        print(f"\n최적 화자 수: {best_n_speakers}명")
        
        return best_labels if best_labels is not None else [0] * n_samples, best_n_speakers
    
    def detect_speaker_changes(self, features, window_size=5):
        """화자 변경점 감지"""
        n_samples = len(features)
        if n_samples < window_size * 2:
            return []
        
        change_points = []
        
        # 슬라이딩 윈도우로 특징 비교
        for i in range(window_size, n_samples - window_size):
            # 이전 윈도우와 다음 윈도우의 평균 특징
            prev_window = np.mean(features[i-window_size:i], axis=0)
            next_window = np.mean(features[i:i+window_size], axis=0)
            
            # 코사인 거리 계산
            distance = cosine(prev_window, next_window)
            
            # 임계값 이상이면 변경점으로 표시
            if distance > 0.3:  # 임계값은 조정 가능
                change_points.append(i)
        
        # 너무 가까운 변경점들 병합
        filtered_change_points = []
        for i, cp in enumerate(change_points):
            if i == 0 or cp - change_points[i-1] > window_size:
                filtered_change_points.append(cp)
        
        return filtered_change_points
    
    def post_process_advanced(self, segments, features, labels):
        """고급 후처리"""
        if not segments or not features:
            return segments
        
        # 1. 화자별 평균 임베딩 계산
        speaker_embeddings = {}
        for i, label in enumerate(labels):
            if label not in speaker_embeddings:
                speaker_embeddings[label] = []
            speaker_embeddings[label].append(features[i])
        
        # 평균 계산
        for speaker in speaker_embeddings:
            speaker_embeddings[speaker] = np.mean(speaker_embeddings[speaker], axis=0)
        
        # 2. 각 세그먼트의 화자 재검증
        refined_segments = []
        for i, (seg, feature, label) in enumerate(zip(segments, features, labels)):
            # 현재 세그먼트와 각 화자의 거리 계산
            distances = {}
            for speaker, embedding in speaker_embeddings.items():
                distances[speaker] = cosine(feature, embedding)
            
            # 가장 가까운 화자 선택
            best_speaker = min(distances, key=distances.get)
            confidence = 1 - distances[best_speaker]
            
            seg_copy = seg.copy()
            seg_copy['speaker'] = f'SPEAKER_{best_speaker}'
            seg_copy['confidence'] = confidence
            refined_segments.append(seg_copy)
        
        # 3. 시간순 정렬 및 병합
        refined_segments = sorted(refined_segments, key=lambda x: x['start'])
        
        # 4. 동적 병합 (임베딩 거리 기반)
        merged = []
        if refined_segments:
            current = refined_segments[0]
            
            for seg in refined_segments[1:]:
                # 같은 화자이고 시간적으로 가까운 경우
                if (seg['speaker'] == current['speaker'] and 
                    seg['start'] - current['end'] < 1.5):
                    
                    # 추가로 신뢰도가 충분히 높은 경우만 병합
                    if current.get('confidence', 0) > 0.7 and seg.get('confidence', 0) > 0.7:
                        current['end'] = seg['end']
                        current['duration'] = current['end'] - current['start']
                        current['confidence'] = (current['confidence'] + seg['confidence']) / 2
                    else:
                        merged.append(current)
                        current = seg
                else:
                    merged.append(current)
                    current = seg
            
            merged.append(current)
        
        # 5. 짧은 구간 처리
        final_segments = []
        for i, seg in enumerate(merged):
            if seg['duration'] < 1.0 and seg.get('confidence', 0) < 0.6:
                # 앞뒤 세그먼트 확인
                if i > 0 and i < len(merged) - 1:
                    prev_speaker = merged[i-1]['speaker']
                    next_speaker = merged[i+1]['speaker']
                    
                    # 앞뒤가 같은 화자면 그 화자로 할당
                    if prev_speaker == next_speaker:
                        seg['speaker'] = prev_speaker
                        seg['confidence'] = 0.5
            
            final_segments.append(seg)
        
        return final_segments
    
    def detect_speakers(self, video_path, min_duration=1.0, num_speakers=None, progress_callback=None):
        """향상된 화자 감지"""
        print("\n=== 향상된 화자 감지 시작 (고도화된 특징 추출) ===")
        print("1/6 단계: 오디오 추출 중...")
        
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return None
        
        try:
            # 1. 다단계 음성 구간 검출
            print("2/6 단계: 다단계 음성 구간 검출 중...")
            segments = self.segment_audio_multi_stage(audio_path, min_duration)
            print(f"  → {len(segments)}개의 음성 구간 감지됨")
            
            if not segments:
                return []
            
            # 2. 고도화된 특징 추출
            print(f"3/6 단계: {len(segments)}개 구간에서 고급 특징 추출 중...")
            print("  (MFCC 40개, 프로소디, 장기 스펙트럴 특징 등)")
            
            audio = AudioSegment.from_wav(audio_path)
            features = []
            valid_segments = []
            
            for i, seg in enumerate(segments):
                if i % 5 == 0:
                    print(f"  → 진행중... {i}/{len(segments)} 구간 처리")
                    if progress_callback:
                        progress_callback(30 + (30 * i / len(segments)))
                
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)
                
                if end_ms > start_ms:
                    audio_segment = audio[start_ms:end_ms]
                    feat = self.extract_enhanced_features(audio_segment, audio.frame_rate)
                    
                    if feat is not None and len(feat) > 0:
                        features.append(feat)
                        valid_segments.append(seg)
            
            if not features:
                return []
            
            print(f"  → {len(features)}개 구간에서 특징 추출 완료")
            
            # 3. 화자 변경점 감지
            print("4/6 단계: 화자 변경점 감지 중...")
            change_points = self.detect_speaker_changes(features)
            print(f"  → {len(change_points)}개 변경점 감지됨")
            
            # 4. 적응형 클러스터링
            print("5/6 단계: 적응형 화자 클러스터링 중...")
            features_array = np.array(features)
            
            if num_speakers is None:
                labels, detected_speakers = self.adaptive_clustering(features_array)
                print(f"  → 자동 감지된 화자 수: {detected_speakers}명")
            else:
                labels, _ = self.adaptive_clustering(
                    features_array, 
                    min_speakers=num_speakers, 
                    max_speakers=num_speakers
                )
                print(f"  → 지정된 화자 수: {num_speakers}명")
            
            # 5. 고급 후처리
            print("6/6 단계: 고급 후처리 및 최적화 중...")
            
            # 세그먼트에 라벨 할당
            for seg, label in zip(valid_segments, labels):
                seg['speaker'] = f'SPEAKER_{label}'
            
            # 고급 후처리 적용
            final_segments = self.post_process_advanced(valid_segments, features, labels)
            
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
    
    def simple_vad_enhanced(self, audio_path, min_duration):
        """향상된 폴백용 VAD - 더 민감한 설정"""
        audio = AudioSegment.from_wav(audio_path)
        
        # 전체 오디오 분석
        print(f"오디오 정보: 길이={len(audio)/1000:.1f}초, 평균에너지={audio.dBFS:.1f}dBFS")
        
        segments = []
        chunk_size = 50  # 50ms로 더 세밀하게
        
        # 동적 임계값 설정
        chunks_energy = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) > 0:
                chunks_energy.append(chunk.dBFS)
        
        if chunks_energy:
            # 에너지 분포의 25 percentile을 임계값으로 사용
            energy_threshold = np.percentile([e for e in chunks_energy if e > -float('inf')], 25)
            print(f"동적 에너지 임계값: {energy_threshold:.1f}dBFS")
        else:
            energy_threshold = audio.dBFS - 10
        
        current_segment = None
        silence_tolerance = 500  # 500ms까지 무음 허용
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            
            if chunk.dBFS > energy_threshold:
                if current_segment is None:
                    current_segment = {
                        'start': i / 1000.0,
                        'last_speech': i
                    }
                else:
                    current_segment['last_speech'] = i
            else:
                if current_segment is not None:
                    # 무음이 너무 길면 세그먼트 종료
                    if i - current_segment['last_speech'] > silence_tolerance:
                        current_segment['end'] = current_segment['last_speech'] / 1000.0
                        current_segment['duration'] = current_segment['end'] - current_segment['start']
                        
                        if current_segment['duration'] >= min_duration:
                            segments.append({
                                'start': current_segment['start'],
                                'end': current_segment['end'],
                                'duration': current_segment['duration']
                            })
                        
                        current_segment = None
        
        # 마지막 구간 처리
        if current_segment is not None:
            current_segment['end'] = len(audio) / 1000.0
            current_segment['duration'] = current_segment['end'] - current_segment['start']
            if current_segment['duration'] >= min_duration:
                segments.append({
                    'start': current_segment['start'],
                    'end': current_segment['end'],
                    'duration': current_segment['duration']
                })
        
        print(f"향상된 simple VAD: {len(segments)}개 구간 검출")
        return segments
    
    def simple_vad(self, audio_path, min_duration):
        """폴백용 간단한 VAD"""
        return self.simple_vad_enhanced(audio_path, min_duration)