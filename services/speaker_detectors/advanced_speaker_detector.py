import os
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition, EncoderClassifier
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedSpeakerDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = None
        self.verification_model = None
        
    def initialize_models(self):
        """사전학습된 화자 임베딩 모델 로드"""
        try:
            # 더 간단한 임베딩 모델 사용
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="pretrained_models/spkrec-xvect-voxceleb"
            )
            
            return True
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            # 대체 방법: wav2vec2 기반 특징 추출
            try:
                from transformers import Wav2Vec2Model, Wav2Vec2Processor
                self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
                self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                self.use_wav2vec = True
                return True
            except:
                return False
    
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
    
    def segment_audio_with_vad(self, audio_path, min_duration=1.0):
        """고급 VAD를 사용한 음성 구간 검출"""
        try:
            # silero-vad 사용 (더 정확한 VAD)
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            
            (get_speech_timestamps, save_audio, read_audio, 
             VADIterator, collect_chunks) = utils
            
            # 오디오 로드
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
            print(f"VAD 실패: {e}")
            # 폴백으로 기본 VAD 사용
            return self.simple_vad(audio_path, min_duration)
    
    def extract_speaker_embeddings(self, audio_path, segments):
        """각 음성 구간에서 화자 임베딩 추출"""
        if not self.embedding_model:
            if not self.initialize_models():
                return None, None
        
        embeddings = []
        valid_segments = []
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 16kHz로 리샘플링
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # 모노로 변환
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            for seg in segments:
                start_sample = int(seg['start'] * 16000)
                end_sample = int(seg['end'] * 16000)
                
                if end_sample > start_sample and end_sample <= waveform.shape[1]:
                    segment_audio = waveform[:, start_sample:end_sample]
                    
                    # 임베딩 추출
                    with torch.no_grad():
                        # 배치 차원 추가
                        if segment_audio.dim() == 2:
                            segment_audio = segment_audio.unsqueeze(0)
                        
                        embedding = self.embedding_model.encode_batch(segment_audio)
                        
                        # numpy로 변환하고 차원 확인
                        emb_array = embedding.squeeze().cpu().numpy()
                        
                        # 1차원으로 평탄화
                        if emb_array.ndim > 1:
                            emb_array = emb_array.flatten()
                        
                        embeddings.append(emb_array)
                        valid_segments.append(seg)
            
            return embeddings, valid_segments
            
        except Exception as e:
            print(f"임베딩 추출 실패: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def cluster_speakers_advanced(self, embeddings, num_speakers=None):
        """고급 클러스터링 알고리즘"""
        if len(embeddings) < 2:
            return [0] * len(embeddings)
        
        # 임베딩 정규화
        embeddings_array = np.array(embeddings)
        
        # 차원 확인 및 수정
        if embeddings_array.ndim > 2:
            # 3차원 이상인 경우 2차원으로 reshape
            embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
        elif embeddings_array.ndim == 1:
            # 1차원인 경우 2차원으로 변환
            embeddings_array = embeddings_array.reshape(-1, 1)
        
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(embeddings_array)
        
        if num_speakers is None:
            # DBSCAN으로 자동 화자 수 감지
            clustering = DBSCAN(
                eps=0.5,
                min_samples=2,
                metric='cosine'
            )
            labels = clustering.fit_predict(normalized_embeddings)
            
            # 노이즈 처리 (-1 라벨)
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            
            num_speakers = len(unique_labels)
            
            # 노이즈를 가장 가까운 클러스터로 할당
            for i, label in enumerate(labels):
                if label == -1:
                    # 가장 가까운 클러스터 찾기
                    min_dist = float('inf')
                    best_label = 0
                    
                    for j, other_label in enumerate(labels):
                        if other_label != -1 and i != j:
                            dist = np.linalg.norm(
                                normalized_embeddings[i] - normalized_embeddings[j]
                            )
                            if dist < min_dist:
                                min_dist = dist
                                best_label = other_label
                    
                    labels[i] = best_label
        else:
            # Agglomerative Clustering으로 지정된 화자 수로 클러스터링
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                affinity='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(normalized_embeddings)
        
        return labels
    
    def refine_speaker_segments(self, segments, labels):
        """화자 라벨 후처리 및 정제"""
        # 시간순 정렬
        sorted_indices = sorted(range(len(segments)), key=lambda i: segments[i]['start'])
        
        # 스무딩: 짧은 구간 사이의 화자 변경 제거
        smoothed_labels = labels.copy()
        
        for i in range(1, len(sorted_indices) - 1):
            curr_idx = sorted_indices[i]
            prev_idx = sorted_indices[i-1]
            next_idx = sorted_indices[i+1]
            
            # 현재 구간이 짧고 앞뒤가 같은 화자인 경우
            if (segments[curr_idx]['duration'] < 2.0 and 
                labels[prev_idx] == labels[next_idx] and
                labels[curr_idx] != labels[prev_idx]):
                
                # 앞뒤 구간과의 시간 간격 확인
                gap_prev = segments[curr_idx]['start'] - segments[prev_idx]['end']
                gap_next = segments[next_idx]['start'] - segments[curr_idx]['end']
                
                if gap_prev < 1.0 and gap_next < 1.0:
                    smoothed_labels[curr_idx] = labels[prev_idx]
        
        return smoothed_labels
    
    def detect_speakers(self, video_path, min_duration=2.0, num_speakers=None):
        """고급 화자 감지"""
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return None
        
        try:
            # 1. VAD로 음성 구간 검출
            segments = self.segment_audio_with_vad(audio_path, min_duration)
            
            if not segments:
                return []
            
            # 2. 화자 임베딩 추출
            embeddings, valid_segments = self.extract_speaker_embeddings(audio_path, segments)
            
            if embeddings is None or not embeddings:
                print("임베딩 추출 실패, 기본 방법으로 폴백")
                # 폴백으로 기본 화자 감지 사용
                from .speaker_detector import SpeakerDetector
                basic_detector = SpeakerDetector()
                return basic_detector.detect_speakers(video_path, min_duration, num_speakers)
            
            # 3. 클러스터링
            labels = self.cluster_speakers_advanced(embeddings, num_speakers)
            
            # 4. 후처리
            refined_labels = self.refine_speaker_segments(valid_segments, labels)
            
            # 5. 결과 정리
            final_segments = []
            for seg, label in zip(valid_segments, refined_labels):
                final_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'speaker': f'SPEAKER_{label}',
                    'duration': seg['duration']
                })
            
            # 6. 같은 화자의 인접 구간 병합
            final_segments = self.merge_adjacent_segments(final_segments)
            
            # 임시 파일 삭제
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return final_segments
            
        except Exception as e:
            print(f"고급 화자 감지 실패: {e}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return None
    
    def merge_adjacent_segments(self, segments, max_gap=0.5):
        """인접한 같은 화자 구간 병합"""
        if not segments:
            return segments
        
        # 시간순 정렬
        segments = sorted(segments, key=lambda x: x['start'])
        
        merged = []
        current = segments[0].copy()
        
        for segment in segments[1:]:
            if (segment['speaker'] == current['speaker'] and 
                segment['start'] - current['end'] <= max_gap):
                # 병합
                current['end'] = segment['end']
                current['duration'] = current['end'] - current['start']
            else:
                merged.append(current)
                current = segment.copy()
        
        merged.append(current)
        return merged
    
    def simple_vad(self, audio_path, min_duration):
        """폴백용 간단한 VAD"""
        # 기존 simple_vad 구현 사용
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(audio_path)
        
        segments = []
        chunk_size = 100  # 100ms
        energy_threshold = audio.dBFS + 10
        
        current_segment = None
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            
            if chunk.dBFS > energy_threshold:
                if current_segment is None:
                    current_segment = {
                        'start': i / 1000.0
                    }
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