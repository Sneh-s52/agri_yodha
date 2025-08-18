"""
Audio processor for handling audio-based queries.
Uses speech recognition and GPT-3.5-turbo for processing.
"""

import io
import os
import wave
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import openai
import librosa
import numpy as np
from pydub import AudioSegment
from .base_processor import BaseProcessor, ProcessingResult

class AudioProcessor(BaseProcessor):
    """Processes audio queries and extracts spoken information."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o"):
        super().__init__(model_name=model_name)
        self.api_key = api_key
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        # Initialize OpenAI client
        openai_key = api_key or os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                openai.api_key = openai_key
                self.client = openai.OpenAI(api_key=openai_key)
                self.logger.info("✅ OpenAI client initialized successfully")
            except Exception as e:
                self.logger.warning(f"❌ Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            self.client = None
    
    def can_process(self, input_data: Union[str, bytes, Path]) -> bool:
        """Check if input is audio-based."""
        if isinstance(input_data, bytes):
            try:
                # Try to load as audio
                AudioSegment.from_file(io.BytesIO(input_data))
                return True
            except:
                return False
        elif isinstance(input_data, Path):
            return input_data.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        elif isinstance(input_data, str):
            # Check if it's a file path
            path = Path(input_data)
            return path.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        return False
    
    def process(self, input_data: Union[str, bytes, Path], **kwargs) -> ProcessingResult:
        """Process audio input and extract spoken information."""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")
        
        # Load audio
        audio = self._load_audio(input_data)
        
        # Extract metadata
        metadata = self.extract_metadata(input_data)
        metadata.update({
            "duration": len(audio) / 1000.0,  # Convert to seconds
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "bit_depth": audio.sample_width * 8
        })
        
        # Transcribe audio
        transcription = self._transcribe_audio(audio, **kwargs)
        
        # Process transcription with GPT-3.5-turbo
        processed_content = self._process_transcription(transcription, **kwargs)
        
        return ProcessingResult(
            content=processed_content,
            metadata=metadata,
            confidence=0.8,  # Moderate confidence for audio processing
            modality="audio",
            processed_data={
                "transcription": transcription,
                "audio_features": self._extract_audio_features(audio),
                "speech_analysis": self._analyze_speech(audio),
                "language_detection": self._detect_language(transcription)
            }
        )
    
    def _load_audio(self, input_data: Union[str, bytes, Path]) -> AudioSegment:
        """Load audio from various input types."""
        if isinstance(input_data, bytes):
            return AudioSegment.from_file(io.BytesIO(input_data))
        elif isinstance(input_data, Path):
            return AudioSegment.from_file(str(input_data))
        elif isinstance(input_data, str):
            # Assume it's a file path
            return AudioSegment.from_file(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _transcribe_audio(self, audio: AudioSegment, **kwargs) -> str:
        """Transcribe audio using OpenAI Whisper API."""
        if not self.client:
            self.logger.warning("No OpenAI client available, using basic audio analysis")
            return self._basic_audio_analysis(audio)
            
        try:
            # Convert to WAV format for API and ensure proper format
            buffer = io.BytesIO()
            # Ensure mono channel and 16kHz sample rate for best results
            audio_processed = audio.set_channels(1).set_frame_rate(16000)
            audio_processed.export(buffer, format="wav")
            buffer.seek(0)
            buffer.name = "audio.wav"  # Give the buffer a filename for OpenAI API
            
            # Use OpenAI Whisper API
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=buffer,
                response_format="text"
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return self._basic_audio_analysis(audio)
    
    def _process_transcription(self, transcription: str, **kwargs) -> str:
        """Process transcription with GPT-3.5-turbo."""
        if not transcription or transcription.strip() == "":
            return "No speech detected in audio."
        
        if not self.client:
            self.logger.warning("No OpenAI client available, returning transcription as-is")
            return transcription
        
        try:
            task = kwargs.get('task', 'general')
            prompts = {
                'general': f"""
                Analyze this transcribed speech and provide a structured summary:
                
                Transcription: {transcription}
                
                Please provide:
                1. Main topic or subject
                2. Key points or questions
                3. Speaker's intent or request
                4. Any specific details or context
                
                Format as a clear, structured summary.
                """,
                'question': f"""
                Analyze this transcribed question:
                
                Question: {transcription}
                
                Please identify:
                1. Question type (what, how, why, when, where, who)
                2. Main subject/topic
                3. Specific requirements
                4. Expected answer format
                
                Format as a structured question analysis.
                """,
                'command': f"""
                Analyze this transcribed command:
                
                Command: {transcription}
                
                Please identify:
                1. Action required
                2. Target or subject
                3. Parameters or conditions
                4. Expected outcome
                
                Format as a structured command analysis.
                """
            }
            
            prompt = prompts.get(task, prompts['general'])
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that processes transcribed speech for further analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error processing transcription: {e}")
            return transcription
    
    def _basic_audio_analysis(self, audio: AudioSegment) -> str:
        """Fallback audio analysis without transcription."""
        return f"""
        Basic Audio Analysis:
        - Duration: {len(audio) / 1000.0:.2f} seconds
        - Sample rate: {audio.frame_rate} Hz
        - Channels: {audio.channels}
        - Bit depth: {audio.sample_width * 8} bits
        - Average volume: {audio.dBFS:.2f} dBFS
        """
    
    def _extract_audio_features(self, audio: AudioSegment) -> Dict[str, Any]:
        """Extract audio features for analysis."""
        try:
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            features = {
                "duration": float(len(audio) / 1000.0),
                "sample_rate": int(audio.frame_rate),
                "channels": int(audio.channels),
                "bit_depth": int(audio.sample_width * 8),
                "average_volume": float(audio.dBFS if audio.dBFS != float('-inf') else -60.0),
                "max_volume": int(audio.max_possible_amplitude),
                "rms": float(np.sqrt(np.mean(samples**2))),
                "zero_crossings": int(np.sum(np.diff(np.sign(samples)) != 0)),
                "spectral_centroid": self._calculate_spectral_centroid(samples, audio.frame_rate),
                "spectral_rolloff": self._calculate_spectral_rolloff(samples, audio.frame_rate)
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting audio features: {e}")
            return {}
    
    def _analyze_speech(self, audio: AudioSegment) -> Dict[str, Any]:
        """Analyze speech characteristics."""
        try:
            # Convert to mono for analysis
            if audio.channels > 1:
                audio_mono = audio.set_channels(1)
            else:
                audio_mono = audio
            
            samples = np.array(audio_mono.get_array_of_samples())
            
            analysis = {
                "speech_detected": self._detect_speech(samples, audio.frame_rate),
                "speaking_rate": self._estimate_speaking_rate(samples, audio.frame_rate),
                "pitch_range": self._analyze_pitch_range(samples, audio.frame_rate),
                "energy_distribution": self._analyze_energy_distribution(samples)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing speech: {e}")
            return {}
    
    def _detect_language(self, transcription: str) -> str:
        """Detect language from transcription (basic implementation)."""
        # This is a basic implementation - in practice, you'd use a language detection library
        # For now, we'll use simple heuristics
        if not transcription:
            return "unknown"
        
        # Simple language detection based on common words
        text_lower = transcription.lower()
        
        # English indicators
        english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
        english_count = sum(1 for word in english_words if word in text_lower)
        
        # Spanish indicators
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no']
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        
        # French indicators
        french_words = ['le', 'la', 'de', 'et', 'en', 'un', 'est', 'que', 'il', 'ne']
        french_count = sum(1 for word in french_words if word in text_lower)
        
        if english_count > spanish_count and english_count > french_count:
            return "english"
        elif spanish_count > english_count and spanish_count > french_count:
            return "spanish"
        elif french_count > english_count and french_count > spanish_count:
            return "french"
        else:
            return "unknown"
    
    def _calculate_spectral_centroid(self, samples: np.ndarray, sample_rate: int) -> float:
        """Calculate spectral centroid."""
        try:
            # Use librosa for spectral analysis
            spectral_centroids = librosa.feature.spectral_centroid(y=samples.astype(float), sr=sample_rate)
            return float(np.mean(spectral_centroids))
        except Exception as e:
            self.logger.error(f"Error calculating spectral centroid: {e}")
            return 0.0
    
    def _calculate_spectral_rolloff(self, samples: np.ndarray, sample_rate: int) -> float:
        """Calculate spectral rolloff."""
        try:
            spectral_rolloff = librosa.feature.spectral_rolloff(y=samples.astype(float), sr=sample_rate)
            return float(np.mean(spectral_rolloff))
        except Exception as e:
            self.logger.error(f"Error calculating spectral rolloff: {e}")
            return 0.0
    
    def _detect_speech(self, samples: np.ndarray, sample_rate: int) -> bool:
        """Detect if audio contains speech."""
        try:
            # Simple speech detection based on energy and frequency characteristics
            # This is a basic implementation - in practice, you'd use more sophisticated methods
            
            # Calculate energy
            energy = float(np.mean(samples**2))
            
            # Calculate zero crossing rate
            zero_crossings = int(np.sum(np.diff(np.sign(samples)) != 0))
            
            # Speech typically has moderate energy and moderate zero crossing rate
            return energy > 1000 and 100 < zero_crossings < 10000
            
        except Exception as e:
            self.logger.error(f"Error detecting speech: {e}")
            return False
    
    def _estimate_speaking_rate(self, samples: np.ndarray, sample_rate: int) -> float:
        """Estimate speaking rate in words per minute."""
        try:
            # This is a simplified estimation
            # In practice, you'd use more sophisticated methods
            
            # Calculate energy envelope
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            energy = []
            for i in range(0, len(samples) - frame_length, hop_length):
                frame = samples[i:i + frame_length]
                energy.append(np.mean(frame**2))
            
            # Count energy peaks (potential syllables)
            threshold = float(np.mean(energy)) * 1.5
            peaks = sum(1 for e in energy if e > threshold)
            
            # Estimate words per minute (rough approximation)
            duration = float(len(samples)) / float(sample_rate)
            syllables_per_second = float(peaks) / duration
            words_per_minute = syllables_per_second * 60 * 0.7  # Assume 0.7 words per syllable
            
            return float(min(max(words_per_minute, 60), 300))  # Clamp between 60-300 WPM
            
        except Exception as e:
            self.logger.error(f"Error estimating speaking rate: {e}")
            return 150.0  # Default to average speaking rate
    
    def _analyze_pitch_range(self, samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Analyze pitch range of speech."""
        try:
            # Use librosa for pitch analysis
            pitches, magnitudes = librosa.piptrack(y=samples.astype(float), sr=sample_rate)
            
            # Get pitch values where magnitude is above threshold
            threshold = np.percentile(magnitudes, 90)
            valid_pitches = pitches[magnitudes > threshold]
            
            if len(valid_pitches) > 0:
                return {
                    "min_pitch": float(np.min(valid_pitches)),
                    "max_pitch": float(np.max(valid_pitches)),
                    "mean_pitch": float(np.mean(valid_pitches)),
                    "pitch_range": float(np.max(valid_pitches) - np.min(valid_pitches))
                }
            else:
                return {"min_pitch": 0.0, "max_pitch": 0.0, "mean_pitch": 0.0, "pitch_range": 0.0}
                
        except Exception as e:
            self.logger.error(f"Error analyzing pitch range: {e}")
            return {"min_pitch": 0.0, "max_pitch": 0.0, "mean_pitch": 0.0, "pitch_range": 0.0}
    
    def _analyze_energy_distribution(self, samples: np.ndarray) -> Dict[str, float]:
        """Analyze energy distribution in audio."""
        try:
            # Calculate energy in different frequency bands
            frame_length = int(0.025 * 16000)  # 25ms frames at 16kHz
            
            # Simple frequency band analysis
            low_energy = np.mean(samples[:len(samples)//3]**2)
            mid_energy = np.mean(samples[len(samples)//3:2*len(samples)//3]**2)
            high_energy = np.mean(samples[2*len(samples)//3:]**2)
            
            total_energy = low_energy + mid_energy + high_energy
            
            return {
                "low_freq_energy": float(low_energy / total_energy if total_energy > 0 else 0),
                "mid_freq_energy": float(mid_energy / total_energy if total_energy > 0 else 0),
                "high_freq_energy": float(high_energy / total_energy if total_energy > 0 else 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing energy distribution: {e}")
            return {"low_freq_energy": 0.33, "mid_freq_energy": 0.33, "high_freq_energy": 0.34}
