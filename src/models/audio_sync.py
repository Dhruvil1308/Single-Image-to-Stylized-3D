import os
import asyncio
import numpy as np

# Suppress librosa future warnings if any
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import edge_tts
    import librosa
except ImportError:
    pass

class AudioSynchronizer:
    def __init__(self, voice="en-US-ChristopherNeural"):
        self.voice = voice
    
    async def _generate_audio(self, text, output_file):
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_file)

    def generate_speech(self, text, output_file):
        """Generates TTS audio using edge-tts."""
        asyncio.run(self._generate_audio(text, output_file))
        return output_file

    def extract_envelope(self, audio_path, fps=30, max_mouth_open=2.5):
        """
        Extracts the amplitude envelope from the audio file.
        Returns an array of mouth-open parameter values for each frame.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        y, sr = librosa.load(audio_path, sr=None)
        
        # We want `fps` frames per second of audio
        samples_per_frame = int(sr / fps)
        
        # Calculate RMS energy per frame
        rms = librosa.feature.rms(
            y=y, 
            frame_length=samples_per_frame, 
            hop_length=samples_per_frame
        )[0]
        
        # Smooth the envelope with a wider window for natural transitions
        window_size = 5
        window = np.ones(window_size) / window_size
        rms_smoothed = np.convolve(rms, window, mode='same')
        
        # Normalize between 0 and 1
        max_val = rms_smoothed.max()
        if max_val > 0.001:
            rms_smoothed = rms_smoothed / max_val
        else:
            rms_smoothed = np.zeros_like(rms_smoothed)
            
        # Non-linear mapping: ease-in/ease-out for natural mouth movement
        # Quiet sounds barely open, loud sounds open wide
        envelope = np.power(rms_smoothed, 1.3)
        
        # Apply temporal smoothing to prevent jitter between frames
        alpha = 0.3  # smoothing factor
        for i in range(1, len(envelope)):
            envelope[i] = alpha * envelope[i] + (1 - alpha) * envelope[i - 1]
        
        # Scale to max_mouth_open
        expressions = envelope * max_mouth_open
        return expressions
