import yt_dlp
import ffmpeg
import numpy as np
import librosa
import soundfile as sf
import io

class Audio_Process:
    # Initial Processing
    def __init__(self, audio_data, sample_rate):

        # Process the data 
        self.y = audio_data
        self.sr = sample_rate

        # Extract Tempo & Beat Structure
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)

        # Harmonic-Percussive Separation
        self.harmonic, self.percussive = librosa.effects.hpss(self.y)

        # Onset Strength
        self.onset_env = librosa.onset.onset_strength(y=self.percussive, sr=self.sr)

        # MFCC for Beat Texture and Groove Quality
        self.mfccs = librosa.feature.mfcc(y=self.percussive, sr=self.sr, n_mfcc=13)

        # Filter Breakbeats by Zero-Crossing Rate
        self.zcr = librosa.feature.zero_crossing_rate(y=self.percussive)