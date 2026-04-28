"""
data_io.py
Purpose: Handles loading and saving of high-resolution audio files and snippets.
"""
import librosa
import soundfile as sf
import numpy as np

def load_audio(filepath, sr=None, start_sec=None, end_sec=None):
    offset = start_sec if start_sec is not None else 0.0
    duration = None
    if start_sec is not None and end_sec is not None:
        duration = end_sec - start_sec
    
    y, sr_out = librosa.load(filepath, sr=sr, offset=offset, duration=duration, mono=True)
    return y, sr_out

def save_audio(filepath, y, sr):
    sf.write(filepath, y, sr)
