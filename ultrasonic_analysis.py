#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ultrasonic Analysis for Musical Instruments (96 kHz etc.)

Features per file:
- PUA_frame: Probability of Ultrasonic Activity given "playing" frames
- PUA_sec:   Probability of Ultrasonic Activity in 1-second windows (given playing)
- Frequency distribution (counts of bins above threshold over the whole track)
- Max frequency per frame (10 dB above noise floor)
- 3 plots per file:
    1) Spectrogram + ultrasonic mask
    2) Frequency distribution
    3) Max frequency per frame
- Folder/file-level CSV summary (ultrasonic_summary.csv)
"""

from pathlib import Path
from typing import Union, Sequence

import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# ========================= CORE ANALYSIS ========================= #

def analyze_ultrasonic(
    path: Union[str, Path],
    audible_max_hz: float = 20_000.0,
    n_fft: int = 4096,
    hop_length: int | None = None,
    min_level_db: float = -80.0,
    snr_db: float = 10.0,
    noise_percentile: float = 10.0,
    pua_window_sec: float = 1.0,
    playing_strong_db: float = -15.0,  # Strong audible-band activity threshold
) -> dict:
    """
    Analyze ultrasonic content for a single audio file.
    """

    path = Path(path)

    # ----- Load audio (no resample, mono) ----- #
    y, sr = librosa.load(path.as_posix(), sr=None, mono=True)
    if hop_length is None:
        hop_length = n_fft // 4

    # ----- STFT + dB (relative to max in file, limited by top_db) ----- #
    S = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    mag = np.abs(S)
    mag_db = librosa.amplitude_to_db(mag, ref=np.max, top_db=120.0)

    # Frequencies & times
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(mag_db.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft
    )

    # ----- Frequency masks ----- #
    audible_mask = freqs <= audible_max_hz
    ultra_mask = freqs > audible_max_hz
    if not np.any(ultra_mask):
        raise ValueError(
            f"Sample rate {sr} too low – no ultrasonic band above {audible_max_hz:.0f} Hz."
        )

    # ----- Noise floors (in dB, relative) ----- #
    global_noise_floor = np.percentile(mag_db, noise_percentile)
    audible_noise_floor = np.percentile(mag_db[audible_mask, :], noise_percentile)
    ultra_noise_floor = np.percentile(mag_db[ultra_mask, :], noise_percentile)

    # Thresholds: ultrasonic/global thresholds based on noise-relative SNR
    ultra_thresh = max(ultra_noise_floor + snr_db, min_level_db)
    global_thresh = max(global_noise_floor + snr_db, min_level_db)

    # ----- "Playing" frames in the audible band ----- #
    aud_max_per_frame = mag_db[audible_mask, :].max(axis=0)
    playing_frames = aud_max_per_frame > playing_strong_db
    n_play_frames = int(playing_frames.sum())

    # ----- Ultrasonic-active frames (any ultrasonic bin above ultra_thresh) ----- #
    ultra_mag_full = mag_db[ultra_mask, :]
    ultra_active_frames = (ultra_mag_full > ultra_thresh).any(axis=0)

    # ----- PUA (frame-level) ----- #
    if n_play_frames == 0:
        pua_frame = np.nan
        n_ultra_frames = 0
    else:
        both = playing_frames & ultra_active_frames
        n_ultra_frames = int(both.sum())
        pua_frame = n_ultra_frames / float(n_play_frames)

    # ----- PUA (second-level windows) ----- #
    T_end = times[-1] if times.size > 0 else 0.0
    if T_end <= 0 or n_play_frames == 0:
        pua_sec = np.nan
    else:
        n_windows = int(np.ceil(T_end / pua_window_sec))
        playing_win = np.zeros(n_windows, dtype=bool)
        ultra_win = np.zeros(n_windows, dtype=bool)

        for i in range(n_windows):
            t_start = i * pua_window_sec
            t_end = (i + 1) * pua_window_sec
            idx = np.where((times >= t_start) & (times < t_end))[0]
            if idx.size == 0:
                continue

            if playing_frames[idx].any():
                playing_win[i] = True
                if (playing_frames[idx] & ultra_active_frames[idx]).any():
                    ultra_win[i] = True

        n_play_win = playing_win.sum()
        if n_play_win == 0:
            pua_sec = np.nan
        else:
            pua_sec = (ultra_win & playing_win).sum() / float(n_play_win)

    # ----- Frequency distribution (counts over track, only during playing frames) ----- #
    freq_counts = np.zeros_like(freqs, dtype=np.int64)
    for t_idx, is_playing in enumerate(playing_frames):
        if not is_playing:
            continue
        mask = mag_db[:, t_idx] > global_thresh
        freq_counts[mask] += 1

    # ----- Max frequency per frame above the global threshold ----- #
    frame_max_freqs = np.full(mag_db.shape[1], np.nan, dtype=float)
    for t_idx, is_playing in enumerate(playing_frames):
        if not is_playing:
            continue
        mask = mag_db[:, t_idx] > global_thresh
        if not np.any(mask):
            continue
        max_idx = np.where(mask)[0].max()
        frame_max_freqs[t_idx] = freqs[max_idx]

    if np.all(np.isnan(frame_max_freqs)):
        max_freq_overall = np.nan
        max_ultra_freq = np.nan
    else:
        max_freq_overall = float(np.nanmax(frame_max_freqs))
        ultra_only = frame_max_freqs.copy()
        ultra_only[ultra_only <= audible_max_hz] = np.nan
        max_ultra_freq = (
            float(np.nanmax(ultra_only))
            if not np.all(np.isnan(ultra_only))
            else np.nan
        )

    results = {
        "file": path.name,
        "path": path.as_posix(),
        "sr": sr,
        "times_s": times,
        "freqs_hz": freqs,
        "mag_db": mag_db,
        "pua_frame": pua_frame,
        "pua_sec": pua_sec,
        "n_play_frames": n_play_frames,
        "n_ultra_frames": n_ultra_frames,
        "freq_counts": freq_counts,
        "frame_max_freqs_hz": frame_max_freqs,
        "max_freq_overall_hz": max_freq_overall,
        "max_ultrasonic_freq_hz": max_ultra_freq,
        "playing_frames": playing_frames,
        "ultra_mask": ultra_mask,
        "thresholds_db": {
            "global": global_thresh,
            "ultra": ultra_thresh,
            "playing_strong_db": playing_strong_db,
        },
        "noise_floors_db": {
            "global": global_noise_floor,
            "audible": audible_noise_floor,
            "ultra": ultra_noise_floor,
        },
        "pua_window_sec": pua_window_sec,
    }

    return results


# ========================= PLOTTING ========================= #

def plot_results(results: dict, audible_max_hz: float, output_dir: Path) -> None:
    """
    Create plots for each analyzed audio file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    freqs = results["freqs_hz"]
    times = results["times_s"]
    mag_db = results["mag_db"]
    playing_frames = results["playing_frames"]
    ultra_mask = results["ultra_mask"]
    freq_counts = results["freq_counts"]
    frame_max_freqs = results["frame_max_freqs_hz"]
    pua_frame = results["pua_frame"]
    pua_sec = results["pua_sec"]
    file_name = results["file"]
    thresholds_db = results["thresholds_db"]
    ultra_thresh = thresholds_db["ultra"]

    # ---------- 1) Spectrogram + Ultrasonic Mask ---------- #
    fig, ax = plt.subplots(figsize=(12, 6))
    extent = (times[0], times[-1], freqs[0], freqs[-1])

    im = ax.imshow(
        mag_db,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    plt.colorbar(im, ax=ax, label="dB (relative to file max)")

    mag_for_mask = mag_db.copy()
    mag_for_mask[:, ~playing_frames] = -1e9
    mask = (mag_for_mask > ultra_thresh) & ultra_mask[:, None]

    mask_array = np.zeros_like(mag_db, dtype=float)
    mask_array[mask] = 1.0

    ax.imshow(
        mask_array,
        origin="lower",
        aspect="auto",
        extent=extent,
        alpha=0.3,
    )

    ax.axhline(audible_max_hz, linestyle="--")
    ax.set_title(
        f"{file_name} – Spectrogram + Ultrasonic Mask\n"
        f"PUA_frame={pua_frame:.3f}, PUA_sec={pua_sec:.3f}, ultra_thresh={ultra_thresh:.1f} dB"
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_name}_spec_ultra.png", dpi=200)
    plt.close(fig)

    # ---------- 2) Frequency Histogram (counts of frames above threshold) ---------- #
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(freqs, freq_counts)
    ax.axvline(audible_max_hz, linestyle="--")

    ultra_thresh = results["thresholds_db"]["ultra"]
    global_thresh = results["thresholds_db"]["global"]
    playing_strong_db = results["thresholds_db"]["playing_strong_db"]

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Count of frames above threshold")

    ax.set_title(
        f"{file_name} – Frequency Histogram\n"
        f"(global_thresh={global_thresh:.1f} dB, "
        f"ultra_thresh={ultra_thresh:.1f} dB, "
        f"playing_strong={playing_strong_db:.1f} dB)"
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"{file_name}_freq_histogram.png", dpi=200)
    plt.close(fig)

    # ---------- 3) Max Frequency per Frame ---------- #
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, frame_max_freqs, marker=".", linestyle="none")
    ax.axhline(audible_max_hz, linestyle="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Max frequency per frame [Hz]")
    clean_name = file_name.replace(".wav", "")
    ax.set_title(f"{clean_name} – Max Frequency per Frame")
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_name}_frame_max_freq.png", dpi=200)
    plt.close(fig)

    # ---------- 4) Histogram of Maximum Frequencies per Frame ---------- #
    valid = np.isfinite(frame_max_freqs)
    valid_vals = frame_max_freqs[valid]

    if valid_vals.size > 0:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(valid_vals, bins=80, edgecolor='black')

        ax.axvline(audible_max_hz, linestyle="--", color="red")
        ax.set_xlabel("Maximum frequency per frame [Hz]")
        ax.set_ylabel("Count of frames")
        ax.set_title(
            f"{file_name} – Histogram of Maximum Frequencies per Frame\n"
            f"(only strong-playing frames)"
        )

        fig.tight_layout()
        fig.savefig(output_dir / f"{file_name}_max_freq_histogram.png", dpi=200)
        plt.close(fig)


# ========================= BATCH PROCESS (FILE / FOLDER) ========================= #

def process_input(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    audible_max_hz: float = 20_000.0,
    n_fft: int = 4096,
    hop_length: int | None = None,
    min_level_db: float = -70.0,
    snr_db: float = 10.0,
    noise_percentile: float = 10.0,
    pua_window_sec: float = 1.0,
    playing_strong_db: float = -25.0,
    exts: Sequence[str] = (".wav", ".mp3", ".flac", ".ogg"),
) -> None:
    """
    If input_path is a file, analyze one file.
    If input_path is a folder, analyze all supported audio files inside it.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files: list[Path] = []

    if input_path.is_file():
        if input_path.suffix.lower() in exts:
            audio_files = [input_path]
            print(f"📄 Single-file mode: {input_path.name}")
        else:
            print(f"❌ Not a supported audio file: {input_path}")
            return
    elif input_path.is_dir():
        print(f"📁 Folder mode: scanning {input_path}")
        for p in sorted(input_path.glob("**/*")):
            if p.suffix.lower() in exts:
                audio_files.append(p)
    else:
        print(f"❌ Path does not exist: {input_path}")
        return

    if not audio_files:
        print("⚠ No supported audio files found.")
        return

    print(f"✅ Found {len(audio_files)} audio file(s) to analyze.")

    rows = []

    for path in tqdm(audio_files, desc="Analyzing files"):
        print(f"🔍 {path.name}")
        res = analyze_ultrasonic(
            path,
            audible_max_hz=audible_max_hz,
            n_fft=n_fft,
            hop_length=hop_length,
            min_level_db=min_level_db,
            snr_db=snr_db,
            noise_percentile=noise_percentile,
            pua_window_sec=pua_window_sec,
            playing_strong_db=playing_strong_db,
        )
        plot_results(res, audible_max_hz=audible_max_hz, output_dir=output_dir)

        rows.append(
            {
                "file": res["file"],
                "path": res["path"],
                "pua_frame": res["pua_frame"],
                "pua_sec": res["pua_sec"],
                "n_play_frames": res["n_play_frames"],
                "n_ultra_frames": res["n_ultra_frames"],
                "max_freq_overall_hz": res["max_freq_overall_hz"],
                "max_ultrasonic_freq_hz": res["max_ultrasonic_freq_hz"],
                "noise_global_db": res["noise_floors_db"]["global"],
                "noise_audible_db": res["noise_floors_db"]["audible"],
                "noise_ultra_db": res["noise_floors_db"]["ultra"],
                "pua_window_sec": res["pua_window_sec"],
                "playing_strong_db": playing_strong_db,
            }
        )

    df = pd.DataFrame(rows)

    # ===== Round values for the CSV output =====
    # 1) Round PUA values to 2 decimal places
    for col in ("pua_frame", "pua_sec"):
        if col in df.columns:
            df[col] = df[col].round(2)

    # 2) Round maximum frequencies to whole numbers
    for col in ("max_freq_overall_hz", "max_ultrasonic_freq_hz"):
        if col in df.columns:
            # round -> Int64 to preserve NaN values if present
            df[col] = df[col].round().astype("Int64")

    csv_path = output_dir / "ultrasonic_summary.csv"

    try:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"💾 Saved summary CSV to: {csv_path}")
    except PermissionError:
        fallback = output_dir / "ultrasonic_summary_fallback.csv"
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"⚠ PermissionError on {csv_path}. Saved instead to: {fallback}")
        print("   The file may be open in Excel or locked by OneDrive. Close it and run again if needed.")


# ========================= MAIN ========================= #

if __name__ == "__main__":
    # Set the paths here:
    input_path = r""
    output_dir = r""

    process_input(
        input_path=input_path,
        output_dir=output_dir,
        audible_max_hz=20_000.0,
        n_fft=4096,
        hop_length=None,
        min_level_db=-70.0,
        snr_db=10.0,
        noise_percentile=10.0,
        pua_window_sec=1.0,
        playing_strong_db=-15.0,
    )
