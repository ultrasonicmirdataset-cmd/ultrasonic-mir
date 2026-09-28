#!/usr/bin/env python3
# %% [markdown]
# # Classical 15-instrument classification: timestamped Random Forest runs
#
# This RF-only version compares three bandwidth conditions using the same audio segments
# and an explicitly balanced two-view linear-frequency representation:
#
# * 44k       : native files from train_44 (44.1 kS/s)
# * 96k_full  : native files from train_96 (96 kS/s)
# * 96k_lp22  : the same 96 kS/s files, low-pass filtered at 22.05 kHz
#
# View A spans 0--22.05 kHz and View U spans 22.05--40 kHz. Random Forest
# receives descriptors computed separately inside each view. No MFCC or
# mel-scaled feature is used. Activity labels are taken from the
# 96 kS/s master exactly as in instrument_activity_threshold_inspector_colab:
# non-overlapping 0.5 s windows are active when RMS >= -70 dBFS.
#
# Every execution creates a unique UTC-timestamped results directory. Feature
# arrays are kept in a separate shared cache so repeated runs never overwrite
# results and do not repeat expensive feature extraction when inputs/config match.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # The experiment still runs if tqdm is unavailable.
    tqdm = None


DEFAULT_INSTRUMENTS: Tuple[str, ...] = (
    "Acoustic_guitar",
    "Bansury",
    "Clarinet",
    "Classic_guitar",
    "Drums",
    "Duble_bass",
    "Electric_guitar",
    "Flute",
    "Harmonica",
    "Kanun",
    "Piano",
    "Saxophone",
    "Trumpet",
    "Violin",
    "Vocals",
)

MODES: Tuple[str, ...] = ("44k", "96k_full", "96k_lp22")
STAT_NAMES: Tuple[str, ...] = ("mean", "std", "p10", "p50", "p90")


@dataclass(frozen=True)
class Config:
    data_root_44: str = "/content/drive/MyDrive/train_44"
    data_root_96: str = "/content/drive/MyDrive/train_96"
    output_root: str = "rf_timestamped_runs"
    shared_cache_dir: str = "results_rf_kmeans_balanced_ultrasound"
    instruments: Tuple[str, ...] = DEFAULT_INSTRUMENTS

    # Exact activity rule from instrument_activity_threshold_inspector_colab.
    activity_window_seconds: float = 0.5
    activity_hop_seconds: float = 0.5
    activity_threshold_dbfs: float = -70.0

    # Contiguous temporal split inside every stem. This retains all 15 classes.
    train_fraction: float = 0.70
    split_guard_seconds: float = 2.0
    max_train_windows_per_class: int = 1200
    max_test_windows_per_class: int = 500

    # Linear STFT and fixed physical-frequency bands; no mel/MFCC transform.
    stft_window_ms: float = 32.0
    stft_hop_ms: float = 8.0
    max_feature_frequency_hz: float = 40000.0
    band_width_hz: float = 2000.0
    audible_ultrasonic_boundary_hz: float = 22050.0
    log_floor_db: float = -120.0
    ultrasonic_presence_floor_db: float = -80.0

    # Same 96LP control used in the existing project notebook.
    lowpass_cutoff_hz: float = 22050.0
    lowpass_numtaps: int = 2049
    lowpass_kaiser_beta: float = 8.6

    # DATA_SEED controls window subsampling. RF_SEEDS change only the forests,
    # allowing several model replicates to share one manifest and feature cache.
    data_seed: int = 42
    rf_seeds: Tuple[int, ...] = (42, 43, 44)
    rf_trees: int = 500
    rf_min_samples_leaf: int = 2


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def canonical_song_id(value: Any) -> int | float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid song ID: {value!r}")
    return int(number) if number.is_integer() else number


def extract_song_id(filename: str) -> Optional[int | float]:
    """Extract song IDs robustly, including extensionless decimal names."""
    name = Path(filename).name.strip()
    known_audio_extensions = {".wav", ".wave", ".flac", ".aif", ".aiff"}
    suffix = Path(name).suffix.casefold()
    if suffix in known_audio_extensions:
        name = name[: -len(suffix)]

    # A right-to-left Drive UI can store 2.duble_bass_2 for intended ID 2.2.
    mirrored = re.fullmatch(r"(\d+)\.(.+?)[_-](\d+)", name)
    if mirrored and any(ch.isalpha() for ch in mirrored.group(2)):
        return canonical_song_id(f"{mirrored.group(3)}.{mirrored.group(1)}")

    matches = re.findall(r"(\d+(?:\.\d+)?)", name)
    return canonical_song_id(matches[-1]) if matches else None


def song_family_id(song_id: int | float) -> int:
    return int(math.floor(float(canonical_song_id(song_id))))


def is_readable_audio(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        sf.info(str(path))
        return True
    except Exception:
        return False


def resolve_instrument_directories(
    root: Path, instruments: Sequence[str]
) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    available = [path for path in root.iterdir() if path.is_dir()]
    by_normalized: Dict[str, List[Path]] = {}
    for path in available:
        by_normalized.setdefault(normalize_name(path.name), []).append(path)

    resolved: Dict[str, Path] = {}
    for instrument in instruments:
        matches = by_normalized.get(normalize_name(instrument), [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one folder for {instrument!r} under {root}; "
                f"found {[path.name for path in matches]}"
            )
        resolved[instrument] = matches[0]
    return resolved


def scan_stems(root: Path, instruments: Sequence[str]) -> Dict[str, Dict[int | float, Path]]:
    directories = resolve_instrument_directories(root, instruments)
    stems: Dict[str, Dict[int | float, Path]] = {name: {} for name in instruments}
    for instrument, directory in directories.items():
        for path in sorted(directory.iterdir()):
            if not is_readable_audio(path):
                continue
            song_id = extract_song_id(path.name)
            if song_id is None:
                warnings.warn(f"No song ID in {path}; skipping")
                continue
            if song_id in stems[instrument]:
                raise RuntimeError(
                    f"Duplicate {instrument}, song {song_id}: "
                    f"{stems[instrument][song_id]} and {path}"
                )
            stems[instrument][song_id] = path
    return stems


def audio_info(path: Path) -> Dict[str, Any]:
    info = sf.info(str(path))
    return {
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "duration_seconds": float(info.frames) / float(info.samplerate),
        "channels": int(info.channels),
    }


def build_pair_inventory(cfg: Config) -> pd.DataFrame:
    stems_44 = scan_stems(Path(cfg.data_root_44), cfg.instruments)
    stems_96 = scan_stems(Path(cfg.data_root_96), cfg.instruments)
    rows: List[Dict[str, Any]] = []

    for instrument in cfg.instruments:
        ids_44 = set(stems_44[instrument])
        ids_96 = set(stems_96[instrument])
        common = sorted(ids_44 & ids_96, key=float)
        if not common:
            raise RuntimeError(f"No paired 44/96 stems for {instrument}")
        for song_id in common:
            path_44 = stems_44[instrument][song_id]
            path_96 = stems_96[instrument][song_id]
            info_44 = audio_info(path_44)
            info_96 = audio_info(path_96)
            if info_44["sample_rate"] != 44100:
                raise ValueError(f"Expected 44100 Hz in {path_44}; found {info_44['sample_rate']}")
            if info_96["sample_rate"] != 96000:
                raise ValueError(f"Expected 96000 Hz in {path_96}; found {info_96['sample_rate']}")
            rows.append(
                {
                    "instrument": instrument,
                    "song_id": song_id,
                    "song_family": song_family_id(song_id),
                    "path_44": str(path_44),
                    "path_96": str(path_96),
                    "duration_44_seconds": info_44["duration_seconds"],
                    "duration_96_seconds": info_96["duration_seconds"],
                    "common_duration_seconds": min(
                        info_44["duration_seconds"], info_96["duration_seconds"]
                    ),
                    "duration_delta_seconds": (
                        info_96["duration_seconds"] - info_44["duration_seconds"]
                    ),
                    "missing_from_44": ",".join(map(str, sorted(ids_96 - ids_44, key=float))),
                    "missing_from_96": ",".join(map(str, sorted(ids_44 - ids_96, key=float))),
                }
            )
    result = pd.DataFrame(rows).sort_values(["instrument", "song_id"]).reset_index(drop=True)
    if result["instrument"].nunique() != len(cfg.instruments):
        raise RuntimeError("The paired inventory does not contain every requested instrument")
    return result


def read_mono_segment(
    handle: sf.SoundFile,
    start_seconds: float,
    duration_seconds: float,
    pad_frames: int = 0,
) -> np.ndarray:
    sr = int(handle.samplerate)
    target_start = int(round(start_seconds * sr))
    target_length = int(round(duration_seconds * sr))
    read_start = target_start - pad_frames
    read_end = target_start + target_length + pad_frames
    left_pad = max(0, -read_start)
    right_pad = max(0, read_end - len(handle))
    read_start = max(0, read_start)
    read_end = min(len(handle), read_end)
    handle.seek(read_start)
    block = handle.read(read_end - read_start, dtype="float32", always_2d=True)
    mono = np.mean(block, axis=1, dtype=np.float32)
    if left_pad or right_pad:
        mono = np.pad(mono, (left_pad, right_pad), mode="constant")
    expected = target_length + 2 * pad_frames
    if len(mono) < expected:
        mono = np.pad(mono, (0, expected - len(mono)), mode="constant")
    elif len(mono) > expected:
        mono = mono[:expected]
    return np.asarray(mono, dtype=np.float32)


def rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-10))


def _stable_subsample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=seed, replace=False)


def build_activity_manifest(inventory: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Select active 0.5 s windows from 96 kS/s masters at -70 dBFS."""
    if not math.isclose(
        cfg.activity_window_seconds,
        cfg.activity_hop_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        warnings.warn("The supplied inspector used equal 0.5 s window and hop")

    rows: List[Dict[str, Any]] = []
    total_candidates = sum(
        max(
            0,
            int(
                math.floor(
                    (float(item.common_duration_seconds) - cfg.activity_window_seconds)
                    / cfg.activity_hop_seconds
                )
            )
            + 1,
        )
        for item in inventory.itertuples(index=False)
    )
    progress = (
        tqdm(total=total_candidates, desc="96k activity scan", unit="window")
        if tqdm is not None
        else None
    )
    try:
        for item in inventory.itertuples(index=False):
            duration = float(item.common_duration_seconds)
            split_time = cfg.train_fraction * duration
            with sf.SoundFile(str(item.path_96), mode="r") as handle:
                sr = int(handle.samplerate)
                if sr != 96000:
                    raise ValueError(f"Activity source must be 96 kHz: {item.path_96}")
                starts = np.arange(
                    0.0,
                    max(0.0, duration - cfg.activity_window_seconds) + 1e-9,
                    cfg.activity_hop_seconds,
                )
                for start in starts:
                    if progress is not None:
                        progress.update(1)
                    end = float(start) + cfg.activity_window_seconds
                    if end <= split_time - cfg.split_guard_seconds:
                        split = "train"
                    elif start >= split_time + cfg.split_guard_seconds:
                        split = "test"
                    else:
                        continue
                    audio = read_mono_segment(
                        handle,
                        float(start),
                        cfg.activity_window_seconds,
                        pad_frames=0,
                    )
                    level = rms_dbfs(audio)
                    if level < cfg.activity_threshold_dbfs:
                        continue
                    start_ms = int(round(float(start) * 1000.0))
                    song_token = str(canonical_song_id(item.song_id))
                    rows.append(
                        {
                            "key": f"{item.instrument}|{song_token}|{start_ms:010d}",
                            "instrument": item.instrument,
                            "song_id": item.song_id,
                            "song_family": int(item.song_family),
                            "start_seconds": float(start),
                            "duration_seconds": cfg.activity_window_seconds,
                            "split": split,
                            "activity_rms_dbfs_96": level,
                            "path_44": str(item.path_44),
                            "path_96": str(item.path_96),
                        }
                    )
            if progress is None:
                print(f"Activity scan: {item.instrument}, song {item.song_id}")
    finally:
        if progress is not None:
            progress.close()

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No active windows passed the 96 kHz RMS criterion")

    selected: List[pd.DataFrame] = []
    for class_index, instrument in enumerate(cfg.instruments):
        for split_index, split in enumerate(("train", "test")):
            part = manifest[
                (manifest["instrument"] == instrument) & (manifest["split"] == split)
            ]
            limit = (
                cfg.max_train_windows_per_class
                if split == "train"
                else cfg.max_test_windows_per_class
            )
            part = _stable_subsample(
                part,
                limit=limit,
                seed=cfg.data_seed + 1009 * class_index + 97 * split_index,
            )
            selected.append(part)

    manifest = pd.concat(selected, ignore_index=True)
    manifest = manifest.sort_values(["split", "instrument", "key"]).reset_index(drop=True)
    coverage = manifest.groupby(["split", "instrument"]).size().unstack(fill_value=0)
    missing = [
        f"{split}/{instrument}"
        for split in ("train", "test")
        for instrument in cfg.instruments
        if int(coverage.get(instrument, pd.Series()).get(split, 0)) == 0
    ]
    if missing:
        raise RuntimeError(f"No active windows for: {', '.join(missing)}")
    return manifest


def activity_manifest_signature(inventory: pd.DataFrame, cfg: Config) -> str:
    payload = {
        "inventory": inventory[
            ["instrument", "song_id", "path_44", "path_96", "common_duration_seconds"]
        ].astype(str).values.tolist(),
        "config": {
            "instruments": list(cfg.instruments),
            "activity_window_seconds": cfg.activity_window_seconds,
            "activity_hop_seconds": cfg.activity_hop_seconds,
            "activity_threshold_dbfs": cfg.activity_threshold_dbfs,
            "train_fraction": cfg.train_fraction,
            "split_guard_seconds": cfg.split_guard_seconds,
            "max_train_windows_per_class": cfg.max_train_windows_per_class,
            "max_test_windows_per_class": cfg.max_test_windows_per_class,
            "data_seed": cfg.data_seed,
        },
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def _legacy_manifest_matches_config(metadata_path: Path, cfg: Config) -> bool:
    if not metadata_path.exists():
        return False
    try:
        old = json.loads(metadata_path.read_text(encoding="utf-8"))["config"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    required = {
        "activity_window_seconds": cfg.activity_window_seconds,
        "activity_hop_seconds": cfg.activity_hop_seconds,
        "activity_threshold_dbfs": cfg.activity_threshold_dbfs,
        "train_fraction": cfg.train_fraction,
        "split_guard_seconds": cfg.split_guard_seconds,
        "max_train_windows_per_class": cfg.max_train_windows_per_class,
        "max_test_windows_per_class": cfg.max_test_windows_per_class,
        "data_seed": cfg.data_seed,
    }
    for key, value in required.items():
        if key == "data_seed":
            old_value = old.get("data_seed", old.get("random_seed"))
        else:
            old_value = old.get(key)
        if old_value != value:
            return False
    return True


def load_or_build_activity_manifest(
    inventory: pd.DataFrame, cfg: Config, shared_cache_dir: Path
) -> pd.DataFrame:
    """Reuse a matching manifest; otherwise scan the 96 kS/s masters once."""
    manifest_cache_dir = shared_cache_dir / "manifest_cache"
    manifest_cache_dir.mkdir(parents=True, exist_ok=True)
    signature = activity_manifest_signature(inventory, cfg)
    cache_path = manifest_cache_dir / f"activity_manifest_{signature}.csv"
    if cache_path.exists():
        print(f"Loading cached activity manifest: {cache_path}")
        return pd.read_csv(cache_path)

    # Reuse the completed v2 run immediately when its configuration matches.
    legacy_manifest = shared_cache_dir / "active_window_manifest.csv"
    legacy_metadata = shared_cache_dir / "run_metadata.json"
    if legacy_manifest.exists() and _legacy_manifest_matches_config(legacy_metadata, cfg):
        print(f"Loading matching legacy activity manifest: {legacy_manifest}")
        return pd.read_csv(legacy_manifest)

    manifest = build_activity_manifest(inventory, cfg)
    manifest.to_csv(cache_path, index=False)
    print(f"Saved shared activity manifest cache: {cache_path}")
    return manifest


def next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def aggregate_stats(values: np.ndarray) -> List[float]:
    values = np.asarray(values, dtype=np.float64)
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 90)),
    ]


def fixed_band_edges(start_hz: float, end_hz: float, width_hz: float) -> np.ndarray:
    """Create near-uniform bands without leaving a tiny final sliver."""
    if not (0.0 <= start_hz < end_hz) or width_hz <= 0.0:
        raise ValueError("Invalid fixed-band limits")
    edges = [float(start_hz)]
    candidate = float(start_hz + width_hz)
    while candidate < end_hz:
        if end_hz - candidate < 0.5 * width_hz:
            break
        edges.append(candidate)
        candidate += width_hz
    edges.append(float(end_hz))
    return np.asarray(edges, dtype=np.float64)


def view_band_edges(cfg: Config) -> Dict[str, np.ndarray]:
    boundary = cfg.audible_ultrasonic_boundary_hz
    return {
        "audible": fixed_band_edges(0.0, boundary, cfg.band_width_hz),
        "ultrasonic": fixed_band_edges(
            boundary, cfg.max_feature_frequency_hz, cfg.band_width_hz
        ),
    }


def _band_label(low_hz: float, high_hz: float) -> str:
    return f"band_{int(round(low_hz)):05d}_{int(round(high_hz)):05d}_branch_rel_db"


BRANCH_DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "centroid_norm",
    "bandwidth_norm",
    "rolloff85_norm",
    "rolloff95_norm",
    "flatness_log10",
    "crest_db",
    "spectral_slope_db",
    "spectral_flux",
    "frame_energy_rel_db",
)


def feature_names(cfg: Config) -> List[str]:
    names: List[str] = []
    for view, edges in view_band_edges(cfg).items():
        for low, high in zip(edges[:-1], edges[1:]):
            for stat_name in STAT_NAMES:
                names.append(f"{view}__{_band_label(low, high)}_{stat_name}")
        for descriptor in BRANCH_DESCRIPTOR_NAMES:
            for stat_name in STAT_NAMES:
                names.append(f"{view}__{descriptor}_{stat_name}")

    for stat_name in STAT_NAMES:
        names.append(f"ultrasonic__to_audible_ratio_db_{stat_name}")
    for threshold_db in (-80, -60, -40):
        names.append(f"ultrasonic__occupancy_above_{abs(threshold_db):03d}db")
    names.extend(
        (
            "ultrasonic__cross_energy_envelope_correlation",
            "ultrasonic__cross_positive_energy_flux_correlation",
        )
    )
    return names


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(valid) < 3:
        return 0.0
    first = first[valid]
    second = second[valid]
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(first, second)[0, 1], -1.0, 1.0))


def _branch_arrays(
    frequencies: np.ndarray,
    power: np.ndarray,
    start_hz: float,
    end_hz: float,
    valid_frames: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return branch energy and framewise descriptors in physical Hz."""
    eps = np.finfo(np.float64).tiny
    mask = (frequencies >= start_hz) & (frequencies < end_hz)
    frame_count = power.shape[1]
    zeros = np.zeros(frame_count, dtype=np.float64)
    if not np.any(mask):
        arrays = {name: zeros.copy() for name in BRANCH_DESCRIPTOR_NAMES}
        arrays["frame_energy_rel_db"] = np.full(frame_count, cfg.log_floor_db)
        return zeros.copy(), arrays

    f = frequencies[mask].astype(np.float64)
    p = power[mask]
    raw_total = np.sum(p, axis=0)
    total = raw_total + eps
    distribution = p / total[None, :]
    width_hz = end_hz - start_hz

    centroid_hz = np.sum(f[:, None] * distribution, axis=0)
    centroid_norm = (centroid_hz - start_hz) / width_hz
    bandwidth_norm = np.sqrt(
        np.sum(np.square(f[:, None] - centroid_hz[None, :]) * distribution, axis=0)
    ) / width_hz
    cumulative = np.cumsum(p, axis=0)
    rolloff85 = (
        f[np.argmax(cumulative >= 0.85 * total[None, :], axis=0)] - start_hz
    ) / width_hz
    rolloff95 = (
        f[np.argmax(cumulative >= 0.95 * total[None, :], axis=0)] - start_hz
    ) / width_hz
    flatness = np.exp(np.mean(np.log(p + eps), axis=0)) / (
        np.mean(p, axis=0) + eps
    )
    flatness_log10 = np.log10(np.maximum(flatness, eps))
    crest_db = 10.0 * np.log10(
        (np.max(p, axis=0) + eps) / (np.mean(p, axis=0) + eps)
    )

    x = (f - start_hz) / width_hz
    x_centered = x - np.mean(x)
    log_power = 10.0 * np.log10(p / total[None, :] + eps)
    spectral_slope = np.sum(x_centered[:, None] * log_power, axis=0) / (
        np.sum(np.square(x_centered)) + eps
    )

    magnitude_distribution = np.sqrt(p)
    magnitude_distribution /= (
        np.sum(magnitude_distribution, axis=0, keepdims=True) + eps
    )
    spectral_flux = np.sqrt(
        np.sum(np.square(np.diff(magnitude_distribution, axis=1)), axis=0)
    )
    spectral_flux = np.concatenate(([0.0], spectral_flux))

    valid_energy = raw_total[np.asarray(valid_frames, dtype=bool)]
    reference_energy = float(np.median(valid_energy)) if len(valid_energy) else eps
    frame_energy_rel_db = 10.0 * np.log10(
        (raw_total + eps) / (reference_energy + eps)
    )
    frame_energy_rel_db = np.clip(frame_energy_rel_db, cfg.log_floor_db, 40.0)

    arrays = {
        "centroid_norm": centroid_norm,
        "bandwidth_norm": bandwidth_norm,
        "rolloff85_norm": rolloff85,
        "rolloff95_norm": rolloff95,
        "flatness_log10": flatness_log10,
        "crest_db": crest_db,
        "spectral_slope_db": spectral_slope,
        "spectral_flux": spectral_flux,
        "frame_energy_rel_db": frame_energy_rel_db,
    }
    invalid = ~np.asarray(valid_frames, dtype=bool)
    for name, array in arrays.items():
        array = np.asarray(array, dtype=np.float64)
        array[invalid] = cfg.log_floor_db if name == "frame_energy_rel_db" else 0.0
        arrays[name] = array
    return raw_total, arrays


def _append_view_features(
    values: List[float],
    frequencies: np.ndarray,
    power: np.ndarray,
    edges: np.ndarray,
    branch_energy: np.ndarray,
    descriptors: Mapping[str, np.ndarray],
    valid_frames: np.ndarray,
    cfg: Config,
) -> None:
    eps = np.finfo(np.float64).tiny
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        if np.any(mask):
            band_energy = np.sum(power[mask], axis=0)
            relative_db = 10.0 * np.log10(
                (band_energy + eps) / (branch_energy + eps)
            )
            relative_db = np.clip(relative_db, cfg.log_floor_db, 20.0)
            relative_db[~valid_frames] = cfg.log_floor_db
        else:
            relative_db = np.full(power.shape[1], cfg.log_floor_db, dtype=np.float64)
        values.extend(aggregate_stats(relative_db))
    for descriptor in BRANCH_DESCRIPTOR_NAMES:
        values.extend(aggregate_stats(descriptors[descriptor]))


def extract_linear_features(
    audio: np.ndarray,
    sample_rate: int,
    cfg: Config,
    enable_ultrasonic_view: bool = True,
) -> np.ndarray:
    """Balanced audible/ultrasonic fixed-Hz features; no mel/MFCC stage."""
    audio = np.asarray(audio, dtype=np.float32)
    nperseg = max(64, int(round(sample_rate * cfg.stft_window_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * cfg.stft_hop_ms / 1000.0)))
    nfft = next_power_of_two(nperseg)
    if len(audio) < nperseg:
        audio = np.pad(audio, (0, nperseg - len(audio)))
    frequencies, _, zxx = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - hop,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    power = np.square(np.abs(zxx).astype(np.float64))
    eps = np.finfo(np.float64).tiny
    boundary = cfg.audible_ultrasonic_boundary_hz
    audible_mask = frequencies < boundary
    ultrasonic_mask = (
        (frequencies >= boundary) & (frequencies < cfg.max_feature_frequency_hz)
    )
    # In 44k and 96LP, remove numerical/stop-band ultrasonic residue.
    if not enable_ultrasonic_view:
        power[frequencies >= boundary, :] = 0.0
    audible_energy = np.sum(power[audible_mask], axis=0)
    ultrasonic_energy = (
        np.sum(power[ultrasonic_mask], axis=0)
        if np.any(ultrasonic_mask)
        else np.zeros(power.shape[1], dtype=np.float64)
    )
    ratio_db = 10.0 * np.log10(
        (ultrasonic_energy + eps) / (audible_energy + eps)
    )
    ratio_db = np.clip(ratio_db, cfg.log_floor_db, 20.0)
    audible_valid = audible_energy > eps
    ultrasonic_valid = ratio_db >= cfg.ultrasonic_presence_floor_db

    edges = view_band_edges(cfg)
    audible_energy, audible_descriptors = _branch_arrays(
        frequencies,
        power,
        0.0,
        boundary,
        audible_valid,
        cfg,
    )
    ultrasonic_energy, ultrasonic_descriptors = _branch_arrays(
        frequencies,
        power,
        boundary,
        cfg.max_feature_frequency_hz,
        ultrasonic_valid,
        cfg,
    )

    values: List[float] = []
    _append_view_features(
        values,
        frequencies,
        power,
        edges["audible"],
        audible_energy,
        audible_descriptors,
        audible_valid,
        cfg,
    )
    _append_view_features(
        values,
        frequencies,
        power,
        edges["ultrasonic"],
        ultrasonic_energy,
        ultrasonic_descriptors,
        ultrasonic_valid,
        cfg,
    )

    values.extend(aggregate_stats(ratio_db))
    for threshold_db in (-80.0, -60.0, -40.0):
        values.append(float(np.mean(ratio_db >= threshold_db)))

    cross_valid = audible_valid & ultrasonic_valid
    audible_envelope_db = 10.0 * np.log10(
        (audible_energy + eps) / (np.median(audible_energy[audible_valid]) + eps)
    )
    ultrasonic_reference = (
        float(np.median(ultrasonic_energy[ultrasonic_valid]))
        if np.any(ultrasonic_valid)
        else eps
    )
    ultrasonic_envelope_db = 10.0 * np.log10(
        (ultrasonic_energy + eps) / (ultrasonic_reference + eps)
    )
    values.append(
        _safe_correlation(
            audible_envelope_db[cross_valid], ultrasonic_envelope_db[cross_valid]
        )
    )
    audible_positive_flux = np.maximum(
        np.diff(audible_envelope_db, prepend=audible_envelope_db[0]), 0.0
    )
    ultrasonic_positive_flux = np.maximum(
        np.diff(ultrasonic_envelope_db, prepend=ultrasonic_envelope_db[0]), 0.0
    )
    values.append(
        _safe_correlation(
            audible_positive_flux[cross_valid],
            ultrasonic_positive_flux[cross_valid],
        )
    )

    result = np.asarray(values, dtype=np.float32)
    expected = len(feature_names(cfg))
    if result.shape != (expected,) or not np.all(np.isfinite(result)):
        raise RuntimeError(
            f"Invalid feature vector: shape={result.shape}, expected={(expected,)}, "
            f"finite={bool(np.all(np.isfinite(result)))}"
        )
    return result


def lowpass_taps(cfg: Config) -> np.ndarray:
    if cfg.lowpass_numtaps % 2 == 0:
        raise ValueError("lowpass_numtaps must be odd")
    return signal.firwin(
        numtaps=cfg.lowpass_numtaps,
        cutoff=cfg.lowpass_cutoff_hz,
        fs=96000,
        window=("kaiser", cfg.lowpass_kaiser_beta),
        pass_zero="lowpass",
    ).astype(np.float32)


def feature_cache_signature(mode: str, manifest: pd.DataFrame, cfg: Config) -> str:
    payload = {
        "feature_version": "balanced_two_view_v2",
        "mode": mode,
        "keys": manifest["key"].tolist(),
        "paths": manifest[["path_44", "path_96"]].astype(str).values.tolist(),
        "feature": {
            key: value
            for key, value in asdict(cfg).items()
            if key
            in {
                "activity_window_seconds",
                "stft_window_ms",
                "stft_hop_ms",
                "max_feature_frequency_hz",
                "band_width_hz",
                "audible_ultrasonic_boundary_hz",
                "log_floor_db",
                "ultrasonic_presence_floor_db",
                "lowpass_cutoff_hz",
                "lowpass_numtaps",
                "lowpass_kaiser_beta",
            }
        },
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def extract_mode_features(
    mode: str, manifest: pd.DataFrame, cfg: Config, shared_cache_dir: Path
) -> Tuple[np.ndarray, List[str]]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    names = feature_names(cfg)
    cache_dir = shared_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = feature_cache_signature(mode, manifest, cfg)
    cache_path = cache_dir / f"{mode}_{signature}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if cached["keys"].tolist() == manifest["key"].tolist():
            print(f"Loading cached {mode} features: {cache_path}")
            cached_names = cached["feature_names"].tolist()
            X_cached = cached["X"].astype(np.float32)
            return enforce_disabled_ultrasonic_constant(
                X_cached, cached_names, mode
            ), cached_names

    path_column = "path_44" if mode == "44k" else "path_96"
    expected_sr = 44100 if mode == "44k" else 96000
    taps = lowpass_taps(cfg) if mode == "96k_lp22" else None
    pad_frames = len(taps) // 2 if taps is not None else 0
    X = np.empty((len(manifest), len(names)), dtype=np.float32)

    grouped = manifest.reset_index().groupby(path_column, sort=True)
    completed = 0
    progress = (
        tqdm(total=len(manifest), desc=f"{mode} features", unit="window")
        if tqdm is not None
        else None
    )
    try:
        for path_string, group in grouped:
            with sf.SoundFile(str(path_string), mode="r") as handle:
                sr = int(handle.samplerate)
                if sr != expected_sr:
                    raise ValueError(f"Expected {expected_sr} Hz in {path_string}; found {sr}")
                for row in group.itertuples(index=False):
                    audio = read_mono_segment(
                        handle,
                        float(row.start_seconds),
                        float(row.duration_seconds),
                        pad_frames=pad_frames,
                    )
                    if taps is not None:
                        audio = signal.oaconvolve(audio, taps, mode="same").astype(np.float32)
                        audio = audio[
                            pad_frames : pad_frames + int(round(row.duration_seconds * sr))
                        ]
                    X[int(row.index)] = extract_linear_features(
                        audio,
                        sr,
                        cfg,
                        enable_ultrasonic_view=(mode == "96k_full"),
                    )
                    completed += 1
                    if progress is not None:
                        progress.update(1)
            if progress is None:
                print(f"{mode}: {completed}/{len(manifest)} windows")
    finally:
        if progress is not None:
            progress.close()

    X = enforce_disabled_ultrasonic_constant(X, names, mode)
    np.savez_compressed(
        cache_path,
        X=X,
        keys=np.asarray(manifest["key"].tolist()),
        feature_names=np.asarray(names),
    )
    return X, names


def enforce_disabled_ultrasonic_constant(
    X: np.ndarray, names: Sequence[str], mode: str
) -> np.ndarray:
    """Make every disabled-U column exactly constant, including silent STFT frames."""
    if mode == "96k_full":
        return X
    ultrasonic_columns = np.asarray(
        [name.startswith("ultrasonic__") for name in names], dtype=bool
    )
    if int(np.sum(ultrasonic_columns)) != 100:
        raise RuntimeError("Expected exactly 100 ultrasonic feature columns")
    result = np.asarray(X, dtype=np.float32).copy()
    result[:, ultrasonic_columns] = 0.0
    return result


def per_class_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    instruments: Sequence[str],
    mode: str,
    algorithm: str,
) -> List[Dict[str, Any]]:
    labels = np.arange(len(instruments))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return [
        {
            "algorithm": algorithm,
            "mode": mode,
            "instrument": instrument,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, instrument in enumerate(instruments)
    ]


def save_confusion_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    instruments: Sequence[str],
    title: str,
    destination: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(instruments)), normalize="true")
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(instruments)), labels=instruments, rotation=55, ha="right")
    ax.set_yticks(np.arange(len(instruments)), labels=instruments)
    ax.set_xlabel("Predicted instrument")
    ax.set_ylabel("True instrument")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Row-normalized proportion")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_mode(
    mode: str,
    X: np.ndarray,
    manifest: pd.DataFrame,
    names: Sequence[str],
    cfg: Config,
    rf_seed: int,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    class_to_index = {name: index for index, name in enumerate(cfg.instruments)}
    y = manifest["instrument"].map(class_to_index).to_numpy(dtype=np.int64)
    train_mask = manifest["split"].eq("train").to_numpy()
    test_mask = manifest["split"].eq("test").to_numpy()
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    keys_test = manifest.loc[test_mask, "key"].to_numpy()
    if set(np.unique(y_train)) != set(range(len(cfg.instruments))):
        raise RuntimeError("Training split does not contain all 15 classes")
    if set(np.unique(y_test)) != set(range(len(cfg.instruments))):
        raise RuntimeError("Test split does not contain all 15 classes")

    overall: List[Dict[str, Any]] = []
    per_class: List[Dict[str, Any]] = []
    prediction_columns: Dict[str, Any] = {
        "key": keys_test,
        "true_instrument": [cfg.instruments[index] for index in y_test],
    }

    # Random Forest: identical feature vector, no scaling required.
    rf = RandomForestClassifier(
        n_estimators=cfg.rf_trees,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=rf_seed,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    overall.append(
        {
            "algorithm": "RandomForest",
            "rf_seed": rf_seed,
            "mode": mode,
            "accuracy": float(accuracy_score(y_test, rf_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, rf_pred)),
            "train_windows": int(len(y_train)),
            "test_windows": int(len(y_test)),
        }
    )
    seed_rows = per_class_rows(
        y_test, rf_pred, cfg.instruments, mode, "RandomForest"
    )
    for row in seed_rows:
        row["rf_seed"] = rf_seed
    per_class.extend(seed_rows)
    prediction_columns["rf_prediction"] = [cfg.instruments[index] for index in rf_pred]
    save_confusion_plot(
        y_test,
        rf_pred,
        cfg.instruments,
        f"Random Forest - {mode} - RF seed {rf_seed}",
        output_dir / "plots" / f"confusion_rf_{mode}.png",
    )
    rf_importance = pd.DataFrame(
        {
            "feature": names,
            "view": [name.split("__", 1)[0] for name in names],
            "importance": rf.feature_importances_,
        }
    )
    rf_importance.sort_values("importance", ascending=False).to_csv(
        output_dir / f"rf_feature_importance_{mode}.csv", index=False
    )
    rf_importance.groupby("view", as_index=False)["importance"].sum().to_csv(
        output_dir / f"rf_view_importance_{mode}.csv", index=False
    )

    return overall, per_class, pd.DataFrame(prediction_columns)


def save_comparisons(overall: pd.DataFrame, per_class: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = ["accuracy", "balanced_accuracy"]
    rows: List[Dict[str, Any]] = []
    for algorithm, group in overall.groupby("algorithm"):
        indexed = group.set_index("mode")
        for metric in metric_columns:
            if indexed[metric].notna().sum() == 0:
                continue
            row: Dict[str, Any] = {"algorithm": algorithm, "metric": metric}
            for mode in MODES:
                row[mode] = float(indexed.loc[mode, metric])
            row["delta_96full_minus_44"] = row["96k_full"] - row["44k"]
            row["delta_96full_minus_96lp22"] = row["96k_full"] - row["96k_lp22"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "overall_mode_comparison.csv", index=False)

    per_class_rows_out: List[Dict[str, Any]] = []
    for (algorithm, instrument), group in per_class.groupby(["algorithm", "instrument"]):
        indexed = group.set_index("mode")
        for metric in ("precision", "recall", "f1"):
            row = {"algorithm": algorithm, "instrument": instrument, "metric": metric}
            for mode in MODES:
                row[mode] = float(indexed.loc[mode, metric])
            row["delta_96full_minus_44"] = row["96k_full"] - row["44k"]
            row["delta_96full_minus_96lp22"] = row["96k_full"] - row["96k_lp22"]
            per_class_rows_out.append(row)
    pd.DataFrame(per_class_rows_out).to_csv(
        output_dir / "per_instrument_mode_comparison.csv", index=False
    )


def save_run_metadata(
    cfg: Config,
    inventory: pd.DataFrame,
    manifest: pd.DataFrame,
    output_dir: Path,
    run_id: str,
    started_at_utc: str,
) -> None:
    names = feature_names(cfg)
    view_counts = {
        view: sum(name.startswith(f"{view}__") for name in names)
        for view in ("audible", "ultrasonic")
    }
    metadata = {
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "algorithm": "RandomForest",
        "config": asdict(cfg),
        "design": {
            "activity_source": "96 kS/s master",
            "activity_rule": "0.5 s RMS >= -70 dBFS, 0.5 s hop",
            "features": (
                "Two separately normalized fixed-Hz views: 0-22.05 kHz and "
                "22.05-40 kHz; branch-specific bands/descriptors; no mel/MFCC"
            ),
            "primary_causal_contrast": "96k_full minus 96k_lp22",
            "practical_contrast": "96k_full minus native 44k",
            "split": "contiguous temporal train/test blocks within each stem with guard",
            "warning": (
                "This split retains all 15 classes but is not an unseen-recording test. "
                "Some instruments, especially Bansury, lack enough independent song "
                "families for a 15-class song-disjoint evaluation."
            ),
        },
        "paired_stems": int(len(inventory)),
        "selected_windows": int(len(manifest)),
        "classes": list(cfg.instruments),
        "feature_count": len(names),
        "feature_count_by_view": view_counts,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def create_run_directory(output_root: Path, cfg: Config) -> Tuple[Path, str, str]:
    started = datetime.now(timezone.utc)
    started_at_utc = started.isoformat()
    timestamp = started.strftime("%Y%m%d_%H%M%S_%f_UTC")
    rf_seed_token = "-".join(str(seed) for seed in cfg.rf_seeds)
    run_id = (
        f"rf_run_{timestamp}_dseed{cfg.data_seed}_rf{rf_seed_token}_"
        f"tr{cfg.max_train_windows_per_class}_te{cfg.max_test_windows_per_class}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / run_id
    output_dir.mkdir(parents=False, exist_ok=False)
    return output_dir, run_id, started_at_utc


def run(cfg: Config, audit_only: bool = False) -> Path:
    elapsed_start = time.perf_counter()
    if not (0.0 < cfg.audible_ultrasonic_boundary_hz < cfg.max_feature_frequency_hz):
        raise ValueError("The audible/ultrasonic boundary must lie inside the feature range")
    if not cfg.rf_seeds:
        raise ValueError("At least one RF seed is required")
    if len(set(cfg.rf_seeds)) != len(cfg.rf_seeds):
        raise ValueError(f"RF seeds must be unique: {cfg.rf_seeds}")
    if any(seed < 0 or seed > 2**32 - 1 for seed in (cfg.data_seed, *cfg.rf_seeds)):
        raise ValueError("Data and RF seeds must be between 0 and 2**32 - 1")
    schema_names = feature_names(cfg)
    schema_counts = {
        view: sum(name.startswith(f"{view}__") for name in schema_names)
        for view in ("audible", "ultrasonic")
    }
    if schema_counts != {"audible": 100, "ultrasonic": 100}:
        raise RuntimeError(f"Unexpected two-view feature schema: {schema_counts}")
    output_root = Path(cfg.output_root).expanduser().resolve()
    shared_cache_dir = Path(cfg.shared_cache_dir).expanduser().resolve()
    shared_cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir, run_id, started_at_utc = create_run_directory(output_root, cfg)
    print(f"Run ID: {run_id}")
    print(f"Unique results directory: {output_dir}")
    print(f"Shared cache directory: {shared_cache_dir}")
    print("Feature schema: 100 audible + 100 ultrasonic features")
    print("Scanning paired 44.1/96 kS/s stems...")
    inventory = build_pair_inventory(cfg)
    inventory.to_csv(output_dir / "paired_stem_inventory.csv", index=False)
    family_counts = (
        inventory.groupby("instrument")["song_family"].nunique().rename("song_families")
    )
    print(family_counts.to_string())
    if (family_counts < 2).any():
        affected = ", ".join(family_counts[family_counts < 2].index)
        warnings.warn(
            "A fully song-disjoint 15-class evaluation is impossible with the current "
            f"inventory. Fewer than two song families: {affected}."
        )
    if audit_only:
        print(f"Audit saved to {output_dir / 'paired_stem_inventory.csv'}")
        return output_dir

    print("Selecting active windows from the 96 kS/s masters...")
    manifest = load_or_build_activity_manifest(inventory, cfg, shared_cache_dir)
    manifest.to_csv(output_dir / "active_window_manifest.csv", index=False)
    counts = manifest.groupby(["instrument", "split"]).size().unstack(fill_value=0)
    counts.to_csv(output_dir / "active_window_counts.csv")
    print(counts.to_string())
    save_run_metadata(
        cfg, inventory, manifest, output_dir, run_id, started_at_utc
    )

    # Load/extract every bandwidth condition only once. All RF seeds below use
    # these exact same arrays, so seed-to-seed differences are model randomness.
    features_by_mode: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for mode in MODES:
        print(f"\n=== Preparing {mode} features ===")
        features_by_mode[mode] = extract_mode_features(
            mode, manifest, cfg, shared_cache_dir / "feature_cache"
        )

    batch_overall: List[pd.DataFrame] = []
    batch_per_instrument: List[pd.DataFrame] = []
    batch_mode_comparisons: List[pd.DataFrame] = []
    for rf_seed in cfg.rf_seeds:
        print(f"\n######## RF seed {rf_seed} ########")
        seed_output_dir = output_dir / f"rf_seed_{rf_seed}"
        seed_output_dir.mkdir(parents=False, exist_ok=False)
        all_overall: List[Dict[str, Any]] = []
        all_per_class: List[Dict[str, Any]] = []
        all_predictions: List[pd.DataFrame] = []
        for mode in MODES:
            print(f"\n=== {mode}; RF seed {rf_seed} ===")
            X, names = features_by_mode[mode]
            overall, per_class, predictions = evaluate_mode(
                mode, X, manifest, names, cfg, rf_seed, seed_output_dir
            )
            all_overall.extend(overall)
            all_per_class.extend(per_class)
            predictions.insert(0, "rf_seed", rf_seed)
            predictions.insert(1, "mode", mode)
            all_predictions.append(predictions)

        overall_frame = pd.DataFrame(all_overall)
        per_class_frame = pd.DataFrame(all_per_class)
        predictions_frame = pd.concat(all_predictions, ignore_index=True)
        overall_frame.to_csv(seed_output_dir / "overall_metrics.csv", index=False)
        per_class_frame.to_csv(
            seed_output_dir / "per_instrument_metrics.csv", index=False
        )
        predictions_frame.to_csv(
            seed_output_dir / "test_predictions.csv", index=False
        )
        save_comparisons(overall_frame, per_class_frame, seed_output_dir)
        mode_comparison = pd.read_csv(
            seed_output_dir / "overall_mode_comparison.csv"
        )
        mode_comparison.insert(1, "rf_seed", rf_seed)
        batch_mode_comparisons.append(mode_comparison)
        batch_overall.append(overall_frame)
        batch_per_instrument.append(per_class_frame)
        (seed_output_dir / "SEED_COMPLETE.txt").write_text(
            f"Completed RF seed {rf_seed}\n", encoding="utf-8"
        )

    all_seed_overall = pd.concat(batch_overall, ignore_index=True)
    all_seed_per_instrument = pd.concat(batch_per_instrument, ignore_index=True)
    all_seed_mode_comparison = pd.concat(batch_mode_comparisons, ignore_index=True)
    all_seed_overall.to_csv(
        output_dir / "all_rf_seeds_overall_metrics.csv", index=False
    )
    all_seed_per_instrument.to_csv(
        output_dir / "all_rf_seeds_per_instrument_metrics.csv", index=False
    )
    all_seed_mode_comparison.to_csv(
        output_dir / "all_rf_seeds_mode_comparison.csv", index=False
    )

    accuracy_rows = all_seed_overall[
        ["rf_seed", "mode", "accuracy", "balanced_accuracy"]
    ].copy()
    accuracy_rows.to_csv(output_dir / "rf_seed_accuracy_results.csv", index=False)
    accuracy_summary = (
        accuracy_rows.groupby("mode", sort=False)[["accuracy", "balanced_accuracy"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    accuracy_summary.columns = [
        "mode",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
    ]
    accuracy_summary["n_rf_seeds"] = len(cfg.rf_seeds)
    accuracy_summary.to_csv(
        output_dir / "rf_seed_accuracy_mean_std.csv", index=False
    )

    print("\nAccuracy for every RF seed:")
    print(accuracy_rows.to_string(index=False))
    print("\nAccuracy mean and standard deviation across RF seeds:")
    print(accuracy_summary.to_string(index=False))
    print(f"\nAll outputs saved under: {output_dir}")
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    elapsed_seconds = time.perf_counter() - elapsed_start
    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completed_at_utc"] = completed_at_utc
    metadata["elapsed_seconds"] = elapsed_seconds
    metadata["overall_results_by_rf_seed"] = all_seed_overall.to_dict(orient="records")
    metadata["accuracy_mean_std_by_mode"] = accuracy_summary.to_dict(orient="records")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "RUN_COMPLETE.txt").write_text(
        (
            f"Completed RF run {run_id}\n"
            f"Data seed: {cfg.data_seed}\n"
            f"RF seeds: {','.join(map(str, cfg.rf_seeds))}\n"
            f"Completed at {completed_at_utc}\n"
            f"Elapsed seconds: {elapsed_seconds:.3f}\n"
        ),
        encoding="utf-8",
    )
    return output_dir


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-44", default=Config.data_root_44)
    parser.add_argument("--data-root-96", default=Config.data_root_96)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Parent directory; every run gets a new timestamped child directory",
    )
    parser.add_argument(
        "--shared-cache-dir",
        default=None,
        help="Persistent cache shared by all timestamped runs",
    )
    parser.add_argument(
        "--instruments",
        default=",".join(DEFAULT_INSTRUMENTS),
        help="Comma-separated instrument folder names",
    )
    parser.add_argument("--max-train-per-class", type=int, default=1200)
    parser.add_argument("--max-test-per-class", type=int, default=500)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="Controls active-window subsampling and therefore the shared feature cache",
    )
    parser.add_argument(
        "--rf-seeds",
        default="42,43,44",
        help="Comma-separated RF random seeds trained on the exact same cached features",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    script_directory = Path(__file__).resolve().parent
    output_root = args.output_root or str(
        script_directory / "rf_timestamped_runs"
    )
    shared_cache_dir = args.shared_cache_dir or str(
        script_directory / "results_rf_kmeans_balanced_ultrasound"
    )
    instruments = tuple(item.strip() for item in args.instruments.split(",") if item.strip())
    if args.seed is not None:
        # Backward compatibility with the previous single-seed notebook.
        data_seed = args.seed
        rf_seeds = (args.seed,)
    else:
        data_seed = args.data_seed
        rf_seeds = tuple(
            int(item.strip()) for item in args.rf_seeds.split(",") if item.strip()
        )
    cfg = Config(
        data_root_44=args.data_root_44,
        data_root_96=args.data_root_96,
        output_root=output_root,
        shared_cache_dir=shared_cache_dir,
        instruments=instruments,
        max_train_windows_per_class=args.max_train_per_class,
        max_test_windows_per_class=args.max_test_per_class,
        rf_trees=args.rf_trees,
        data_seed=data_seed,
        rf_seeds=rf_seeds,
    )
    run(cfg, audit_only=args.audit_only)


if __name__ == "__main__":
    main()
