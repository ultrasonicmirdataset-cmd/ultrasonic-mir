#!/usr/bin/env python3
"""Recording-level isolated-source classification (15 classes).

Run on the extracted local/Colab copies of train_96 and train_44, never on
Drive URLs. This script writes results only to --output, never to the inputs.

Example (after mounting Google Drive in Colab):
    !pip install numpy scipy scikit-learn soundfile torch
    !python isolated_rf_cnn6.py \
        --train96 '/content/drive/MyDrive/train_96' \
        --train44 '/content/drive/MyDrive/train_44' \
        --output '/content/drive/MyDrive/ultrasonic_mir_runs/experiment_5_1'

First run with --audit-only. The proposed split is whole tracks 1,3,4,5 for
training; 2,6 for evaluation; and 2.2 entirely excluded. Thus all windows
from a given track, including across instruments, stay in one partition.
Every classifier and bandwidth condition uses the same 96-kHz-derived window
IDs. Conditions: native 44.1-kHz, full-band 96-kHz, and 96-kHz low-pass at
22.05 kHz. --derive44 resamples *every* 44.1-kHz window from its 96-kHz
master if the provided train_44 directory cannot be verified as paired.

The defaults use a 200-tree RF on up to 300/150 balanced 0.5-s windows,
32/8-ms linear STFT, fixed 2-kHz bands up to 40 kHz and three seeds. CNN6
uses four 5x5 convolution blocks, per-frequency BN, frequency
average, temporal max+mean, 512-dimensional embedding and fixed 50 epochs.
Only the final epoch is evaluated; evaluation does not select checkpoints.
The -70 dBFS full-band RMS activity rule comes from the earlier Drive code;
it does not set the ultrasonic-noise visualization threshold.

Requirements: Python >=3.10, numpy, scipy, scikit-learn, soundfile (preferred),
torch (only for CNN6). PCM WAVs also work through Python's built-in wave
module if soundfile is absent. Audio is not included with this code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import wave
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

try:
    import soundfile as sf
except ImportError:
    sf = None


INSTRUMENTS = (
    "Acoustic_guitar", "Bansury", "Clarinet", "Classic_guitar", "Drums",
    "Duble_bass", "Electric_guitar", "Flute", "Harmonica", "Kanun",
    "Piano", "Saxophone", "Trumpet", "Violin", "Vocals",
)
TRAIN_TRACKS = ("1", "3", "4", "5")
EVAL_TRACKS = ("2", "6")
EXCLUDED_TRACKS = ("2.2",)
CONDITIONS = ("44", "96", "96lp", "ultra20")
DEFAULT_CONDITIONS = ("44", "96", "96lp")


@dataclass(frozen=True)
class Config:
    train96: Path
    train44: Path | None
    output: Path
    derive44: bool = False
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS
    models: tuple[str, ...] = ("rf", "cnn6")
    seeds: tuple[int, ...] = (42, 43, 44)
    data_seed: int = 42
    window_seconds: float = 0.5
    activity_dbfs: float = -70.0
    rf_train_per_class: int = 300
    cnn_train_pool_per_class: int = 1200
    eval_per_class: int = 150
    rf_trees: int = 200
    epochs: int = 50
    cnn_samples_per_class_epoch: int = 256
    batch_size: int = 16
    workers: int = 0
    learning_rate: float = 2e-4
    duration_tolerance_seconds: float = 0.10


def track_id(filename: str) -> str | None:
    name = filename.strip()
    name = re.sub(r"\.(wav|wave|flac|aiff?)$", "", name, flags=re.I)
    # This ambiguous Drive basename has been used to encode song 2.2; do not
    # accidentally treat it as a second song-2 training/evaluation stem.
    if re.fullmatch(r"2\.duble_bass_2", name, flags=re.I):
        return "2.2"
    match = re.search(r"(?:^|[_\s-])(2\.2|[1-6])$", name)
    return match.group(1) if match else None


class AudioFile:
    """Seekable mono reader; soundfile also supports extensionless Drive WAVs."""

    def __init__(self, path: Path):
        self.path = path
        self.handle: Any = None
        self.backend = "soundfile" if sf is not None else "wave"

    def __enter__(self) -> AudioFile:
        if self.backend == "soundfile":
            self.handle = sf.SoundFile(str(self.path))
            self.samplerate = int(self.handle.samplerate)
            self.frames = int(self.handle.frames)
            self.channels = int(self.handle.channels)
        else:
            self.handle = wave.open(str(self.path), "rb")
            self.samplerate = int(self.handle.getframerate())
            self.frames = int(self.handle.getnframes())
            self.channels = int(self.handle.getnchannels())
            self.sample_width = int(self.handle.getsampwidth())
            if self.handle.getcomptype() != "NONE":
                raise ValueError(f"Install soundfile to read compressed WAV: {self.path}")
        return self

    def __exit__(self, *_: object) -> None:
        if self.handle is not None:
            self.handle.close()

    def read_mono(self, start: int, frames: int) -> np.ndarray:
        if frames < 0:
            raise ValueError("frames must be nonnegative")
        left = max(0, -start)
        begin = max(0, start)
        usable = max(0, min(self.frames - begin, frames - left))
        out = np.zeros(frames, dtype=np.float32)
        if usable == 0:
            return out
        if self.backend == "soundfile":
            self.handle.seek(begin)
            block = self.handle.read(usable, dtype="float32", always_2d=True)
        else:
            self.handle.setpos(begin)
            raw = self.handle.readframes(usable)
            if self.sample_width == 1:
                values = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
            elif self.sample_width == 2:
                values = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
            elif self.sample_width == 3:
                octets = np.frombuffer(raw, np.uint8).reshape(-1, 3)
                integers = (octets[:, 0].astype(np.int32)
                            | (octets[:, 1].astype(np.int32) << 8)
                            | (octets[:, 2].astype(np.int32) << 16))
                integers = (integers ^ 0x800000) - 0x800000
                values = integers.astype(np.float32) / 8388608
            elif self.sample_width == 4:
                values = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
            else:
                raise ValueError(f"Install soundfile for WAV sample width {self.sample_width}")
            block = values.reshape(-1, self.channels)
        if len(block) != usable:
            raise IOError(f"Short read: {self.path}, frame {begin}")
        out[left:left + usable] = np.mean(block, axis=1, dtype=np.float32)
        return out


def file_info(path: Path) -> dict[str, Any]:
    with AudioFile(path) as audio:
        return {"sample_rate": audio.samplerate, "frames": audio.frames,
                "duration": audio.frames / audio.samplerate,
                "channels": audio.channels, "size_bytes": path.stat().st_size}


def scan_root(root: Path) -> dict[tuple[str, str], Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Cannot find input folder: {root}")
    folders = {p.name.casefold(): p for p in root.iterdir() if p.is_dir()}
    missing = [name for name in INSTRUMENTS if name.casefold() not in folders]
    if missing:
        raise ValueError(f"Missing instrument folders under {root}: {missing}")
    found: dict[tuple[str, str], Path] = {}
    for name in INSTRUMENTS:
        for path in sorted(folders[name.casefold()].iterdir()):
            if not path.is_file():
                continue
            track = track_id(path.name)
            if track not in (*TRAIN_TRACKS, *EVAL_TRACKS, *EXCLUDED_TRACKS):
                continue
            key = (name, track)
            if key in found:
                raise ValueError(f"Duplicate {key} in {root}: {found[key]} and {path}")
            found[key] = path
    return found


def audit(cfg: Config) -> tuple[list[dict[str, Any]], list[str]]:
    p96 = scan_root(cfg.train96)
    p44 = {} if cfg.derive44 else scan_root(cfg.train44)  # type: ignore[arg-type]
    issues: list[str] = []
    inventory: list[dict[str, Any]] = []
    for label in INSTRUMENTS:
        for track in (*TRAIN_TRACKS, *EVAL_TRACKS):
            key = label, track
            file96, file44 = p96.get(key), p44.get(key)
            if file96 is None and file44 is None:
                continue
            if file96 is None:
                issues.append(f"{label}/{track}: present only in train_44")
                continue
            if file44 is None and not cfg.derive44:
                issues.append(f"{label}/{track}: missing matching train_44 stem")
                continue
            try:
                info96 = file_info(file96)
                info44 = None if cfg.derive44 else file_info(file44)  # type: ignore[arg-type]
            except (ValueError, OSError, RuntimeError, wave.Error) as exc:
                issues.append(f"{label}/{track}: unreadable audio: {exc}")
                continue
            if info96["sample_rate"] != 96000:
                issues.append(f"{file96}: expected 96000 Hz, got {info96['sample_rate']}")
            if info44 and info44["sample_rate"] != 44100:
                issues.append(f"{file44}: expected 44100 Hz, got {info44['sample_rate']}")
            if info44 and abs(info96["duration"] - info44["duration"]) > cfg.duration_tolerance_seconds:
                issues.append(f"{label}/{track}: 44/96 durations differ by "
                              f"{abs(info96['duration']-info44['duration']):.3f} s "
                              "(check alignment and source pairing)")
            inventory.append({"instrument": label, "track": track,
                              "split": "train" if track in TRAIN_TRACKS else "eval",
                              "path96": str(file96), "path44": str(file44) if file44 else "",
                              "duration96_s": info96["duration"],
                              "duration44_s": info44["duration"] if info44 else "",
                              "sample_rate96": info96["sample_rate"],
                              "sample_rate44": info44["sample_rate"] if info44 else "",
                              "size96": info96["size_bytes"],
                              "size44": info44["size_bytes"] if info44 else ""})
    for label in INSTRUMENTS:
        for split in ("train", "eval"):
            if not any(row["instrument"] == label and row["split"] == split
                       for row in inventory):
                issues.append(f"{label}: no {split} recording in the selected split")
    if not cfg.derive44:
        for label in INSTRUMENTS:
            if (label, "2.2") in p96 and (label, "2.2") not in p44:
                print(f"NOTE: {label}/2.2 is excluded, absent from train_44")
    return inventory, issues


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    if not rows and fields is None:
        return
    names = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scan_activity(inventory: Sequence[dict[str, Any]], cfg: Config) -> list[dict[str, Any]]:
    """Apply the same full-band 96-kHz RMS decision before any bandwidth change."""
    candidates: list[dict[str, Any]] = []
    count = int(round(96000 * cfg.window_seconds))
    for row in inventory:
        with AudioFile(Path(row["path96"])) as audio:
            maximum = float(row["duration96_s"])
            if row["duration44_s"] != "":
                maximum = min(maximum, float(row["duration44_s"]))
            total = min(int(maximum / cfg.window_seconds), audio.frames // count)
            for index in range(total):
                chunk = audio.read_mono(index * count, count)
                rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
                rms_dbfs = 20 * math.log10(max(rms, 1e-10))
                if rms_dbfs >= cfg.activity_dbfs:
                    candidates.append({"instrument": row["instrument"],
                                       "track": row["track"], "split": row["split"],
                                       "start_s": round(index * cfg.window_seconds, 8),
                                       "rms_dbfs96": round(rms_dbfs, 3),
                                       "path96": row["path96"], "path44": row["path44"],
                                       "class_index": INSTRUMENTS.index(row["instrument"])})
        print(f"Activity scan: {row['instrument']}/{row['track']}", flush=True)
    return candidates


def balanced_order(rows: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Cycle over stems so a class with several tracks does not use only one."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["track"]].append(row)
    for values in groups.values():
        rng.shuffle(values)
    order: list[dict[str, Any]] = []
    while any(groups.values()):
        keys = sorted(key for key, values in groups.items() if values)
        rng.shuffle(keys)
        for key in keys:
            order.append(groups[key].pop())
    return order


def choose_windows(candidates: Sequence[dict[str, Any]], cfg: Config
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[(row["instrument"], row["split"])].append(row)
    gaps = [f"{label}/{split}" for label in INSTRUMENTS for split in ("train", "eval")
            if not grouped[(label, split)]]
    if gaps:
        raise RuntimeError(f"No active windows for: {', '.join(gaps)}")
    rf_n = min(cfg.rf_train_per_class, *(len(grouped[(label, "train")]) for label in INSTRUMENTS))
    cnn_n = min(cfg.cnn_train_pool_per_class, *(len(grouped[(label, "train")]) for label in INSTRUMENTS))
    eval_n = min(cfg.eval_per_class, *(len(grouped[(label, "eval")]) for label in INSTRUMENTS))
    rf_rows, cnn_rows, eval_rows = [], [], []
    for index, label in enumerate(INSTRUMENTS):
        train = balanced_order(grouped[(label, "train")], cfg.data_seed + 1009 * index)
        validation = balanced_order(grouped[(label, "eval")], cfg.data_seed + 2003 * index)
        rf_rows.extend(train[:rf_n])
        cnn_rows.extend(train[:cnn_n])
        eval_rows.extend(validation[:eval_n])
    print(f"Selected per class: RF train={rf_n}, CNN train={cnn_n}, "
          f"evaluation={eval_n}; same evaluation windows for all conditions.")
    return rf_rows, cnn_rows, eval_rows


def lowpass_taps() -> np.ndarray:
    return signal.firwin(2049, 22050, fs=96000,
                         window=("kaiser", 8.6)).astype(np.float32)


def load_window(row: dict[str, Any], condition: str, cfg: Config,
                taps: np.ndarray | None = None) -> np.ndarray:
    seconds = float(row["start_s"])
    if condition == "44" and not cfg.derive44:
        sample_rate = 44100
        path = Path(row["path44"])
        with AudioFile(path) as audio:
            return audio.read_mono(round(seconds * sample_rate),
                                   round(cfg.window_seconds * sample_rate))
    sample_rate = 96000
    path = Path(row["path96"])
    length = round(cfg.window_seconds * sample_rate)
    if condition == "96lp":
        if taps is None:
            taps = lowpass_taps()
        pad = len(taps) // 2
    elif condition == "44":
        pad = 9600  # 100 ms of resampling context on each side
    else:
        pad = 0
    with AudioFile(path) as audio:
        chunk = audio.read_mono(round(seconds * sample_rate) - pad,
                                length + 2 * pad)
    if condition == "96lp":
        # Valid convolution with symmetric context is zero-phase at the clip.
        return signal.oaconvolve(chunk, taps, mode="valid").astype(np.float32)
    if condition == "44":
        resampled = signal.resample_poly(chunk, 147, 320)
        start = round(pad * 44100 / 96000)
        result = resampled[start:start + round(cfg.window_seconds * 44100)]
        if len(result) != round(cfg.window_seconds * 44100):
            raise RuntimeError("44.1-kHz resampling returned a short window")
        return result.astype(np.float32)
    return chunk


def summary_stats(values: np.ndarray) -> list[float]:
    a = np.asarray(values, np.float64)
    return [float(np.mean(a)), float(np.std(a)),
            float(np.percentile(a, 10)), float(np.median(a)),
            float(np.percentile(a, 90))]


def rf_features(audio: np.ndarray, sample_rate: int, condition: str) -> np.ndarray:
    """Linear-frequency descriptors; never compute mel or MFCC features."""
    win = round(sample_rate * 0.032)
    hop = round(sample_rate * 0.008)
    freqs, _, spectrum = signal.stft(audio, fs=sample_rate, window="hann",
        nperseg=win, noverlap=win - hop, nfft=win, boundary=None, padded=False)
    power = np.abs(spectrum).astype(np.float64) ** 2
    if condition == "ultra20":
        power[freqs <= 20000] = 0.0
    elif condition != "96":
        power[freqs >= 22050] = 0.0  # Prevent stop-band residual as a cue.
    eps = 1e-20
    feature_ceiling = 48000 if condition == "ultra20" else 40000
    total = power[freqs < feature_ceiling].sum(axis=0) + eps
    features: list[float] = []
    for lo in range(0, feature_ceiling, 2000):
        mask = (freqs >= lo) & (freqs < lo + 2000)
        band = power[mask].sum(axis=0) if mask.any() else np.zeros_like(total)
        relative = np.clip(10 * np.log10((band + eps) / total), -120, 10)
        features.extend(summary_stats(relative))

    branch_energy: list[np.ndarray] = []
    branches = ((0, 20000), (20000, 48000)) if condition == "ultra20" else (
        (0, 22050), (22050, 40000))
    for low, high in branches:
        mask = (freqs >= low) & (freqs < high)
        p = power[mask]
        f = freqs[mask]
        energy = p.sum(axis=0) if p.size else np.zeros_like(total)
        branch_energy.append(energy)
        if not p.size or (high > 22050 and condition in ("44", "96lp")):
            features.extend([0.0] * (8 * 5))
            continue
        denom = energy + eps
        probability = p / denom
        centroid = (f[:, None] * probability).sum(axis=0)
        width = np.sqrt((((f[:, None] - centroid) ** 2) * probability).sum(axis=0))
        cumulative = np.cumsum(p, axis=0)
        rolloff = f[np.argmax(cumulative >= 0.85 * energy[None, :], axis=0)]
        flatness = np.exp(np.mean(np.log(p + eps), axis=0)) / (p.mean(axis=0) + eps)
        crest = p.max(axis=0) / (p.mean(axis=0) + eps)
        log_spectrum = 10 * np.log10(p + eps)
        normalized_f = (f - f.mean()) / max(high - low, 1)
        slope = (normalized_f[:, None] * log_spectrum).sum(axis=0) / (
            float((normalized_f ** 2).sum()) + eps)
        magnitude = np.sqrt(p)
        magnitude /= magnitude.sum(axis=0, keepdims=True) + eps
        flux = np.r_[0.0, np.sqrt((np.diff(magnitude, axis=1) ** 2).sum(axis=0))]
        for descriptor in (centroid / high, width / high, rolloff / high,
                           flatness, np.log10(crest + eps), slope, flux,
                           10 * np.log10(energy + eps)):
            features.extend(summary_stats(descriptor))
    audible, ultrasonic = branch_energy
    ratio_db = np.clip(10 * np.log10((ultrasonic + eps) / (audible + eps)), -120, 20)
    features.extend(summary_stats(ratio_db))
    result = np.asarray(features, np.float32)
    expected_features = 205 if condition == "ultra20" else 185
    if result.shape != (expected_features,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"Unexpected RF feature vector: {result.shape}")
    return result


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]]]:
    labels = np.arange(len(INSTRUMENTS))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    overall = {"accuracy": float(accuracy_score(y_true, y_pred)),
               "macro_f1": float(np.mean(f1))}
    detail = [{"instrument": label, "precision": float(precision[i]),
               "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
              for i, label in enumerate(INSTRUMENTS)]
    return overall, detail


def save_evaluation(run_dir: Path, model: str, condition: str, seed: int,
                    true: np.ndarray, pred: np.ndarray,
                    rows: Sequence[dict[str, Any]],
                    overall_rows: list[dict[str, Any]],
                    class_rows: list[dict[str, Any]]) -> None:
    overall, detail = metrics(true, pred)
    overall_rows.append({"model": model, "condition": condition, "seed": seed, **overall})
    class_rows.extend({"model": model, "condition": condition, "seed": seed, **d}
                      for d in detail)
    prefix = f"{model}_{condition}_seed{seed}"
    write_csv(run_dir / f"{prefix}_predictions.csv", [
        {"instrument": r["instrument"], "track": r["track"],
         "start_s": r["start_s"], "true": int(y), "predicted": int(p)}
        for r, y, p in zip(rows, true, pred)])
    cm = confusion_matrix(true, pred, labels=np.arange(len(INSTRUMENTS)))
    write_csv(run_dir / f"{prefix}_confusion.csv", [
        {"instrument": name, **{other: int(cm[i, j]) for j, other in enumerate(INSTRUMENTS)}}
        for i, name in enumerate(INSTRUMENTS)])
    print(f"{prefix}: accuracy={overall['accuracy']:.4f}, macro-F1={overall['macro_f1']:.4f}",
          flush=True)


def run_rf(cfg: Config, train: Sequence[dict[str, Any]], validation: Sequence[dict[str, Any]],
           run_dir: Path, overall: list[dict[str, Any]], per_class: list[dict[str, Any]]) -> None:
    for condition in cfg.conditions:
        print(f"Extracting RF features: {condition}", flush=True)
        sr = 44100 if condition == "44" else 96000
        taps = lowpass_taps() if condition == "96lp" else None
        x_train = np.stack([rf_features(load_window(r, condition, cfg, taps), sr, condition)
                            for r in train])
        x_val = np.stack([rf_features(load_window(r, condition, cfg, taps), sr, condition)
                          for r in validation])
        y_train = np.asarray([r["class_index"] for r in train])
        y_val = np.asarray([r["class_index"] for r in validation])
        np.savez_compressed(run_dir / f"rf_features_{condition}.npz", x_train=x_train,
                            x_val=x_val, y_train=y_train, y_val=y_val)
        for seed in cfg.seeds:
            forest = RandomForestClassifier(n_estimators=cfg.rf_trees,
                       random_state=seed, n_jobs=-1)
            forest.fit(x_train, y_train)
            pred = forest.predict(x_val)
            save_evaluation(run_dir, "rf", condition, seed, y_val, pred,
                            validation, overall, per_class)


def run_cnn(cfg: Config, train: Sequence[dict[str, Any]], validation: Sequence[dict[str, Any]],
            run_dir: Path, overall: list[dict[str, Any]], per_class: list[dict[str, Any]]) -> None:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, Dataset, Sampler
    except ImportError as exc:
        raise RuntimeError("CNN6 requires PyTorch: pip install torch") from exc

    class WindowDataset(Dataset):
        def __init__(self, rows: Sequence[dict[str, Any]], condition: str):
            self.rows = list(rows)
            self.condition = condition
            self.taps = lowpass_taps() if condition == "96lp" else None

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int) -> tuple[Any, int]:
            row = self.rows[i]
            audio = load_window(row, self.condition, cfg, self.taps)
            return torch.from_numpy(audio), int(row["class_index"])

    class BalancedSampler(Sampler[int]):
        def __init__(self, rows: Sequence[dict[str, Any]], seed: int):
            self.by_label = [np.flatnonzero([r["class_index"] == label for r in rows])
                             for label in range(len(INSTRUMENTS))]
            self.seed = seed
            self.epoch = 0

        def __len__(self) -> int:
            return cfg.cnn_samples_per_class_epoch * len(INSTRUMENTS)

        def __iter__(self) -> Iterable[int]:
            rng = np.random.default_rng(self.seed + 1009 * self.epoch)
            order = np.concatenate([rng.choice(indices,
                size=cfg.cnn_samples_per_class_epoch,
                replace=len(indices) < cfg.cnn_samples_per_class_epoch)
                for indices in self.by_label])
            rng.shuffle(order)
            return iter(order.tolist())

    def make_model(sample_rate: int, band_mode: str) -> Any:
        n_fft = round(sample_rate * 0.032)
        hop = round(sample_rate * 0.008)
        n_bins = n_fft // 2 + 1

        class Block(nn.Module):
            def __init__(self, in_ch: int, out_ch: int):
                super().__init__()
                self.conv = nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                nn.init.xavier_uniform_(self.conv.weight)
                nn.init.ones_(self.bn.weight)
                nn.init.zeros_(self.bn.bias)

            def forward(self, x: Any) -> Any:
                return F.avg_pool2d(F.relu_(self.bn(self.conv(x))), 2)

        class CNN6(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_fft, self.hop = n_fft, hop
                self.register_buffer("hann", torch.hann_window(n_fft), persistent=False)
                frequencies = torch.fft.rfftfreq(n_fft, 1 / sample_rate)
                if band_mode == "lowpass":
                    keep = frequencies < 22050
                elif band_mode == "ultrasonic":
                    keep = frequencies > 20000
                else:
                    keep = torch.ones_like(frequencies, dtype=torch.bool)
                self.register_buffer("keep", keep, persistent=False)
                self.bn0 = nn.BatchNorm2d(n_bins)
                self.blocks = nn.ModuleList([Block(1, 64), Block(64, 128),
                                             Block(128, 256), Block(256, 512)])
                self.fc1 = nn.Linear(512, 512)
                self.classifier = nn.Linear(512, len(INSTRUMENTS))
                nn.init.ones_(self.bn0.weight)
                nn.init.zeros_(self.bn0.bias)
                for layer in (self.fc1, self.classifier):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

            def forward(self, audio: Any) -> Any:
                with torch.autocast(device_type=audio.device.type, enabled=False):
                    spec = torch.stft(audio.float(), n_fft=self.n_fft,
                         hop_length=self.hop, window=self.hann.float(),
                         center=True, pad_mode="reflect", return_complex=True)
                    power = spec.abs().square()
                    if band_mode != "full":
                        power = power.masked_fill(~self.keep[None, :, None], 0.0)
                    x = torch.log(power + 1e-10).transpose(1, 2).unsqueeze(1)
                x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
                for block in self.blocks:
                    x = F.dropout(block(x), p=0.2, training=self.training)
                x = torch.mean(x, dim=3)
                x = torch.max(x, dim=2).values + torch.mean(x, dim=2)
                x = F.relu_(self.fc1(F.dropout(x, p=0.5, training=self.training)))
                return self.classifier(x)

        return CNN6()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    for condition in cfg.conditions:
        sample_rate = 44100 if condition == "44" else 96000
        ds_train = WindowDataset(train, condition)
        ds_val = WindowDataset(validation, condition)
        val_loader = DataLoader(ds_val, batch_size=cfg.batch_size,
                                num_workers=cfg.workers, shuffle=False)
        for seed in cfg.seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if amp:
                torch.cuda.manual_seed_all(seed)
            sampler = BalancedSampler(train, seed)
            train_loader = DataLoader(ds_train, batch_size=cfg.batch_size,
                                      sampler=sampler, num_workers=cfg.workers,
                                      pin_memory=amp)
            band_mode = {"96lp": "lowpass", "ultra20": "ultrasonic"}.get(condition, "full")
            model = make_model(sample_rate, band_mode).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                          weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            history = []
            for epoch in range(cfg.epochs):
                sampler.epoch = epoch
                number = epoch + 1
                if number <= 3:
                    lr_ratio = number / 3
                elif number <= 30:
                    lr_ratio = 1.0
                else:
                    progress = (number - 30) / max(1, cfg.epochs - 30)
                    lr_ratio = 0.10 + 0.90 * (1 + math.cos(math.pi * progress)) / 2
                for group in optimizer.param_groups:
                    group["lr"] = cfg.learning_rate * lr_ratio
                model.train()
                total_loss = total = 0
                for audio, labels in train_loader:
                    audio, labels = audio.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=amp):
                        logits = model(audio)
                        loss = F.cross_entropy(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += float(loss.detach()) * len(labels)
                    total += len(labels)
                history.append({"epoch": number, "train_loss": total_loss / total,
                                "learning_rate": optimizer.param_groups[0]["lr"]})
                print(f"CNN6 {condition} seed={seed} epoch={number}/{cfg.epochs} "
                      f"loss={total_loss / total:.4f}", flush=True)
            prefix = f"cnn6_{condition}_seed{seed}"
            torch.save({"state_dict": model.state_dict(), "epoch": cfg.epochs,
                        "classes": INSTRUMENTS, "condition": condition,
                        "sample_rate": sample_rate}, run_dir / f"{prefix}_final.pt")
            write_csv(run_dir / f"{prefix}_history.csv", history)
            # Crucially: evaluate once after the fixed final epoch, never pick
            # a model or threshold using the held-out recordings.
            model.eval()
            predictions, truths = [], []
            with torch.inference_mode():
                for audio, labels in val_loader:
                    logits = model(audio.to(device))
                    predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
                    truths.extend(labels.tolist())
            save_evaluation(run_dir, "cnn6", condition, seed,
                            np.asarray(truths), np.asarray(predictions),
                            validation, overall, per_class)
            del model, optimizer
            if amp:
                torch.cuda.empty_cache()


def summarize(rows: Sequence[dict[str, Any]], keys: Sequence[str],
              metrics_names: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for names, items in sorted(groups.items()):
        record = dict(zip(keys, names))
        record["runs"] = len(items)
        for metric in metrics_names:
            vals = [float(r[metric]) for r in items]
            record[metric + "_mean"] = statistics.mean(vals)
            record[metric + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        result.append(record)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train96", type=Path, default=Path("/content/drive/MyDrive/train_96"))
    parser.add_argument("--train44", type=Path, default=Path("/content/drive/MyDrive/train_44"))
    parser.add_argument("--derive44", action="store_true",
                        help="Derive ALL 44.1-kHz windows from 96-kHz masters; ignore train_44.")
    parser.add_argument("--output", type=Path, default=Path("./experiment_5_1_results"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--models", default="rf,cnn6")
    parser.add_argument("--conditions", default="44,96,96lp")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--rf-train-per-class", type=int, default=300)
    parser.add_argument("--cnn-train-pool-per-class", type=int, default=1200)
    parser.add_argument("--eval-per-class", type=int, default=150)
    parser.add_argument("--cnn-samples-per-class-epoch", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--duration-tolerance-seconds", type=float, default=0.10)
    args = parser.parse_args()
    model_names = tuple(x.strip().lower() for x in args.models.split(",") if x.strip())
    conditions = tuple(x.strip().lower() for x in args.conditions.split(",") if x.strip())
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    if not model_names or set(model_names) - {"rf", "cnn6"}:
        parser.error("--models must contain rf and/or cnn6")
    if not conditions or set(conditions) - set(CONDITIONS):
        parser.error("--conditions must contain 44,96,96lp,ultra20")
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain distinct integers")
    if min(args.epochs, args.rf_trees, args.rf_train_per_class,
           args.cnn_train_pool_per_class, args.eval_per_class,
           args.cnn_samples_per_class_epoch, args.batch_size) <= 0 or args.workers < 0:
        parser.error("training sizes/epochs/trees/batch size must be positive")
    if args.duration_tolerance_seconds < 0:
        parser.error("duration tolerance must be nonnegative")
    cfg = Config(train96=args.train96, train44=None if args.derive44 else args.train44,
                 output=args.output, derive44=args.derive44, models=model_names,
                 conditions=conditions, seeds=seeds, rf_trees=args.rf_trees,
                 epochs=args.epochs, rf_train_per_class=args.rf_train_per_class,
                 cnn_train_pool_per_class=args.cnn_train_pool_per_class,
                 eval_per_class=args.eval_per_class,
                 cnn_samples_per_class_epoch=args.cnn_samples_per_class_epoch,
                 batch_size=args.batch_size, workers=args.workers,
                 duration_tolerance_seconds=args.duration_tolerance_seconds)
    inventory, issues = audit(cfg)
    cfg.output.mkdir(parents=True, exist_ok=True)
    audit_path = cfg.output / "experiment_5_1_audit.json"
    audit_path.write_text(json.dumps({"issues": issues, "inventory": inventory,
                      "train_tracks": TRAIN_TRACKS, "eval_tracks": EVAL_TRACKS,
                      "excluded_tracks": EXCLUDED_TRACKS, "derive44": cfg.derive44},
                      indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit: {audit_path}; {len(inventory)} paired stems, {len(issues)} issues")
    if issues:
        for issue in issues:
            print("  - " + issue)
        if not args.audit_only:
            raise RuntimeError("Dataset audit failed; no training started. See audit JSON.")
    if args.audit_only:
        return

    candidates = scan_activity(inventory, cfg)
    rf_train, cnn_train, validation = choose_windows(candidates, cfg)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = cfg.output / f"experiment_5_1_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_csv(run_dir / "paired_stem_inventory.csv", inventory)
    write_csv(run_dir / "window_manifest_rf_train.csv", rf_train)
    write_csv(run_dir / "window_manifest_cnn_train.csv", cnn_train)
    write_csv(run_dir / "window_manifest_eval.csv", validation)
    run_config = {**{key: str(value) if isinstance(value, Path) else value
                     for key, value in asdict(cfg).items()},
                  "train_tracks": TRAIN_TRACKS, "eval_tracks": EVAL_TRACKS,
                  "excluded_tracks": EXCLUDED_TRACKS, "classes": INSTRUMENTS,
                  "note": "Independent rerun; previous reported percentages are not assumed."}
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    overall: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    if "rf" in cfg.models:
        run_rf(cfg, rf_train, validation, run_dir, overall, per_class)
    if "cnn6" in cfg.models:
        run_cnn(cfg, cnn_train, validation, run_dir, overall, per_class)
    write_csv(run_dir / "per_seed_overall.csv", overall)
    write_csv(run_dir / "per_seed_per_instrument.csv", per_class)
    write_csv(run_dir / "summary_overall_mean_sd.csv",
              summarize(overall, ("model", "condition"), ("accuracy", "macro_f1")))
    write_csv(run_dir / "summary_per_instrument_mean_sd.csv",
              summarize(per_class, ("model", "condition", "instrument"),
                        ("precision", "recall", "f1")))
    print(f"Isolated-classification results: {run_dir}")


if __name__ == "__main__":
    main()
