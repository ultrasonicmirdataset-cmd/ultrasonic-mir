"""Ultrasonic-only clean-instrument classification with the legacy CNN6 body.

The experiment uses only isolated 96 kS/s stems. Activity is decided from the
unmodified full-band waveform in non-overlapping 0.5 s windows, using the same
RMS rule as ``instrument_activity_threshold_inspector_colab.ipynb``. Only after
an active window has been selected are STFT bins below the ultrasonic cutoff
masked before the CNN sees the example.

The convolutional body matches the polyphonic notebook's historical
``Cnn6LinearSpec`` implementation: per-linear-frequency BatchNorm, four 5x5
convolution blocks with average pooling, frequency averaging, max+mean temporal
pooling, and a 512-unit fully connected layer. The head here is multiclass
because every clean window belongs to exactly one instrument.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm


ALL_INSTRUMENTS: Tuple[str, ...] = (
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


@dataclass
class ExperimentConfig:
    # Drive paths.
    drive_96_root: str = "/content/drive/MyDrive/train_96"
    drive_output_root: str = (
        "/content/drive/MyDrive/96k project/over_20_classification/runs"
    )
    local_stem_cache_root: str = "/content/ultrasonic_only_cnn6_stems"
    use_local_stem_cache: bool = True
    force_rebuild_local_cache: bool = False

    # Twelve-class experiment: exclude Bansuri, piano, and electric guitar.
    # Each can be restored independently for exploratory comparisons.
    include_bansury: bool = False
    include_electric_guitar: bool = False
    include_piano: bool = False

    # Activity is measured on the original full-band 96 kS/s stem.
    expected_sample_rate: int = 96_000
    clip_seconds: float = 0.5
    activity_threshold_dbfs: float = -70.0

    # Ultrasonic-only model input. None keeps everything up to Nyquist.
    ultrasonic_cutoff_hz: float = 20_000.0
    ultrasonic_upper_hz: Optional[float] = None

    # Contiguous temporal split within each stem. This retains every class.
    train_fraction: float = 0.70
    split_guard_seconds: float = 2.0
    data_seed: int = 42
    test_windows_per_class: int = 150
    max_train_pool_per_class: Optional[int] = None

    # Three independent model initializations over one fixed data split.
    model_seeds: Tuple[int, ...] = (42, 43, 44)

    # Full-training duration and LR schedule used by the polyphonic CNN6 code.
    epochs: int = 50
    train_samples_per_class_per_epoch: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-4
    warmup_epochs: int = 3
    lr_hold_epochs: int = 30
    minimum_lr_ratio: float = 0.10
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    random_gain_db_min: float = -6.0
    random_gain_db_max: float = 6.0
    peak_limit: float = 0.99

    # Linear STFT and the legacy CNN6 body.
    stft_window_ms: float = 32.0
    stft_hop_ms: float = 8.0
    stft_center: bool = True
    log_epsilon: float = 1e-10
    embedding_size: int = 512
    cnn_dropout: float = 0.20
    embedding_dropout: float = 0.50

    # Runtime.
    num_workers: int = 6
    prefetch_factor: int = 4
    pin_memory: bool = True
    use_amp: bool = True


def selected_instruments(cfg: ExperimentConfig) -> Tuple[str, ...]:
    excluded = set()
    if not cfg.include_bansury:
        excluded.add("Bansury")
    if not cfg.include_electric_guitar:
        excluded.add("Electric_guitar")
    if not cfg.include_piano:
        excluded.add("Piano")
    return tuple(name for name in ALL_INSTRUMENTS if name not in excluded)


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in value if character.isalnum())


def safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("._") or "item"


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.expected_sample_rate != 96_000:
        raise ValueError("This experiment is defined for the 96 kS/s master stems.")
    if not math.isclose(cfg.clip_seconds, 0.5):
        raise ValueError("The activity notebook and CNN6 experiment use 0.5 s clips.")
    if not 0.0 < cfg.train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    if cfg.split_guard_seconds < 0:
        raise ValueError("split_guard_seconds cannot be negative.")
    if not 0.0 < cfg.ultrasonic_cutoff_hz < cfg.expected_sample_rate / 2:
        raise ValueError("ultrasonic_cutoff_hz must be between 0 and Nyquist.")
    if cfg.ultrasonic_upper_hz is not None:
        if not cfg.ultrasonic_cutoff_hz < cfg.ultrasonic_upper_hz <= cfg.expected_sample_rate / 2:
            raise ValueError("ultrasonic_upper_hz must be above the cutoff and at most Nyquist.")
    if cfg.test_windows_per_class < 1:
        raise ValueError("test_windows_per_class must be positive.")
    if cfg.train_samples_per_class_per_epoch < 1:
        raise ValueError("train_samples_per_class_per_epoch must be positive.")
    if cfg.epochs < 1:
        raise ValueError("epochs must be positive.")
    if not 0 <= cfg.warmup_epochs <= cfg.lr_hold_epochs <= cfg.epochs:
        raise ValueError(
            "Expected 0 <= warmup_epochs <= lr_hold_epochs <= epochs."
        )
    if not 0.0 < cfg.minimum_lr_ratio <= 1.0:
        raise ValueError("minimum_lr_ratio must be in (0, 1].")
    if not cfg.model_seeds or len(set(cfg.model_seeds)) != len(cfg.model_seeds):
        raise ValueError("model_seeds must contain distinct seed values.")


def _is_audio_file(path: Path) -> bool:
    if not path.is_file():
        return False
    # A few files in this Drive are stored with audio MIME type but no visible
    # filename extension. Let libsndfile inspect those by header as well.
    if path.suffix and path.suffix.casefold() not in {
        ".wav",
        ".flac",
        ".aif",
        ".aiff",
    }:
        return False
    try:
        sf.info(str(path))
        return True
    except Exception:
        return False


def _resolve_instrument_directories(
    root: Path, instruments: Sequence[str]
) -> Dict[str, Path]:
    available = {
        normalize_name(path.name): path
        for path in root.iterdir()
        if path.is_dir()
    }
    result: Dict[str, Path] = {}
    missing: List[str] = []
    for instrument in instruments:
        directory = available.get(normalize_name(instrument))
        if directory is None:
            missing.append(instrument)
        else:
            result[instrument] = directory
    if missing:
        raise FileNotFoundError(
            f"Missing instrument folders under {root}: {', '.join(missing)}"
        )
    return result


def _extract_song_id(path: Path) -> Optional[str]:
    match = re.search(r"(\d+(?:\.\d+)?)$", path.stem)
    return match.group(1) if match else None


def scan_stems(root: Path, instruments: Sequence[str]) -> Dict[str, List[Path]]:
    directories = _resolve_instrument_directories(root, instruments)
    result: Dict[str, List[Path]] = {}
    for instrument, directory in directories.items():
        paths = sorted(
            (path for path in directory.iterdir() if _is_audio_file(path)),
            key=lambda path: path.name.casefold(),
        )
        if not paths:
            raise FileNotFoundError(f"No readable audio stems in {directory}")
        result[instrument] = paths
    return result


def _copy_if_needed(source: Path, destination: Path, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not force and destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    if temporary.stat().st_size != source.stat().st_size:
        raise IOError(f"Incomplete local copy: {source} -> {temporary}")
    os.replace(temporary, destination)


def prepare_local_stem_catalog(
    source_catalog: Mapping[str, Sequence[Path]], cfg: ExperimentConfig
) -> Dict[str, List[Path]]:
    if not cfg.use_local_stem_cache:
        return {key: list(value) for key, value in source_catalog.items()}

    local_root = Path(cfg.local_stem_cache_root)
    local_root.mkdir(parents=True, exist_ok=True)
    total = sum(len(paths) for paths in source_catalog.values())
    progress = tqdm(total=total, desc="Caching clean stems on Colab SSD")
    result: Dict[str, List[Path]] = {}
    for instrument, paths in source_catalog.items():
        local_paths: List[Path] = []
        for source in paths:
            destination = local_root / safe_slug(instrument) / source.name
            _copy_if_needed(source, destination, cfg.force_rebuild_local_cache)
            local_paths.append(destination)
            progress.update(1)
        result[instrument] = local_paths
    progress.close()
    return result


def _catalog_signature(
    catalog: Mapping[str, Sequence[Path]], cfg: ExperimentConfig
) -> Tuple[str, Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for instrument, paths in sorted(catalog.items()):
        for path in paths:
            info = sf.info(str(path))
            files.append(
                {
                    "instrument": instrument,
                    "file": path.name,
                    "size": int(path.stat().st_size),
                    "frames": int(info.frames),
                    "sample_rate": int(info.samplerate),
                }
            )
    payload = {
        "version": "fullband_activity_temporal_split_v1",
        "files": files,
        "clip_seconds": cfg.clip_seconds,
        "activity_threshold_dbfs": cfg.activity_threshold_dbfs,
        "train_fraction": cfg.train_fraction,
        "split_guard_seconds": cfg.split_guard_seconds,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest, payload


def build_or_load_activity_manifest(
    catalog: Mapping[str, Sequence[Path]],
    output_root: Path,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Apply the activity notebook's full-band 0.5 s RMS decision."""

    signature, signature_payload = _catalog_signature(catalog, cfg)
    cache_dir = output_root / "_activity_manifest_cache" / signature[:20]
    cache_path = cache_dir / "active_windows.csv"
    metadata_path = cache_dir / "metadata.json"
    if cache_path.is_file() and metadata_path.is_file():
        print(f"Reusing activity manifest: {cache_path}")
        return pd.read_csv(cache_path)

    rows: List[Dict[str, Any]] = []
    total_files = sum(len(paths) for paths in catalog.values())
    progress = tqdm(total=total_files, desc="Full-band 0.5 s RMS activity scan")
    for instrument, paths in catalog.items():
        for path in paths:
            with sf.SoundFile(str(path), mode="r") as handle:
                sample_rate = int(handle.samplerate)
                if sample_rate != cfg.expected_sample_rate:
                    raise RuntimeError(
                        f"Sample-rate mismatch for {path}: {sample_rate} != {cfg.expected_sample_rate}"
                    )
                total_frames = int(len(handle))
                duration = total_frames / sample_rate
                block_frames = int(round(cfg.clip_seconds * sample_rate))
                split_time = cfg.train_fraction * duration
                block_index = 0
                while True:
                    block = handle.read(block_frames, dtype="float32", always_2d=True)
                    if len(block) < block_frames:
                        break
                    start = block_index * cfg.clip_seconds
                    end = start + cfg.clip_seconds
                    block_index += 1

                    # Same channel averaging and RMS convention as the inspector.
                    mono = np.mean(block, axis=1, dtype=np.float64)
                    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
                    rms_dbfs = 20.0 * math.log10(max(rms, 1e-10))
                    if rms_dbfs < cfg.activity_threshold_dbfs:
                        continue

                    if end <= split_time - cfg.split_guard_seconds:
                        split = "train"
                    elif start >= split_time + cfg.split_guard_seconds:
                        split = "test"
                    else:
                        continue

                    rows.append(
                        {
                            "instrument": instrument,
                            "class_index": -1,
                            "file": path.name,
                            "path": str(path),
                            "song_id": _extract_song_id(path),
                            "start_seconds": float(start),
                            "end_seconds": float(end),
                            "rms_dbfs_fullband": float(rms_dbfs),
                            "split": split,
                        }
                    )
            progress.update(1)
    progress.close()

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No active windows passed the full-band RMS threshold.")
    class_to_index = {
        instrument: index for index, instrument in enumerate(catalog.keys())
    }
    manifest["class_index"] = manifest["instrument"].map(class_to_index).astype(int)
    manifest = manifest.sort_values(
        ["split", "instrument", "file", "start_seconds"]
    ).reset_index(drop=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cache_path, index=False)
    save_json(
        metadata_path,
        {
            **signature_payload,
            "signature": signature,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_rows": len(manifest),
        },
    )
    return manifest


def select_balanced_experiment_windows(
    manifest: pd.DataFrame, instruments: Sequence[str], cfg: ExperimentConfig
) -> pd.DataFrame:
    selected: List[pd.DataFrame] = []
    problems: List[str] = []
    for class_index, instrument in enumerate(instruments):
        train = manifest[
            (manifest["instrument"] == instrument) & (manifest["split"] == "train")
        ].copy()
        test = manifest[
            (manifest["instrument"] == instrument) & (manifest["split"] == "test")
        ].copy()
        if train.empty:
            problems.append(f"{instrument}: no active training windows")
            continue
        if len(test) < cfg.test_windows_per_class:
            problems.append(
                f"{instrument}: {len(test)} test windows, need {cfg.test_windows_per_class}"
            )
            continue

        train = train.sample(
            frac=1.0,
            random_state=cfg.data_seed + 1009 * class_index,
        )
        if cfg.max_train_pool_per_class is not None:
            train = train.head(cfg.max_train_pool_per_class)
        test = test.sample(
            n=cfg.test_windows_per_class,
            replace=False,
            random_state=cfg.data_seed + 2003 * class_index,
        )
        selected.extend([train, test])

    if problems:
        raise RuntimeError("Invalid activity/split coverage:\n- " + "\n- ".join(problems))

    result = pd.concat(selected, ignore_index=True)
    class_to_index = {name: index for index, name in enumerate(instruments)}
    result["class_index"] = result["instrument"].map(class_to_index).astype(int)
    result = result.sort_values(
        ["split", "instrument", "file", "start_seconds"]
    ).reset_index(drop=True)
    return result


class CleanWindowDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        cfg: ExperimentConfig,
        training: bool,
        seed: int,
    ) -> None:
        self.rows = manifest.to_dict("records")
        self.cfg = cfg
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.frames = int(round(cfg.clip_seconds * cfg.expected_sample_rate))

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        start_frame = int(round(float(row["start_seconds"]) * self.cfg.expected_sample_rate))
        with sf.SoundFile(str(row["path"]), mode="r") as handle:
            handle.seek(start_frame)
            block = handle.read(self.frames, dtype="float32", always_2d=True)
        if len(block) != self.frames:
            raise RuntimeError(
                f"Short read for {row['path']} at {row['start_seconds']} s: "
                f"{len(block)} != {self.frames}"
            )
        audio = np.mean(block, axis=1, dtype=np.float32)

        if self.training:
            token = (
                f"gain|{self.seed}|{self.epoch}|{index}|{row['instrument']}|"
                f"{row['file']}|{row['start_seconds']}"
            )
            digest = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(digest)
            gain_db = float(
                rng.uniform(self.cfg.random_gain_db_min, self.cfg.random_gain_db_max)
            )
            audio = audio * (10.0 ** (gain_db / 20.0))
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            if peak > self.cfg.peak_limit > 0:
                audio *= self.cfg.peak_limit / peak

        return {
            "audio": torch.from_numpy(audio.astype(np.float32, copy=False)),
            "target": int(row["class_index"]),
            "manifest_index": int(index),
        }


class BalancedClassSampler(Sampler[int]):
    def __init__(
        self,
        manifest: pd.DataFrame,
        class_count: int,
        samples_per_class: int,
        seed: int,
    ) -> None:
        self.by_class: Dict[int, np.ndarray] = {}
        labels = manifest["class_index"].to_numpy(dtype=np.int64)
        for class_index in range(class_count):
            pool = np.flatnonzero(labels == class_index)
            if len(pool) == 0:
                raise RuntimeError(f"Empty training pool for class index {class_index}")
            self.by_class[class_index] = pool
        self.class_count = int(class_count)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.class_count * self.samples_per_class

    def __iter__(self) -> Iterable[int]:
        rng = np.random.default_rng(
            (self.seed + (self.epoch + 1) * 2_000_003) % (2**32)
        )
        values: List[int] = []
        for class_index in range(self.class_count):
            pool = self.by_class[class_index]
            chosen = rng.choice(
                pool,
                size=self.samples_per_class,
                replace=len(pool) < self.samples_per_class,
            )
            values.extend(int(item) for item in chosen)
        rng.shuffle(values)
        return iter(values)


def init_layer(layer: nn.Module) -> None:
    if hasattr(layer, "weight") and getattr(layer, "weight") is not None:
        nn.init.xavier_uniform_(getattr(layer, "weight"))
    if hasattr(layer, "bias") and getattr(layer, "bias") is not None:
        nn.init.zeros_(getattr(layer, "bias"))


def init_bn(layer: nn.Module) -> None:
    if hasattr(layer, "bias") and getattr(layer, "bias") is not None:
        nn.init.zeros_(getattr(layer, "bias"))
    if hasattr(layer, "weight") and getattr(layer, "weight") is not None:
        nn.init.ones_(getattr(layer, "weight"))


class ConvBlock5x5(nn.Module):
    """The one-convolution block used by the polyphonic notebook's CNN6."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(
        self,
        x: torch.Tensor,
        pool_size: Tuple[int, int] = (2, 2),
        pool_type: str = "avg",
    ) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == "avg":
            return F.avg_pool2d(x, kernel_size=pool_size)
        if pool_type == "max":
            return F.max_pool2d(x, kernel_size=pool_size)
        if pool_type == "avg+max":
            return F.avg_pool2d(x, pool_size) + F.max_pool2d(x, pool_size)
        raise ValueError(f"Unsupported pool_type={pool_type!r}")


class Cnn6LinearSpecUltrasonicClassifier(nn.Module):
    """The legacy CNN6 body with a multiclass clean-stem output head."""

    def __init__(self, cfg: ExperimentConfig, classes_num: int) -> None:
        super().__init__()
        self.sample_rate = int(cfg.expected_sample_rate)
        self.window_size = int(round(self.sample_rate * cfg.stft_window_ms / 1000.0))
        self.hop_size = int(round(self.sample_rate * cfg.stft_hop_ms / 1000.0))
        self.freq_bins = self.window_size // 2 + 1
        self.log_epsilon = float(cfg.log_epsilon)
        self.stft_center = bool(cfg.stft_center)
        self.register_buffer(
            "hann_window", torch.hann_window(self.window_size), persistent=False
        )
        frequencies = torch.fft.rfftfreq(
            self.window_size, d=1.0 / self.sample_rate
        )
        keep = frequencies >= float(cfg.ultrasonic_cutoff_hz)
        if cfg.ultrasonic_upper_hz is not None:
            keep &= frequencies <= float(cfg.ultrasonic_upper_hz)
        self.register_buffer("ultrasonic_keep", keep, persistent=True)

        # Exact historical input normalization: one BN channel per linear bin.
        self.bn0 = nn.BatchNorm2d(self.freq_bins)
        init_bn(self.bn0)
        self.conv_block1 = ConvBlock5x5(1, 64)
        self.conv_block2 = ConvBlock5x5(64, 128)
        self.conv_block3 = ConvBlock5x5(128, 256)
        self.conv_block4 = ConvBlock5x5(256, 512)
        self.fc1 = nn.Linear(512, cfg.embedding_size, bias=True)
        self.classifier = nn.Linear(cfg.embedding_size, classes_num, bias=True)
        init_layer(self.fc1)
        init_layer(self.classifier)
        self.cnn_dropout = float(cfg.cnn_dropout)
        self.embedding_dropout = float(cfg.embedding_dropout)

    def extract_log_linear_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        device_type = waveform.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            complex_spec = torch.stft(
                waveform.float(),
                n_fft=self.window_size,
                hop_length=self.hop_size,
                win_length=self.window_size,
                window=self.hann_window.float(),
                center=self.stft_center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            power = complex_spec.abs().square()

            # The activity decision has already been made on full-band audio.
            # This mask is the only bandwidth operation seen by the classifier.
            power = power.masked_fill(~self.ultrasonic_keep[None, :, None], 0.0)
            x = torch.log(power + self.log_epsilon)
            x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, T, F]
        return x

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.extract_log_linear_spectrogram(waveform)
        x = x.transpose(1, 3)  # [B, F, T, 1]
        x = self.bn0(x)
        x = x.transpose(1, 3)  # [B, 1, T, F]

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=self.cnn_dropout, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=self.cnn_dropout, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=self.cnn_dropout, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=self.cnn_dropout, training=self.training)

        x = torch.mean(x, dim=3)  # historical global frequency average
        x = torch.max(x, dim=2).values + torch.mean(x, dim=2)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        return F.relu_(self.fc1(x))

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.extract_embedding(waveform)
        return {"logits": self.classifier(embedding), "embedding": embedding}


def _loader_kwargs(cfg: ExperimentConfig) -> Dict[str, Any]:
    if cfg.num_workers <= 0:
        return {}
    return {
        "prefetch_factor": cfg.prefetch_factor,
        "persistent_workers": False,
    }


def make_loaders(
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    class_count: int,
    seed: int,
    cfg: ExperimentConfig,
) -> Tuple[DataLoader, DataLoader, CleanWindowDataset, BalancedClassSampler]:
    train_dataset = CleanWindowDataset(train_manifest, cfg, training=True, seed=seed)
    test_dataset = CleanWindowDataset(test_manifest, cfg, training=False, seed=seed)
    sampler = BalancedClassSampler(
        train_manifest,
        class_count,
        cfg.train_samples_per_class_per_epoch,
        seed=seed + 31_337,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
        **_loader_kwargs(cfg),
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, test_loader, train_dataset, sampler


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    truth: List[np.ndarray] = []
    prediction: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    for batch in loader:
        audio = batch["audio"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(audio)["logits"]
        probability = torch.softmax(logits.float(), dim=1)
        truth.append(batch["target"].numpy())
        prediction.append(torch.argmax(probability, dim=1).cpu().numpy())
        probabilities.append(probability.cpu().numpy())
    y_true = np.concatenate(truth)
    y_pred = np.concatenate(prediction)
    y_prob = np.concatenate(probabilities)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return metrics, y_true, y_pred, y_prob


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _plot_history(history: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[1].plot(history["epoch"], 100 * history["train_accuracy"], label="Train")
    axes[1].plot(history["epoch"], 100 * history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percent")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_confusion(
    matrix: np.ndarray, instruments: Sequence[str], path: Path
) -> None:
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sum, 1)
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(instruments)), instruments, rotation=60, ha="right")
    axis.set_yticks(range(len(instruments)), instruments)
    axis.set_xlabel("Predicted instrument")
    axis.set_ylabel("True instrument")
    axis.set_title("Ultrasonic-only CNN6 confusion matrix (row normalized)")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def train_one_seed(
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    instruments: Sequence[str],
    seed: int,
    seed_dir: Path,
    cfg: ExperimentConfig,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray]:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    train_loader, test_loader, train_dataset, sampler = make_loaders(
        train_manifest, test_manifest, len(instruments), seed, cfg
    )
    model = Cnn6LinearSpecUltrasonicClassifier(cfg, len(instruments)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    def lr_schedule(epoch_index: int) -> float:
        """Polyphonic CNN6 schedule: warm-up, hold, then cosine decay."""
        epoch_number = int(epoch_index) + 1
        if cfg.warmup_epochs > 0 and epoch_number <= cfg.warmup_epochs:
            return epoch_number / cfg.warmup_epochs
        if epoch_number <= cfg.lr_hold_epochs:
            return 1.0
        cosine_epochs = max(1, cfg.epochs - cfg.lr_hold_epochs)
        progress = min(
            1.0,
            max(0.0, (epoch_number - cfg.lr_hold_epochs) / cosine_epochs),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.minimum_lr_ratio + (1.0 - cfg.minimum_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_schedule,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_accuracy = -math.inf
    best_epoch = 0
    history_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 78)
    print(f"CNN6 MODEL SEED {seed} | device={device} | AMP={amp_enabled}")
    print("=" * 78)
    for epoch in range(cfg.epochs):
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        train_dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(
            train_loader,
            desc=f"seed {seed} epoch {epoch + 1}/{cfg.epochs}",
            leave=False,
        )
        for batch in progress:
            audio = batch["audio"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(audio)["logits"]
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * len(target)
            correct += int((torch.argmax(logits.detach(), dim=1) == target).sum().item())
            total += len(target)
            progress.set_postfix(
                loss=f"{running_loss / max(total, 1):.4f}",
                acc=f"{correct / max(total, 1):.3f}",
            )

        validation, _, _, _ = evaluate_model(
            model, test_loader, device, amp_enabled
        )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(total, 1),
            "train_accuracy": correct / max(total, 1),
            "val_accuracy": validation["accuracy"],
            "val_balanced_accuracy": validation["balanced_accuracy"],
            "val_macro_f1": validation["macro_f1"],
            "learning_rate": epoch_learning_rate,
        }
        history_rows.append(row)
        history = pd.DataFrame(history_rows)
        history.to_csv(seed_dir / "training_history.csv", index=False)
        print(
            f"seed={seed} epoch={epoch + 1:02d}/{cfg.epochs} | "
            f"loss={row['train_loss']:.4f} | train={100 * row['train_accuracy']:.2f}% | "
            f"val={100 * row['val_accuracy']:.2f}% | "
            f"macro-F1={100 * row['val_macro_f1']:.2f}%",
            flush=True,
        )

        checkpoint = {
            "epoch": epoch + 1,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history_rows,
            "instruments": list(instruments),
            "config": asdict(cfg),
        }
        _atomic_torch_save(checkpoint, seed_dir / "last_model.pth")
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch + 1
            _atomic_torch_save(checkpoint, seed_dir / "best_model.pth")

    best = torch.load(seed_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    final, y_true, y_pred, y_prob = evaluate_model(
        model, test_loader, device, amp_enabled
    )
    labels = np.arange(len(instruments))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_instrument = pd.DataFrame(
        {
            "instrument": list(instruments),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
            "model_seed": seed,
        }
    )
    per_instrument.to_csv(seed_dir / "per_instrument_metrics.csv", index=False)

    overall = {
        "model_seed": seed,
        "best_epoch": best_epoch,
        **final,
        "train_windows_available": len(train_manifest),
        "test_windows": len(test_manifest),
        "classes": len(instruments),
    }
    pd.DataFrame([overall]).to_csv(seed_dir / "overall_metrics.csv", index=False)

    prediction_rows = test_manifest.reset_index(drop=True).copy()
    prediction_rows["true_index"] = y_true
    prediction_rows["predicted_index"] = y_pred
    prediction_rows["predicted_instrument"] = [instruments[index] for index in y_pred]
    prediction_rows["correct"] = (y_true == y_pred).astype(np.uint8)
    prediction_rows["predicted_probability"] = y_prob[np.arange(len(y_pred)), y_pred]
    prediction_rows.to_csv(seed_dir / "predictions.csv", index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matrix, index=instruments, columns=instruments).to_csv(
        seed_dir / "confusion_matrix_counts.csv"
    )
    _plot_confusion(matrix, instruments, seed_dir / "confusion_matrix.png")
    _plot_history(pd.DataFrame(history_rows), seed_dir / "training_history.png")
    (seed_dir / "SEED_COMPLETE.txt").write_text(
        f"Completed UTC: {datetime.now(timezone.utc).isoformat()}\n"
        f"Seed: {seed}\nAccuracy: {final['accuracy']:.8f}\n",
        encoding="utf-8",
    )

    del model, optimizer, scheduler, scaler, train_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return overall, per_instrument, pd.DataFrame(history_rows), matrix


def _make_run_directory(
    output_root: Path, instruments: Sequence[str], cfg: ExperimentConfig
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    optional_tag = (
        f"eg{int(cfg.include_electric_guitar)}_p{int(cfg.include_piano)}"
    )
    seeds_tag = "-".join(str(seed) for seed in cfg.model_seeds)
    name = (
        f"cnn6_ultrasonic_only_{timestamp}_over{cfg.ultrasonic_cutoff_hz:g}Hz_"
        f"c{len(instruments)}_{optional_tag}_dseed{cfg.data_seed}_mseed{seeds_tag}"
    )
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _aggregate_results(
    overall_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[pd.DataFrame],
    matrices: Sequence[np.ndarray],
    instruments: Sequence[str],
    run_dir: Path,
) -> Dict[str, Any]:
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(run_dir / "all_seeds_overall_metrics.csv", index=False)
    per_seed = pd.concat(per_seed_rows, ignore_index=True)
    per_seed.to_csv(run_dir / "all_seeds_per_instrument_metrics.csv", index=False)

    numeric = ["accuracy", "balanced_accuracy", "macro_f1"]
    summary_rows: List[Dict[str, Any]] = []
    for metric in numeric:
        values = overall[metric].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "runs": len(values),
            }
        )
    overall_summary = pd.DataFrame(summary_rows)
    overall_summary.to_csv(run_dir / "all_seeds_mean_std.csv", index=False)

    aggregate = (
        per_seed.groupby("instrument", sort=False)[["precision", "recall", "f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        "instrument",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
    ]
    aggregate = aggregate.set_index("instrument").loc[list(instruments)].reset_index()
    aggregate.to_csv(run_dir / "per_instrument_mean_std.csv", index=False)
    aggregate[["instrument", "f1_mean", "f1_std"]].to_csv(
        run_dir / "paper_table_ultrasonic_only_f1.csv", index=False
    )

    summed_matrix = np.sum(np.stack(matrices, axis=0), axis=0)
    pd.DataFrame(summed_matrix, index=instruments, columns=instruments).to_csv(
        run_dir / "all_seeds_confusion_matrix_counts.csv"
    )
    _plot_confusion(
        summed_matrix,
        instruments,
        run_dir / "all_seeds_confusion_matrix.png",
    )

    return {
        row["metric"]: {"mean": row["mean"], "std": row["std"]}
        for row in summary_rows
    }


def describe_configuration(cfg: ExperimentConfig) -> Dict[str, Any]:
    instruments = selected_instruments(cfg)
    return {
        "classes": len(instruments),
        "instruments": instruments,
        "excluded": tuple(name for name in ALL_INSTRUMENTS if name not in instruments),
        "activity": (
            f"full-band RMS >= {cfg.activity_threshold_dbfs:.1f} dBFS in complete "
            f"{cfg.clip_seconds:.1f} s windows"
        ),
        "model_input": (
            f"96 kS/s linear STFT bins >= {cfg.ultrasonic_cutoff_hz:g} Hz"
        ),
        "split": (
            f"{100 * cfg.train_fraction:.0f}/{100 * (1 - cfg.train_fraction):.0f} "
            f"temporal per stem, {cfg.split_guard_seconds:g} s guard on each side"
        ),
        "test_windows_per_class": cfg.test_windows_per_class,
        "model_seeds": cfg.model_seeds,
        "epochs": cfg.epochs,
    }


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    validate_config(cfg)
    instruments = selected_instruments(cfg)
    source_root = Path(cfg.drive_96_root)
    output_root = Path(cfg.drive_output_root)
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"96 kS/s data root not found: {source_root}. Mount Google Drive first."
        )

    print("=" * 80)
    print("ULTRASONIC-ONLY CLEAN-INSTRUMENT CNN6 CLASSIFICATION")
    print("=" * 80)
    for key, value in describe_configuration(cfg).items():
        print(f"{key}: {value}")

    source_catalog = scan_stems(source_root, instruments)
    local_catalog = prepare_local_stem_catalog(source_catalog, cfg)
    manifest = build_or_load_activity_manifest(local_catalog, output_root, cfg)
    experiment_manifest = select_balanced_experiment_windows(
        manifest, instruments, cfg
    )
    counts = (
        experiment_manifest.groupby(["instrument", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(instruments)
    )
    print("\nSelected active-window counts:")
    print(counts.to_string())

    run_dir = _make_run_directory(output_root, instruments, cfg)
    print(f"\nTimestamped output: {run_dir}")
    save_json(
        run_dir / "run_config.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
            "resolved_instruments": instruments,
            "architecture": (
                "legacy Cnn6LinearSpec: per-frequency BN, four 5x5 blocks, "
                "frequency mean, temporal max+mean, 512-unit fc"
            ),
            "activity_source": "unmodified full-band 96 kS/s waveform",
            "bandwidth_operation": (
                "STFT power bins below cutoff are set to zero only after activity selection"
            ),
        },
    )
    manifest.to_csv(run_dir / "all_active_windows_before_balancing.csv", index=False)
    experiment_manifest.to_csv(run_dir / "experiment_window_manifest.csv", index=False)
    counts.reset_index().to_csv(run_dir / "window_counts.csv", index=False)

    train_manifest = experiment_manifest[
        experiment_manifest["split"] == "train"
    ].reset_index(drop=True)
    test_manifest = experiment_manifest[
        experiment_manifest["split"] == "test"
    ].reset_index(drop=True)

    overall_rows: List[Mapping[str, Any]] = []
    per_seed_rows: List[pd.DataFrame] = []
    matrices: List[np.ndarray] = []
    for seed in cfg.model_seeds:
        seed_dir = run_dir / f"model_seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=False)
        overall, per_instrument, _, matrix = train_one_seed(
            train_manifest,
            test_manifest,
            instruments,
            seed,
            seed_dir,
            cfg,
        )
        overall_rows.append(overall)
        per_seed_rows.append(per_instrument)
        matrices.append(matrix)

    summary = _aggregate_results(
        overall_rows, per_seed_rows, matrices, instruments, run_dir
    )
    (run_dir / "RUN_COMPLETE.txt").write_text(
        f"Completed UTC: {datetime.now(timezone.utc).isoformat()}\n"
        f"Classes: {len(instruments)}\n"
        f"Seeds: {list(cfg.model_seeds)}\n"
        f"Accuracy mean: {summary['accuracy']['mean']:.8f}\n"
        f"Accuracy std: {summary['accuracy']['std']:.8f}\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("THREE-SEED SUMMARY")
    print("=" * 80)
    for metric, values in summary.items():
        print(
            f"{metric:>18}: {100 * values['mean']:.2f}% "
            f"+/- {100 * values['std']:.2f}%"
        )
    print(f"Full results: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "instruments": instruments,
        "summary": summary,
    }


def model_smoke_test() -> Dict[str, Any]:
    """Fast CPU/GPU shape check; does not read Drive or train a model."""
    cfg = ExperimentConfig(model_seeds=(42,), num_workers=0)
    model = Cnn6LinearSpecUltrasonicClassifier(
        cfg, classes_num=len(selected_instruments(cfg))
    )
    model.eval()
    waveform = torch.zeros(2, int(cfg.expected_sample_rate * cfg.clip_seconds))
    with torch.no_grad():
        output = model(waveform)
    kept = int(model.ultrasonic_keep.sum().item())
    return {
        "logits_shape": tuple(output["logits"].shape),
        "embedding_shape": tuple(output["embedding"].shape),
        "frequency_bins": model.freq_bins,
        "ultrasonic_bins_kept": kept,
        "cutoff_hz": cfg.ultrasonic_cutoff_hz,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Train the three-seed experiment")
    parser.add_argument("--data-root", help="Path to train_96")
    parser.add_argument("--output-root", help="Directory for timestamped runs")
    parser.add_argument("--local-cache-root", help="Local audio cache directory")
    args = parser.parse_args()
    if args.run:
        config = ExperimentConfig()
        if args.data_root:
            config.drive_96_root = args.data_root
        if args.output_root:
            config.drive_output_root = args.output_root
        if args.local_cache_root:
            config.local_stem_cache_root = args.local_cache_root
        print(run_experiment(config))
    else:
        print(model_smoke_test())
