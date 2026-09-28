"""Independent-recording ultrasonic silence control with CNN6.

This negative-control experiment uses the raw studio recordings under
``valid_96`` rather than DAW-gated stems.  A complete raw recording is assigned
to exactly one side of the split.  Quiet windows are selected only when the
maximum per-channel RMS in the audible 0--20 kHz band is strictly below
-80 dBFS.  The classifier then receives only the 20--48 kHz STFT bins.

The experiment is intentionally conservative: exact/long digital-zero patterns
are rejected, decoded-audio hashes detect duplicate recordings, train and
validation files are disjoint, and equal numbers of unique windows are selected
for every admitted class on both sides of the split.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm

import ultrasonic_only_cnn6 as base


RAW_FOLDER_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "Acoustic_guitar": ("Acustic guitar", "Acoustic guitar"),
    "Bansury": ("Bansuri", "Bansury"),
    "Clarinet": ("klarinet", "Clarinet"),
    "Classic_guitar": ("Classic guitar", "Classical guitar"),
    "Drums": ("Drums",),
    "Duble_bass": ("Duble bass", "Double bass"),
    "Electric_guitar": ("Electric guitar",),
    "Flute": ("Fluet", "Flute"),
    "Harmonica": ("Harmonica",),
    "Kanun": ("Kanun",),
    "Piano": ("Piano",),
    "Saxophone": ("Saxophon tenor", "Saxophone tenor", "Saxophone"),
    "Trumpet": ("Trumpt", "Trumpet"),
    "Violin": ("Violin",),
    "Vocals": ("Vocal", "Vocals"),
}


@dataclass
class RawSilenceConfig(base.ExperimentConfig):
    drive_96_root: str = "/content/drive/MyDrive/valid_96"
    drive_output_root: str = (
        "/content/drive/MyDrive/96k project/over_20_classification/raw_silence_runs"
    )
    local_stem_cache_root: str = "/content/ultrasonic_raw_silence_recordings"

    quiet_threshold_dbfs: float = -80.0
    audible_upper_hz: float = 20_000.0
    maximum_exact_zero_fraction: float = 0.95
    maximum_exact_zero_run_seconds: float = 0.100

    # The default split is deterministic and independent of signal values: sort
    # each instrument's filenames and hold out the last complete recording.
    validation_filename_by_instrument: Dict[str, str] = field(default_factory=dict)
    minimum_recordings_per_class: int = 2
    minimum_unique_train_windows_per_class: int = 100
    minimum_unique_validation_windows_per_class: int = 100
    maximum_balanced_train_windows_per_class: Optional[int] = None
    test_windows_per_class: int = 150
    experiment_instruments: Optional[Tuple[str, ...]] = None

    ultrasonic_cutoff_hz: float = 20_000.0
    ultrasonic_upper_hz: Optional[float] = 48_000.0
    include_electric_guitar: bool = True
    include_piano: bool = True
    require_cuda_for_training: bool = True
    force_rebuild_quiet_manifest: bool = False


def validate_config(cfg: RawSilenceConfig) -> None:
    base.validate_config(cfg)
    if not 0.0 < cfg.audible_upper_hz <= cfg.ultrasonic_cutoff_hz:
        raise ValueError("audible_upper_hz must be in (0, ultrasonic_cutoff_hz].")
    if cfg.minimum_recordings_per_class < 2:
        raise ValueError("Independent validation requires at least two recordings.")
    if cfg.minimum_unique_train_windows_per_class < 1:
        raise ValueError("minimum_unique_train_windows_per_class must be positive.")
    if cfg.minimum_unique_validation_windows_per_class < 1:
        raise ValueError(
            "minimum_unique_validation_windows_per_class must be positive."
        )
    if cfg.test_windows_per_class < cfg.minimum_unique_validation_windows_per_class:
        raise ValueError(
            "test_windows_per_class cannot be below the validation minimum."
        )
    if not 0.0 <= cfg.maximum_exact_zero_fraction <= 1.0:
        raise ValueError("maximum_exact_zero_fraction must be in [0, 1].")
    if cfg.maximum_exact_zero_run_seconds < 0:
        raise ValueError("maximum_exact_zero_run_seconds cannot be negative.")
    if cfg.experiment_instruments is not None:
        unknown = sorted(set(cfg.experiment_instruments) - set(base.ALL_INSTRUMENTS))
        if unknown:
            raise ValueError(f"Unknown experiment instruments: {unknown}")


def _resolve_raw_catalog(root: Path) -> Dict[str, List[Path]]:
    available = {
        base.normalize_name(path.name): path for path in root.iterdir() if path.is_dir()
    }
    catalog: Dict[str, List[Path]] = {}
    missing: List[str] = []
    for instrument in base.ALL_INSTRUMENTS:
        directory: Optional[Path] = None
        for alias in RAW_FOLDER_ALIASES[instrument]:
            directory = available.get(base.normalize_name(alias))
            if directory is not None:
                break
        if directory is None:
            missing.append(instrument)
            continue
        files = sorted(
            [path for path in directory.iterdir() if base._is_audio_file(path)],
            key=lambda path: (base.normalize_name(path.name), path.name.casefold()),
        )
        if not files:
            missing.append(instrument)
        else:
            catalog[instrument] = files
    if missing:
        raise FileNotFoundError(
            "Missing raw recording folders/files for: " + ", ".join(missing)
        )
    return catalog


def _split_assignments(
    catalog: Mapping[str, Sequence[Path]], cfg: RawSilenceConfig
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for instrument in base.ALL_INSTRUMENTS:
        paths = list(catalog[instrument])
        override = cfg.validation_filename_by_instrument.get(instrument)
        if override:
            matches = [path for path in paths if path.name == override]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Validation override for {instrument} did not match exactly one "
                    f"file: {override!r}"
                )
            validation_path: Optional[Path] = matches[0]
        elif len(paths) >= cfg.minimum_recordings_per_class:
            validation_path = paths[-1]
        else:
            validation_path = None

        for recording_index, path in enumerate(paths, start=1):
            split = "validation" if validation_path == path else "train"
            if validation_path is None:
                split = "insufficient_recordings"
            info = sf.info(str(path))
            rows.append(
                {
                    "instrument": instrument,
                    "recording_index": recording_index,
                    "file": path.name,
                    "path": str(path),
                    "split": split,
                    "bytes": int(path.stat().st_size),
                    "frames": int(info.frames),
                    "sample_rate": int(info.samplerate),
                    "channels": int(info.channels),
                    "duration_seconds": float(info.frames / info.samplerate),
                }
            )
    return pd.DataFrame(rows)


def _longest_true_run(mask: np.ndarray) -> int:
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return 0
    boundaries = np.flatnonzero(np.diff(positions) > 1)
    starts = np.r_[0, boundaries + 1]
    ends = np.r_[boundaries, len(positions) - 1]
    return int(np.max(positions[ends] - positions[starts] + 1))


def _band_rms(
    block: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    include_high: bool,
) -> Tuple[np.ndarray, float]:
    values = block.astype(np.float64, copy=False)
    frame_count = int(values.shape[0])
    spectrum = np.fft.rfft(values, axis=0)
    frequencies = np.fft.rfftfreq(frame_count, d=1.0 / sample_rate)
    if include_high:
        keep = (frequencies >= low_hz) & (frequencies <= high_hz)
    else:
        keep = (frequencies >= low_hz) & (frequencies < high_hz)
    weights = np.full(len(frequencies), 2.0, dtype=np.float64)
    weights[0] = 1.0
    if frame_count % 2 == 0:
        weights[-1] = 1.0
    channel_power = weights[keep, None] * np.square(np.abs(spectrum[keep]))
    channel_rms = np.sqrt(np.sum(channel_power, axis=0)) / frame_count
    mono_spectrum = np.mean(spectrum, axis=1)
    mono_power = weights[keep] * np.square(np.abs(mono_spectrum[keep]))
    mono_rms = float(np.sqrt(np.sum(mono_power)) / frame_count)
    return channel_rms, mono_rms


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-10))


def _window_metrics(block: np.ndarray, cfg: RawSilenceConfig) -> Dict[str, Any]:
    values = block.astype(np.float64, copy=False)
    mono = np.mean(values, axis=1)
    audible_channel_rms, audible_mono_rms = _band_rms(
        values, cfg.expected_sample_rate, 0.0, cfg.audible_upper_hz, False
    )
    _, ultrasonic_mono_rms = _band_rms(
        values,
        cfg.expected_sample_rate,
        cfg.ultrasonic_cutoff_hz,
        cfg.ultrasonic_upper_hz or cfg.expected_sample_rate / 2,
        True,
    )
    maximum_audible_channel_dbfs = max(_dbfs(value) for value in audible_channel_rms)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    all_channels_zero = np.all(values == 0.0, axis=1)
    exact_zero_fraction = float(np.mean(all_channels_zero))
    longest_zero_seconds = (
        _longest_true_run(all_channels_zero) / cfg.expected_sample_rate
    )
    digital_zero = bool(peak == 0.0)
    mono_rms = float(np.sqrt(np.mean(np.square(mono))))
    mono_cancellation = bool(mono_rms == 0.0 and peak > 0.0)
    excessive_zero_fraction = bool(
        exact_zero_fraction > cfg.maximum_exact_zero_fraction
    )
    excessive_zero_run = bool(
        longest_zero_seconds >= cfg.maximum_exact_zero_run_seconds
    )
    zero_rejected = bool(
        digital_zero
        or mono_cancellation
        or excessive_zero_fraction
        or excessive_zero_run
    )
    passes_quiet = bool(
        maximum_audible_channel_dbfs < cfg.quiet_threshold_dbfs
        and not zero_rejected
    )
    if digital_zero:
        decision = "digital_zero"
    elif mono_cancellation:
        decision = "exact_mono_cancellation"
    elif excessive_zero_fraction:
        decision = "excessive_zero_fraction"
    elif excessive_zero_run:
        decision = "long_zero_run"
    elif maximum_audible_channel_dbfs >= cfg.quiet_threshold_dbfs:
        decision = "audible_rms_at_or_above_quiet_threshold"
    else:
        decision = "accepted"
    return {
        "rms_dbfs_fullband_mono": _dbfs(mono_rms),
        "rms_dbfs_audible_0_20k_mono": _dbfs(audible_mono_rms),
        "rms_dbfs_audible_0_20k_max_channel": maximum_audible_channel_dbfs,
        "rms_dbfs_ultrasonic_20_48k_mono": _dbfs(ultrasonic_mono_rms),
        "peak_absolute": peak,
        "exact_zero_fraction": exact_zero_fraction,
        "longest_exact_zero_run_seconds": longest_zero_seconds,
        "is_digital_zero": digital_zero,
        "is_exact_mono_cancellation": mono_cancellation,
        "excessive_exact_zero_fraction": excessive_zero_fraction,
        "excessive_exact_zero_run": excessive_zero_run,
        "passes_quiet": passes_quiet,
        "decision": decision,
    }


def _catalog_signature(
    assignments: pd.DataFrame, cfg: RawSilenceConfig
) -> Tuple[str, Dict[str, Any]]:
    files = assignments[
        [
            "instrument",
            "file",
            "split",
            "bytes",
            "frames",
            "sample_rate",
            "channels",
        ]
    ].to_dict("records")
    payload = {
        "version": "raw_studio_independent_recording_silence_control_v1",
        "files": files,
        "clip_seconds": cfg.clip_seconds,
        "quiet_threshold_dbfs": cfg.quiet_threshold_dbfs,
        "selection_band_hz": [0.0, cfg.audible_upper_hz],
        "selection_metric": "maximum_per_channel_rfft_band_rms",
        "model_band_hz": [cfg.ultrasonic_cutoff_hz, cfg.ultrasonic_upper_hz],
        "zero_fraction_limit": cfg.maximum_exact_zero_fraction,
        "zero_run_seconds_limit": cfg.maximum_exact_zero_run_seconds,
    }
    digest = hashlib.sha256(
        json.dumps(base.json_ready(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest, payload


def build_or_load_manifest(
    catalog: Mapping[str, Sequence[Path]],
    assignments: pd.DataFrame,
    output_root: Path,
    cfg: RawSilenceConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    signature, signature_payload = _catalog_signature(assignments, cfg)
    cache_dir = output_root / "_raw_silence_manifest_cache" / signature[:20]
    manifest_path = cache_dir / "all_scanned_windows.csv"
    files_path = cache_dir / "recording_audit.csv"
    metadata_path = cache_dir / "metadata.json"
    if (
        not cfg.force_rebuild_quiet_manifest
        and manifest_path.is_file()
        and files_path.is_file()
        and metadata_path.is_file()
    ):
        print(f"Reusing raw-silence manifest: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        for column in (
            "is_digital_zero",
            "is_exact_mono_cancellation",
            "excessive_exact_zero_fraction",
            "excessive_exact_zero_run",
            "passes_quiet",
        ):
            manifest[column] = manifest[column].astype(bool)
        current_paths = {
            (row.instrument, row.file): row.path
            for row in assignments.itertuples(index=False)
        }
        manifest["path"] = [
            current_paths[(instrument, file_name)]
            for instrument, file_name in zip(
                manifest["instrument"], manifest["file"]
            )
        ]
        return manifest, pd.read_csv(files_path)

    rows: List[Dict[str, Any]] = []
    file_rows: List[Dict[str, Any]] = []
    progress = tqdm(total=len(assignments), desc="Raw studio quiet-window census")
    block_frames = int(round(cfg.clip_seconds * cfg.expected_sample_rate))
    for assignment in assignments.itertuples(index=False):
        path = Path(assignment.path)
        progress.set_postfix(instrument=assignment.instrument, file=assignment.file)
        decoded_hash = hashlib.sha256()
        quiet_count = 0
        complete_windows = 0
        with sf.SoundFile(str(path), mode="r") as handle:
            if int(handle.samplerate) != cfg.expected_sample_rate:
                raise RuntimeError(
                    f"Sample-rate mismatch for {path}: {handle.samplerate} != "
                    f"{cfg.expected_sample_rate}"
                )
            block_index = 0
            while True:
                block = handle.read(block_frames, dtype="float32", always_2d=True)
                if len(block) == 0:
                    break
                decoded_hash.update(np.ascontiguousarray(block).tobytes())
                if len(block) < block_frames:
                    break
                start = block_index * cfg.clip_seconds
                block_index += 1
                metrics = _window_metrics(block, cfg)
                complete_windows += 1
                quiet_count += int(metrics["passes_quiet"])
                rows.append(
                    {
                        "instrument": assignment.instrument,
                        "class_index": -1,
                        "recording_index": assignment.recording_index,
                        "file": assignment.file,
                        "path": str(path),
                        "split": assignment.split,
                        "start_seconds": float(start),
                        "end_seconds": float(start + cfg.clip_seconds),
                        **metrics,
                    }
                )
        file_rows.append(
            {
                **assignment._asdict(),
                "decoded_audio_sha256": decoded_hash.hexdigest(),
                "complete_windows": complete_windows,
                "accepted_quiet_windows": quiet_count,
                "accepted_quiet_seconds": quiet_count * cfg.clip_seconds,
            }
        )
        progress.update(1)
    progress.close()

    manifest = pd.DataFrame(rows)
    recording_audit = pd.DataFrame(file_rows)
    if manifest.empty:
        raise RuntimeError("No complete raw-recording windows were found.")
    duplicate_hashes = recording_audit[
        recording_audit.duplicated("decoded_audio_sha256", keep=False)
    ]
    if not duplicate_hashes.empty:
        raise RuntimeError(
            "Duplicate decoded recordings detected:\n"
            + duplicate_hashes[
                ["instrument", "file", "split", "decoded_audio_sha256"]
            ].to_string(index=False)
        )
    class_to_index = {
        instrument: index for index, instrument in enumerate(base.ALL_INSTRUMENTS)
    }
    manifest["class_index"] = manifest["instrument"].map(class_to_index).astype(int)
    manifest = manifest.sort_values(
        ["split", "instrument", "file", "start_seconds"]
    ).reset_index(drop=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    recording_audit.to_csv(files_path, index=False)
    base.save_json(
        metadata_path,
        {
            **signature_payload,
            "signature": signature,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_rows": len(manifest),
        },
    )
    return manifest, recording_audit


def availability_table(
    manifest: pd.DataFrame,
    recording_audit: pd.DataFrame,
    cfg: RawSilenceConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for instrument in base.ALL_INSTRUMENTS:
        files = recording_audit[recording_audit["instrument"] == instrument]
        group = manifest[
            (manifest["instrument"] == instrument) & manifest["passes_quiet"]
        ]
        train_count = int((group["split"] == "train").sum())
        validation_count = int((group["split"] == "validation").sum())
        recording_count = len(files)
        eligible = bool(
            recording_count >= cfg.minimum_recordings_per_class
            and train_count >= cfg.minimum_unique_train_windows_per_class
            and validation_count
            >= cfg.minimum_unique_validation_windows_per_class
        )
        reasons: List[str] = []
        if recording_count < cfg.minimum_recordings_per_class:
            reasons.append(
                f"recordings {recording_count}<{cfg.minimum_recordings_per_class}"
            )
        if train_count < cfg.minimum_unique_train_windows_per_class:
            reasons.append(
                f"train quiet {train_count}<"
                f"{cfg.minimum_unique_train_windows_per_class}"
            )
        if validation_count < cfg.minimum_unique_validation_windows_per_class:
            reasons.append(
                f"validation quiet {validation_count}<"
                f"{cfg.minimum_unique_validation_windows_per_class}"
            )
        rows.append(
            {
                "instrument": instrument,
                "raw_recordings": recording_count,
                "train_recordings": int((files["split"] == "train").sum()),
                "validation_recordings": int(
                    (files["split"] == "validation").sum()
                ),
                "quiet_train_windows": train_count,
                "quiet_validation_windows": validation_count,
                "quiet_train_seconds": train_count * cfg.clip_seconds,
                "quiet_validation_seconds": validation_count * cfg.clip_seconds,
                "eligible": eligible,
                "exclusion_reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def _prepare(
    cfg: RawSilenceConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_config(cfg)
    source_root = Path(cfg.drive_96_root)
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Raw 96-kS/s root not found: {source_root}. Mount Drive first."
        )
    source_catalog = _resolve_raw_catalog(source_root)
    local_catalog = base.prepare_local_stem_catalog(source_catalog, cfg)
    assignments = _split_assignments(local_catalog, cfg)
    manifest, recording_audit = build_or_load_manifest(
        local_catalog, assignments, Path(cfg.drive_output_root), cfg
    )
    table = availability_table(manifest, recording_audit, cfg)
    return manifest, recording_audit, table


def resolve_instruments(
    table: pd.DataFrame, cfg: RawSilenceConfig
) -> Tuple[str, ...]:
    eligible = tuple(table.loc[table["eligible"], "instrument"].astype(str))
    if cfg.experiment_instruments is None:
        selected = eligible
    else:
        selected = tuple(cfg.experiment_instruments)
        invalid = [name for name in selected if name not in eligible]
        if invalid:
            details = table.set_index("instrument").loc[
                invalid,
                [
                    "raw_recordings",
                    "quiet_train_windows",
                    "quiet_validation_windows",
                    "exclusion_reason",
                ],
            ]
            raise RuntimeError(
                "Requested instruments are not eligible:\n" + details.to_string()
            )
    if len(selected) < 2:
        raise RuntimeError(
            f"Only {len(selected)} class(es) have independent recordings and "
            "enough quiet windows. Inspect the census before training."
        )
    return selected


def select_equal_windows(
    manifest: pd.DataFrame,
    instruments: Sequence[str],
    cfg: RawSilenceConfig,
) -> Tuple[pd.DataFrame, int, int]:
    quiet = manifest[manifest["passes_quiet"]].copy()
    train_counts = (
        quiet[quiet["split"] == "train"]
        .groupby("instrument")
        .size()
        .reindex(instruments, fill_value=0)
    )
    validation_counts = (
        quiet[quiet["split"] == "validation"]
        .groupby("instrument")
        .size()
        .reindex(instruments, fill_value=0)
    )
    train_per_class = int(train_counts.min())
    if cfg.maximum_balanced_train_windows_per_class is not None:
        train_per_class = min(
            train_per_class, cfg.maximum_balanced_train_windows_per_class
        )
    validation_per_class = min(
        int(validation_counts.min()), cfg.test_windows_per_class
    )
    if train_per_class < cfg.minimum_unique_train_windows_per_class:
        raise RuntimeError("Balanced training support fell below the minimum.")
    if validation_per_class < cfg.minimum_unique_validation_windows_per_class:
        raise RuntimeError("Balanced validation support fell below the minimum.")

    selected: List[pd.DataFrame] = []
    for class_index, instrument in enumerate(instruments):
        train = quiet[
            (quiet["instrument"] == instrument) & (quiet["split"] == "train")
        ].sample(
            n=train_per_class,
            replace=False,
            random_state=cfg.data_seed + 1009 * class_index,
        )
        validation = quiet[
            (quiet["instrument"] == instrument)
            & (quiet["split"] == "validation")
        ].sample(
            n=validation_per_class,
            replace=False,
            random_state=cfg.data_seed + 2003 * class_index,
        )
        selected.extend([train, validation])
    result = pd.concat(selected, ignore_index=True)
    class_to_index = {name: index for index, name in enumerate(instruments)}
    result["class_index"] = result["instrument"].map(class_to_index).astype(int)
    return (
        result.sort_values(
            ["split", "instrument", "file", "start_seconds"]
        ).reset_index(drop=True),
        train_per_class,
        validation_per_class,
    )


def audit_selected(
    selected: pd.DataFrame,
    recording_audit: pd.DataFrame,
    instruments: Sequence[str],
    cfg: RawSilenceConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    threshold_column = "rms_dbfs_audible_0_20k_max_channel"
    if not bool(selected["passes_quiet"].all()):
        raise AssertionError("At least one selected window failed the quiet gate.")
    maximum_selected = float(selected[threshold_column].max())
    if maximum_selected >= cfg.quiet_threshold_dbfs:
        raise AssertionError(
            f"Selected audible RMS reaches {maximum_selected:.5f} dBFS; required "
            f"< {cfg.quiet_threshold_dbfs:g} dBFS."
        )
    train_files = set(selected.loc[selected["split"] == "train", "path"])
    validation_files = set(
        selected.loc[selected["split"] == "validation", "path"]
    )
    overlap = train_files & validation_files
    if overlap:
        raise AssertionError(f"Train/validation file leakage: {sorted(overlap)}")
    used_files = recording_audit[
        recording_audit["instrument"].isin(instruments)
    ]
    duplicate_hashes = used_files[
        used_files.duplicated("decoded_audio_sha256", keep=False)
    ]
    if not duplicate_hashes.empty:
        raise AssertionError("Duplicate decoded recordings entered the experiment.")
    summary = (
        selected.groupby(["instrument", "split"])
        .agg(
            windows=("file", "size"),
            recordings=("file", "nunique"),
            audible_max_channel_dbfs_min=(threshold_column, "min"),
            audible_max_channel_dbfs_median=(threshold_column, "median"),
            audible_max_channel_dbfs_max=(threshold_column, "max"),
            ultrasonic_mono_dbfs_median=(
                "rms_dbfs_ultrasonic_20_48k_mono", "median"
            ),
        )
        .reset_index()
    )
    assertions = {
        "all_selected_windows_pass_quiet": True,
        "quiet_selection_band_hz": [0.0, cfg.audible_upper_hz],
        "quiet_metric": "maximum per-channel RMS",
        "strict_threshold_dbfs": cfg.quiet_threshold_dbfs,
        "maximum_selected_audible_rms_dbfs": maximum_selected,
        "classifier_input_band_hz": [
            cfg.ultrasonic_cutoff_hz,
            cfg.ultrasonic_upper_hz,
        ],
        "train_validation_file_overlap": 0,
        "duplicate_decoded_recordings": 0,
        "classes": list(instruments),
    }
    return summary, assertions


def describe_configuration(cfg: RawSilenceConfig) -> Dict[str, Any]:
    return {
        "raw_root": cfg.drive_96_root,
        "census_classes": len(base.ALL_INSTRUMENTS),
        "quiet_selection": (
            f"maximum channel RMS in 0--{cfg.audible_upper_hz / 1000:g} kHz "
            f"< {cfg.quiet_threshold_dbfs:g} dBFS"
        ),
        "split": "one complete raw recording held out per instrument",
        "balancing": "equal unique train and validation windows across classes",
        "model_input": "96 kS/s linear STFT, 20--48 kHz only",
        "minimum_train_windows": cfg.minimum_unique_train_windows_per_class,
        "minimum_validation_windows": (
            cfg.minimum_unique_validation_windows_per_class
        ),
        "model_seeds": cfg.model_seeds,
        "epochs": cfg.epochs,
    }


def preview_availability(cfg: RawSilenceConfig) -> Dict[str, Any]:
    manifest, recording_audit, table = _prepare(cfg)
    eligible = tuple(table.loc[table["eligible"], "instrument"].astype(str))
    print("\nIndependent raw-recording assignments:")
    print(
        recording_audit[
            [
                "instrument",
                "file",
                "split",
                "duration_seconds",
                "accepted_quiet_windows",
                "accepted_quiet_seconds",
            ]
        ].to_string(index=False)
    )
    print("\nAll-instrument quiet-window census:")
    print(table.to_string(index=False))
    print(f"\nEligible classes ({len(eligible)}): {eligible}")
    return {
        "manifest": manifest,
        "recording_audit": recording_audit,
        "availability": table,
        "eligible": eligible,
    }


def _make_run_directory(
    output_root: Path, instruments: Sequence[str], cfg: RawSilenceConfig
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    seeds = "-".join(str(seed) for seed in cfg.model_seeds)
    run_dir = output_root / (
        f"cnn6_raw_silence_{timestamp}_below{abs(cfg.quiet_threshold_dbfs):g}dbfs_"
        f"c{len(instruments)}_mseed{seeds}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_experiment(cfg: RawSilenceConfig) -> Dict[str, Any]:
    validate_config(cfg)
    if cfg.require_cuda_for_training and not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU runtime before training.")
    print("=" * 80)
    print("RAW-STUDIO ULTRASONIC SILENCE CONTROL — CNN6")
    print("=" * 80)
    for key, value in describe_configuration(cfg).items():
        print(f"{key}: {value}")

    manifest, recording_audit, table = _prepare(cfg)
    instruments = resolve_instruments(table, cfg)
    selected, train_per_class, validation_per_class = select_equal_windows(
        manifest, instruments, cfg
    )
    audit_summary, assertions = audit_selected(
        selected, recording_audit, instruments, cfg
    )
    run_dir = _make_run_directory(Path(cfg.drive_output_root), instruments, cfg)
    chance = 1.0 / len(instruments)

    base.save_json(
        run_dir / "run_config.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
            "resolved_instruments": instruments,
            "chance_accuracy": chance,
            "train_unique_windows_per_class": train_per_class,
            "validation_unique_windows_per_class": validation_per_class,
            "scientific_question": (
                "Can an unseen raw recording's 20--48 kHz residual identify the "
                "instrument label when all channels are audibly quiet?"
            ),
        },
    )
    recording_audit.to_csv(run_dir / "recording_audit.csv", index=False)
    table.to_csv(run_dir / "availability_all15.csv", index=False)
    selected.to_csv(run_dir / "selected_equal_window_manifest.csv", index=False)
    audit_summary.to_csv(run_dir / "quiet_selection_audit.csv", index=False)
    base.save_json(run_dir / "hard_assertions.json", assertions)

    print(f"\nTimestamped output: {run_dir}")
    print(f"Classes ({len(instruments)}): {instruments}")
    print(f"Chance accuracy: {100 * chance:.2f}%")
    print(f"Equal unique train windows/class: {train_per_class}")
    print(f"Equal unique validation windows/class: {validation_per_class}")
    print("\nHard-audit summary:")
    print(audit_summary.to_string(index=False))
    print(
        "\nPASS: independent recording files, no duplicate decoded audio, no "
        "train/validation overlap, and every selected audible channel is below "
        f"{cfg.quiet_threshold_dbfs:g} dBFS."
    )

    train_manifest = selected[selected["split"] == "train"].reset_index(drop=True)
    validation_manifest = selected[
        selected["split"] == "validation"
    ].reset_index(drop=True)
    overall_rows: List[Mapping[str, Any]] = []
    per_seed_rows: List[pd.DataFrame] = []
    matrices: List[np.ndarray] = []
    for seed in cfg.model_seeds:
        seed_dir = run_dir / f"model_seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=False)
        overall, per_instrument, _, matrix = base.train_one_seed(
            train_manifest,
            validation_manifest,
            instruments,
            seed,
            seed_dir,
            cfg,
        )
        overall_rows.append(overall)
        per_seed_rows.append(per_instrument)
        matrices.append(matrix)
    summary = base._aggregate_results(
        overall_rows, per_seed_rows, matrices, instruments, run_dir
    )
    inherited = run_dir / "paper_table_ultrasonic_only_f1.csv"
    if inherited.is_file():
        os.replace(inherited, run_dir / "paper_table_raw_silence_control_f1.csv")
    (run_dir / "RUN_COMPLETE.txt").write_text(
        f"Completed UTC: {datetime.now(timezone.utc).isoformat()}\n"
        f"Classes: {len(instruments)}\n"
        f"Chance: {chance:.8f}\n"
        f"Train unique windows/class: {train_per_class}\n"
        f"Validation unique windows/class: {validation_per_class}\n"
        f"Seeds: {list(cfg.model_seeds)}\n"
        f"Accuracy mean: {summary['accuracy']['mean']:.8f}\n"
        f"Accuracy std: {summary['accuracy']['std']:.8f}\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 80)
    print("RAW-SILENCE CONTROL SUMMARY")
    print("=" * 80)
    for metric, values in summary.items():
        print(
            f"{metric:>18}: {100 * values['mean']:.2f}% "
            f"+/- {100 * values['std']:.2f}%"
        )
    print(f"Full results: {run_dir}")
    gc.collect()
    return {
        "run_dir": str(run_dir),
        "instruments": instruments,
        "chance_accuracy": chance,
        "train_windows_per_class": train_per_class,
        "validation_windows_per_class": validation_per_class,
        "summary": summary,
    }


def protocol_smoke_test() -> Dict[str, Any]:
    cfg = RawSilenceConfig(model_seeds=(42,), num_workers=0)
    frames = int(cfg.expected_sample_rate * cfg.clip_seconds)
    time = np.arange(frames, dtype=np.float64) / cfg.expected_sample_rate
    quiet_amplitude = math.sqrt(2.0) * 10 ** (-85.0 / 20.0)
    quiet = (quiet_amplitude * np.sin(2 * np.pi * 1_000 * time))[:, None]
    quiet = np.repeat(quiet.astype(np.float32), 2, axis=1)
    one_loud_channel = quiet.copy()
    loud_amplitude = math.sqrt(2.0) * 10 ** (-60.0 / 20.0)
    one_loud_channel[:, 1] = loud_amplitude * np.sin(2 * np.pi * 1_000 * time)
    ultrasonic = (
        loud_amplitude * np.sin(2 * np.pi * 30_000 * time)
    ).astype(np.float32)[:, None]
    ultrasonic = np.repeat(ultrasonic, 2, axis=1)
    zero = np.zeros_like(quiet)
    qm = _window_metrics(quiet, cfg)
    lm = _window_metrics(one_loud_channel, cfg)
    um = _window_metrics(ultrasonic, cfg)
    zm = _window_metrics(zero, cfg)
    if not qm["passes_quiet"]:
        raise AssertionError("A -85 dBFS audible window should pass.")
    if lm["passes_quiet"]:
        raise AssertionError("A loud audible channel incorrectly passed.")
    if not um["passes_quiet"]:
        raise AssertionError("The selection gate incorrectly inspected ultrasound.")
    if zm["passes_quiet"] or not zm["is_digital_zero"]:
        raise AssertionError("Digital-zero rejection failed.")
    return {
        "quiet_nonzero_passes": True,
        "one_loud_channel_rejected": True,
        "ultrasonic_band_not_used_for_selection": True,
        "digital_zero_rejected": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Train the raw-recording control")
    parser.add_argument("--data-root", help="Path to unedited valid_96 recordings")
    parser.add_argument("--output-root", help="Directory for timestamped runs")
    parser.add_argument("--local-cache-root", help="Local audio cache directory")
    args = parser.parse_args()
    if args.run:
        config = RawSilenceConfig()
        if args.data_root:
            config.drive_96_root = args.data_root
        if args.output_root:
            config.drive_output_root = args.output_root
        if args.local_cache_root:
            config.local_stem_cache_root = args.local_cache_root
        print(run_experiment(config))
    else:
        print(protocol_smoke_test())
