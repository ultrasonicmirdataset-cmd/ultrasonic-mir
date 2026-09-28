#!/usr/bin/env python3
"""Raw quiet-window control: 12 classes, 96-kHz raw recordings.

The gate is a 0.5-s, full-band RMS strictly below -80 dBFS, computed
across all channels and samples. Entirely digital-zero windows are excluded.
The CNN6 sees only STFT bins above 20 kHz. Whole raw recordings
belong to one split: train tracks 1/2/4/5, held-out tracks 3/6. Track 2.2
is excluded. The fixed final epoch is evaluated once; no held-out checkpoint
selection. Requires an explicit mapping from every raw recording to its song
family; studio filenames do not reliably encode song numbers. There must be
train and evaluation recordings for every class, and tracks 3 and 6 must
both occur among the held-out recordings. This script never alters input data.

Keep isolated_rf_cnn6.py and
ultrasonic12_cnn6.py next to this script. In Colab:
    !pip install numpy scipy scikit-learn soundfile torch
    !python raw_quiet12_cnn6.py \\
      --raw-root /content/drive/MyDrive/valid_96 \\
      --create-map-template /content/raw_track_map.json
    # Fill in each null "track" with the verified song ID, e.g. "3".
    !python raw_quiet12_cnn6.py \\
      --raw-root /content/drive/MyDrive/valid_96 \\
      --track-map /content/raw_track_map.json --audit-only
    !python raw_quiet12_cnn6.py \\
      --raw-root /content/drive/MyDrive/valid_96 \\
      --track-map /content/raw_track_map.json \\
      --output /content/drive/MyDrive/ultrasonic_mir_runs/experiment_5_4
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import isolated_rf_cnn6 as core
from ultrasonic12_cnn6 import TWELVE_INSTRUMENTS


RAW_FOLDER_ALIASES: dict[str, tuple[str, ...]] = {
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
QUIET_INSTRUMENTS = TWELVE_INSTRUMENTS
EXCLUDED_INSTRUMENTS = frozenset(set(core.INSTRUMENTS) - set(QUIET_INSTRUMENTS))
TRAIN_TRACKS = ("1", "2", "4", "5")
EVAL_TRACKS = ("3", "6")
KNOWN_TRACKS = (*TRAIN_TRACKS, *EVAL_TRACKS, *core.EXCLUDED_TRACKS)
AUDIO_SUFFIXES = {".wav", ".wave", ".flac", ".aif", ".aiff"}


def catalog_raw(root: Path) -> tuple[dict[str, list[Path]], list[str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Raw-data directory does not exist: {root}")
    folders = {p.name.casefold(): p for p in root.iterdir() if p.is_dir()}
    catalog: dict[str, list[Path]] = {}
    issues: list[str] = []
    for label in QUIET_INSTRUMENTS:
        matches = {folders[alias.casefold()] for alias in RAW_FOLDER_ALIASES[label]
                   if alias.casefold() in folders}
        if len(matches) != 1:
            issues.append(f"{label}: expected exactly one raw folder; found {len(matches)}")
            continue
        directory = next(iter(matches))
        paths = sorted((p for p in directory.iterdir() if p.is_file() and
                        (p.suffix.lower() in AUDIO_SUFFIXES or not p.suffix)),
                       key=lambda p: p.name.casefold())
        if not paths:
            issues.append(f"{label}: no raw audio files in {directory}")
        catalog[label] = paths
    return catalog, issues


def template(root: Path, catalog: dict[str, list[Path]]) -> dict[str, Any]:
    return {"schema": "raw-song-map-v1",
            "instructions": "For every file, replace null with its verified track ID: 1,2,3,4,5,6 or 2.2 (excluded). Never infer from filename order.",
            "entries": [{"instrument": label, "relative_path": str(path.relative_to(root)),
                         "track": None}
                        for label in QUIET_INSTRUMENTS
                        for path in catalog.get(label, [])]}


def assignments(root: Path, catalog: dict[str, list[Path]], map_path: Path
               ) -> tuple[list[dict[str, Any]], list[str]]:
    content = json.loads(map_path.read_text(encoding="utf-8"))
    schema = content.get("schema")
    # Accept earlier prefixed v1 maps while generating a generic schema now.
    if (not isinstance(schema, str) or
            not (schema == "raw-song-map-v1" or schema.endswith("-raw-song-map-v1")) or
            not isinstance(content.get("entries"), list)):
        raise ValueError("Track map must use the generated raw-song-map-v1 schema")
    known = {str(path.relative_to(root)): (label, path) for label, paths in catalog.items()
             for path in paths}
    issues: list[str] = []
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for entry in content["entries"]:
        if not isinstance(entry, dict):
            issues.append("Non-object entry in track map")
            continue
        name = entry.get("relative_path")
        label_in_map = entry.get("instrument")
        # An existing 15-class map can still be used: ignore ONLY real files
        # in the three excluded instrument folders, not arbitrary unknown paths.
        if (isinstance(name, str) and isinstance(label_in_map, str) and
                label_in_map in EXCLUDED_INSTRUMENTS):
            components = Path(name).parts
            if (len(components) == 2 and
                    components[0].casefold() in
                    {alias.casefold() for alias in RAW_FOLDER_ALIASES[label_in_map]} and
                    (root / name).is_file()):
                continue
        if not isinstance(name, str) or name not in known:
            issues.append(f"Unknown raw file in map: {name!r}")
            continue
        if name in seen:
            issues.append(f"Raw file mapped twice: {name}")
            continue
        seen.add(name)
        label, path = known[name]
        if entry.get("instrument") != label:
            issues.append(f"{name}: wrong instrument label ({entry.get('instrument')!r} != {label!r})")
            continue
        track = entry.get("track")
        if track not in KNOWN_TRACKS:
            issues.append(f"{name}: track must be a verified string in {KNOWN_TRACKS}; got {track!r}")
            continue
        if track == "2.2":
            continue
        try:
            info = core.file_info(path)
        except Exception as exc:
            issues.append(f"{name}: cannot read raw audio: {exc}")
            continue
        if info["sample_rate"] != 96000:
            issues.append(f"{name}: expected raw 96000 Hz, got {info['sample_rate']}")
        rows.append({"instrument": label, "track": track,
                     "split": "eval" if track in EVAL_TRACKS else "train",
                     "path96": str(path), "relative_path": name,
                     "sample_rate96": info["sample_rate"],
                     "duration96_s": info["duration"],
                     "channels": info["channels"], "size_bytes": info["size_bytes"]})
    for missing in sorted(set(known) - seen):
        issues.append(f"Raw file has no track assignment: {missing}")
    if not set(EVAL_TRACKS).issubset({r["track"] for r in rows if r["split"] == "eval"}):
        issues.append("Both held-out track families 3 and 6 must be present")
    for label in QUIET_INSTRUMENTS:
        by_class = [r for r in rows if r["instrument"] == label]
        for split in ("train", "eval"):
            if not any(r["split"] == split for r in by_class):
                issues.append(f"{label}: missing distinct raw recording for {split}")
    return rows, issues


def read_channels(audio: core.AudioFile, start: int, frames: int) -> np.ndarray:
    """Preserve all raw channels for the full-band RMS gate."""
    if audio.backend == "soundfile":
        audio.handle.seek(start)
        result = audio.handle.read(frames, dtype="float32", always_2d=True)
    else:
        audio.handle.setpos(start)
        raw = audio.handle.readframes(frames)
        width = audio.sample_width
        if width == 1:
            values = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
        elif width == 2:
            values = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
        elif width == 3:
            octets = np.frombuffer(raw, np.uint8).reshape(-1, 3)
            integers = (octets[:, 0].astype(np.int32) |
                        (octets[:, 1].astype(np.int32) << 8) |
                        (octets[:, 2].astype(np.int32) << 16))
            integers = (integers ^ 0x800000) - 0x800000
            values = integers.astype(np.float32) / 8388608
        elif width == 4:
            values = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
        else:
            raise ValueError(f"Install soundfile to read raw width {width}: {audio.path}")
        result = values.reshape(-1, audio.channels)
    if result.shape != (frames, audio.channels):
        raise IOError(f"Short raw read at frame {start}: {audio.path}")
    return result


def quiet_windows(rows: list[dict[str, Any]], window_s: float, threshold_db: float
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, counts = [], []
    for row in rows:
        tally = defaultdict(int)
        with core.AudioFile(Path(row["path96"])) as audio:
            frame_count = round(audio.samplerate * window_s)
            for idx in range(audio.frames // frame_count):
                block = read_channels(audio, idx * frame_count, frame_count)
                tally["complete_windows"] += 1
                rms = float(np.sqrt(np.mean(np.square(block.astype(np.float64)))))
                dbfs = 20 * math.log10(max(rms, 1e-20))
                if dbfs >= threshold_db:
                    tally["not_quiet"] += 1
                    continue
                if rms == 0.0:
                    tally["digital_zero_rejected"] += 1
                    continue
                tally["eligible"] += 1
                candidates.append({"instrument": row["instrument"],
                                   "track": row["track"], "split": row["split"],
                                   "path96": row["path96"], "path44": "",
                                   "start_s": round(idx * window_s, 8),
                                   "rms_dbfs96": round(dbfs, 3),
                                   "class_index": QUIET_INSTRUMENTS.index(row["instrument"])})
        counts.append({"instrument": row["instrument"], "track": row["track"],
                       "split": row["split"], "relative_path": row["relative_path"],
                       **{field: tally[field] for field in ("complete_windows", "not_quiet",
                           "digital_zero_rejected", "eligible")}})
        print(f"Raw quiet scan: {row['instrument']}/{row['track']}: {tally['eligible']} eligible", flush=True)
    return candidates, counts


def balance_quiet(candidates: list[dict[str, Any]], seed: int, max_train: int,
                  max_eval: int, min_train: int, min_eval: int
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[(row["instrument"], row["split"])].append(row)
    gaps = [f"{label}/{split}={len(grouped[(label, split)])}"
            for label in QUIET_INSTRUMENTS for split, minimum in (("train", min_train), ("eval", min_eval))
            if len(grouped[(label, split)]) < minimum]
    if gaps:
        raise RuntimeError("Too few genuine quiet windows for: " + ", ".join(gaps))
    n_train = min(max_train, *(len(grouped[(label, "train")]) for label in QUIET_INSTRUMENTS))
    n_eval = min(max_eval, *(len(grouped[(label, "eval")]) for label in QUIET_INSTRUMENTS))
    train, evaluation = [], []
    for idx, label in enumerate(QUIET_INSTRUMENTS):
        train.extend(core.balanced_order(grouped[(label, "train")], seed + 101 * idx)[:n_train])
        evaluation.extend(core.balanced_order(grouped[(label, "eval")], seed + 131 * idx)[:n_eval])
    print(f"Balanced quiet windows per class: train={n_train}, held-out={n_eval}")
    return train, evaluation


def assess(run_dir: Path, overall: list[dict[str, Any]]) -> None:
    chance = 1 / len(QUIET_INSTRUMENTS)
    mean = float(np.mean([row["accuracy"] for row in overall]))
    result = {
        "heldout_accuracy_by_seed": {str(row["seed"]): row["accuracy"] for row in overall},
        "mean_heldout_accuracy": mean, "chance_accuracy": chance,
        "difference_from_uniform_chance": mean - chance,
        "interpretation": "Compare each seed and class with uniform chance; no numerical threshold is imposed for near-chance performance. Training-loss convergence and held-out generalization are distinct; check the saved epoch history as well.",
    }
    (run_dir / "quiet_control_assessment.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(f"Quiet control: accuracy={mean:.4f}, uniform chance={chance:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, default=Path("/content/drive/MyDrive/valid_96"))
    p.add_argument("--track-map", type=Path)
    p.add_argument("--create-map-template", type=Path)
    p.add_argument("--output", type=Path, default=Path("./experiment_5_4_results"))
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--window-seconds", type=float, default=0.5)
    p.add_argument("--quiet-dbfs", type=float, default=-80.0)
    p.add_argument("--max-train-per-class", type=int, default=1200)
    p.add_argument("--max-eval-per-class", type=int, default=150)
    p.add_argument("--min-train-per-class", type=int, default=1)
    p.add_argument("--min-eval-per-class", type=int, default=1)
    p.add_argument("--cnn-samples-per-class-epoch", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    if not (0 < args.window_seconds <= 10) or (
        args.epochs <= 0 or args.workers < 0 or
        min(args.max_train_per_class, args.max_eval_per_class, args.min_train_per_class,
            args.min_eval_per_class, args.cnn_samples_per_class_epoch, args.batch_size) <= 0
    ):
        p.error("Invalid window, quiet gate, zero gate, or training parameters")
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if len(seeds) != len(set(seeds)) or not seeds:
        p.error("Seeds must be distinct integers")
    catalog, issues = catalog_raw(args.raw_root)
    if args.create_map_template:
        if args.create_map_template.exists():
            p.error(f"Refusing to overwrite existing mapping: {args.create_map_template}")
        args.create_map_template.parent.mkdir(parents=True, exist_ok=True)
        args.create_map_template.write_text(json.dumps(template(args.raw_root, catalog),
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote editable raw track map: {args.create_map_template}")
        for issue in issues:
            print("  - " + issue)
        return
    if not args.track_map:
        p.error("Pass --track-map with verified song IDs (or use --create-map-template)")
    inventory, mapping_issues = assignments(args.raw_root, catalog, args.track_map)
    issues.extend(mapping_issues)
    args.output.mkdir(parents=True, exist_ok=True)
    audit_path = args.output / "experiment_5_4_audit.json"
    audit = {"classes": QUIET_INSTRUMENTS, "train_tracks": TRAIN_TRACKS,
             "heldout_tracks": EVAL_TRACKS, "excluded_tracks": core.EXCLUDED_TRACKS,
             "inventory": inventory, "issues": issues}
    if not issues:
        candidates, counts = quiet_windows(inventory,
            args.window_seconds, args.quiet_dbfs)
        audit["raw_window_counts"] = counts
        counts_by_class = defaultdict(int)
        for row in candidates:
            counts_by_class[(row["instrument"], row["split"])] += 1
        for label in QUIET_INSTRUMENTS:
            for split, minimum in (("train", args.min_train_per_class),
                                   ("eval", args.min_eval_per_class)):
                if counts_by_class[(label, split)] < minimum:
                    issues.append(f"{label}/{split}: {counts_by_class[(label, split)]} usable quiet windows; need {minimum}")
    else:
        candidates = []
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Raw audit: {len(inventory)} recordings, {len(issues)} issues; {audit_path}")
    for issue in issues:
        print("  - " + issue)
    if args.audit_only:
        return
    if issues:
        raise RuntimeError("Raw quiet audit failed; no training started")
    train, validation = balance_quiet(candidates, 42, args.max_train_per_class,
        args.max_eval_per_class, args.min_train_per_class, args.min_eval_per_class)
    cfg = core.Config(train96=args.raw_root, train44=None, output=args.output,
        derive44=True, conditions=("ultra20",), models=("cnn6",), seeds=seeds,
        window_seconds=args.window_seconds,
        cnn_train_pool_per_class=args.max_train_per_class,
        eval_per_class=args.max_eval_per_class,
        cnn_samples_per_class_epoch=args.cnn_samples_per_class_epoch,
        batch_size=args.batch_size, workers=args.workers, epochs=args.epochs)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = args.output / f"experiment_5_4_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    core.write_csv(run_dir / "raw_inventory.csv", inventory)
    core.write_csv(run_dir / "quiet_window_counts.csv", audit["raw_window_counts"])
    core.write_csv(run_dir / "train_windows.csv", train)
    core.write_csv(run_dir / "heldout_windows.csv", validation)
    (run_dir / "config.json").write_text(json.dumps({"experiment": "raw_quiet_12",
        "classes": QUIET_INSTRUMENTS,
        "class_count": len(QUIET_INSTRUMENTS), "frequency_band_hz": [20000, 48000],
        "quiet_gate": "full_band_all_channels_rms", "quiet_threshold_dbfs": args.quiet_dbfs,
        "zero_gate": "exclude_all_zero_windows",
        "track_map": str(args.track_map), **{key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()}}, indent=2), encoding="utf-8")
    overall: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    core.INSTRUMENTS = QUIET_INSTRUMENTS
    core.run_cnn(cfg, train, validation, run_dir, overall, per_class)
    core.write_csv(run_dir / "per_seed_overall.csv", overall)
    core.write_csv(run_dir / "per_seed_per_class.csv", per_class)
    core.write_csv(run_dir / "summary_overall_mean_sd.csv",
        core.summarize(overall, ("model", "condition"), ("accuracy", "macro_f1")))
    core.write_csv(run_dir / "summary_per_class_mean_sd.csv",
        core.summarize(per_class, ("model", "condition", "instrument"),
                       ("precision", "recall", "f1")))
    assess(run_dir, overall)
    print(f"Raw quiet-control outputs: {run_dir}")


if __name__ == "__main__":
    main()
