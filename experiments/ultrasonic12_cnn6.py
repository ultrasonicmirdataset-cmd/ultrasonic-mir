#!/usr/bin/env python3
"""Recording-level 12-class isolated-source recognition, 20–48 kHz.

Keep isolated_rf_cnn6.py in the same directory. This script
reuses its 96-kHz full-band activity selection and CNN6 training schedule,
but masks all STFT power bins below 20 kHz before the model input. No 44.1-kHz
audio or 96LP control is relevant to this 96-kHz ultrasonic-only experiment.

Training tracks: 1,3,4,5. Held-out evaluation tracks: 2,6. Track 2.2 is
excluded, including when it is an independent rendition of song 2. Each
recording is assigned before its windows are extracted. The same evaluation
windows are used for all three seeds, with evaluation after fixed epoch 50.

The default experiment uses CNN6. Pass --models rf,cnn6 for an additional
exploratory RF comparison using 2-kHz linear bands across 20–48 kHz.

Example:
    python ultrasonic12_cnn6.py \
      --train96 /content/drive/MyDrive/train_96 \
      --output /content/drive/MyDrive/ultrasonic_mir_runs/experiment_5_3 --audit-only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import isolated_rf_cnn6 as core


TWELVE_INSTRUMENTS = tuple(name for name in core.INSTRUMENTS
                           if name not in {"Bansury", "Electric_guitar", "Piano"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train96", type=Path, default=Path("/content/drive/MyDrive/train_96"))
    parser.add_argument("--output", type=Path, default=Path("./experiment_5_3_results"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--models", default="cnn6", help="cnn6, rf, or rf,cnn6")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--rf-train-per-class", type=int, default=300)
    parser.add_argument("--cnn-train-pool-per-class", type=int, default=1200)
    parser.add_argument("--eval-per-class", type=int, default=150)
    parser.add_argument("--cnn-samples-per-class-epoch", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    models = tuple(v.strip().lower() for v in args.models.split(",") if v.strip())
    seeds = tuple(int(v) for v in args.seeds.split(",") if v.strip())
    if not models or set(models) - {"rf", "cnn6"}:
        parser.error("--models must be cnn6, rf, or rf,cnn6")
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain distinct integer values")
    if min(args.epochs, args.rf_trees, args.rf_train_per_class,
           args.cnn_train_pool_per_class, args.eval_per_class,
           args.cnn_samples_per_class_epoch, args.batch_size) <= 0 or args.workers < 0:
        parser.error("training sizes, epochs, trees and batch size must be positive")

    # All imported core functions read INSTRUMENTS at call time. Restrict the
    # class set only in this standalone process, without modifying the 5.1 file.
    core.INSTRUMENTS = TWELVE_INSTRUMENTS
    cfg = core.Config(train96=args.train96, train44=None, derive44=True,
        output=args.output, conditions=("ultra20",), models=models, seeds=seeds,
        epochs=args.epochs, rf_trees=args.rf_trees,
        rf_train_per_class=args.rf_train_per_class,
        cnn_train_pool_per_class=args.cnn_train_pool_per_class,
        eval_per_class=args.eval_per_class,
        cnn_samples_per_class_epoch=args.cnn_samples_per_class_epoch,
        batch_size=args.batch_size, workers=args.workers)
    inventory, issues = core.audit(cfg)
    cfg.output.mkdir(parents=True, exist_ok=True)
    audit_path = cfg.output / "experiment_5_3_audit.json"
    audit_path.write_text(json.dumps({"classes": TWELVE_INSTRUMENTS,
        "train_tracks": core.TRAIN_TRACKS, "eval_tracks": core.EVAL_TRACKS,
        "excluded_tracks": core.EXCLUDED_TRACKS, "inventory": inventory,
        "issues": issues}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"12-class audit: {len(inventory)} recordings, {len(issues)} issues; {audit_path}")
    for issue in issues:
        print("  - " + issue)
    if args.audit_only:
        return
    if issues:
        raise RuntimeError("Ultrasonic-only audit failed; no training started")

    candidates = core.scan_activity(inventory, cfg)
    rf_train, cnn_train, validation = core.choose_windows(candidates, cfg)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = cfg.output / f"experiment_5_3_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    core.write_csv(run_dir / "recordings.csv", inventory)
    core.write_csv(run_dir / "rf_train_windows.csv", rf_train)
    core.write_csv(run_dir / "cnn6_train_windows.csv", cnn_train)
    core.write_csv(run_dir / "heldout_windows.csv", validation)
    configuration = {k: str(v) if isinstance(v, Path) else v
                     for k, v in asdict(cfg).items()}
    configuration.update({"classes": TWELVE_INSTRUMENTS,
        "frequency_band_hz": [20000, 48000], "experiment": "section_5_3",
        "note": "Reported scores require an actual run on audited audio."})
    (run_dir / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    overall, per_class = [], []
    if "rf" in models:
        core.run_rf(cfg, rf_train, validation, run_dir, overall, per_class)
    if "cnn6" in models:
        core.run_cnn(cfg, cnn_train, validation, run_dir, overall, per_class)
    core.write_csv(run_dir / "per_seed_overall.csv", overall)
    core.write_csv(run_dir / "per_seed_per_class.csv", per_class)
    core.write_csv(run_dir / "summary_overall_mean_sd.csv",
        core.summarize(overall, ("model", "condition"), ("accuracy", "macro_f1")))
    core.write_csv(run_dir / "summary_per_class_mean_sd.csv",
        core.summarize(per_class, ("model", "condition", "instrument"),
                       ("precision", "recall", "f1")))
    print(f"12-class ultrasonic-only results: {run_dir}")


if __name__ == "__main__":
    main()
