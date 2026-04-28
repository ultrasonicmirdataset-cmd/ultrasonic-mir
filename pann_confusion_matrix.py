#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Confusion matrix and evaluation script for PANN-based instrument classification.

This script evaluates a trained checkpoint and saves:
    1. Segment-level confusion matrix
    2. File-level confusion matrix using majority voting
    3. Classification reports
    4. JSON metrics summaries

Expected project structure:
    project/
    ├── pann_confusion_matrix.py
    ├── models.py
    └── utils.py

The models.py file must contain the model architectures used during training.
The selected model and sample rate must match the training configuration.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import models as pann_models


MODEL_CLASS_NAMES = {
    "cnn6": "Cnn6",
    "cnn10": "Cnn10",
    "cnn14": "Cnn14",
    "cnn14_no_specaug": "Cnn14_no_specaug",
    "cnn14_no_dropout": "Cnn14_no_dropout",
    "cnn14_emb512": "Cnn14_emb512",
    "cnn14_emb128": "Cnn14_emb128",
    "cnn14_emb32": "Cnn14_emb32",
    "resnet22": "ResNet22",
    "resnet38": "ResNet38",
    "resnet54": "ResNet54",
    "mobilenetv1": "MobileNetV1",
    "mobilenetv2": "MobileNetV2",
    "leenet11": "LeeNet11",
    "leenet24": "LeeNet24",
}


AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".aiff", ".aif")


class InstrumentEvalDataset(Dataset):
    """
    Folder-based evaluation dataset.

    Expected folder structure:
        test_root/
        ├── instrument_1/
        │   ├── file_1.wav
        │   └── file_2.wav
        ├── instrument_2/
        │   ├── file_1.wav
        │   └── file_2.wav
        └── ...

    Each subfolder is treated as one class.
    """

    def __init__(
        self,
        root_dir: str,
        target_sr: int,
        segment_seconds: float = 1.0,
        overlap: float = 0.5,
    ):
        self.root_dir = Path(root_dir)
        self.target_sr = target_sr
        self.segment_samples = int(segment_seconds * target_sr)
        self.step = max(1, int(self.segment_samples * (1.0 - overlap)))

        self.samples = []
        self.class_names = []
        self.label_map: Dict[str, int] = {}

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Test data path does not exist: {self.root_dir}")

        self._build_index(segment_seconds=segment_seconds, overlap=overlap)

    def _build_index(self, segment_seconds: float, overlap: float) -> None:
        print(f"\nLoading dataset from: {self.root_dir}")
        print(f"Target sample rate: {self.target_sr}")
        print(f"Segment seconds: {segment_seconds}")
        print(f"Overlap: {overlap}")
        print(f"Segment samples: {self.segment_samples}")
        print(f"Step samples: {self.step}")

        class_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        if not class_folders:
            raise RuntimeError("No class folders were found in the test data path.")

        for idx, class_dir in enumerate(class_folders):
            class_name = class_dir.name
            self.label_map[class_name] = idx
            self.class_names.append(class_name)

        for class_dir in tqdm(class_folders, desc="Indexing folders"):
            class_name = class_dir.name
            label = self.label_map[class_name]

            for audio_path in sorted(class_dir.rglob("*")):
                if not audio_path.is_file():
                    continue

                if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue

                try:
                    info = sf.info(audio_path.as_posix())
                    original_sr = info.samplerate
                    total_frames = info.frames
                except Exception as exc:
                    print(f"Skipping {audio_path}: {exc}")
                    continue

                if original_sr == self.target_sr:
                    total_target_samples = total_frames
                else:
                    duration_sec = total_frames / original_sr
                    total_target_samples = int(duration_sec * self.target_sr)

                if total_target_samples < self.segment_samples:
                    continue

                for start in range(0, total_target_samples - self.segment_samples + 1, self.step):
                    self.samples.append(
                        {
                            "path": audio_path.as_posix(),
                            "label": label,
                            "start": start,
                        }
                    )

        print(f"\nTotal segments: {len(self.samples)}")
        print(f"Classes: {self.label_map}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.samples[idx]
        path = item["path"]
        label = item["label"]
        start = item["start"]

        audio = self._load_segment(path=path, start=start)

        x = torch.tensor(audio, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y, path

    def _load_segment(self, path: str, start: int) -> np.ndarray:
        try:
            info = sf.info(path)
            original_sr = info.samplerate

            if original_sr == self.target_sr:
                audio, _ = sf.read(
                    path,
                    start=start,
                    stop=start + self.segment_samples,
                    dtype="float32",
                )
            else:
                audio, _ = sf.read(path, dtype="float32")
                audio = self._to_mono(audio)
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)
                audio = audio[start:start + self.segment_samples]

        except Exception:
            audio, original_sr = librosa.load(path, sr=None, mono=True)
            if original_sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)
            audio = audio[start:start + self.segment_samples]

        audio = self._to_mono(audio)
        audio = self._pad_or_trim(audio)
        audio = self._peak_normalize(audio)

        return audio.astype(np.float32)

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio

    def _pad_or_trim(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < self.segment_samples:
            audio = np.pad(audio, (0, self.segment_samples - len(audio)))
        elif len(audio) > self.segment_samples:
            audio = audio[: self.segment_samples]
        return audio

    @staticmethod
    def _peak_normalize(audio: np.ndarray) -> np.ndarray:
        return audio / (np.max(np.abs(audio)) + 1e-6)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_name: str,
    sr: int,
    num_classes: int,
    window_ms: float,
    mel_bins: int,
    fmin: float,
    fmax: float,
) -> nn.Module:
    if model_name not in MODEL_CLASS_NAMES:
        available = ", ".join(sorted(MODEL_CLASS_NAMES.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    class_name = MODEL_CLASS_NAMES[model_name]

    if not hasattr(pann_models, class_name):
        raise AttributeError(
            f"models.py does not contain '{class_name}'. "
            f"Check that the selected model exists in your models.py file."
        )

    model_class = getattr(pann_models, class_name)

    window_size = int(sr * window_ms / 1000.0)
    hop_size = window_size // 4

    print("\nModel configuration:")
    print(f"Model: {model_name}")
    print(f"Sample rate: {sr}")
    print(f"Window size: {window_size}")
    print(f"Hop size: {hop_size}")
    print(f"Mel bins: {mel_bins}")
    print(f"Frequency range: {fmin} Hz to {fmax} Hz")
    print(f"Classes: {num_classes}")

    return model_class(
        sample_rate=sr,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
        fmin=fmin,
        fmax=fmax,
        classes_num=num_classes,
    )


def load_weights(model: nn.Module, weights_path: str, device: torch.device) -> nn.Module:
    if not weights_path:
        raise ValueError("Please provide --weights_path.")

    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("Loaded checkpoint with model_state_dict.")
    else:
        state_dict = checkpoint
        print("Loaded raw state_dict.")

    model.load_state_dict(state_dict, strict=True)

    return model


def extract_model_output(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "clipwise_output" not in output:
        raise KeyError("Model output dictionary must contain 'clipwise_output'.")

    scores = output["clipwise_output"]

    if scores.min().detach().item() >= 0.0 and scores.max().detach().item() <= 1.0:
        scores = torch.logit(scores.clamp(1e-6, 1.0 - 1e-6))

    return scores


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: Path,
    title: str,
    normalize: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    display_values = cm.astype(float)

    if normalize:
        row_sums = display_values.sum(axis=1, keepdims=True)
        display_values = np.divide(
            display_values,
            row_sums,
            out=np.zeros_like(display_values),
            where=row_sums != 0,
        )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=display_values,
        display_labels=class_names,
    )

    values_format = ".2f" if normalize else "d"
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False, values_format=values_format)

    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    prefix: str,
    output_dir: Path,
) -> None:
    acc = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    report_txt = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm_path = output_dir / f"{prefix}_confusion_matrix.png"
    cm_norm_path = output_dir / f"{prefix}_confusion_matrix_normalized.png"
    txt_path = output_dir / f"{prefix}_classification_report.txt"
    json_path = output_dir / f"{prefix}_metrics.json"

    save_confusion_matrix(
        cm=cm,
        class_names=class_names,
        out_path=cm_path,
        title=f"{prefix.replace('_', ' ').title()} Confusion Matrix",
        normalize=False,
    )

    save_confusion_matrix(
        cm=cm,
        class_names=class_names,
        out_path=cm_norm_path,
        title=f"{prefix.replace('_', ' ').title()} Normalized Confusion Matrix",
        normalize=True,
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": float(acc),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),
            },
            f,
            indent=2,
        )

    print(f"\n=== {prefix} ===")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Precision macro:   {precision_macro:.4f}")
    print(f"Recall macro:      {recall_macro:.4f}")
    print(f"F1-score macro:    {f1_macro:.4f}")
    print("\nClassification report:\n")
    print(report_txt)
    print(f"Saved confusion matrix to:            {cm_path}")
    print(f"Saved normalized confusion matrix to: {cm_norm_path}")
    print(f"Saved report to:                      {txt_path}")
    print(f"Saved metrics JSON to:                {json_path}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: Path,
) -> None:
    model.eval()

    segment_true = []
    segment_pred = []
    segment_paths = []

    print("\nStarting evaluation...\n")

    for x, y, file_paths in tqdm(dataloader, desc="Evaluating"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        output = model(x)
        logits = extract_model_output(output)
        preds = torch.argmax(logits, dim=1)

        segment_true.extend(y.cpu().numpy().tolist())
        segment_pred.extend(preds.cpu().numpy().tolist())
        segment_paths.extend(list(file_paths))

    save_metrics(
        y_true=segment_true,
        y_pred=segment_pred,
        class_names=class_names,
        prefix="segment_level",
        output_dir=output_dir,
    )

    file_true, file_pred = majority_vote_by_file(
        segment_true=segment_true,
        segment_pred=segment_pred,
        segment_paths=segment_paths,
    )

    save_metrics(
        y_true=file_true,
        y_pred=file_pred,
        class_names=class_names,
        prefix="file_level",
        output_dir=output_dir,
    )


def majority_vote_by_file(
    segment_true: List[int],
    segment_pred: List[int],
    segment_paths: List[str],
) -> Tuple[List[int], List[int]]:
    file_to_true = {}
    file_to_preds = {}

    for true_label, pred_label, file_path in zip(segment_true, segment_pred, segment_paths):
        file_to_true[file_path] = true_label
        file_to_preds.setdefault(file_path, []).append(pred_label)

    file_true = []
    file_pred = []

    for file_path in sorted(file_to_preds.keys()):
        voted_pred = Counter(file_to_preds[file_path]).most_common(1)[0][0]
        file_true.append(file_to_true[file_path])
        file_pred.append(voted_pred)

    return file_true, file_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PANN classifier and generate confusion matrices."
    )

    parser.add_argument("--weights_path", type=str, default="", help="Path to the trained checkpoint.")
    parser.add_argument("--test_data_path", type=str, default="", help="Path to the test dataset root folder.")
    parser.add_argument("--output_dir", type=str, default="", help="Directory for saving evaluation results.")

    parser.add_argument(
        "--model",
        type=str,
        default="cnn6",
        choices=sorted(MODEL_CLASS_NAMES.keys()),
        help="Model architecture used during training.",
    )

    parser.add_argument(
        "--sr",
        type=int,
        default=96000,
        choices=[16000, 44100, 48000, 96000],
        help="Target sample rate. This must match the training sample rate.",
    )

    parser.add_argument("--segment_seconds", type=float, default=1.0, help="Segment length in seconds.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Segment overlap ratio.")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers.")

    parser.add_argument("--window_ms", type=float, default=32.0, help="STFT window length in milliseconds.")
    parser.add_argument("--mel_bins", type=int, default=64, help="Number of mel bins.")
    parser.add_argument("--fmin", type=float, default=50.0, help="Minimum frequency for log-mel extraction.")
    parser.add_argument("--fmax", type=float, default=None, help="Maximum frequency for log-mel extraction.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.weights_path:
        raise ValueError("Please provide --weights_path.")

    if not args.test_data_path:
        raise ValueError("Please provide --test_data_path.")

    if not args.output_dir:
        raise ValueError("Please provide --output_dir.")

    if args.fmax is None:
        args.fmax = args.sr / 2.0

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = InstrumentEvalDataset(
        root_dir=args.test_data_path,
        target_sr=args.sr,
        segment_seconds=args.segment_seconds,
        overlap=args.overlap,
    )

    if len(dataset) == 0:
        raise RuntimeError("No valid test samples were found.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model(
        model_name=args.model,
        sr=args.sr,
        num_classes=len(dataset.class_names),
        window_ms=args.window_ms,
        mel_bins=args.mel_bins,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    model = load_weights(
        model=model,
        weights_path=args.weights_path,
        device=device,
    )

    model.to(device)

    print(f"\nLoaded weights from: {args.weights_path}")
    print(f"Saving results to:   {output_dir}")

    evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=dataset.class_names,
        output_dir=output_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
