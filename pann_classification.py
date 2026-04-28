#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PANN-based instrument classification training script.

This file is designed to run as a regular Python script, not as a Colab notebook.
It imports the model architecture from a separate models.py module.

Expected project structure:
    project/
    ├── pann_classification.py
    ├── models.py
    └── utils.py

Note:
    The provided models.py file imports do_mixup, interpolate, and pad_framewise_output
    from utils.py. Make sure utils.py is also available in the same directory.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split

from models import (
    Cnn6,
    Cnn10,
    Cnn14,
    Cnn14_no_specaug,
    Cnn14_no_dropout,
    Cnn14_emb512,
    Cnn14_emb128,
    Cnn14_emb32,
    ResNet22,
    ResNet38,
    ResNet54,
    MobileNetV1,
    MobileNetV2,
    LeeNet11,
    LeeNet24,
)


MODEL_REGISTRY = {
    "cnn6": Cnn6,
    "cnn10": Cnn10,
    "cnn14": Cnn14,
    "cnn14_no_specaug": Cnn14_no_specaug,
    "cnn14_no_dropout": Cnn14_no_dropout,
    "cnn14_emb512": Cnn14_emb512,
    "cnn14_emb128": Cnn14_emb128,
    "cnn14_emb32": Cnn14_emb32,
    "resnet22": ResNet22,
    "resnet38": ResNet38,
    "resnet54": ResNet54,
    "mobilenetv1": MobileNetV1,
    "mobilenetv2": MobileNetV2,
    "leenet11": LeeNet11,
    "leenet24": LeeNet24,
}


AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif")


class InstrumentDataset(Dataset):
    """
    Folder-based instrument dataset.

    Expected folder structure:
        dataset_root/
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
        sr: int,
        segment_seconds: float = 1.0,
        overlap: float = 0.5,
        use_rms_filter: bool = False,
        rms_threshold: float = 0.02,
    ):
        self.root_dir = Path(root_dir)
        self.sr = sr
        self.use_rms_filter = use_rms_filter
        self.rms_threshold = rms_threshold

        self.segment_samples = int(segment_seconds * sr)
        self.step = int(self.segment_samples * (1.0 - overlap))

        if self.step <= 0:
            raise ValueError("overlap must be smaller than 1.0")

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.root_dir}")

        self.index = []
        self.label_map: Dict[str, int] = {}

        self._build_index()

    def _build_index(self) -> None:
        print(f"\nIndexing dataset from: {self.root_dir}")

        label_idx = 0

        for instrument_dir in sorted(self.root_dir.iterdir()):
            if not instrument_dir.is_dir():
                continue

            instrument_name = instrument_dir.name
            self.label_map[instrument_name] = label_idx

            for audio_path in sorted(instrument_dir.rglob("*")):
                if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue

                try:
                    info = sf.info(audio_path.as_posix())
                    total_samples = info.frames
                    file_sr = info.samplerate

                    if file_sr != self.sr:
                        duration_sec = total_samples / file_sr
                        total_samples = int(duration_sec * self.sr)

                except Exception as exc:
                    print(f"Skipping file {audio_path}: {exc}")
                    continue

                if total_samples < self.segment_samples:
                    continue

                for start in range(0, total_samples - self.segment_samples + 1, self.step):
                    self.index.append(
                        {
                            "path": audio_path.as_posix(),
                            "label": label_idx,
                            "start": start,
                        }
                    )

            label_idx += 1

        if not self.index:
            raise RuntimeError("No valid audio segments were found.")

        print(f"Indexed segments: {len(self.index)}")
        print(f"Classes: {self.label_map}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.index[idx]
        path = item["path"]
        label = item["label"]
        start = item["start"]

        audio = self._load_segment(path, start)

        if self.use_rms_filter:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < self.rms_threshold:
                return self.__getitem__((idx + 1) % len(self.index))

        x = torch.tensor(audio, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y

    def _load_segment(self, path: str, start: int) -> np.ndarray:
        try:
            info = sf.info(path)
            file_sr = info.samplerate

            if file_sr == self.sr:
                audio, _ = sf.read(
                    path,
                    start=start,
                    stop=start + self.segment_samples,
                    dtype="float32",
                )
            else:
                audio, _ = sf.read(path, dtype="float32")
                audio = self._to_mono(audio)
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sr)
                audio = audio[start:start + self.segment_samples]

        except Exception:
            audio, file_sr = librosa.load(path, sr=None, mono=True)
            if file_sr != self.sr:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sr)
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
            pad_width = self.segment_samples - len(audio)
            audio = np.pad(audio, (0, pad_width))
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
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    window_size = int(sr * window_ms / 1000.0)
    hop_size = window_size // 4

    model_class = MODEL_REGISTRY[model_name]

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


def extract_model_output(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Convert model output to a tensor suitable for CrossEntropyLoss.

    Most PANN-style models return sigmoid probabilities under 'clipwise_output'.
    CrossEntropyLoss expects logits, so probabilities are converted back to logits.
    """

    if "clipwise_output" not in output:
        raise KeyError("Model output dictionary must contain 'clipwise_output'.")

    scores = output["clipwise_output"]

    if scores.min().detach().item() >= 0.0 and scores.max().detach().item() <= 1.0:
        scores = torch.logit(scores.clamp(1e-6, 1.0 - 1e-6))

    return scores


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            output = model(x)
            logits = extract_model_output(output)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    mean_loss = total_loss / max(len(loader), 1)
    accuracy = correct / total if total > 0 else 0.0

    return mean_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            output = model(x)
            logits = extract_model_output(output)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total if total > 0 else 0.0


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    save_path: str,
    learning_rate: float,
) -> None:
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0

    print("\nStarting training...\n")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_acc = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if save_path and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                save_file.as_posix(),
            )

            print(f"Saved best checkpoint to: {save_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PANN-based instrument classifier."
    )

    parser.add_argument("--data_path", type=str, default="", help="Path to the dataset root folder.")
    parser.add_argument("--save_path", type=str, default="", help="Path for saving the best checkpoint.")

    parser.add_argument(
        "--model",
        type=str,
        default="cnn6",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model architecture to train.",
    )

    parser.add_argument("--sr", type=int, default=96000, help="Target sample rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")

    parser.add_argument("--segment_seconds", type=float, default=1.0, help="Segment length in seconds.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Segment overlap ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")

    parser.add_argument("--window_ms", type=float, default=32.0, help="STFT window length in milliseconds.")
    parser.add_argument("--mel_bins", type=int, default=64, help="Number of mel bins.")
    parser.add_argument("--fmin", type=float, default=50.0, help="Minimum frequency for log-mel extraction.")
    parser.add_argument("--fmax", type=float, default=None, help="Maximum frequency for log-mel extraction.")

    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("--use_rms_filter", action="store_true", help="Skip very low-energy segments.")
    parser.add_argument("--rms_threshold", type=float, default=0.02, help="RMS threshold for segment filtering.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_path:
        raise ValueError("Please provide --data_path or set a default DATA_PATH in the script.")

    if args.fmax is None:
        args.fmax = args.sr / 2.0

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = InstrumentDataset(
        root_dir=args.data_path,
        sr=args.sr,
        segment_seconds=args.segment_seconds,
        overlap=args.overlap,
        use_rms_filter=args.use_rms_filter,
        rms_threshold=args.rms_threshold,
    )

    val_size = int(args.val_ratio * len(dataset))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    pin_memory = torch.cuda.is_available()
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = build_model(
        model_name=args.model,
        sr=args.sr,
        num_classes=len(dataset.label_map),
        window_ms=args.window_ms,
        mel_bins=args.mel_bins,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        save_path=args.save_path,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
