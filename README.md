# Ultrasonic MIR

Code repository for ultrasonic Music Information Retrieval (MIR) analysis.

This repository contains the code used to analyze whether ultrasonic frequency content, preserved in high-sample-rate audio recordings, can support MIR tasks such as instrument classification and source separation.

The project focuses on comparing conventional audio-rate representations with high-resolution recordings, especially 16 kHz, 44.1 kHz, and 96 kHz settings.

## Repository Structure

```text
ultrasonic-mir/
├── NMF_separation/
│   └── README.md
├── SDR_SIR_NMF.py
├── models.py
├── pann_classification.py
├── pann_confusion_matrix.py
├── requirements.txt
└── ultrasonic_analysis.py
```

## Files

### `ultrasonic_analysis.py`

Performs ultrasonic activity analysis on audio files or folders.

Main outputs include:

- ultrasonic activity measures
- frame-level and second-level PUA statistics
- frequency-distribution plots
- maximum-frequency-per-frame plots
- CSV summary file

The input and output paths are intentionally left empty in the script and should be set by the user before running.

### `pann_classification.py`

Trains PANN-based instrument classification models.

The script supports multiple sample-rate settings:

- 16 kHz
- 44.1 kHz
- 48 kHz
- 96 kHz

It imports the model architectures from `models.py`.

Example usage:

```bash
python pann_classification.py \
  --data_path "" \
  --save_path "" \
  --model cnn6 \
  --sr 96000
```

Supported models include:

```text
cnn6
cnn10
cnn14
cnn14_no_specaug
cnn14_no_dropout
cnn14_emb512
cnn14_emb128
cnn14_emb32
resnet22
resnet38
resnet54
mobilenetv1
mobilenetv2
leenet11
leenet24
```

### `pann_confusion_matrix.py`

Evaluates a trained PANN classification model and generates confusion matrices.

The script saves:

- segment-level confusion matrix
- normalized segment-level confusion matrix
- file-level confusion matrix using majority voting
- normalized file-level confusion matrix
- classification report
- JSON metrics summary

Example usage:

```bash
python pann_confusion_matrix.py \
  --weights_path "" \
  --test_data_path "" \
  --output_dir "" \
  --model cnn6 \
  --sr 96000
```

The selected model and sample rate must match the training configuration.

### `models.py`

Contains the neural-network architectures used by the PANN classification pipeline.

The classification and evaluation scripts import model classes from this file.

### `SDR_SIR_NMF.py`

Computes source-separation evaluation metrics for NMF-based experiments.

This file is used for evaluating separation quality using metrics such as SDR and SIR.

### `NMF_separation/`

Contains the NMF source-separation code.

This folder has its own separate README file with specific usage instructions.

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Requirements

The main dependencies are listed in `requirements.txt`:

```text
numpy
scipy
pandas
matplotlib
librosa
soundfile
torch
torchaudio
torchlibrosa
scikit-learn
tqdm
```

## Dataset Format

For classification experiments, the expected dataset structure is:

```text
dataset_root/
├── instrument_1/
│   ├── audio_1.wav
│   └── audio_2.wav
├── instrument_2/
│   ├── audio_1.wav
│   └── audio_2.wav
└── ...
```

Each subfolder is treated as one class.

## Notes

- Paths are intentionally left empty in the public scripts.
- Audio files, trained checkpoints, generated figures, and CSV result files should usually not be committed to the repository.
- For reproducibility, use the same model name, sample rate, window configuration, and class-folder order during training and evaluation.
- The NMF pipeline has a separate README inside the `NMF_separation/` folder.

## Citation

If you use this code, please cite the related paper once available.
