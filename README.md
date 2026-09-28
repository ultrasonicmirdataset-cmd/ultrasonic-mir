# Ultrasonic MIR

Code for studying ultrasonic content in instrument recordings and its use in music information retrieval. The repository contains ultrasonic analysis, classification, and NMF source-separation tools, as well as 96-kS/s classification experiments. The newer experiment scripts complement the original tools.

## Find the right code

| Task | Entry point | What it does |
| --- | --- | --- |
| Inspect ultrasonic energy and PUA | [`ultrasonic_analysis.py`](ultrasonic_analysis.py) | Measures ultrasonic activity in one recording or a folder; writes per-file plots and `ultrasonic_summary.csv`. |
| Inspect when an instrument plays | [`experiments/instrument_activity_threshold_inspector_colab.ipynb`](experiments/instrument_activity_threshold_inspector_colab.ipynb) | Colab tool for 0.5-s stem RMS curves, thresholded activity intervals, figures and CSVs. Diagnostic; does not train or alter labels. |
| Original PANN instrument classifier | [`pann_classification.py`](pann_classification.py), [`models.py`](models.py), [`pann_confusion_matrix.py`](pann_confusion_matrix.py) | Trains and evaluates the original classifier. Its random segment-level validation differs from the later recording-level protocol. |
| Original NMF source separation | [`NMF_separation/nmf_separation_UI.ipynb`](NMF_separation/nmf_separation_UI.ipynb), [`NMF_separation/`](NMF_separation/) | Interactive separation notebook, with local audio I/O, DSP, NMF, clustering and reconstruction modules. |
| NMF separation metrics | [`SDR_SIR_NMF.py`](SDR_SIR_NMF.py) | Computes SDR, SI-SDR and SIR for supplied estimates and references at 44.1 and 96 kS/s. |
| Isolated, polyphonic and control experiments | [`experiments/README.md`](experiments/README.md) | Original 96-kS/s experiments and updated recording-disjoint implementations, including Random Forest, CNN6, ultrasonic-only classification, and raw quiet windows. |

## Installation and data

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
```

For the separate NMF notebook, install its additional packages with `python -m pip install -r NMF_separation/requirements.txt`. Colab notebooks also require their own runtime setup and a mounted Drive.

Audio stems, raw recordings, checkpoints and result bundles are not included. Prepare authorized local copies or mount Drive. The isolated classification scripts expect instrument directories under `train_96` and, when using the paired audio, `train_44`:

```text
train_96/                     train_44/
  Acoustic_guitar/              Acoustic_guitar/
    Acoustic_guitar_1             Acoustic_guitar_1
    ...                           ...
  Bansury/                     Bansury/
    ...                           ...
  ...                          ...
```

The example reflects Drive's extensionless WAV names; `.wav` names are also recognized by the newer classification scripts. Keep the historical folder names such as `Bansury`, `Duble_bass`, and `Classic_guitar`. The newer raw quiet-window experiment instead expects the unedited instrument folders under `valid_96`, with a verified file-to-recording map; see its [protocol](experiments/RECORDING_SPLIT_PROTOCOL.md). The two polyphonic notebooks require aligned mixes and stems and expose their own Colab configuration cells.

## Original analysis and separation tools

In [`ultrasonic_analysis.py`](ultrasonic_analysis.py), set `input_path` to an audio file or folder and `output_dir` to a writable folder in the script's `__main__` block. Then run `python ultrasonic_analysis.py`. It writes ultrasonic masks, frequency distributions, maximum-frequency plots, and a CSV summary. Its PUA and plotting thresholds are analysis settings; they do not determine labels in the newer classification protocol.

For the original classifier, supply a directory of instrument subfolders and a checkpoint destination:

```bash
python pann_classification.py --data_path /path/to/instrument_folders \
  --save_path /path/to/model.pth --model cnn6 --sr 96000

python pann_confusion_matrix.py --weights_path /path/to/model.pth \
  --test_data_path /path/to/test_instrument_folders \
  --output_dir /path/to/evaluation --model cnn6 --sr 96000
```

Match the model, sample rate, feature settings and class order between training and evaluation. The original training script splits *segments* at random; its validation score should not be presented as a held-out-recording score.

For NMF, open [`NMF_separation/nmf_separation_UI.ipynb`](NMF_separation/nmf_separation_UI.ipynb) in a notebook environment with its working directory set to `NMF_separation/`; it imports adjacent modules (`data_io.py`, `dsp.py`, `nmf_engine.py`, `clustering.py`, `reconstruction.py`) and presents a file selection and separation dashboard. Provide the audio and select the separation settings in the notebook. To assess separated signals, fill the seven input/output paths near the start of [`SDR_SIR_NMF.py`](SDR_SIR_NMF.py), then run `python SDR_SIR_NMF.py`. This is a separate evaluation script, not a classifier.

## Recording-disjoint experiments

The updated entry points live directly under [`experiments/`](experiments/), alongside the older experiment code:

| Experiment | New entry point | Input and split |
| --- | --- | --- |
| 15-class RF and CNN6 | [`isolated_rf_cnn6.py`](experiments/isolated_rf_cnn6.py) | Isolated stems; families 1, 3, 4, 5 train; 2, 6 held out; compare 44.1, full 96, and 96 low-pass at 22.05 kHz. |
| Polyphonic CNN6 | [`polyphonic_cnn6_recording_disjoint.ipynb`](experiments/polyphonic_cnn6_recording_disjoint.ipynb) | One target and held-out song family per run; matching bandwidth modes; configurable hard negatives. |
| 12-class ultrasonic-only CNN6 | [`ultrasonic12_cnn6.py`](experiments/ultrasonic12_cnn6.py) | Isolated 96-kS/s stems; families 1, 3, 4, 5 train; 2, 6 held out; spectral bins above 20 kHz. |
| 12-class raw quiet-window control | [`raw_quiet12_cnn6.py`](experiments/raw_quiet12_cnn6.py) | Unedited 96-kS/s recordings; same 12 labels as the ultrasonic-only experiment, but train families 1, 2, 4, 5 and hold out 3, 6; requires a verified raw file-to-family mapping. |

The checked-in settings exclude track 2.2. The polyphonic notebook can be reconfigured to include decimal IDs, in which case it keeps them in the same train/evaluation partition as their base recording family. For audit and training commands, required recording coverage, output manifests, historical entry points and interpretation limits, read [`experiments/README.md`](experiments/README.md) and [`experiments/RECORDING_SPLIT_PROTOCOL.md`](experiments/RECORDING_SPLIT_PROTOCOL.md). Audit the full local dataset before starting a long run. Code inspection or a synthetic smoke run alone does not verify results on the complete recordings.

For example, from the repository root, audit the isolated 15-class and ultrasonic-only 12-class experiments against a local copy of the 96-kS/s stems:

```bash
python experiments/isolated_rf_cnn6.py \
  --train96 /path/to/train_96 --derive44 \
  --output /path/to/results/isolated --audit-only

python experiments/ultrasonic12_cnn6.py \
  --train96 /path/to/train_96 \
  --output /path/to/results/ultrasonic12 --audit-only
```

These audits inspect recording coverage and audio metadata; they do not prove that each class has usable active 0.5-s windows. Check that coverage before removing `--audit-only` for training. For raw quiet windows, create and verify a file-to-family map before auditing; the exact commands are in [the protocol](experiments/RECORDING_SPLIT_PROTOCOL.md#commands). The control uses the same 12 labels as the ultrasonic-only run, with validation families 3 and 6.
