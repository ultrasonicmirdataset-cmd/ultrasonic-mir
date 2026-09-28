# Ultrasonic MIR

Code for studying ultrasonic content in instrument recordings and its use in music information retrieval. The repository contains the original ultrasonic analysis, classification, and NMF source-separation tools, as well as the later 96-kS/s classification experiments. These components serve different analyses in the paper; the newer experiment scripts do not replace the original tools.

## Find the right code

| Task | Entry point | What it does |
| --- | --- | --- |
| Inspect ultrasonic energy and PUA | [`ultrasonic_analysis.py`](ultrasonic_analysis.py) | Measures ultrasonic activity in one recording or a folder; writes per-file plots and `ultrasonic_summary.csv`. |
| Inspect when an instrument plays | [`experiments/instrument_activity_threshold_inspector_colab.ipynb`](experiments/instrument_activity_threshold_inspector_colab.ipynb) | Colab tool for 0.5-s stem RMS curves, thresholded activity intervals, figures and CSVs. Diagnostic; does not train or alter labels. |
| Original PANN instrument classifier | [`pann_classification.py`](pann_classification.py), [`models.py`](models.py), [`pann_confusion_matrix.py`](pann_confusion_matrix.py) | Trains and evaluates the original classifier. Its random segment-level validation differs from the later recording-level protocol. |
| Original NMF source separation | [`NMF_separation/nmf_separation_UI.ipynb`](NMF_separation/nmf_separation_UI.ipynb), [`NMF_separation/`](NMF_separation/) | Interactive separation notebook, with local audio I/O, DSP, NMF, clustering and reconstruction modules. |
| NMF separation metrics | [`SDR_SIR_NMF.py`](SDR_SIR_NMF.py) | Computes SDR, SI-SDR and SIR for supplied estimates and references at 44.1 and 96 kS/s. |
| Later isolated, polyphonic and control experiments | [`experiments/README.md`](experiments/README.md) | Both the earlier 96-kS/s experiments and the ICASSP 2027 recording-disjoint implementations, including Random Forest, CNN6, ultrasonic-only classification, and raw quiet windows. |

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

The example reflects Drive's extensionless WAV names; `.wav` names are also recognized by the newer classification scripts. Keep the historical folder names such as `Bansury`, `Duble_bass`, and `Classic_guitar`. The newer raw quiet-window experiment instead expects the unedited instrument folders under `valid_96`, with a verified file-to-recording map; see its [protocol](experiments/ICASSP2027_PROTOCOL.md). The two polyphonic notebooks require aligned mixes and stems and expose their own Colab configuration cells.

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

## ICASSP 2027 experiments

The updated entry points live directly under [`experiments/`](experiments/), alongside the older experiment code:

| Experiment | New entry point | Input and split |
| --- | --- | --- |
| Section 5.1, 15-class RF and CNN6 | [`ICASSP2027_experiment_5_1_rf_cnn6.py`](experiments/ICASSP2027_experiment_5_1_rf_cnn6.py) | Isolated stems; families 1, 3, 4, 5 train; 2, 6 held out; compare 44.1, full 96, and 96 low-pass at 22.05 kHz. |
| Section 5.2, polyphonic CNN6 | [`ICASSP2027_polyphonic_CNN6_recording_disjoint.ipynb`](experiments/ICASSP2027_polyphonic_CNN6_recording_disjoint.ipynb) | One target and held-out song family per run; matching bandwidth modes; configurable hard negatives. |
| Section 5.3, 12-class ultrasonic-only | [`ICASSP2027_experiment_5_3_ultrasonic12.py`](experiments/ICASSP2027_experiment_5_3_ultrasonic12.py) | Isolated 96-kS/s stems; families 1, 3, 4, 5 train; 2, 6 held out; spectral bins above 20 kHz. |
| Section 5.3 raw quiet-window control, 12 classes | [`ICASSP2027_experiment_5_4_raw_quiet12.py`](experiments/ICASSP2027_experiment_5_4_raw_quiet12.py) | Unedited 96-kS/s recordings; same 12 labels as the ultrasonic-only experiment, but train families 1, 2, 4, 5 and hold out 3, 6; requires a verified raw file-to-family mapping. |

The checked-in settings exclude track 2.2. The polyphonic notebook can be reconfigured to include decimal IDs, in which case it keeps them in the same train/evaluation partition as their base recording family. For audit and training commands, required recording coverage, output manifests, historical entry points and interpretation limits, read [`experiments/README.md`](experiments/README.md) and [`experiments/ICASSP2027_PROTOCOL.md`](experiments/ICASSP2027_PROTOCOL.md). Audit the full local dataset before starting a long run. Code inspection or a synthetic smoke run alone does not verify the results on the complete recordings or reproduce a paper table.

For example, from the repository root, audit Sections 5.1 and 5.3 against a local copy of the 96-kS/s stems:

```bash
python experiments/ICASSP2027_experiment_5_1_rf_cnn6.py \
  --train96 /path/to/train_96 --derive44 \
  --output /path/to/results/section_5_1 --audit-only

python experiments/ICASSP2027_experiment_5_3_ultrasonic12.py \
  --train96 /path/to/train_96 \
  --output /path/to/results/section_5_3 --audit-only
```

These 5.1 and 5.3 audits inspect recording coverage and audio metadata; they do not prove that each class has usable active 0.5-s windows. Check that coverage before removing `--audit-only` for training. For raw quiet windows, create and verify a file-to-family map before auditing; the exact commands are in [the protocol](experiments/ICASSP2027_PROTOCOL.md#commands). The manuscript describes a recording-disjoint quiet control alongside the 12-class ultrasonic-only run but does not separately state the quiet-control labels or song IDs. The control here uses the same 12 labels and the previously requested validation families 3 and 6.

## Citation

Please cite the associated paper when bibliographic details are available.
