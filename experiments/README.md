# Extended 96-kS/s experiments

This directory holds the earlier extended experiments from the project's 96K
workspace, the updated ICASSP 2027 recording-disjoint implementations, and a
diagnostic notebook for instrument playing time. The original ultrasonic
analysis, PANN classification and NMF source-separation tools are documented
in the [repository README](../README.md). The recordings, checkpoints and
large result bundles are not included in this repository. Run the scripts on
authorized local copies or a mounted Drive and keep outputs outside the input
audio directories.

## Updated ICASSP 2027 entry points

These files are placed **directly in `experiments/`**. The older entry points
below remain available. See [the detailed protocol](ICASSP2027_PROTOCOL.md)
for exact requirements, flags, audits and interpretation limits.

| Task | Entry point | Recording families and input |
| --- | --- | --- |
| 5.1: isolated RF and CNN6, 15 classes | [`ICASSP2027_experiment_5_1_rf_cnn6.py`](ICASSP2027_experiment_5_1_rf_cnn6.py) | Train 1, 3, 4, 5; evaluate 2, 6. Paired 44.1-kS/s, 96-kS/s full band, and 96-kS/s low-pass at 22.05 kHz. |
| 5.2: polyphonic ON/OFF CNN6 | [`ICASSP2027_polyphonic_CNN6_recording_disjoint.ipynb`](ICASSP2027_polyphonic_CNN6_recording_disjoint.ipynb) | Choose one target instrument and one independent validation song family per run; choose 44k, 96k_full, or 96k_lp22 and optional hard negatives. |
| 5.3: ultrasonic-only CNN6, 12 classes | [`ICASSP2027_experiment_5_3_ultrasonic12.py`](ICASSP2027_experiment_5_3_ultrasonic12.py) | Train 1, 3, 4, 5; evaluate 2, 6; 96-kS/s signal above 20 kHz. Excludes `Bansury`, `Electric_guitar`, and `Piano`. |
| 5.3: raw quiet-window control, 12 classes | [`ICASSP2027_experiment_5_4_raw_quiet12.py`](ICASSP2027_experiment_5_4_raw_quiet12.py) | The same 12 labels as ultrasonic-only; train 1, 2, 4, 5; evaluate 3, 6; unedited `valid_96` audio with a verified recording-family map; full-band RMS gate and ultrasonic CNN6 input. |

The manuscript's Section 5.3 describes recording-level separation for the
quiet control but does not separately name its class count or track numbers.
The updated script uses the same 12 labels as the ultrasonic-only experiment
and the previously requested held-out families 3 and 6, whereas the
ultrasonic-only script holds out 2 and 6. It tests whether quiet raw windows
still carry class-specific ultrasonic cues, without assuming that held-out
accuracy is at chance level.

The three new Python scripts exclude family 2.2, and the checked-in polyphonic
notebook selects only song IDs 1–6. If `song_ids` is reconfigured to include
2.2, the notebook keeps it in the same train/evaluation partition as family
2. The ultrasonic-only experiment cannot be evaluated at 44.1 kS/s or with
a 22.05-kHz low-pass because those
inputs have no >20-kHz content. The quiet control also takes 96-kS/s raw
audio and extracts >20-kHz content. The polyphonic comparison supports all
three bandwidth settings. The 5.1 and 5.3 class sets and audio selection
criteria differ, so their results are not interchangeable.

Run the [audits](ICASSP2027_PROTOCOL.md#commands) before training. For the
5.1 and 5.3 scripts, `--audit-only` checks the available recordings and their
metadata; it does not check that every class has usable active 0.5-s windows.
The `train_96` inventory checked on September 28, 2026 includes the newly
uploaded `Bansury` take 5 and `Vocals` take 2. Thus all 15 classes have a
training and held-out recording for the proposed 5.1 split, and the 12
classes have coverage for 5.3 at the filename level. The new takes have not
been decoded or checked for independence. Drive revision metadata lists the
original filenames of `bansury_5` and `vocals_2` as copies of `bansury_2.2`
and `vocals_2.2`, respectively, with matching byte counts. Verify the actual
recording identities before accepting those renamed files as different takes;
file-name coverage alone cannot establish the intended split. `train_44` still
lacks `Bansury`/5,
`Vocals`/2 and `Harmonica`/5. Use `--derive44` to obtain the paired 44.1-kS/s
condition from the 96-kS/s recordings rather than using the incomplete
`train_44` folder.

For the 12 selected classes, the `valid_96` inventory includes at least two
raw files per class at the latest check. However, the filenames do not give
verified song-family IDs. Some are named as copies, so confirm recording
independence and assign every included file to a verified family before
claiming the required 1/2/4/5 versus 3/6 split. The map must establish at
least one train and one held-out raw recording per class; an audit and a
complete run must also find eligible quiet windows on both sides. File counts
alone cannot establish a held-out score or convergence result.

## Playing-time and threshold inspection

[`instrument_activity_threshold_inspector_colab.ipynb`](instrument_activity_threshold_inspector_colab.ipynb)
is a diagnostic Colab notebook for answering when each instrument plays. It
mounts Drive, reads the isolated 96-kS/s masters from
`/content/drive/MyDrive/train_96`, and computes 0.5-s RMS windows for selected
instruments. It accepts the Drive's extensionless WAV names as well as `.wav`
files, and recognizes 2.2 as a distinct take. Its adjustable threshold
defaults to -70 dBFS. Each run saves
per-stem plots and raw activity CSVs, per-instrument and combined summaries,
and metadata under timestamped folders in
`/content/drive/MyDrive/96k project/threshold_inspections`. Set its input
and output paths in the configuration cell if your Drive layout differs.
This notebook does not train a model or change existing labels.

## Earlier extended experiments

| Experiment | Entry point | Input | What it tests |
| --- | --- | --- | --- |
| Polyphonic ON/OFF detection | `polyphonic/instrument_mix_crnn_colab.ipynb` | `train_44`, `train_96`, aligned multitrack mixes/stems | Binary activity of selected instruments inside mixtures at 44.1, full-band 96, or 96-kS/s low-pass input. |
| Ultrasonic-only classification | `ultrasonic_only/ultrasonic_only_cnn6.py` | `train_96` isolated stems | Twelve classes at 20–48 kHz; activity selected on unfiltered full-band 0.5-s windows. |
| Raw-recording quiet control | `ultrasonic_only/ultrasonic_raw_silence_cnn6.py` | `valid_96` unedited recordings | Whether 20–48-kHz residuals in audibly quiet, non-digital-zero windows predict instrument identity across independent recordings. |
| Random Forest bandwidth comparison | `random_forest/classic_instrument_rf_timestamped.py` | Paired `train_44` and `train_96` stems | Fifteen-class 44.1/full 96/96-low-pass comparison using matched active windows and linear-frequency features. |

Install the root `requirements.txt` before using the Python scripts. The
notebook also installs its Colab dependencies. Each directory name in the
dataset is an instrument label, using the project's historical spellings
(`Bansury`, `Duble_bass`, `Classic_guitar`, etc.).

## Polyphonic notebook

Open `polyphonic/instrument_mix_crnn_colab.ipynb` in Colab and edit the **first
configuration cell** before training. Its checked-in preset is one target,
`Clarinet`, at `44k`, with song family 5 held out. Set
`SELECTED_INSTRUMENTS` and `AUDIO_MODE` for each paired comparison, and check
the full configuration saved in each run directory. The notebook includes
isolated-stem pretraining, natural and synthetic mixtures, optional hard
negatives, and natural plus selected-only validation. It is a single-run
notebook; three independent model runs require three recorded seeds/runs.
Execution logs and checkpoints were removed from the committed notebook.

## Ultrasonic-only, twelve classes

The default excludes `Bansury` (Bansuri), `Electric_guitar`, and `Piano` from
the fifteen available labels. The remaining twelve are acoustic guitar,
clarinet, classical guitar, drums, double bass, flute, harmonica, kanun,
saxophone, trumpet, violin, and vocals. The cutoff is **20 kHz**, not
22.05 kHz. The script's `__main__` without `--run` only checks the model shape.

```bash
python experiments/ultrasonic_only/ultrasonic_only_cnn6.py
python experiments/ultrasonic_only/ultrasonic_only_cnn6.py --run \
  --data-root /path/to/train_96 \
  --output-root /path/to/ultrasonic_runs \
  --local-cache-root /path/to/local_audio_cache
```

The supplied source used thirteen classes, excluding only piano and electric
guitar. Its completed September 15 run therefore **cannot** be labeled a
twelve-class result. The twelve-class configuration above requires a new run.
The clean-stem script partitions time within each stem with a two-second guard;
that does not establish generalization to a different recording of the same
instrument.

## Quiet windows from raw recordings

The control imports the classifier above, so keep both files in the same
directory. It reads the unedited files under `valid_96`, assigns entire raw
recordings to train or validation, selects windows only from the audible band
below -80 dBFS on every channel, rejects digital zeros, and gives the model
only 20–48-kHz content. Eligible classes are determined by a census of
independent recordings and valid quiet windows; their number is not assumed.
Without `--run`, the command checks the audible gate and zero rejection on
synthetic signals without reading the dataset.

```bash
python experiments/ultrasonic_only/ultrasonic_raw_silence_cnn6.py
python experiments/ultrasonic_only/ultrasonic_raw_silence_cnn6.py --run \
  --data-root /path/to/valid_96 \
  --output-root /path/to/raw_silence_runs \
  --local-cache-root /path/to/local_raw_audio_cache
```

The older `quiet_matched_runs` in the Drive workspace are a **stem** control
with three eligible classes; they are not output from this raw-recording
script. No completed `raw_silence_runs` bundle was found during this import.
Consequently, this repository does not yet document a measured chance-level
raw-control result.

## Random Forest

The RF script is standalone. It derives activity from the native 96-kS/s
master, pairs the same 0.5-s windows across 44.1 kS/s, full 96 kS/s and a
96-kS/s low-pass-at-22.05-kHz condition, then trains three forests. Use the
following settings to match the inspected September 13 **RF run config**:

```bash
python experiments/random_forest/classic_instrument_rf_timestamped.py \
  --data-root-44 /path/to/train_44 \
  --data-root-96 /path/to/train_96 \
  --output-root /path/to/rf_timestamped_runs \
  --shared-cache-dir /path/to/rf_feature_cache \
  --rf-trees 200 --max-train-per-class 300 --max-test-per-class 150 \
  --rf-seeds 42,43,44
```

Use `--audit-only` to inspect the inventory before computing features or
training. The inspected completed run used fifteen labels and a guarded
**temporal split within each stem**, not a recording-disjoint split. This is a
material difference from the current paper's Section 3 description and must
be reconciled before claiming exact paper reproduction. The code itself warns
that some classes lack the independent song families needed for a fully
recording-disjoint fifteen-class test.

## Status of earlier Drive runs

The historical thirteen-class ultrasonic run, three-class stem quiet run,
and RF run are distinct experiments with different class sets and splits. Do
not pool their scores or use their tables as the twelve-class or raw-recording
results. The manuscript's recording-disjoint claims require new full-data runs
of the updated entry points above, with audited manifests; earlier within-stem
splits cannot establish that protocol. No new numerical results are claimed by
importing this code.
