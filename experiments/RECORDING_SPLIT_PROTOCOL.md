# Recording-level instrument classification

This is a new version of the instrument-classification experiments, placed
directly in `experiments/`. The earlier repository entry points remain in
their original locations. These files are **protocol implementations**;
real-data CNN6 runs and final raw-file mapping are still outstanding.

Keep the three Python files together: the 12-class and raw-control scripts
import the 15-class script as their common audio, feature, model, and reporting
module. Use Python 3.10+ and the root `requirements.txt`. The polyphonic
notebook runs in Colab and installs its extra dependencies in its first cell.

| Task / implementation | Entry point | Train families | Held-out families | Classes / input |
| --- | --- | --- | --- | --- |
| Isolated RF and CNN6 | `isolated_rf_cnn6.py` | 1, 3, 4, 5 | 2, 6 | 15; paired 44.1 kS/s, full 96 kS/s, 96 kS/s LP at 22.05 kHz |
| Ultrasonic-only | `ultrasonic12_cnn6.py` | 1, 3, 4, 5 | 2, 6 | 12; 96 kS/s, bins strictly above 20 kHz; no Bansury, electric guitar, or piano |
| Raw quiet-window control | `raw_quiet12_cnn6.py` | 1, 2, 4, 5 | 3, 6 | Same 12 labels as ultrasonic-only; full-band raw RMS strictly below -80 dBFS, non-digital-zero; CNN6 sees only bins above 20 kHz |
| Polyphonic ON/OFF | `polyphonic_cnn6_recording_disjoint.ipynb` | Other selected families | Selected family | One target per run; 44k, 96k_full, or 96k_lp22 |

The quiet control uses the same 12 labels as the ultrasonic-only experiment,
with validation families 3 and 6 (ultrasonic-only uses 2 and 6). It computes
the actual score without assuming a chance-level result.

Track 2.2 is excluded throughout. Recording families are assigned before
extracting any half-second windows. For CNN6, the code trains to a fixed
50 epochs by default and reports final-epoch evaluation rather than picking
the best checkpoint on the held-out recordings. Three seeds, 42/43/44, are the
defaults for the classification scripts. The polyphonic notebook exposes
`RUN_SEED` for separate runs with the same three seeds. Its original Drive
notebook has not been overwritten.

## Commands

After mounting the audio in Colab or making authorized local copies, first
audit the input folders. Use an output folder separate from the recordings:

```bash
python isolated_rf_cnn6.py \
  --train96 /path/to/train_96 --derive44 \
  --output /path/to/results/isolated --audit-only

python ultrasonic12_cnn6.py \
  --train96 /path/to/train_96 \
  --output /path/to/results/ultrasonic12 --audit-only

python raw_quiet12_cnn6.py \
  --raw-root /path/to/valid_96 \
  --create-map-template /path/to/raw_track_map.json
```

Fill each generated `track: null` with its **verified** recording family for
every included file across the 12 selected labels; do not infer a family
from the raw filename or its sort order. Then run the quiet preflight:

```bash
python raw_quiet12_cnn6.py \
  --raw-root /path/to/valid_96 --track-map /path/to/raw_track_map.json \
  --output /path/to/results/raw_quiet --audit-only
```

After an audit has no issues, repeat the corresponding command without
`--audit-only` to train. The raw quiet script accepts the newly generated
mapping as input; no private raw filenames or unfinished mappings are shipped
with the repository. It requires a separate train and evaluation raw recording
and at least one eligible quiet window on each side for every class, then
balances the window counts. Full-band RMS is computed across all samples and
channels. Exactly zero-valued windows are excluded; there are no additional
zero-fraction, zero-run, or duplicate-window filters. It reports actual
held-out accuracy and epoch-by-epoch training loss, without assuming a
chance-level result.

The 44.1-kS/s condition is derived from the *same* native 96-kS/s
audio. The documented `--derive44` option enforces that pairing digitally.
If you instead use `--train44`, the code currently checks track IDs, sample
rates, and duration agreement only: verify waveform pairing separately
before treating such a run as matched. The ultrasonic-only experiment uses
96-kS/s audio exclusively; 44.1-kS/s and LP22 inputs cannot represent its
20–48-kHz range.

For the polyphonic notebook, set one target in `SELECTED_INSTRUMENTS`, choose
`VAL_SONG_ID` where it has usable ON/OFF examples, select `AUDIO_MODE` and
`RUN_SEED`, and set or disable target-specific hard negatives. Hold the
validation family and augmentation settings fixed across 44k, 96k_full, and
96k_lp22 for each target. Its isolated-stem pretraining and hard-negative
mixtures are implementation choices that should be documented with any
reported results. Run and aggregate all targets and seeds before comparing
results across configurations.

## Readiness and provenance

The three scripts compiled, the notebook's function and configuration cells
executed, and the RF entry points completed synthetic-data runs. All three
CNN6 entry points also completed a one-epoch CPU smoke run on synthetic data,
including every advertised isolated-classification bandwidth. This checks runtime
paths, not a full training job. CNN6 and the polyphonic notebook have **not**
been trained on the full real dataset under this protocol. One actual isolated
`train_96/Acoustic_guitar/1` WAV was decoded at 96 kS/s: the activity scan
found 1,010 half-second windows, and a selected active window passed loading
and finite RF-feature extraction for full 96, 96-LP22, >20-kHz, and derived
44.1-kS/s paths. This is a one-file I/O and feature check, not an end-to-end
real-data experiment. The raw `valid_96` files remain unavailable for an
authenticated full-file read here, so the quiet-window gate has only been
tested end to end on synthetic audio. The Drive inventory
checked on September 28, 2026 includes newly uploaded `Bansury` take 5 and
`Vocals` take 2. At the file-metadata level, all 15 classes now have train
and evaluation takes in `train_96` for isolated classification, as do the 12
classes in ultrasonic-only classification. Audio readability, activity-window counts, and independence of
the new takes have not been verified. Drive revision metadata lists their
original filenames as `עותק של bansury_2.2` for `bansury_5` and
`עותק של vocals_2.2` for `vocals_2`; both pairs have matching byte sizes.
Confirm the underlying recording identity before accepting those renamed files
as distinct takes. `train_44` still lacks `Bansury`/5,
`Vocals`/2, and `Harmonica`/5; use `--derive44` to make the 44.1-kS/s
condition from the 96-kS/s recordings. The raw `valid_96` filenames do not
establish recording family 3/6. The 12 selected
classes each have at least two raw filenames at the latest inventory check,
but a verified map and eligible quiet-window counts remain outstanding.
Some filenames prefixed `עותק של` ("copy of") also need independence checked
before treating pairs as separate takes. The isolated and ultrasonic `--audit-only`
flags check file metadata and split coverage, while a training run also scans
windows for sufficient active examples. Do not describe the
previously reported percentages or runs using within-recording temporal
splits as results from this new split.
