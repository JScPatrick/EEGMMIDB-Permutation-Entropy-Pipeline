# EEGMMIDB Permutation-Entropy (PE) Decoding Pipeline

Pipeline used to benchmark permutation-entropy features on the EEGMMIDB. It decodes motor execution (ME) and motor imagery (MI) tasks using:

* Temporal PE
* HV spatial PE (horizontal/vertical triplets)
* Cross-like spatial PE (plus-shaped neighboring electrodes)
* Spatiotemporal PE (center, up, right, and left electrodes combined with a time-lagged center electrode signal)
* Combined PE feature models, plus ablations and sanity checks (within-pair label permutation and capacity-matched controls)

## What this repository does

* Loads EEGMMIDB EDF files (subjects S001–S109)
* Preprocesses EEG (EEG-only, resampling, average reference, optional notch)
* Extracts paired baseline/task epochs per cue
* Applies band-invariant artifact rejection and trial capping
* Computes PE features under two scopes: whole scalp and sensorimotor ROI
* Runs within-subject decoding using group-aware CV (trial-pair grouped)
* Writes per-subject and group summaries, plus figures

## Requirements

* Python 3.9+ is recommended
* Dependencies: see `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

## Data

You can download EEGMMIDB (PhysioNet MI/ME dataset) and then place it in:

```text
./data/EEGMMIDB/S001
./data/EEGMMIDB/S002
./data/EEGMMIDB/S003
...
```

## Configuration

You can copy the example configuration and then edit paths as needed:

```bash
cp config.example.json config.json
```

Important settings:

* `DATA_ROOT`: path containing `S001`, `S002`, `S003`, ...
* `OUT_FOLDER`: where outputs are written (default: `./outputs/...`)
* `BANDS`, epoch windows, rejection thresholds, CV folds, etc. (see `run_pipeline.py`)

The data path can also be set through the environment variable:

```bash
export EEGMMIDB_ROOT="/path/to/EEGMMIDB"
```

## To run the code

```bash
python run_pipeline.py
```

## Outputs

A run folder is created under `OUT_FOLDER` and will contain files such as:

* `run_config.txt`, `run_config.json`
* `subject_summary.csv`
* `decoding_by_subject.csv`
* `decoding_group_summary.csv`
* `family_comparisons_stats.csv`
* `feature_condition_summary.csv`
* `ablation_by_subject.csv`, `ablation_group_summary.csv`
* `permutation_stats.csv`
* `matched_subjects_MI_ME_summary.csv`
* `cross_count_distribution_summary.csv`
* `capacity_matched_controls.csv`
* Figures: `Fig_decoding_bar_with_dots_<band>_<scope>.png`

## Other notes

* This pipeline uses deterministic RNG (`RNG_SEED`)
* This pipeline uses group-aware CV (trial-pair grouping) when available (`StratifiedGroupKFold`)
