"""
Permutation-Entropy (PE) Decoding Pipeline for EEGMMIDB Public Data
Cross-like spatial and spatiotemporal ordinal patterns benchmarked against temporal and horizontal/vertical spatial PE

Important design choices used:
- Group-aware CV (trial-pair grouping) through StratifiedGroupKFold when available.
- Band-invariant trial rejection and band-invariant trial capping:
  Rejection and capping are computed once (on a fixed rejection band) and then applied to all analysis bands.
  This prevents different bands from silently using different trial subsets.
- MI vs ME fairness supports:
  (a) contrast-specific PTP thresholds, (b) optional ROI-only PTP check, (c) explicit matched MI/ME subject summaries.
- Two analysis scopes:
  "whole" vs "sensorimotor" (feature extraction scope, not just rejection).
- Sensorimotor scope supports through novel families via motor-centered cross construction.
- Additions for Strengthening:
  1) Ablation to test whether “novel features add complementary variance”.
  2) Within-pair label permutation test (swap baseline/task within each trial group) as a leakage/chance sanity check.
  3) Motor-relevant band analysis (mu, beta, mu+beta, broad).
  4) Sensorimotor ROI vs whole-scalp as an analysis factor.
  5) Clear novelty test framing (“do cross/spatiotemporal outperform temporal/HV under rigorous CV?”).
  6) Robustness grid runner as an option (multiple config variants in one run).
  7) Minimal interpretability output hooks (delta-PE summaries by scope/family).
  8) Minimal “ready-to-run repo” items: single script, config-first, deterministic RNG, comprehensive outputs.

Outputs (per run folder):
- run_config.txt / run_config.json
- subject_summary.csv
- decoding_by_subject.csv
- decoding_group_summary.csv
- family_comparisons_stats.csv
- feature_condition_summary.csv
- ablation_by_subject.csv / ablation_group_summary.csv
- permutation_stats.csv
- matched_subjects_MI_ME_summary.csv
- cross_count_distribution_summary.csv
- capacity_matched_controls.csv
- Fig_decoding_bar_with_dots_<band>_<scope>.png
"""

import os, re, math, json, warnings, platform
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from itertools import permutations

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import mne

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except Exception:
    HAS_SGK = False

from scipy import stats


# -------------------------------
# CONFIG
# -------------------------------
@dataclass
class Config:
    # ---- run control ----
    DEBUG: bool = False
    DEBUG_N_SUBJECTS: int = 10
    DEBUG_SUBJECTS: Tuple[str, ...] = ()
    SUBJECTS_INCLUDE: Optional[Tuple[str, ...]] = None

    # ---- paths ----
    DATA_ROOT: str = os.environ.get("EEGMMIDB_ROOT", "./data/EEGMMIDB")
    OUT_FOLDER: str = os.environ.get("OUT_FOLDER", "./outputs/pe_cross_spatiotemporal_results")

    # ---- EEGMMIDB run IDs ----
    ME_RUNS: Tuple[int, ...] = (3, 5, 7, 9, 11, 13)
    MI_RUNS: Tuple[int, ...] = (4, 6, 8, 10, 12, 14)

    # ---- resampling ----
    TARGET_SFREQ: float = 160.0

    # ---- analysis bands ----
    BANDS: Tuple[Tuple[str, float, float], ...] = (
        ("broad_1_40", 1.0, 40.0),
        ("mu_8_12", 8.0, 12.0),
        ("beta_13_30", 13.0, 30.0),
        ("mubeta_8_30", 8.0, 30.0),
    )

    DO_NOTCH: bool = False
    NOTCH_HZ: float = 60.0

    # ---- rejection band (fixed; used for rejection + capping only) ----
    REJECT_BAND: Tuple[str, float, float] = ("reject_1_40", 1.0, 40.0)

    # ---- epoch windows relative to cue (seconds) ----
    BASELINE_TMIN: float = -3.0
    BASELINE_TMAX: float = 0.0
    TASK_TMIN: float = 0.5
    TASK_TMAX: float = 3.5

    # ---- artifact rejection thresholds ----
    REJECT_PTP_UV: float = 150.0
    REJECT_PTP_UV_MI: float = 150.0
    REJECT_PTP_UV_ME: float = 300.0

    PTP_EXCEED_FRACTION: float = 0.20
    PTP_CHECK_MODE: str = "either"  # "either" or "both"

    PTP_USE_ROI_ONLY: bool = True
    PTP_ROI_CHANNELS: Tuple[str, ...] = (
        "C3", "C4", "CZ", "FC3", "FC4", "CP3", "CP4", "C1", "C2"
    )

    FLAT_STD_UV: float = 0.5
    FLAT_MAX_FRACTION_CH: float = 0.2

    # ---- PE parameters ----
    TEMPORAL_M: int = 4
    TEMPORAL_TAU_MS: float = 0.0
    TEMPORAL_TAU_SAMPLES: int = 1

    SPATIOTEMPORAL_DT_MS: float = 0.0
    SPATIOTEMPORAL_DT_SAMPLES: int = 1

    # ---- trial capping (band-invariant; applied once using REJECT_BAND) ----
    MAX_TRIALS_PER_RUN_PER_CONTRAST: int = 40

    # ---- family-specific downsampling ----
    DOWNSAMPLE_HV: int = 2
    DOWNSAMPLE_CROSS: int = 1
    DOWNSAMPLE_ST: int = 1

    # ---- scopes (analysis factor) ----
    SCOPES: Tuple[str, ...] = ("whole", "sensorimotor")
    SENSORIMOTOR_CHANNELS: Tuple[str, ...] = (
        "C3", "C4", "CZ", "C1", "C2", "FC3", "FC4", "CP3", "CP4"
    )

    # Motor-centered cross construction for sensorimotor novelty families
    MOTOR_CROSS_CENTERS: Tuple[str, ...] = ("C3", "C4", "CZ", "C1", "C2")

    # Minimum crosses required per scope
    MIN_CROSSES_REQUIRED_WHOLE: int = 10
    MIN_CROSSES_REQUIRED_SENSOR: int = 3

    # ---- decoding ----
    CV_FOLDS: int = 5
    MIN_TRIALS_PER_CONTRAST: int = 8
    RNG_SEED: int = 0

    # ---- strengthening additions ----
    RUN_ABLATION: bool = True

    RUN_PERMUTATION_TEST: bool = True
    N_PERMUTATIONS: int = 200
    PERMUTE_ONLY_FAMILIES: Tuple[str, ...] = ("all_features",)

    RUN_CAPACITY_CONTROLS: bool = True

    # Optional robustness grid: list of small override dicts; empty => single run
    ROBUSTNESS_GRID: Tuple[Dict[str, Any], ...] = ()


def load_config_from_json(path: str, cfg_obj: Config) -> Config:
    """Optionally override Config fields from a JSON file (e.g., config.json)."""
    if not os.path.exists(path):
        return cfg_obj

    with open(path, "r") as f:
        overrides = json.load(f)

    for k, v in overrides.items():
        if hasattr(cfg_obj, k):
            setattr(cfg_obj, k, v)
        else:
            print(f"[WARN] Unknown config key in {path}: {k} (ignored)")
    return cfg_obj

cfg = Config()
cfg = load_config_from_json("config.json", cfg)

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


# -------------------------------
# JSON-safe config writer
# -------------------------------
def _jsonable(x):
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def write_run_config(out_dir: str, cfg_obj: Config):
    cfg_dict = _jsonable(asdict(cfg_obj))
    info = {
        "run_datetime_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "DATA_ROOT": cfg_obj.DATA_ROOT,
        "results_dir": out_dir,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "mne_version": getattr(mne, "__version__", "unknown"),
        "RNG_SEED": cfg_obj.RNG_SEED,
        "config": cfg_dict,
    }

    with open(os.path.join(out_dir, "run_config.txt"), "w") as f:
        f.write("RUN CONFIG — EEGMMIDB PE Decoding Pipeline\n\n")
        for k in ["run_datetime_local", "DATA_ROOT", "results_dir", "platform", "python_version", "mne_version", "RNG_SEED"]:
            f.write(f"{k}: {info[k]}\n")
        f.write("\nCONFIG VALUES:\n")
        for k, v in cfg_dict.items():
            f.write(f"  {k}: {v}\n")

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(info, f, indent=2)


# -------------------------------
# IO helpers
# -------------------------------
def list_subjects(root: str, cfg_obj: Config) -> List[str]:
    subs = sorted([d for d in os.listdir(root) if re.match(r"^S\d{3}$", d)])

    if cfg_obj.SUBJECTS_INCLUDE:
        include = set(cfg_obj.SUBJECTS_INCLUDE)
        subs = [s for s in subs if s in include]
        return subs

    if cfg_obj.DEBUG:
        if cfg_obj.DEBUG_SUBJECTS and len(cfg_obj.DEBUG_SUBJECTS) > 0:
            include = set(cfg_obj.DEBUG_SUBJECTS)
            subs = [s for s in subs if s in include]
        else:
            subs = subs[: int(cfg_obj.DEBUG_N_SUBJECTS)]
    return subs


def edf_path(cfg_obj: Config, subj: str, run: int) -> str:
    return os.path.join(cfg_obj.DATA_ROOT, subj, f"{subj}R{run:02d}.edf")


# -------------------------------
# Channel normalization / rename
# -------------------------------
def norm_name(name: str) -> str:
    s = str(name).upper().strip()
    s = s.replace("EEG", "").replace("REF", "")
    s = re.sub(r"[\s\-\:\._/\\]+", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def canonical_1020_mapping(raw_ch_names: List[str]) -> Dict[str, str]:
    std_mont = mne.channels.make_standard_montage("standard_1020")
    std_map = {norm_name(n): n for n in std_mont.ch_names}
    rename = {}
    for ch in raw_ch_names:
        k = norm_name(ch)
        if k in std_map:
            rename[ch] = std_map[k]
    return rename


# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_raw(cfg_obj: Config, raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, float, float]:
    sfreq_original = float(raw.info["sfreq"])
    raw.pick(picks="eeg")

    ren = canonical_1020_mapping(raw.ch_names)
    if ren:
        raw.rename_channels(ren)

    try:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                        on_missing="ignore", verbose=False)
    except Exception:
        pass

    if abs(float(raw.info["sfreq"]) - float(cfg_obj.TARGET_SFREQ)) > 1e-6:
        raw.resample(cfg_obj.TARGET_SFREQ, npad="auto", verbose=False)

    raw.set_eeg_reference("average", verbose=False)
    sfreq_used = float(raw.info["sfreq"])
    return raw, sfreq_original, sfreq_used


def bandpass_raw(cfg_obj: Config, raw_prepared: mne.io.BaseRaw, l_freq: float, h_freq: float) -> mne.io.BaseRaw:
    r = raw_prepared.copy()
    r.filter(l_freq, h_freq, verbose=False)
    if cfg_obj.DO_NOTCH:
        r.notch_filter(freqs=[cfg_obj.NOTCH_HZ], verbose=False)
    return r


# -------------------------------
# Trial-paired epoch extraction
# -------------------------------
def extract_paired_epochs_from_cues(
    cfg_obj: Config,
    raw: mne.io.BaseRaw,
    run_number: int,
    cue_labels: Tuple[str, ...] = ("T1", "T2"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sf = float(raw.info["sfreq"])
    want = set([str(x).strip() for x in cue_labels])

    if raw.annotations is None or len(raw.annotations) == 0:
        return (np.empty((0, raw.get_data().shape[0], 0)),
                np.empty((0, raw.get_data().shape[0], 0)),
                np.empty((0,), dtype=object))

    b0 = int(round(cfg_obj.BASELINE_TMIN * sf))
    b1 = int(round(cfg_obj.BASELINE_TMAX * sf))
    t0 = int(round(cfg_obj.TASK_TMIN * sf))
    t1 = int(round(cfg_obj.TASK_TMAX * sf))
    b_len = b1 - b0
    t_len = t1 - t0
    if b_len <= 0 or t_len <= 0:
        raise ValueError("Bad epoch window lengths.")

    Xb, Xt, gids = [], [], []
    cue_idx = 0
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        if str(desc).strip() not in want:
            continue
        cue = int(round(onset * sf))
        bs, be = cue + b0, cue + b1
        ts, te = cue + t0, cue + t1
        if bs < 0 or ts < 0 or be > raw.n_times or te > raw.n_times:
            cue_idx += 1
            continue

        seg_b = raw.get_data(start=bs, stop=be)
        seg_t = raw.get_data(start=ts, stop=te)
        if seg_b.shape[1] == b_len and seg_t.shape[1] == t_len:
            Xb.append(seg_b)
            Xt.append(seg_t)
            gids.append(f"run{run_number:02d}_cue{cue_idx:04d}_s{cue}")
        cue_idx += 1

    if len(Xb) == 0:
        return (np.empty((0, raw.get_data().shape[0], b_len)),
                np.empty((0, raw.get_data().shape[0], t_len)),
                np.empty((0,), dtype=object))

    return np.stack(Xb, axis=0), np.stack(Xt, axis=0), np.array(gids, dtype=object)


# -------------------------------
# Artifact rejection (band-invariant use)
# -------------------------------
def reject_bad_trials(
    cfg_obj: Config,
    Xb: np.ndarray,
    Xt: np.ndarray,
    trial_ids: np.ndarray,
    contrast_type: str,
    ch_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:

    n0 = int(Xb.shape[0])
    if n0 == 0:
        return Xb, Xt, trial_ids, {"n_raw": 0, "n_after_ptp": 0, "n_after_flat": 0}

    ptp_uv = getattr(cfg_obj, f"REJECT_PTP_UV_{contrast_type}", getattr(cfg_obj, "REJECT_PTP_UV", 150.0))
    ptp_thresh = float(ptp_uv) * 1e-6

    flat_std = float(cfg_obj.FLAT_STD_UV) * 1e-6
    frac_thr = float(cfg_obj.PTP_EXCEED_FRACTION)
    mode = str(cfg_obj.PTP_CHECK_MODE).lower().strip()

    roi_idx = None
    if bool(getattr(cfg_obj, "PTP_USE_ROI_ONLY", False)):
        roi_set = {c.upper() for c in getattr(cfg_obj, "PTP_ROI_CHANNELS", ())}
        idx = [i for i, nm in enumerate(ch_names) if nm.upper() in roi_set]
        if len(idx) >= 3:
            roi_idx = np.array(idx, dtype=int)

    keep_ptp = []
    for i in range(n0):
        eb = Xb[i]
        et = Xt[i]
        eb_chk = eb[roi_idx, :] if roi_idx is not None else eb
        et_chk = et[roi_idx, :] if roi_idx is not None else et

        frac_b = float(np.mean(np.ptp(eb_chk, axis=1) > ptp_thresh))
        frac_t = float(np.mean(np.ptp(et_chk, axis=1) > ptp_thresh))

        if mode == "both":
            bad = (frac_b > frac_thr) and (frac_t > frac_thr)
        else:
            bad = (frac_b > frac_thr) or (frac_t > frac_thr)

        if not bad:
            keep_ptp.append(i)

    Xb1, Xt1, ids1 = Xb[keep_ptp], Xt[keep_ptp], trial_ids[keep_ptp]
    n1 = int(Xb1.shape[0])

    keep_flat = []
    for i in range(n1):
        eb = Xb1[i]
        et = Xt1[i]
        frac_flat_b = float(np.mean(np.std(eb, axis=1) < flat_std))
        frac_flat_t = float(np.mean(np.std(et, axis=1) < flat_std))
        if (frac_flat_b > cfg_obj.FLAT_MAX_FRACTION_CH) or (frac_flat_t > cfg_obj.FLAT_MAX_FRACTION_CH):
            continue
        keep_flat.append(i)

    Xb2, Xt2, ids2 = Xb1[keep_flat], Xt1[keep_flat], ids1[keep_flat]
    n2 = int(Xb2.shape[0])

    return Xb2, Xt2, ids2, {"n_raw": n0, "n_after_ptp": n1, "n_after_flat": n2}


def cap_trial_ids(
    rng: np.random.Generator,
    trial_ids: np.ndarray,
    max_trials: int
) -> np.ndarray:
    if max_trials is None:
        return trial_ids
    u = np.unique(trial_ids)
    if u.size <= max_trials:
        return u
    return rng.choice(u, size=int(max_trials), replace=False)


# -------------------------------
# Ordinal pattern ID mapping
# -------------------------------
class PatternIdMapper:
    def __init__(self, m: int):
        self.m = m
        self.pow = (m ** np.arange(m - 1, -1, -1)).astype(int)
        size = m ** m
        lut = -np.ones(size, dtype=int)
        for idx, perm in enumerate(permutations(range(m))):
            code = int(np.dot(np.array(perm, dtype=int), self.pow))
            lut[code] = idx
        self.lut = lut
        self.n_patterns = math.factorial(m)

    def ids_from_vectors(self, V: np.ndarray) -> np.ndarray:
        perms = np.argsort(V, axis=0, kind="mergesort").astype(int)
        codes = (perms.T * self.pow).sum(axis=1)
        return self.lut[codes]


def perm_entropy_from_ids(ids: np.ndarray, n_patterns: int) -> float:
    if ids.size == 0:
        return np.nan
    ids = ids[ids >= 0]
    if ids.size == 0:
        return np.nan
    counts = np.bincount(ids, minlength=n_patterns).astype(float)
    p = counts[counts > 0]
    p /= p.sum()
    H = -np.sum(p * np.log2(p))
    Hmax = math.log2(n_patterns) if n_patterns > 1 else 1.0
    return float(H / Hmax)


mapper_m3 = PatternIdMapper(3)
mapper_m4 = PatternIdMapper(4)
mapper_m5 = PatternIdMapper(5)


# -------------------------------
# Layouts
# -------------------------------
def get_channel_xy(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, List[str]]:
    pos = {}
    try:
        mont = raw.get_montage()
        if mont is not None:
            ch_pos = mont.get_positions().get("ch_pos", {})
            for k, v in ch_pos.items():
                if k in raw.ch_names and np.all(np.isfinite(v[:2])):
                    pos[k] = np.array([float(v[0]), float(v[1])], dtype=float)
    except Exception:
        pass

    names, xy = [], []
    for ch in raw.ch_names:
        if ch in pos:
            names.append(ch)
            xy.append(pos[ch])
            continue
        try:
            idx = raw.ch_names.index(ch)
            loc = raw.info["chs"][idx].get("loc", None)
            if loc is not None and len(loc) >= 2 and np.all(np.isfinite(loc[:2])):
                names.append(ch)
                xy.append([float(loc[0]), float(loc[1])])
        except Exception:
            continue

    if len(xy) == 0:
        return np.empty((0, 2)), []
    return np.array(xy, dtype=float), names


def compute_cross_neighbors(raw: mne.io.BaseRaw, min_dir_margin: float = 1e-9) -> List[Tuple[int, int, int, int, int]]:
    xy, names = get_channel_xy(raw)
    if xy.shape[0] == 0:
        return []

    name_to_idx = {n: raw.ch_names.index(n) for n in names}
    coords = {n: xy[i] for i, n in enumerate(names)}

    crosses = []
    for c_name in names:
        x0, y0 = coords[c_name]
        up = [n for n in names if coords[n][1] > y0 + min_dir_margin]
        dn = [n for n in names if coords[n][1] < y0 - min_dir_margin]
        rt = [n for n in names if coords[n][0] > x0 + min_dir_margin]
        lf = [n for n in names if coords[n][0] < x0 - min_dir_margin]

        def pick(cands, axis):
            if not cands:
                return None
            best, best_key = None, None
            for n in cands:
                x, y = coords[n]
                primary = (y - y0) if axis == "y+" else (y0 - y) if axis == "y-" else (x - x0) if axis == "x+" else (x0 - x)
                ortho = abs((x - x0) if axis.startswith("y") else (y - y0))
                dist = math.hypot(x - x0, y - y0)
                key = (primary, ortho, dist)
                if best_key is None or key < best_key:
                    best_key, best = key, n
            return best

        u = pick(up, "y+")
        d = pick(dn, "y-")
        l = pick(lf, "x-")
        r = pick(rt, "x+")
        if None in (u, d, l, r):
            continue

        crosses.append((name_to_idx[c_name], name_to_idx[u], name_to_idx[d], name_to_idx[l], name_to_idx[r]))

    out, seen = [], set()
    for t in crosses:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def spatial_triplets_hv(raw: mne.io.BaseRaw, n_bins: int = 7) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    xy, names = get_channel_xy(raw)
    if xy.shape[0] == 0:
        return [], []

    raw_idx = np.array([raw.ch_names.index(n) for n in names], dtype=int)
    x = xy[:, 0]
    y = xy[:, 1]

    y_bins = np.linspace(y.min() - 1e-12, y.max() + 1e-12, n_bins + 1)
    x_bins = np.linspace(x.min() - 1e-12, x.max() + 1e-12, n_bins + 1)

    row = np.digitize(y, y_bins) - 1
    col = np.digitize(x, x_bins) - 1

    horiz = []
    for rid in np.unique(row):
        idx = np.where(row == rid)[0]
        if idx.size < 3:
            continue
        s = idx[np.argsort(x[idx])]
        for i in range(len(s) - 2):
            horiz.append((raw_idx[s[i]], raw_idx[s[i + 1]], raw_idx[s[i + 2]]))

    vert = []
    for cid in np.unique(col):
        idx = np.where(col == cid)[0]
        if idx.size < 3:
            continue
        s = idx[np.argsort(y[idx])]
        for i in range(len(s) - 2):
            vert.append((raw_idx[s[i]], raw_idx[s[i + 1]], raw_idx[s[i + 2]]))

    def dedupe(L):
        out, seen = [], set()
        for t in L:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    return dedupe(horiz), dedupe(vert)


def filter_crosses_motor_centered(
    crosses: List[Tuple[int, int, int, int, int]],
    ch_names: List[str],
    motor_centers: Tuple[str, ...]
) -> List[Tuple[int, int, int, int, int]]:
    motor_set = {c.upper() for c in motor_centers}
    out = []
    for c, u, d, l, r in crosses:
        if ch_names[c].upper() in motor_set:
            out.append((c, u, d, l, r))
    return out


# -------------------------------
# PE feature computation (by scope)
# -------------------------------
def tau_samples_from_cfg(cfg_obj: Config, sfreq: float) -> int:
    if cfg_obj.TEMPORAL_TAU_MS and cfg_obj.TEMPORAL_TAU_MS > 0:
        return max(1, int(round((cfg_obj.TEMPORAL_TAU_MS / 1000.0) * sfreq)))
    return max(1, int(cfg_obj.TEMPORAL_TAU_SAMPLES))


def dt_samples_from_cfg(cfg_obj: Config, sfreq: float) -> int:
    if cfg_obj.SPATIOTEMPORAL_DT_MS and cfg_obj.SPATIOTEMPORAL_DT_MS > 0:
        return max(1, int(round((cfg_obj.SPATIOTEMPORAL_DT_MS / 1000.0) * sfreq)))
    return max(1, int(cfg_obj.SPATIOTEMPORAL_DT_SAMPLES))


def temporal_pe_by_channel(cfg_obj: Config, epoch: np.ndarray, tau: int) -> np.ndarray:
    m = int(cfg_obj.TEMPORAL_M)
    n_ch, n_samp = epoch.shape
    n_vec = n_samp - (m - 1) * tau
    if n_vec <= 10:
        return np.full(n_ch, np.nan, dtype=float)

    pes = np.full(n_ch, np.nan, dtype=float)
    for ch in range(n_ch):
        x = epoch[ch]
        V = np.vstack([x[i * tau:i * tau + n_vec] for i in range(m)])
        ids = mapper_m4.ids_from_vectors(V)
        pes[ch] = perm_entropy_from_ids(ids, mapper_m4.n_patterns)
    return pes


def temporal_pe_features(cfg_obj: Config, epoch: np.ndarray, tau: int) -> Tuple[float, float]:
    pes = temporal_pe_by_channel(cfg_obj, epoch, tau=tau)
    return float(np.nanmean(pes)), float(np.nanstd(pes))


def spatial_pe_triplets(cfg_obj: Config, epoch: np.ndarray, triplets: List[Tuple[int, int, int]], ds: int) -> float:
    if not triplets:
        return np.nan
    step = max(1, int(ds))
    ep = epoch[:, ::step]
    ids_all = []
    for a, b, c in triplets:
        V = np.vstack([ep[a], ep[b], ep[c]])
        ids_all.append(mapper_m3.ids_from_vectors(V))
    ids = np.concatenate(ids_all, axis=0)
    return perm_entropy_from_ids(ids, mapper_m3.n_patterns)


def cross_like_spatial_pe(cfg_obj: Config, epoch: np.ndarray, crosses: List[Tuple[int, int, int, int, int]], ds: int) -> float:
    if not crosses:
        return np.nan
    step = max(1, int(ds))
    ep = epoch[:, ::step]
    ids_all = []
    for c, u, d, l, r in crosses:
        V = np.vstack([ep[c], ep[u], ep[d], ep[l], ep[r]])
        ids_all.append(mapper_m5.ids_from_vectors(V))
    ids = np.concatenate(ids_all, axis=0)
    return perm_entropy_from_ids(ids, mapper_m5.n_patterns)


def spatiotemporal_pe(cfg_obj: Config, epoch: np.ndarray, crosses: List[Tuple[int, int, int, int, int]], dt_samples: int, ds: int) -> float:
    if not crosses:
        return np.nan
    step = max(1, int(ds))
    ep = epoch[:, ::step]
    dt_eff = max(1, int(round(dt_samples / step)))
    T = ep.shape[1]
    if T <= dt_eff + 5:
        return np.nan

    ids_all = []
    for c, u, d, l, r in crosses:
        x0 = ep[c, :-dt_eff]
        x1 = ep[c, dt_eff:]
        up = ep[u, :-dt_eff]
        lf = ep[l, :-dt_eff]
        rt = ep[r, :-dt_eff]
        V = np.vstack([x0, x1, up, lf, rt])
        ids_all.append(mapper_m5.ids_from_vectors(V))
    ids = np.concatenate(ids_all, axis=0)
    return perm_entropy_from_ids(ids, mapper_m5.n_patterns)


# -------------------------------
# Decoding + stats utilities
# -------------------------------
def within_subject_decode_groupcv(cfg_obj: Config, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Optional[Dict[str, float]]:
    if len(np.unique(y)) < 2:
        return None
    if groups is None or len(groups) != len(y):
        return None

    groups0 = set(groups[y == 0])
    groups1 = set(groups[y == 1])
    min_groups = min(len(groups0), len(groups1))
    n_splits = min(int(cfg_obj.CV_FOLDS), int(min_groups))
    if n_splits < 2:
        return None

    if not HAS_SGK:
        from sklearn.model_selection import GroupKFold
        cv = GroupKFold(n_splits=n_splits)
    else:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg_obj.RNG_SEED)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))
    ])

    scoring = {"acc": "accuracy", "bal_acc": "balanced_accuracy", "f1": "f1", "auc": "roc_auc"}
    out = cross_validate(clf, X, y, cv=cv, groups=groups, scoring=scoring, error_score=np.nan)
    return {k: float(np.nanmean(out[f"test_{k}"])) for k in scoring.keys()}


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.array(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(q, 1.0)
    return out


def paired_effect_size_cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    sd = np.nanstd(d, ddof=1)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.nan
    return float(np.nanmean(d) / sd)


def compute_family_comparisons(dec_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, band, contrast), sub in dec_df.groupby(["scope", "band", "contrast"]):
        pivot = sub.pivot_table(index="subject", columns="family", values="bal_acc", aggfunc="mean")
        pairs = [
            ("cross_like", "temporal"),
            ("spatiotemporal", "temporal"),
            ("cross_like", "hv_spatial"),
            ("spatiotemporal", "hv_spatial"),
        ]

        pvals = []
        tmp = []
        for a, b in pairs:
            if a not in pivot.columns or b not in pivot.columns:
                continue
            xa = pivot[a].values
            xb = pivot[b].values
            ok = np.isfinite(xa) & np.isfinite(xb)
            if np.sum(ok) < 5:
                continue
            t, p = stats.ttest_rel(xa[ok], xb[ok], nan_policy="omit")
            d = paired_effect_size_cohen_d(xa[ok], xb[ok])
            tmp.append((a, b, int(np.sum(ok)), float(t), float(p), d))
            pvals.append(float(p))

        if len(tmp) == 0:
            continue

        qvals = bh_fdr(np.array(pvals, dtype=float))
        for (a, b, n, t, p, d), q in zip(tmp, qvals):
            rows.append({
                "scope": scope,
                "band": band,
                "contrast": contrast,
                "comparison": f"{a}_minus_{b}",
                "n_subjects": n,
                "t_stat": t,
                "p_value": p,
                "p_fdr_bh": float(q),
                "cohen_d_paired": d
            })

    return pd.DataFrame(rows)


# -------------------------------
# Permutation test (within trial pairs)
# -------------------------------
def permute_labels_within_pairs(rng: np.random.Generator, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    y_perm = y.copy()
    u = np.unique(groups)
    for g in u:
        idx = np.where(groups == g)[0]
        if idx.size != 2:
            continue
        if rng.random() < 0.5:
            y_perm[idx] = y_perm[idx][::-1]
    return y_perm


def run_permutation_test(
    cfg_obj: Config,
    rng: np.random.Generator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    observed_bal_acc: float,
    n_perm: int
) -> Dict[str, float]:
    perm_scores = []
    for _ in range(int(n_perm)):
        y_perm = permute_labels_within_pairs(rng, y, groups)
        m = within_subject_decode_groupcv(cfg_obj, X, y_perm, groups)
        perm_scores.append(np.nan if m is None else float(m["bal_acc"]))
    perm_scores = np.array(perm_scores, dtype=float)
    denom = float(np.sum(np.isfinite(perm_scores)))
    p = float((1.0 + np.sum(perm_scores >= observed_bal_acc)) / (1.0 + denom)) if denom > 0 else np.nan
    return {
        "perm_mean_bal_acc": float(np.nanmean(perm_scores)),
        "perm_std_bal_acc": float(np.nanstd(perm_scores)),
        "p_value_perm_ge_observed": p
    }


# -------------------------------
# Capacity-matched controls (noise + shuffled-novel)
# -------------------------------
def shuffle_novel_within_trial_pairs(
    rng: np.random.Generator,
    X_novel: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray
) -> np.ndarray:
    Xn = X_novel.copy()
    u = np.unique(groups)

    valid_trials = []
    base_rows = []
    task_rows = []
    for g in u:
        idx = np.where(groups == g)[0]
        if idx.size != 2:
            continue
        y0 = y[idx]
        if not (np.sum(y0 == 0) == 1 and np.sum(y0 == 1) == 1):
            continue
        b = idx[np.where(y0 == 0)[0][0]]
        t = idx[np.where(y0 == 1)[0][0]]
        valid_trials.append(g)
        base_rows.append(b)
        task_rows.append(t)

    if len(valid_trials) < 2:
        return Xn

    base_rows = np.array(base_rows, dtype=int)
    task_rows = np.array(task_rows, dtype=int)

    novel_base = Xn[base_rows].copy()
    novel_task = Xn[task_rows].copy()

    perm = rng.permutation(len(valid_trials))
    novel_base_shuf = novel_base[perm]
    novel_task_shuf = novel_task[perm]

    Xn[base_rows] = novel_base_shuf
    Xn[task_rows] = novel_task_shuf
    return Xn


# -------------------------------
# Cross-count distribution summary
# -------------------------------
def summarize_cross_counts(subject_df: pd.DataFrame, cfg_obj: Config) -> pd.DataFrame:
    def _summ(scope: str, col: str, min_req: int) -> Dict[str, Any]:
        if col not in subject_df.columns:
            return {
                "scope": scope,
                "metric": col,
                "n_subjects_total": int(subject_df.shape[0]),
                "n_subjects_with_value": 0,
                "mean": np.nan,
                "sd": np.nan,
                "min": np.nan,
                "p05": np.nan,
                "p25": np.nan,
                "median": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "max": np.nan,
                "min_required": int(min_req),
                "frac_below_min_required": np.nan,
            }

        x = pd.to_numeric(subject_df[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x)
        xs = x[ok]
        if xs.size == 0:
            return {
                "scope": scope,
                "metric": col,
                "n_subjects_total": int(subject_df.shape[0]),
                "n_subjects_with_value": 0,
                "mean": np.nan,
                "sd": np.nan,
                "min": np.nan,
                "p05": np.nan,
                "p25": np.nan,
                "median": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "max": np.nan,
                "min_required": int(min_req),
                "frac_below_min_required": np.nan,
            }

        q05, q25, q50, q75, q95 = np.percentile(xs, [5, 25, 50, 75, 95])
        frac_below = float(np.mean(xs < float(min_req))) if min_req is not None else np.nan

        return {
            "scope": scope,
            "metric": col,
            "n_subjects_total": int(subject_df.shape[0]),
            "n_subjects_with_value": int(xs.size),
            "mean": float(np.mean(xs)),
            "sd": float(np.std(xs, ddof=1)) if xs.size >= 2 else np.nan,
            "min": float(np.min(xs)),
            "p05": float(q05),
            "p25": float(q25),
            "median": float(q50),
            "p75": float(q75),
            "p95": float(q95),
            "max": float(np.max(xs)),
            "min_required": int(min_req),
            "frac_below_min_required": float(frac_below),
        }

    rows = []
    rows.append(_summ("whole", "whole_n_crosses", int(cfg_obj.MIN_CROSSES_REQUIRED_WHOLE)))
    rows.append(_summ("sensorimotor", "sensorimotor_n_crosses", int(cfg_obj.MIN_CROSSES_REQUIRED_SENSOR)))
    if "sensorimotor_n_crosses_motor_centered" in subject_df.columns:
        rows.append(_summ("sensorimotor", "sensorimotor_n_crosses_motor_centered", int(cfg_obj.MIN_CROSSES_REQUIRED_SENSOR)))
    return pd.DataFrame(rows)


# -------------------------------
# Plotting
# -------------------------------
def plot_decoding_figure(dec_df: pd.DataFrame, out_png: str, band: str, scope: str):
    fam_order = ["temporal", "hv_spatial", "cross_like", "spatiotemporal", "all_features"]
    contrasts = ["ME_vs_baseline", "MI_vs_baseline"]

    plt.figure(figsize=(11, 4.6))
    rng = np.random.default_rng(cfg.RNG_SEED)

    for i, contrast in enumerate(contrasts):
        ax = plt.subplot(1, 2, i + 1)
        sub = dec_df[(dec_df["contrast"] == contrast) & (dec_df["band"] == band) & (dec_df["scope"] == scope)].copy()
        sub["family"] = pd.Categorical(sub["family"], categories=fam_order, ordered=True)
        sub = sub.sort_values("family")

        means = sub.groupby("family", observed=False)["bal_acc"].mean().reindex(fam_order)
        sems = sub.groupby("family", observed=False)["bal_acc"].apply(
            lambda x: np.nanstd(x, ddof=1) / np.sqrt(max(int(np.sum(np.isfinite(x))), 1))
        ).reindex(fam_order)

        x = np.arange(len(fam_order))
        ax.bar(x, means.values, yerr=sems.values, capsize=4)

        for j, fam in enumerate(fam_order):
            vals = sub[sub["family"] == fam]["bal_acc"].values
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=vals.size)
            ax.scatter(np.full(vals.size, j) + jitter, vals, s=18, alpha=0.55)

        ax.set_xticks(x)
        ax.set_xticklabels(fam_order, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Balanced Accuracy (Group CV)")
        ax.set_title(contrast.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)

    plt.suptitle(f"Within-subject decoding using PE feature families (Group CV)\nBand: {band} | Scope: {scope}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------------------
# Core pipeline (single run)
# -------------------------------
def run_single(cfg_obj: Config, out_dir: str):
    if not os.path.isdir(cfg_obj.DATA_ROOT):
        raise SystemExit(f"DATA_ROOT does not exist: {cfg_obj.DATA_ROOT}")

    os.makedirs(out_dir, exist_ok=True)
    write_run_config(out_dir, cfg_obj)

    rng = np.random.default_rng(cfg_obj.RNG_SEED)
    subjects = list_subjects(cfg_obj.DATA_ROOT, cfg_obj)

    families = {
        "temporal": ["temporal_pe_mean", "temporal_pe_std"],
        "hv_spatial": ["spatial_pe_horizontal", "spatial_pe_vertical"],
        "cross_like": ["cross_like_spatial_pe"],
        "spatiotemporal": ["spatiotemporal_pe"],
        "all_features": [
            "temporal_pe_mean", "temporal_pe_std",
            "spatial_pe_horizontal", "spatial_pe_vertical",
            "cross_like_spatial_pe", "spatiotemporal_pe"
        ]
    }
    all_cols = families["all_features"]
    base_cols = ["temporal_pe_mean", "temporal_pe_std", "spatial_pe_horizontal", "spatial_pe_vertical"]
    novel_cols = ["cross_like_spatial_pe", "spatiotemporal_pe"]

    ablations = {
        "all_minus_temporal": ["spatial_pe_horizontal", "spatial_pe_vertical", "cross_like_spatial_pe", "spatiotemporal_pe"],
        "all_minus_hv": ["temporal_pe_mean", "temporal_pe_std", "cross_like_spatial_pe", "spatiotemporal_pe"],
        "all_minus_cross": ["temporal_pe_mean", "temporal_pe_std", "spatial_pe_horizontal", "spatial_pe_vertical", "spatiotemporal_pe"],
        "all_minus_st": ["temporal_pe_mean", "temporal_pe_std", "spatial_pe_horizontal", "spatial_pe_vertical", "cross_like_spatial_pe"],
        "all_minus_novel": ["temporal_pe_mean", "temporal_pe_std", "spatial_pe_horizontal", "spatial_pe_vertical"],
    }

    subject_rows = []
    decode_rows = []
    feat_condition_rows = []
    ablation_rows = []
    perm_rows = []
    capacity_rows = []

    for subj in tqdm(subjects, desc="Subjects"):
        subj_rec: Dict[str, Any] = {"subject": subj, "bad_runs": 0}
        sfreq_original_any = None
        sfreq_used_any = None

        scope_to_idx: Dict[str, np.ndarray] = {}
        kept_trial_ids: Dict[str, Dict[int, np.ndarray]] = {"ME": {}, "MI": {}}
        layout_audit: Dict[str, Dict[str, int]] = {}

        def process_run(run_number: int, contrast_type: str):
            nonlocal sfreq_original_any, sfreq_used_any

            p = edf_path(cfg_obj, subj, run_number)
            if not os.path.exists(p):
                return

            try:
                raw0 = mne.io.read_raw_edf(p, preload=True, verbose=False)
                raw_prepared, sfreq_orig, sfreq_used = preprocess_raw(cfg_obj, raw0)
                sfreq_original_any = sfreq_orig if sfreq_original_any is None else sfreq_original_any
                sfreq_used_any = sfreq_used if sfreq_used_any is None else sfreq_used_any

                if not scope_to_idx:
                    ch_names = list(raw_prepared.ch_names)
                    for scope in cfg_obj.SCOPES:
                        if scope == "whole":
                            scope_to_idx[scope] = np.arange(len(ch_names), dtype=int)
                        else:
                            roi_set = {c.upper() for c in cfg_obj.SENSORIMOTOR_CHANNELS}
                            idx = [i for i, nm in enumerate(ch_names) if nm.upper() in roi_set]
                            scope_to_idx[scope] = np.array(idx, dtype=int)

                _, rl, rh = cfg_obj.REJECT_BAND
                raw_reject = bandpass_raw(cfg_obj, raw_prepared, rl, rh)

                Xb_r, Xt_r, trial_ids_r = extract_paired_epochs_from_cues(cfg_obj, raw_reject, run_number=run_number)
                Xb2, Xt2, ids2, logd = reject_bad_trials(
                    cfg_obj, Xb_r, Xt_r, trial_ids_r, contrast_type=contrast_type, ch_names=list(raw_reject.ch_names)
                )

                capped_ids = cap_trial_ids(rng, ids2, max_trials=cfg_obj.MAX_TRIALS_PER_RUN_PER_CONTRAST)
                kept_trial_ids[contrast_type][run_number] = np.array(sorted(capped_ids), dtype=object)

                subj_rec[f"{contrast_type}_run{run_number:02d}_rej_n_raw"] = int(logd["n_raw"])
                subj_rec[f"{contrast_type}_run{run_number:02d}_rej_n_after_ptp"] = int(logd["n_after_ptp"])
                subj_rec[f"{contrast_type}_run{run_number:02d}_rej_n_after_flat"] = int(logd["n_after_flat"])
                subj_rec[f"{contrast_type}_run{run_number:02d}_n_after_cap"] = int(len(capped_ids))

            except Exception as e:
                subj_rec["bad_runs"] += 1
                if subj_rec["bad_runs"] <= 3:
                    print(f"[{subj} run {run_number:02d} {contrast_type}] ERROR: {type(e).__name__}: {e}")

        for run in cfg_obj.ME_RUNS:
            process_run(run, "ME")
        for run in cfg_obj.MI_RUNS:
            process_run(run, "MI")

        subj_rec["sfreq_original_any"] = sfreq_original_any
        subj_rec["sfreq_used_any"] = sfreq_used_any

        for ct in ["ME", "MI"]:
            all_ids = []
            for _, ids in kept_trial_ids[ct].items():
                if ids is None:
                    continue
                all_ids.extend(list(ids))
            subj_rec[f"{ct}_n_trials_total_after_rejectcap"] = int(len(np.unique(np.array(all_ids, dtype=object)))) if all_ids else 0

        if sfreq_used_any is None:
            subject_rows.append(subj_rec)
            continue

        rep_run = None
        for run in list(cfg_obj.ME_RUNS) + list(cfg_obj.MI_RUNS):
            if os.path.exists(edf_path(cfg_obj, subj, run)):
                rep_run = run
                break

        raw_rep = None
        if rep_run is not None:
            try:
                raw0 = mne.io.read_raw_edf(edf_path(cfg_obj, subj, rep_run), preload=True, verbose=False)
                raw_rep, _, _ = preprocess_raw(cfg_obj, raw0)
            except Exception:
                raw_rep = None

        if raw_rep is not None:
            for scope in cfg_obj.SCOPES:
                idx = scope_to_idx.get(scope, np.array([], dtype=int))
                if idx.size < 3:
                    layout_audit[scope] = {"n_channels": int(idx.size), "n_crosses": 0, "n_crosses_motor_centered": 0,
                                           "n_horiz_triplets": 0, "n_vert_triplets": 0}
                    continue
                scope_ch = [raw_rep.ch_names[i] for i in idx]
                raw_scope = raw_rep.copy().pick_channels(scope_ch)
                crosses = compute_cross_neighbors(raw_scope)
                hv_h, hv_v = spatial_triplets_hv(raw_scope)
                crosses_motor = filter_crosses_motor_centered(crosses, raw_scope.ch_names, cfg_obj.MOTOR_CROSS_CENTERS) if scope == "sensorimotor" else crosses
                layout_audit[scope] = {
                    "n_channels": int(len(raw_scope.ch_names)),
                    "n_crosses": int(len(crosses)),
                    "n_crosses_motor_centered": int(len(crosses_motor)),
                    "n_horiz_triplets": int(len(hv_h)),
                    "n_vert_triplets": int(len(hv_v)),
                }

            for scope, d in layout_audit.items():
                for k, v in d.items():
                    subj_rec[f"{scope}_{k}"] = v

        for (band_name, low, high) in cfg_obj.BANDS:
            store = {scope: {"ME": {"base": [], "task": [], "trial_ids": []},
                             "MI": {"base": [], "task": [], "trial_ids": []}}
                     for scope in cfg_obj.SCOPES}

            for contrast_type, runs in [("ME", cfg_obj.ME_RUNS), ("MI", cfg_obj.MI_RUNS)]:
                for run_number in runs:
                    if run_number not in kept_trial_ids[contrast_type]:
                        continue
                    keep_ids_run = kept_trial_ids[contrast_type][run_number]
                    if keep_ids_run is None or len(keep_ids_run) == 0:
                        continue

                    p = edf_path(cfg_obj, subj, run_number)
                    if not os.path.exists(p):
                        continue

                    try:
                        raw0 = mne.io.read_raw_edf(p, preload=True, verbose=False)
                        raw_prepared, _, _ = preprocess_raw(cfg_obj, raw0)
                        rb = bandpass_raw(cfg_obj, raw_prepared, low, high)

                        Xb, Xt, trial_ids = extract_paired_epochs_from_cues(cfg_obj, rb, run_number=run_number)
                        if trial_ids.size == 0:
                            continue

                        idx_map = {tid: i for i, tid in enumerate(trial_ids)}
                        sel = [idx_map[tid] for tid in keep_ids_run if tid in idx_map]
                        if len(sel) == 0:
                            continue

                        Xb = Xb[sel]
                        Xt = Xt[sel]
                        ids = trial_ids[sel]

                        for scope in cfg_obj.SCOPES:
                            idx = scope_to_idx.get(scope, np.array([], dtype=int))
                            if idx.size < 3:
                                continue

                            scope_ch = [rb.ch_names[i] for i in idx]
                            rb_scope = rb.copy().pick_channels(scope_ch)

                            hv_h, hv_v = spatial_triplets_hv(rb_scope)
                            layouts = {
                                "crosses": compute_cross_neighbors(rb_scope),
                                "horiz": hv_h,
                                "vert": hv_v,
                            }

                            if scope == "sensorimotor":
                                layouts["crosses"] = filter_crosses_motor_centered(
                                    layouts["crosses"], rb_scope.ch_names, cfg_obj.MOTOR_CROSS_CENTERS
                                )

                            min_crosses = cfg_obj.MIN_CROSSES_REQUIRED_SENSOR if scope == "sensorimotor" else cfg_obj.MIN_CROSSES_REQUIRED_WHOLE
                            crosses_ok = int(len(layouts["crosses"])) >= int(min_crosses)

                            sfreq = float(rb_scope.info["sfreq"])
                            tau = tau_samples_from_cfg(cfg_obj, sfreq)
                            dt = dt_samples_from_cfg(cfg_obj, sfreq)

                            for i_trial in range(Xb.shape[0]):
                                eb = Xb[i_trial][idx, :]
                                et = Xt[i_trial][idx, :]

                                t_mean_b, t_std_b = temporal_pe_features(cfg_obj, eb, tau=tau)
                                t_mean_t, t_std_t = temporal_pe_features(cfg_obj, et, tau=tau)

                                hv_h_b = float(spatial_pe_triplets(cfg_obj, eb, layouts["horiz"], ds=cfg_obj.DOWNSAMPLE_HV))
                                hv_v_b = float(spatial_pe_triplets(cfg_obj, eb, layouts["vert"], ds=cfg_obj.DOWNSAMPLE_HV))
                                hv_h_t = float(spatial_pe_triplets(cfg_obj, et, layouts["horiz"], ds=cfg_obj.DOWNSAMPLE_HV))
                                hv_v_t = float(spatial_pe_triplets(cfg_obj, et, layouts["vert"], ds=cfg_obj.DOWNSAMPLE_HV))

                                cross_b = float(cross_like_spatial_pe(cfg_obj, eb, layouts["crosses"], ds=cfg_obj.DOWNSAMPLE_CROSS)) if crosses_ok else np.nan
                                cross_t = float(cross_like_spatial_pe(cfg_obj, et, layouts["crosses"], ds=cfg_obj.DOWNSAMPLE_CROSS)) if crosses_ok else np.nan

                                st_b = float(spatiotemporal_pe(cfg_obj, eb, layouts["crosses"], dt_samples=dt, ds=cfg_obj.DOWNSAMPLE_ST)) if crosses_ok else np.nan
                                st_t = float(spatiotemporal_pe(cfg_obj, et, layouts["crosses"], dt_samples=dt, ds=cfg_obj.DOWNSAMPLE_ST)) if crosses_ok else np.nan

                                base_feat = {
                                    "temporal_pe_mean": t_mean_b,
                                    "temporal_pe_std": t_std_b,
                                    "spatial_pe_horizontal": hv_h_b,
                                    "spatial_pe_vertical": hv_v_b,
                                    "cross_like_spatial_pe": cross_b,
                                    "spatiotemporal_pe": st_b,
                                }
                                task_feat = {
                                    "temporal_pe_mean": t_mean_t,
                                    "temporal_pe_std": t_std_t,
                                    "spatial_pe_horizontal": hv_h_t,
                                    "spatial_pe_vertical": hv_v_t,
                                    "cross_like_spatial_pe": cross_t,
                                    "spatiotemporal_pe": st_t,
                                }

                                store[scope][contrast_type]["base"].append(base_feat)
                                store[scope][contrast_type]["task"].append(task_feat)
                                store[scope][contrast_type]["trial_ids"].append(ids[i_trial])

                    except Exception:
                        continue

            for scope in cfg_obj.SCOPES:
                for contrast_type, contrast_name in [("ME", "ME_vs_baseline"), ("MI", "MI_vs_baseline")]:
                    base_list = store[scope][contrast_type]["base"]
                    task_list = store[scope][contrast_type]["task"]
                    trial_ids = np.array(store[scope][contrast_type]["trial_ids"], dtype=object)

                    n_trials = int(len(np.unique(trial_ids))) if trial_ids.size > 0 else 0
                    subj_rec[f"{band_name}_{scope}_{contrast_type}_n_trials"] = n_trials

                    if n_trials < int(cfg_obj.MIN_TRIALS_PER_CONTRAST):
                        continue

                    X_base = np.array([[d.get(c, np.nan) for c in all_cols] for d in base_list], dtype=float)
                    X_task = np.array([[d.get(c, np.nan) for c in all_cols] for d in task_list], dtype=float)
                    X = np.vstack([X_base, X_task])
                    y = np.hstack([np.zeros(X_base.shape[0], dtype=int), np.ones(X_task.shape[0], dtype=int)])
                    groups = np.hstack([trial_ids, trial_ids])

                    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
                    X, y, groups = X[ok], y[ok], groups[ok]
                    if len(np.unique(y)) < 2:
                        continue

                    base_means = np.nanmean(X_base, axis=0) if X_base.size else np.full(len(all_cols), np.nan)
                    task_means = np.nanmean(X_task, axis=0) if X_task.size else np.full(len(all_cols), np.nan)
                    col_idx = {c: i for i, c in enumerate(all_cols)}
                    for fam, fam_cols in families.items():
                        b = float(np.nanmean([base_means[col_idx[c]] for c in fam_cols]))
                        t = float(np.nanmean([task_means[col_idx[c]] for c in fam_cols]))
                        denom = b if (np.isfinite(b) and abs(b) > 1e-12) else np.nan
                        dpe = float((t - b) / denom) if np.isfinite(denom) else np.nan
                        feat_condition_rows.append({
                            "subject": subj,
                            "scope": scope,
                            "band": band_name,
                            "contrast": contrast_name,
                            "family": fam,
                            "pe_base_mean": b,
                            "pe_task_mean": t,
                            "delta_pe_ratio": dpe
                        })

                    for fam, fam_cols in families.items():
                        idx = [col_idx[c] for c in fam_cols]
                        Xfam = X[:, idx]
                        metrics = within_subject_decode_groupcv(cfg_obj, Xfam, y, groups)
                        if metrics is None:
                            continue

                        row = {
                            "subject": subj,
                            "scope": scope,
                            "band": band_name,
                            "contrast": contrast_name,
                            "family": fam,
                            "n_samples": int(Xfam.shape[0]),
                            "n_trials": int(len(np.unique(groups))),
                            "n_folds_used": int(min(cfg_obj.CV_FOLDS, len(np.unique(groups)))),
                            **metrics
                        }
                        decode_rows.append(row)

                        if cfg_obj.RUN_PERMUTATION_TEST and (fam in cfg_obj.PERMUTE_ONLY_FAMILIES):
                            perm = run_permutation_test(
                                cfg_obj, rng, Xfam, y, groups, observed_bal_acc=float(metrics["bal_acc"]),
                                n_perm=int(cfg_obj.N_PERMUTATIONS)
                            )
                            perm_rows.append({
                                "subject": subj,
                                "scope": scope,
                                "band": band_name,
                                "contrast": contrast_name,
                                "family": fam,
                                "observed_bal_acc": float(metrics["bal_acc"]),
                                **perm
                            })

                    def decode_with_cols(cols: List[str]) -> Optional[float]:
                        idx2 = [col_idx[c] for c in cols]
                        X2 = X[:, idx2]
                        m = within_subject_decode_groupcv(cfg_obj, X2, y, groups)
                        return None if m is None else float(m["bal_acc"])

                    if cfg_obj.RUN_ABLATION:
                        base_all = decode_with_cols(families["all_features"])
                        if base_all is not None:
                            for abl_name, cols in ablations.items():
                                abl = decode_with_cols(cols)
                                ablation_rows.append({
                                    "subject": subj,
                                    "scope": scope,
                                    "band": band_name,
                                    "contrast": contrast_name,
                                    "ablation": abl_name,
                                    "bal_acc_all_features": float(base_all),
                                    "bal_acc_ablated": (np.nan if abl is None else float(abl)),
                                    "delta_all_minus_ablated": (np.nan if abl is None else float(base_all - abl))
                                })

                    if cfg_obj.RUN_CAPACITY_CONTROLS:
                        bal_all = decode_with_cols(all_cols)
                        bal_base = decode_with_cols(base_cols)
                        if (bal_all is not None) and (bal_base is not None):
                            k_extra = int(len(all_cols) - len(base_cols))
                            Xb = X[:, [col_idx[c] for c in base_cols]]
                            Xn = X[:, [col_idx[c] for c in novel_cols]]

                            noise = rng.standard_normal((Xb.shape[0], k_extra))
                            X_noise = np.hstack([Xb, noise])
                            m_noise = within_subject_decode_groupcv(cfg_obj, X_noise, y, groups)
                            bal_noise = np.nan if m_noise is None else float(m_noise["bal_acc"])

                            Xn_shuf = shuffle_novel_within_trial_pairs(rng, Xn, y, groups)
                            X_shuf = np.hstack([Xb, Xn_shuf])
                            m_shuf = within_subject_decode_groupcv(cfg_obj, X_shuf, y, groups)
                            bal_shuf = np.nan if m_shuf is None else float(m_shuf["bal_acc"])

                            capacity_rows.append({
                                "subject": subj,
                                "scope": scope,
                                "band": band_name,
                                "contrast": contrast_name,
                                "n_samples": int(X.shape[0]),
                                "n_trials": int(len(np.unique(groups))),
                                "bal_acc_base_temporal_hv": float(bal_base),
                                "bal_acc_all_features": float(bal_all),
                                "bal_acc_base_plus_noise": float(bal_noise),
                                "bal_acc_base_plus_shuffled_novel": float(bal_shuf),
                                "delta_all_minus_base": float(bal_all - bal_base),
                                "delta_all_minus_noise": float(bal_all - bal_noise) if np.isfinite(bal_noise) else np.nan,
                                "delta_all_minus_shufnovel": float(bal_all - bal_shuf) if np.isfinite(bal_shuf) else np.nan,
                            })

        subject_rows.append(subj_rec)

    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(os.path.join(out_dir, "subject_summary.csv"), index=False)

    cross_df = summarize_cross_counts(subject_df, cfg_obj)
    cross_df.to_csv(os.path.join(out_dir, "cross_count_distribution_summary.csv"), index=False)

    dec_df = pd.DataFrame(decode_rows)
    dec_df.to_csv(os.path.join(out_dir, "decoding_by_subject.csv"), index=False)

    feat_df = pd.DataFrame(feat_condition_rows)
    feat_df.to_csv(os.path.join(out_dir, "feature_condition_summary.csv"), index=False)

    if len(ablation_rows) > 0:
        abl_df = pd.DataFrame(ablation_rows)
        abl_df.to_csv(os.path.join(out_dir, "ablation_by_subject.csv"), index=False)
    else:
        abl_df = pd.DataFrame([])

    if len(perm_rows) > 0:
        perm_df = pd.DataFrame(perm_rows)
        perm_df.to_csv(os.path.join(out_dir, "permutation_stats.csv"), index=False)
    else:
        perm_df = pd.DataFrame([])

    if len(capacity_rows) > 0:
        cap_df = pd.DataFrame(capacity_rows)
        cap_df.to_csv(os.path.join(out_dir, "capacity_matched_controls.csv"), index=False)
    else:
        pd.DataFrame([]).to_csv(os.path.join(out_dir, "capacity_matched_controls.csv"), index=False)

    matched_rows = []
    if len(dec_df) > 0:
        for (scope, band), sub in dec_df[dec_df["family"] == "all_features"].groupby(["scope", "band"]):
            s_me = set(sub[sub["contrast"] == "ME_vs_baseline"]["subject"].unique())
            s_mi = set(sub[sub["contrast"] == "MI_vs_baseline"]["subject"].unique())
            inter = sorted(list(s_me & s_mi))
            matched_rows.append({
                "scope": scope,
                "band": band,
                "n_subjects_ME": int(len(s_me)),
                "n_subjects_MI": int(len(s_mi)),
                "n_subjects_matched_MI_ME": int(len(inter)),
            })
    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(os.path.join(out_dir, "matched_subjects_MI_ME_summary.csv"), index=False)

    if len(dec_df) > 0:
        group = (
            dec_df.groupby(["scope", "band", "contrast", "family"])
            .agg(
                n_subjects=("subject", "nunique"),
                acc_mean=("acc", "mean"),
                bal_acc_mean=("bal_acc", "mean"),
                f1_mean=("f1", "mean"),
                auc_mean=("auc", "mean"),
                acc_sem=("acc", lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(max(int(np.sum(np.isfinite(x))), 1)))),
                bal_acc_sem=("bal_acc", lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(max(int(np.sum(np.isfinite(x))), 1)))),
                f1_sem=("f1", lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(max(int(np.sum(np.isfinite(x))), 1)))),
                auc_sem=("auc", lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(max(int(np.sum(np.isfinite(x))), 1))))
            )
            .reset_index()
        )
        group.to_csv(os.path.join(out_dir, "decoding_group_summary.csv"), index=False)

        stats_df = compute_family_comparisons(dec_df)
        stats_df.to_csv(os.path.join(out_dir, "family_comparisons_stats.csv"), index=False)

        if len(abl_df) > 0:
            abl_group = (
                abl_df.groupby(["scope", "band", "contrast", "ablation"])
                .agg(
                    n_subjects=("subject", "nunique"),
                    delta_mean=("delta_all_minus_ablated", "mean"),
                    delta_sem=("delta_all_minus_ablated", lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(max(int(np.sum(np.isfinite(x))), 1)))),
                )
                .reset_index()
            )
            abl_group.to_csv(os.path.join(out_dir, "ablation_group_summary.csv"), index=False)
        else:
            pd.DataFrame([]).to_csv(os.path.join(out_dir, "ablation_group_summary.csv"), index=False)

        for (band_name, _, _) in cfg_obj.BANDS:
            for scope in cfg_obj.SCOPES:
                out_png = os.path.join(out_dir, f"Fig_decoding_bar_with_dots_{band_name}_{scope}.png")
                plot_decoding_figure(dec_df, out_png=out_png, band=band_name, scope=scope)
    else:
        pd.DataFrame([]).to_csv(os.path.join(out_dir, "decoding_group_summary.csv"), index=False)
        pd.DataFrame([]).to_csv(os.path.join(out_dir, "family_comparisons_stats.csv"), index=False)
        pd.DataFrame([]).to_csv(os.path.join(out_dir, "ablation_group_summary.csv"), index=False)
        print("[WARN] No decoding results produced (likely too few valid trials per subject).")

    print("\n✅ DONE")
    print("Results folder:", out_dir)
    print("Group CV:", "StratifiedGroupKFold" if HAS_SGK else "GroupKFold (fallback)")


# -------------------------------
# Robustness grid runner
# -------------------------------
def run_pipeline(cfg_obj: Config):
    base_out = cfg_obj.OUT_FOLDER
    os.makedirs(base_out, exist_ok=True)

    if cfg_obj.ROBUSTNESS_GRID and len(cfg_obj.ROBUSTNESS_GRID) > 0:
        for i, overrides in enumerate(cfg_obj.ROBUSTNESS_GRID):
            cfg2 = Config(**{**asdict(cfg_obj), **overrides})
            tag = overrides.get("TAG", f"grid{i:02d}")
            out_dir = os.path.join(base_out, str(tag))
            run_single(cfg2, out_dir)
    else:
        run_single(cfg_obj, base_out)


if __name__ == "__main__":
    run_pipeline(cfg)
