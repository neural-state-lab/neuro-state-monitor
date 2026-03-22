"""UCI EEG Alcohol dataset loader for the addiction classifier.

Loads the UCI EEG database (122 subjects: 77 alcoholic, 45 control).
64 channels at 256 Hz. Visual stimulus paradigm with single/paired images.

Dataset format: Each trial is a gzip-compressed file with 64 channels × 256
samples. File naming convention encodes subject group (co=control, al=alcoholic).

Source: https://archive.ics.uci.edu/dataset/121/eeg+database
"""

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split


logger = structlog.get_logger(__name__)

DEFAULT_UCI_DIR = Path(__file__).parents[2] / "data" / "raw" / "uci_eeg"

# Standard 10-20 channel names for the UCI dataset (64 channels)
UCI_CHANNEL_NAMES = [
    "FP1",
    "FP2",
    "F7",
    "F8",
    "AF1",
    "AF2",
    "FZ",
    "F4",
    "F3",
    "FC6",
    "FC5",
    "FC2",
    "FC1",
    "T8",
    "T7",
    "CZ",
    "C3",
    "C4",
    "CP5",
    "CP6",
    "CP1",
    "CP2",
    "P3",
    "P4",
    "PZ",
    "P8",
    "P7",
    "PO2",
    "PO1",
    "O2",
    "O1",
    "X",
    "AF7",
    "AF8",
    "F5",
    "F6",
    "FT7",
    "FT8",
    "FPZ",
    "FC4",
    "FC3",
    "C6",
    "C5",
    "F2",
    "F1",
    "TP8",
    "TP7",
    "AFZ",
    "CP3",
    "CP4",
    "P5",
    "P6",
    "C1",
    "C2",
    "PO7",
    "PO8",
    "FCZ",
    "POZ",
    "OZ",
    "P2",
    "P1",
    "CPZ",
    "nd",
    "Y",
]

SFREQ = 256.0
N_CHANNELS = 64
N_SAMPLES_PER_TRIAL = 256


@dataclass
class AddictionConfig:
    """Configuration for UCI EEG data loading."""

    data_dir: Path = DEFAULT_UCI_DIR
    # Split ratios (by subject)
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    # Preprocessing
    l_freq: float = 0.5
    h_freq: float = 45.0
    notch_freq: float = 60.0
    ica_n_components: Optional[int] = 20
    resample_sfreq: Optional[float] = None


@dataclass
class AddictionDataset:
    """Container for addiction classifier data."""

    data: np.ndarray  # (n_trials, n_channels, n_samples)
    labels: np.ndarray  # 1 = alcoholic, 0 = control
    subjects: np.ndarray  # subject ID per trial
    metadata: pd.DataFrame  # trial-level metadata


@dataclass
class AddictionSplit:
    """Train/validation/test split."""

    train: AddictionDataset
    val: AddictionDataset
    test: AddictionDataset
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]


def parse_trial_file(filepath: Path) -> Optional[dict]:
    """Parse a single UCI EEG trial file.

    Each file is gzip-compressed with the format:
    ```
    # <channel_name> chan <n> ...
    <trial> <channel> <sample> <value>
    ```

    Returns dict with channel data or None if parsing fails.
    """
    try:
        if filepath.suffix == ".gz":
            with gzip.open(filepath, "rt") as f:
                lines = f.readlines()
        else:
            with open(filepath) as f:
                lines = f.readlines()
    except Exception as exc:
        logger.warning("failed_to_read_trial", path=str(filepath), error=str(exc))
        return None

    data = np.zeros((N_CHANNELS, N_SAMPLES_PER_TRIAL))
    channel_idx = -1
    sample_idx = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            # Channel header: # <channel_name> chan <n> ...
            channel_idx += 1
            sample_idx = 0
            continue

        parts = line.split()
        if len(parts) >= 4:
            try:
                value = float(parts[3])
                if channel_idx < N_CHANNELS and sample_idx < N_SAMPLES_PER_TRIAL:
                    data[channel_idx, sample_idx] = value
                sample_idx += 1
            except (ValueError, IndexError):
                continue

    if channel_idx < 0:
        return None

    return {"data": data}


def get_subject_info(subject_dir: Path) -> dict:
    """Extract subject info from directory name.

    Naming convention:
    - "co2a0000364" → control subject
    - "al2a0000365" → alcoholic subject
    - First 2 chars: "co" = control, "al" = alcoholic
    """
    name = subject_dir.name
    group = "alcoholic" if name.startswith("a") else "control"
    label = 1 if group == "alcoholic" else 0

    return {
        "subject_id": name,
        "group": group,
        "label": label,
    }


def load_subject_trials(
    subject_dir: Path,
) -> Optional[tuple[np.ndarray, dict]]:
    """Load all trials for a single subject.

    Returns (data_array, subject_info) or None if loading fails.
    data_array shape: (n_trials, n_channels, n_samples)
    """
    info = get_subject_info(subject_dir)

    trial_files = sorted(subject_dir.glob("*.gz"))
    if not trial_files:
        trial_files = sorted(subject_dir.glob("*.rd.*"))

    if not trial_files:
        logger.warning("no_trials_found", subject=info["subject_id"])
        return None

    trials = []
    for trial_file in trial_files:
        result = parse_trial_file(trial_file)
        if result is not None:
            trials.append(result["data"])

    if not trials:
        logger.warning("no_valid_trials", subject=info["subject_id"])
        return None

    data = np.array(trials)

    logger.info(
        "loaded_subject",
        subject=info["subject_id"],
        group=info["group"],
        n_trials=len(trials),
    )
    return data, info


def load_uci_dataset(
    config: Optional[AddictionConfig] = None,
) -> AddictionDataset:
    """Load the full UCI EEG dataset.

    Scans SMNI_CMI_TRAIN and SMNI_CMI_TEST directories for subject folders.
    """
    if config is None:
        config = AddictionConfig()

    data_dir = config.data_dir

    # Find subject directories in train and test splits
    subject_dirs = []
    for split_dir in ["SMNI_CMI_TRAIN", "SMNI_CMI_TEST"]:
        split_path = data_dir / split_dir
        if split_path.exists():
            subject_dirs.extend([d for d in sorted(split_path.iterdir()) if d.is_dir()])

    # Also check root directory for subject folders
    if not subject_dirs:
        subject_dirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]

    if not subject_dirs:
        raise FileNotFoundError(
            f"No subject directories found in {data_dir}. "
            f"Run 'python data/download_uci.py' first."
        )

    all_data = []
    all_labels = []
    all_subjects = []
    all_metadata = []

    for subject_dir in subject_dirs:
        result = load_subject_trials(subject_dir)
        if result is None:
            continue

        data, info = result
        n_trials = data.shape[0]

        all_data.append(data)
        all_labels.extend([info["label"]] * n_trials)
        all_subjects.extend([info["subject_id"]] * n_trials)
        all_metadata.extend(
            [
                {
                    "subject_id": info["subject_id"],
                    "group": info["group"],
                    "label": info["label"],
                }
            ]
            * n_trials
        )

    if not all_data:
        raise ValueError("No valid subject data could be loaded")

    dataset = AddictionDataset(
        data=np.concatenate(all_data, axis=0),
        labels=np.array(all_labels),
        subjects=np.array(all_subjects),
        metadata=pd.DataFrame(all_metadata),
    )

    n_alcoholic = (dataset.labels == 1).sum()
    n_control = (dataset.labels == 0).sum()

    logger.info(
        "loaded_uci_dataset",
        n_subjects=len(set(all_subjects)),
        n_trials=len(dataset.labels),
        n_alcoholic=int(n_alcoholic),
        n_control=int(n_control),
    )
    return dataset


def dataset_to_epochs(
    dataset: AddictionDataset,
    preprocess: bool = True,
    config: Optional[AddictionConfig] = None,
) -> mne.Epochs:
    """Convert raw trial data to MNE Epochs.

    Creates an MNE EpochsArray from the numpy data, optionally
    applying preprocessing.
    """
    if config is None:
        config = AddictionConfig()

    # Use standard channel names (skip non-EEG: "X", "Y", "nd")
    ch_names = [ch for ch in UCI_CHANNEL_NAMES if ch not in ("X", "Y", "nd")]
    eeg_indices = [
        i for i, ch in enumerate(UCI_CHANNEL_NAMES) if ch not in ("X", "Y", "nd")
    ]

    # Filter to EEG channels only
    data = dataset.data[:, eeg_indices, :]

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=SFREQ,
        ch_types="eeg",
    )

    epochs = mne.EpochsArray(data, info, verbose=False)

    if preprocess:
        # Apply filtering to the epochs data
        epochs.filter(
            l_freq=config.l_freq,
            h_freq=config.h_freq,
            verbose=False,
        )

    logger.info(
        "created_epochs",
        n_epochs=len(epochs),
        n_channels=len(ch_names),
        sfreq=SFREQ,
    )
    return epochs


def split_by_subject(
    dataset: AddictionDataset,
    config: Optional[AddictionConfig] = None,
) -> AddictionSplit:
    """Split dataset by subject into train/validation/test.

    No subject appears in more than one split.
    """
    if config is None:
        config = AddictionConfig()

    unique_subjects = np.unique(dataset.subjects)

    # Get label per subject (for stratification)
    subject_labels = np.array(
        [dataset.labels[dataset.subjects == s][0] for s in unique_subjects]
    )

    # First split: separate test subjects (stratified by group)
    remaining_subj, test_subj = train_test_split(
        unique_subjects,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=subject_labels,
    )

    remaining_labels = np.array(
        [dataset.labels[dataset.subjects == s][0] for s in remaining_subj]
    )

    # Second split: val from remaining
    adjusted_val = config.val_size / (1 - config.test_size)
    train_subj, val_subj = train_test_split(
        remaining_subj,
        test_size=adjusted_val,
        random_state=config.random_state,
        stratify=remaining_labels,
    )

    train = _filter_by_subjects(dataset, train_subj)
    val = _filter_by_subjects(dataset, val_subj)
    test = _filter_by_subjects(dataset, test_subj)

    logger.info(
        "split_by_subject",
        train=f"{len(train_subj)} subj / {len(train.labels)} trials",
        val=f"{len(val_subj)} subj / {len(val.labels)} trials",
        test=f"{len(test_subj)} subj / {len(test.labels)} trials",
    )

    return AddictionSplit(
        train=train,
        val=val,
        test=test,
        train_subjects=list(train_subj),
        val_subjects=list(val_subj),
        test_subjects=list(test_subj),
    )


def _filter_by_subjects(
    dataset: AddictionDataset,
    subjects: np.ndarray,
) -> AddictionDataset:
    """Filter dataset to specified subjects."""
    mask = np.isin(dataset.subjects, subjects)

    return AddictionDataset(
        data=dataset.data[mask],
        labels=dataset.labels[mask],
        subjects=dataset.subjects[mask],
        metadata=dataset.metadata.iloc[np.where(mask)[0]].reset_index(drop=True),
    )
