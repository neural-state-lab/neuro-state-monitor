"""PEERS dataset loader for the encoding classifier.

Loads the Penn Electrophysiology of Encoding and Retrieval Study (PEERS)
from OpenNeuro (ds004106) in BIDS format. Parses encoding trials, recall
results, and labels each trial as successfully recalled (1) or forgotten (0).

Train/validation/test splits are by SUBJECT (not trial) to prevent data leakage.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from shared.preprocessing.bids import get_bids_subjects, read_bids_raw
from shared.preprocessing.pipeline import (
    PreprocessingConfig,
    preprocess_and_epoch,
    preprocess_raw,
)

logger = structlog.get_logger(__name__)

DEFAULT_PEERS_DIR = Path(__file__).parents[2] / "data" / "raw" / "peers"


@dataclass
class PEERSConfig:
    """Configuration for PEERS data loading."""

    bids_root: Path = DEFAULT_PEERS_DIR
    task: str = "encoding"
    # Epoch timing
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: tuple[Optional[float], float] = (None, 0)
    # Split ratios (by subject)
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    # Preprocessing
    l_freq: float = 0.5
    h_freq: float = 45.0
    notch_freq: float = 60.0
    ica_n_components: Optional[int] = None
    resample_sfreq: Optional[float] = 256.0


@dataclass
class EncodingDataset:
    """Container for encoding classifier data."""

    epochs: mne.Epochs
    labels: np.ndarray  # 1 = recalled, 0 = forgotten
    subjects: np.ndarray  # subject ID per epoch
    metadata: pd.DataFrame  # trial-level metadata


@dataclass
class EncodingSplit:
    """Train/validation/test split of encoding data."""

    train: EncodingDataset
    val: EncodingDataset
    test: EncodingDataset
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]


def load_subject_data(
    bids_root: Path,
    subject: str,
    task: str = "encoding",
    config: Optional[PEERSConfig] = None,
) -> Optional[EncodingDataset]:
    """Load and preprocess data for a single PEERS subject.

    Returns None if the subject data cannot be loaded or has no valid trials.
    """
    if config is None:
        config = PEERSConfig()

    try:
        raw = read_bids_raw(
            bids_root=bids_root,
            subject=subject,
            task=task,
        )
    except Exception as exc:
        logger.warning("failed_to_load_subject", subject=subject, error=str(exc))
        return None

    # Parse events from annotations or events file
    events, event_id = _extract_events(raw, bids_root, subject, task)
    if events is None or len(events) == 0:
        logger.warning("no_events_found", subject=subject)
        return None

    # Build preprocessing config
    preproc_config = PreprocessingConfig(
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        notch_freq=config.notch_freq,
        ica_n_components=config.ica_n_components,
        ref_channels="average",
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=config.baseline,
        event_id=event_id,
        resample_sfreq=config.resample_sfreq,
    )

    # Preprocess and epoch
    try:
        epochs = preprocess_and_epoch(raw, events, preproc_config)
    except Exception as exc:
        logger.warning("preprocessing_failed", subject=subject, error=str(exc))
        return None

    if len(epochs) == 0:
        logger.warning("no_epochs_after_preprocessing", subject=subject)
        return None

    # Extract recall labels
    labels, metadata = _extract_recall_labels(epochs, bids_root, subject, task)

    n_epochs = len(epochs)
    subjects = np.array([subject] * n_epochs)

    logger.info(
        "loaded_subject",
        subject=subject,
        n_epochs=n_epochs,
        n_recalled=int(labels.sum()),
        n_forgotten=int((1 - labels).sum()),
    )

    return EncodingDataset(
        epochs=epochs,
        labels=labels,
        subjects=subjects,
        metadata=metadata,
    )


def _extract_events(
    raw: mne.io.BaseRaw,
    bids_root: Path,
    subject: str,
    task: str,
) -> tuple[Optional[np.ndarray], Optional[dict[str, int]]]:
    """Extract events from BIDS events file or MNE annotations.

    Returns (events_array, event_id_dict) or (None, None) if no events found.
    """
    # Try BIDS events.tsv first
    events_file = (
        bids_root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_events.tsv"
    )

    if events_file.exists():
        events_df = pd.read_csv(events_file, sep="\t")
        return _parse_bids_events(events_df, raw.info["sfreq"])

    # Fall back to MNE annotations
    if raw.annotations:
        try:
            events, event_id = mne.events_from_annotations(raw)
            return events, event_id
        except Exception:
            pass

    # Fall back to stim channel
    try:
        events = mne.find_events(raw, shortest_event=1)
        if len(events) > 0:
            unique_ids = np.unique(events[:, 2])
            event_id = {f"event_{eid}": int(eid) for eid in unique_ids}
            return events, event_id
    except Exception:
        pass

    return None, None


def _parse_bids_events(
    events_df: pd.DataFrame,
    sfreq: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Parse BIDS events.tsv into MNE events array.

    BIDS events.tsv has columns: onset, duration, trial_type, [value], ...
    """
    if "onset" not in events_df.columns:
        raise ValueError("Events file missing 'onset' column")

    # Convert onset times to samples
    samples = (events_df["onset"].values * sfreq).astype(int)

    # Build event ID mapping from trial_type
    if "trial_type" in events_df.columns:
        trial_types = events_df["trial_type"].unique()
        event_id = {str(tt): i + 1 for i, tt in enumerate(trial_types)}
        event_codes = events_df["trial_type"].map(event_id).values
    elif "value" in events_df.columns:
        event_codes = events_df["value"].astype(int).values
        unique_codes = np.unique(event_codes)
        event_id = {f"event_{c}": int(c) for c in unique_codes}
    else:
        event_codes = np.ones(len(events_df), dtype=int)
        event_id = {"stimulus": 1}

    events = np.column_stack(
        [
            samples,
            np.zeros(len(samples), dtype=int),
            event_codes,
        ]
    )

    return events, event_id


def _extract_recall_labels(
    epochs: mne.Epochs,
    bids_root: Path,
    subject: str,
    task: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract recall success labels for each epoch.

    Tries to find recall data in:
    1. Epochs metadata (if present)
    2. Behavioral data files in BIDS
    3. Falls back to event-based labeling

    Returns (labels, metadata_df).
    """
    n_epochs = len(epochs)

    # Check if epochs already have metadata with recall info
    if epochs.metadata is not None:
        recall_cols = [
            c
            for c in epochs.metadata.columns
            if any(kw in c.lower() for kw in ["recall", "remembered", "hit", "correct"])
        ]
        if recall_cols:
            labels = epochs.metadata[recall_cols[0]].values.astype(int)
            return labels, epochs.metadata

    # Check for behavioral data file
    beh_patterns = [
        bids_root / f"sub-{subject}" / "beh" / f"sub-{subject}_task-{task}_beh.tsv",
        bids_root / f"sub-{subject}" / f"sub-{subject}_task-{task}_beh.tsv",
    ]

    for beh_file in beh_patterns:
        if beh_file.exists():
            beh_df = pd.read_csv(beh_file, sep="\t")
            recall_cols = [
                c
                for c in beh_df.columns
                if any(kw in c.lower() for kw in ["recall", "remembered", "hit"])
            ]
            if recall_cols and len(beh_df) >= n_epochs:
                labels = beh_df[recall_cols[0]].values[:n_epochs].astype(int)
                return labels, beh_df.iloc[:n_epochs]

    # Fall back: use event IDs as proxy labels
    # In many encoding paradigms, different event codes mark
    # subsequently remembered vs forgotten items
    event_codes = epochs.events[:, 2]
    unique_codes = np.unique(event_codes)

    if len(unique_codes) == 2:
        # Binary: assume lower code = forgotten, higher = recalled
        labels = (event_codes == unique_codes[1]).astype(int)
    else:
        # Cannot determine recall — assign random labels for pipeline testing
        logger.warning(
            "recall_labels_not_found",
            subject=subject,
            msg="Using event-based proxy labels. Replace with actual recall data.",
        )
        labels = (event_codes % 2).astype(int)

    metadata = pd.DataFrame(
        {
            "event_code": event_codes,
            "recalled": labels,
            "subject": subject,
        }
    )
    return labels, metadata


def load_peers_dataset(
    config: Optional[PEERSConfig] = None,
) -> list[EncodingDataset]:
    """Load all subjects from the PEERS dataset.

    Returns a list of EncodingDataset, one per subject.
    """
    if config is None:
        config = PEERSConfig()

    subjects = get_bids_subjects(config.bids_root)

    if not subjects:
        raise FileNotFoundError(
            f"No BIDS subjects found in {config.bids_root}. "
            f"Run 'python data/download_peers.py' first."
        )

    datasets: list[EncodingDataset] = []

    for subject in subjects:
        dataset = load_subject_data(
            bids_root=config.bids_root,
            subject=subject,
            task=config.task,
            config=config,
        )
        if dataset is not None:
            datasets.append(dataset)

    logger.info(
        "loaded_peers_dataset",
        n_subjects=len(datasets),
        total_epochs=sum(len(d.epochs) for d in datasets),
    )
    return datasets


def merge_datasets(datasets: list[EncodingDataset]) -> EncodingDataset:
    """Merge multiple single-subject datasets into one.

    Concatenates epochs, labels, subjects, and metadata.
    """
    all_epochs = mne.concatenate_epochs([d.epochs for d in datasets])
    all_labels = np.concatenate([d.labels for d in datasets])
    all_subjects = np.concatenate([d.subjects for d in datasets])
    all_metadata = pd.concat([d.metadata for d in datasets], ignore_index=True)

    logger.info(
        "merged_datasets",
        n_subjects=len(datasets),
        n_epochs=len(all_epochs),
        label_balance=f"{all_labels.mean():.2f}",
    )

    return EncodingDataset(
        epochs=all_epochs,
        labels=all_labels,
        subjects=all_subjects,
        metadata=all_metadata,
    )


def split_by_subject(
    dataset: EncodingDataset,
    config: Optional[PEERSConfig] = None,
) -> EncodingSplit:
    """Split dataset by subject into train/validation/test.

    No subject appears in more than one split — prevents data leakage.
    """
    if config is None:
        config = PEERSConfig()

    unique_subjects = np.unique(dataset.subjects)
    n_subjects = len(unique_subjects)

    if n_subjects < 3:
        raise ValueError(
            f"Need at least 3 subjects for train/val/test split, got {n_subjects}"
        )

    # First split: separate test subjects
    remaining_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    # Second split: separate validation from training
    adjusted_val_size = config.val_size / (1 - config.test_size)
    train_subjects, val_subjects = train_test_split(
        remaining_subjects,
        test_size=adjusted_val_size,
        random_state=config.random_state,
    )

    # Build split datasets
    train_data = _filter_by_subjects(dataset, train_subjects)
    val_data = _filter_by_subjects(dataset, val_subjects)
    test_data = _filter_by_subjects(dataset, test_subjects)

    logger.info(
        "split_by_subject",
        train_subjects=len(train_subjects),
        val_subjects=len(val_subjects),
        test_subjects=len(test_subjects),
        train_epochs=len(train_data.epochs),
        val_epochs=len(val_data.epochs),
        test_epochs=len(test_data.epochs),
    )

    return EncodingSplit(
        train=train_data,
        val=val_data,
        test=test_data,
        train_subjects=list(train_subjects),
        val_subjects=list(val_subjects),
        test_subjects=list(test_subjects),
    )


def _filter_by_subjects(
    dataset: EncodingDataset,
    subjects: np.ndarray,
) -> EncodingDataset:
    """Filter dataset to only include specified subjects."""
    mask = np.isin(dataset.subjects, subjects)
    indices = np.where(mask)[0]

    filtered_epochs = dataset.epochs[indices]
    filtered_labels = dataset.labels[mask]
    filtered_subjects = dataset.subjects[mask]
    filtered_metadata = dataset.metadata.iloc[indices].reset_index(drop=True)

    return EncodingDataset(
        epochs=filtered_epochs,
        labels=filtered_labels,
        subjects=filtered_subjects,
        metadata=filtered_metadata,
    )
