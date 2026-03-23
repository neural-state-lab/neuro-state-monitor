"""PTSD/trauma data loader for fMRI and EEG data.

Loads fMRI NIfTI data via Nilearn for resting-state connectivity analysis.
Also supports EEG-based reconsolidation window detection via shared pipeline.

fMRI data sources:
- ENIGMA-PGC PTSD consortium (requires data access agreement)
- OpenNeuro fear conditioning datasets (publicly available)

Assumes fMRIPrep preprocessing was run on raw fMRI data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TraumaConfig:
    """Configuration for PTSD data loading."""

    fmri_dir: Optional[Path] = None
    eeg_dir: Optional[Path] = None
    atlas: str = "schaefer_2018"  # brain parcellation atlas
    n_rois: int = 100  # number of brain regions
    confound_strategy: str = "simple"  # fMRIPrep confound strategy
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


@dataclass
class TraumaDataset:
    """Container for PTSD classifier data."""

    connectivity_matrices: np.ndarray  # (n_subjects, n_rois, n_rois)
    labels: np.ndarray  # 1 = PTSD, 0 = control
    subjects: np.ndarray
    metadata: pd.DataFrame


def load_fmri_connectivity(
    fmri_dir: Path,
    config: Optional[TraumaConfig] = None,
) -> TraumaDataset:
    """Load fMRI data and compute resting-state connectivity matrices.

    Uses Nilearn for atlas parcellation and connectivity estimation.
    Assumes fMRIPrep-preprocessed NIfTI files in BIDS derivatives format.
    """
    if config is None:
        config = TraumaConfig()

    try:
        from nilearn import datasets, maskers
    except ImportError:
        raise RuntimeError("Nilearn not installed. Install with: pip install nilearn")

    # Load atlas for parcellation
    if config.atlas == "schaefer_2018":
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=config.n_rois)
        atlas_maps = atlas.maps
    else:
        raise ValueError(f"Unsupported atlas: {config.atlas}")

    # Find fMRIPrep output files
    fmri_dir = Path(fmri_dir)
    func_files = sorted(
        fmri_dir.rglob("*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    )

    if not func_files:
        raise FileNotFoundError(
            f"No fMRIPrep preprocessed files found in {fmri_dir}. "
            "Run fMRIPrep first or check path."
        )

    # Extract time series and compute connectivity per subject
    masker = maskers.NiftiLabelsMasker(
        labels_img=atlas_maps,
        standardize="zscore_sample",
        verbose=0,
    )

    matrices = []
    labels = []
    subjects = []
    metadata_rows = []

    for func_file in func_files:
        subject_id = _extract_subject_id(func_file)
        label = _get_ptsd_label(func_file, fmri_dir)

        try:
            # Extract ROI time series
            confounds = _load_confounds(func_file, config.confound_strategy)
            time_series = masker.fit_transform(str(func_file), confounds=confounds)

            # Compute connectivity matrix
            conn_matrix = np.corrcoef(time_series.T)

            matrices.append(conn_matrix)
            labels.append(label)
            subjects.append(subject_id)
            metadata_rows.append(
                {
                    "subject_id": subject_id,
                    "label": label,
                    "group": "ptsd" if label == 1 else "control",
                    "n_timepoints": time_series.shape[0],
                    "n_rois": time_series.shape[1],
                }
            )

            logger.info(
                "loaded_fmri_subject",
                subject=subject_id,
                group="ptsd" if label == 1 else "control",
                n_timepoints=time_series.shape[0],
            )
        except Exception as exc:
            logger.warning("fmri_load_failed", subject=subject_id, error=str(exc))

    if not matrices:
        raise ValueError("No valid fMRI data could be loaded")

    dataset = TraumaDataset(
        connectivity_matrices=np.array(matrices),
        labels=np.array(labels),
        subjects=np.array(subjects),
        metadata=pd.DataFrame(metadata_rows),
    )

    logger.info(
        "loaded_fmri_dataset",
        n_subjects=len(subjects),
        n_ptsd=int(sum(labels)),
        n_control=int(len(labels) - sum(labels)),
    )
    return dataset


def _extract_subject_id(filepath: Path) -> str:
    """Extract subject ID from BIDS filename."""
    name = filepath.name
    for part in name.split("_"):
        if part.startswith("sub-"):
            return part.replace("sub-", "")
    return filepath.parent.parent.name.replace("sub-", "")


def _get_ptsd_label(func_file: Path, bids_root: Path) -> int:
    """Get PTSD/control label from participants.tsv or directory structure."""
    participants_file = bids_root / "participants.tsv"

    if participants_file.exists():
        df = pd.read_csv(participants_file, sep="\t")
        subject_id = _extract_subject_id(func_file)
        sub_key = f"sub-{subject_id}"

        if "participant_id" in df.columns and "group" in df.columns:
            match = df[df["participant_id"] == sub_key]
            if len(match) > 0:
                group = match.iloc[0]["group"].lower()
                return 1 if "ptsd" in group or "patient" in group else 0

    # Default: try to infer from directory structure
    path_str = str(func_file).lower()
    if "ptsd" in path_str or "patient" in path_str:
        return 1
    return 0


def _load_confounds(func_file: Path, strategy: str) -> Optional[np.ndarray]:
    """Load fMRIPrep confound regressors."""
    confounds_file = str(func_file).replace(
        "_bold.nii.gz", "_desc-confounds_timeseries.tsv"
    )

    if not Path(confounds_file).exists():
        return None

    try:
        from nilearn.interfaces.fmriprep import load_confounds

        confounds, _ = load_confounds(str(func_file), strategy=(strategy,))
        return confounds
    except Exception:
        return None
