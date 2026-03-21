"""BIDS format conversion and loading utilities.

Uses MNE-BIDS for reading/writing BIDS-compliant EEG datasets.
"""

from pathlib import Path
from typing import Optional

import mne
import structlog

logger = structlog.get_logger(__name__)


def read_bids_raw(
    bids_root: Path,
    subject: str,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    datatype: str = "eeg",
) -> mne.io.BaseRaw:
    """Read raw EEG data from a BIDS-formatted dataset."""
    from mne_bids import BIDSPath, read_raw_bids

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype=datatype,
        root=bids_root,
    )

    raw = read_raw_bids(bids_path, verbose="WARNING")

    logger.info(
        "loaded_bids_raw",
        subject=subject,
        task=task,
        n_channels=len(raw.ch_names),
        sfreq=raw.info["sfreq"],
    )
    return raw


def get_bids_subjects(bids_root: Path) -> list[str]:
    """List all subject IDs in a BIDS dataset."""
    bids_root = Path(bids_root)
    subjects = sorted(
        d.name.replace("sub-", "")
        for d in bids_root.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    logger.info("found_bids_subjects", n_subjects=len(subjects))
    return subjects


def get_bids_tasks(bids_root: Path, subject: str) -> list[str]:
    """List available tasks for a given subject."""

    sub_dir = Path(bids_root) / f"sub-{subject}"
    tasks: set[str] = set()

    for eeg_dir in sub_dir.rglob("eeg"):
        for f in eeg_dir.iterdir():
            if "task-" in f.name:
                task = f.name.split("task-")[1].split("_")[0]
                tasks.add(task)

    return sorted(tasks)


def write_bids(
    raw: mne.io.BaseRaw,
    bids_root: Path,
    subject: str,
    task: str,
    session: Optional[str] = None,
    run: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Write raw EEG data to BIDS format."""
    from mne_bids import BIDSPath, write_raw_bids

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype="eeg",
        root=bids_root,
    )

    write_raw_bids(
        raw,
        bids_path,
        overwrite=overwrite,
        verbose="WARNING",
    )

    logger.info(
        "wrote_bids",
        subject=subject,
        task=task,
        path=str(bids_path),
    )
    return Path(bids_path.fpath)
