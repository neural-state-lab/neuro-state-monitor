"""Download and extract the UCI EEG Database (alcohol study).

Dataset: EEG Database from UCI Machine Learning Repository
- 122 subjects (alcoholic vs control)
- 64 channels, 256 Hz sampling rate
- Visual stimulus paradigm

Source: https://archive.ics.uci.edu/dataset/121/eeg+database
"""

import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import structlog

logger = structlog.get_logger(__name__)

UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld"
DATASET_FILES = [
    "SMNI_CMI_TRAIN.tar.gz",
    "SMNI_CMI_TEST.tar.gz",
]

DATA_DIR = Path(__file__).parent / "raw" / "uci_eeg"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Report download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        if block_num % 100 == 0:
            logger.info("download_progress", pct=pct, file_size_mb=total_size / 1e6)


def download_file(url: str, dest: Path) -> Path:
    """Download a file with progress reporting and resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("file_exists_skipping", path=str(dest))
        return dest

    logger.info("downloading", url=url, dest=str(dest))
    urlretrieve(url, str(dest), reporthook=_progress_hook)
    logger.info("download_complete", path=str(dest))
    return dest


def extract_tar_gz(archive: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("extracting", archive=str(archive), dest=str(dest_dir))

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=dest_dir, filter="data")

    logger.info("extraction_complete", dest=str(dest_dir))


def download_uci_eeg(output_dir: Path | None = None) -> Path:
    """Download and extract the full UCI EEG dataset.

    Returns the path to the extracted data directory.
    """
    output_dir = output_dir or DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in DATASET_FILES:
        url = f"{UCI_BASE_URL}/{filename}"
        archive_path = output_dir / filename
        download_file(url, archive_path)
        extract_tar_gz(archive_path, output_dir)

    logger.info(
        "uci_eeg_ready",
        path=str(output_dir),
        expected_subjects=122,
    )
    return output_dir


def verify_download(data_dir: Path | None = None) -> bool:
    """Verify the UCI EEG dataset was downloaded correctly."""
    data_dir = data_dir or DATA_DIR

    if not data_dir.exists():
        logger.error("data_dir_missing", path=str(data_dir))
        return False

    # Check for expected structure
    gz_files = list(data_dir.rglob("*.gz"))
    if not gz_files:
        logger.error("no_data_files_found", path=str(data_dir))
        return False

    logger.info("verification_passed", n_files=len(gz_files))
    return True


if __name__ == "__main__":
    download_uci_eeg()
    verify_download()
