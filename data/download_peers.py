"""Download the PEERS dataset from OpenNeuro.

Dataset: Penn Electrophysiology of Encoding and Retrieval Study (PEERS)
- Free recall EEG in BIDS format
- Multiple subjects with encoding + retrieval tasks
- Available on OpenNeuro

Source: https://openneuro.org/datasets/ds004106
Requires: openneuro-py or datalad for full download
"""

import subprocess
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

OPENNEURO_DATASET_ID = "ds004106"
DATA_DIR = Path(__file__).parent / "raw" / "peers"


def _check_openneuro_cli() -> bool:
    """Check if openneuro-py is available."""
    try:
        import openneuro

        return True
    except ImportError:
        return False


def _check_datalad() -> bool:
    """Check if datalad is available."""
    try:
        result = subprocess.run(
            ["datalad", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_peers_openneuro(
    output_dir: Path | None = None,
    include: list[str] | None = None,
) -> Path:
    """Download PEERS dataset using openneuro-py.

    Args:
        output_dir: Where to save the dataset. Defaults to data/raw/peers.
        include: Optional list of glob patterns to selectively download.
                 e.g., ["sub-*/eeg/*"] to only get EEG files.
    """
    output_dir = output_dir or DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if _check_openneuro_cli():
        import openneuro

        logger.info(
            "downloading_peers",
            dataset=OPENNEURO_DATASET_ID,
            dest=str(output_dir),
        )

        openneuro.download(
            dataset=OPENNEURO_DATASET_ID,
            target_dir=str(output_dir),
            include=include,
        )

        logger.info("peers_download_complete", path=str(output_dir))
    else:
        logger.error(
            "openneuro_not_installed",
            hint="pip install openneuro-py",
        )
        raise RuntimeError(
            "openneuro-py not installed. Install with: pip install openneuro-py"
        )

    return output_dir


def download_peers_datalad(output_dir: Path | None = None) -> Path:
    """Download PEERS dataset using datalad (alternative method)."""
    output_dir = output_dir or DATA_DIR

    if not _check_datalad():
        raise RuntimeError("datalad not installed. Install with: pip install datalad")

    url = f"https://github.com/OpenNeuroDatasets/{OPENNEURO_DATASET_ID}.git"
    logger.info("cloning_with_datalad", url=url, dest=str(output_dir))

    subprocess.run(
        ["datalad", "clone", url, str(output_dir)],
        check=True,
    )

    # Get the actual data files
    subprocess.run(
        ["datalad", "get", "-r", "."],
        cwd=str(output_dir),
        check=True,
    )

    logger.info("peers_datalad_complete", path=str(output_dir))
    return output_dir


def download_peers(
    output_dir: Path | None = None,
    method: str = "auto",
) -> Path:
    """Download PEERS dataset using best available method.

    Args:
        output_dir: Where to save the dataset.
        method: "openneuro", "datalad", or "auto" (try openneuro first).
    """
    output_dir = output_dir or DATA_DIR

    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info("peers_already_downloaded", path=str(output_dir))
        return output_dir

    if method == "openneuro" or (method == "auto" and _check_openneuro_cli()):
        return download_peers_openneuro(output_dir)
    elif method == "datalad" or (method == "auto" and _check_datalad()):
        return download_peers_datalad(output_dir)
    else:
        logger.error(
            "no_download_method_available",
            hint="Install openneuro-py or datalad",
        )
        raise RuntimeError(
            "No download method available. Install openneuro-py or datalad.\n"
            "  pip install openneuro-py\n"
            "  # or\n"
            "  pip install datalad"
        )


def verify_peers(data_dir: Path | None = None) -> bool:
    """Verify the PEERS dataset structure."""
    data_dir = data_dir or DATA_DIR

    if not data_dir.exists():
        logger.error("data_dir_missing", path=str(data_dir))
        return False

    # Check for BIDS structure markers
    required_files = ["dataset_description.json", "participants.tsv"]
    missing = [f for f in required_files if not (data_dir / f).exists()]

    if missing:
        logger.warning("missing_bids_files", missing=missing)
        return False

    # Count subjects
    subjects = [
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")
    ]
    logger.info(
        "peers_verification",
        n_subjects=len(subjects),
        has_bids_structure=True,
    )
    return len(subjects) > 0


if __name__ == "__main__":
    download_peers()
    verify_peers()
