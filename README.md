# Neuro State Monitor

A unified EEG-based Neural State Monitor — one system with four pluggable classifiers that detect specific cognitive/emotional brain states and trigger appropriate interventions.

| Classifier | Brain State | EEG Signature | Intervention |
|-----------|------------|---------------|-------------|
| Encoding quality | High encoding (learning) | Alpha/beta desynchronization | Adapt learning pace |
| Addiction craving | Craving state | Resting beta, alpha connectivity | Alert, distraction, clinical notification |
| Trauma/PTSD | Memory lability (reconsolidation window) | Theta, P300 at retrieval | Time therapeutic intervention |
| Sleep reactivation (TMR) | Memory replay during sleep | Hippocampal ripple + spindle | Cue positive memories, suppress negative |

All four classifiers share the same preprocessing pipeline, feature extraction modules, MLOps infrastructure, and deployment architecture.

## Architecture

```
neuro-state-monitor/
├── shared/                  # Reusable across all classifiers
│   ├── preprocessing/       # MNE-Python EEG pipeline
│   ├── features/            # Spectral, connectivity, ERP, microstate
│   ├── evaluation/          # Metrics, cross-validation, fairness
│   └── experiment.py        # MLflow tracking helpers
├── classifiers/
│   ├── encoding/            # Encoding success prediction (PEERS)
│   ├── addiction/           # Craving detection (UCI EEG)
│   ├── trauma/              # PTSD/reconsolidation (ENIGMA)
│   └── tmr/                 # Sleep reactivation (TMR)
├── data/                    # Download scripts (raw data never committed)
├── experiments/             # YAML experiment configs
├── infrastructure/          # Docker Compose (MLflow server)
├── notebooks/               # Exploration notebooks per classifier
└── tests/                   # pytest with synthetic EEG fixtures
```

## Quick Start

```bash
# Install dependencies
poetry install --no-root

# Download datasets
python data/download_uci.py
python data/download_peers.py

# Start MLflow tracking server
cd infrastructure && docker-compose up -d

# Train encoding classifier
python -m classifiers.encoding.train --model svm --cv kfold

# Run tests
poetry run python -m pytest tests/ -v
```

## Tech Stack

- **Signal processing:** MNE-Python, NeuroKit2
- **ML/DL:** Braindecode (PyTorch), scikit-learn
- **Experiment tracking:** MLflow
- **Data format:** BIDS (MNE-BIDS)
- **Deployment:** Docker, BentoML
- **Language:** Python 3.10+
- **Package manager:** Poetry
- **Testing:** pytest + GitHub Actions CI

## Datasets

Raw data is never committed. Download scripts in `data/`.

| Dataset | Source | Classifier |
|---------|--------|-----------|
| UCI EEG (alcohol) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/121/eeg+database) | Addiction |
| PEERS | [OpenNeuro ds004106](https://openneuro.org/datasets/ds004106) | Encoding |
| ENIGMA-PGC PTSD | Consortium request | Trauma |
| Wang et al. TMR | Author request | Sleep |

## Development

```bash
# Format code
black .
isort .

# Lint
ruff check .

# Run tests
poetry run python -m pytest tests/ -v
```

## License

MIT
