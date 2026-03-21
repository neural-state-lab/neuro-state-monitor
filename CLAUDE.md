# Neuro State Monitor

## What This Is

A unified EEG-based Neural State Monitor — one system with four pluggable classifiers that detect specific cognitive/emotional brain states and trigger appropriate interventions. All four classifiers share the same preprocessing pipeline, MLOps infrastructure, and deployment architecture.

| Classifier | Brain State | EEG Signature | Intervention |
|-----------|------------|---------------|-------------|
| Encoding quality | High encoding (learning) | Alpha/beta desynchronization | Adapt learning pace |
| Addiction craving | Craving state | Resting beta ↑, alpha connectivity | Alert, distraction, clinical notification |
| Trauma/PTSD | Memory lability (reconsolidation window) | Theta ↑, P300 ↑ at retrieval | Time therapeutic intervention |
| Sleep reactivation (TMR) | Memory replay during sleep | Hippocampal ripple + spindle | Cue positive memories, suppress negative |

No open-source integrated system like this exists. This is original research.

## Architecture

```
neuro-state-monitor/
├── shared/
│   ├── preprocessing/        # MNE-Python EEG pipeline (all classifiers use this)
│   ├── features/             # Feature extraction (spectral, connectivity, ERP)
│   ├── evaluation/           # Benchmarking framework, metrics, cross-subject validation
│   ├── serving/              # BentoML/TorchServe real-time inference API
│   └── monitoring/           # Drift detection, signal quality, alerting
├── classifiers/
│   ├── encoding/             # Goal 3: PEERS dataset, encoding success prediction
│   ├── addiction/            # Goal 1: UCI EEG, craving detection
│   ├── trauma/              # Goal 2: ENIGMA, PTSD/reconsolidation
│   └── tmr/                 # Goal 4: Sleep reactivation, TMR orchestration
├── data/
│   ├── README.md            # Download instructions per dataset
│   └── raw/                 # .gitignore'd — raw data never committed
├── notebooks/               # Exploration notebooks, per-goal subdirectories
├── experiments/             # MLflow experiment configs
├── infrastructure/
│   ├── Dockerfile
│   ├── docker-compose.yml   # MLflow server + model serving
│   └── mlflow/              # MLflow backend configs
├── papers/                  # LaTeX/markdown drafts
├── benchmarks/              # Open benchmark definitions
└── tests/                   # Unit + integration tests
```

## Tech Stack

- **Signal processing:** MNE-Python, NeuroKit2
- **ML/DL:** Braindecode (PyTorch), scikit-learn
- **Experiment tracking:** MLflow or W&B
- **Data versioning:** DVC
- **Data format:** NWB (PyNWB), BIDS (MNE-BIDS)
- **Deployment:** BentoML or TorchServe, Docker, Kubernetes
- **Monitoring:** NannyML, Prometheus + Grafana
- **Language:** Python 3.10+
- **Package manager:** Poetry
- **Testing:** pytest

## Build Sequence

The classifiers are built in a specific order — each one reuses infrastructure from the previous:

1. **Encoding classifier** (PEERS dataset) — builds the entire core pipeline
2. **Addiction classifier** (UCI EEG) — same pipeline, different features
3. **PTSD classifier** (ENIGMA rsfMRI) — extends to fMRI via Nilearn
4. **TMR orchestration** (sleep EEG) — most ambitious, benefits from all prior work

## Datasets

Raw data is NEVER committed. Each dataset has download instructions in `data/README.md`.

### Phase 1 (start here)
- **PEERS** (UPenn) — free recall EEG, BIDS format, OpenNeuro — encoding classifier
- **UCI EEG Database** (alcohol) — 122 subjects, 64-ch, UCI ML Repository — addiction classifier

### Phase 2
- **ABCD Study** — 10K adolescents, fMRI, NIMH Data Archive (apply for access)
- **ENIGMA-PGC PTSD** — 7K subjects, multi-site, consortium request

### Phase 3
- **Wang et al. 2019 TMR EEG** — sleep TMR, requestable
- **CRCNS hippocampus** — rodent electrophysiology, crcns.org

### Transfer learning sources
- DEAP, DREAMER, SEED (emotion EEG — requestable)

## Code Conventions

- Type hints on ALL function signatures
- Docstrings on public API functions
- Black formatting (line length 88)
- isort (profile=black)
- pytest for testing
- No print statements — use structlog
- Every notebook must have a clear title and purpose in the first cell

## Git Strategy

- Small, focused commits with conventional prefixes: feat:, fix:, docs:, chore:, test:
- Never commit to main — feature branches with PRs
- Never commit raw data, API keys, or model weights
- Two contributors: rajnavakoti (research + implementation), brother's account (MLOps infra — starts contributing later)

## Research Context

- Full roadmap: `/Users/raj.navakoti/Desktop/github/NEURO-ROADMAP.md`
- HLR agent memory draft: `ddc-framework/research/hlr-agent-memory-draft.md`
- DDC paper (arxiv:2603.14057) — the methodology that inspired this work
- This project serves dual purpose: original research + brother's EB-2 NIW immigration profile (MLOps for neuroscience/addiction = national interest)
