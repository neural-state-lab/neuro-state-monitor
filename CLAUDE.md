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

### Core
- **Language:** Python 3.10+ (constrained <3.13 for braindecode compat)
- **Package manager:** Poetry
- **Testing:** pytest + GitHub Actions CI
- **Experiment tracking:** MLflow
- **Logging:** structlog

### Signal Processing & Preprocessing
- **MNE-Python** — raw EEG loading, filtering, epoching, re-referencing
- **mne-icalabel** — ICA component auto-labeling (ICLabel neural net, 7 component types)
- **autoreject** — automated per-channel epoch rejection thresholds via cross-validation
- **MNE-BIDS** — BIDS format reading/writing
- **MNE-Connectivity** — PLV, coherence, spectral connectivity

### Feature Extraction
- **specparam** (formerly FOOOF) — parameterized spectral analysis (separates 1/f from oscillations)
- **antropy** — entropy/complexity features (permutation entropy, sample entropy, Hjorth, LZC)
- **pycrostates** — EEG microstate analysis (JOSS-reviewed, replaces custom k-means)
- **NeuroKit2** — multimodal physiological signals (ECG/EDA/respiration for wearable TMR)

### ML/DL Models
- **Braindecode** — EEGNet, ShallowFBCSPNet (PyTorch-based EEG deep learning)
- **scikit-learn** — SVM, Random Forest, preprocessing pipelines
- **TorchEEG** — EEG augmentation transforms, 30+ architectures, LOSO splits
- **selfEEG** — SSL pre-training (SimCLR/BYOL) for cross-subject generalization

### Sleep/TMR Pipeline
- **YASA** — automated sleep staging, spindle detection, slow oscillation detection (eLife 2021, 381 citations)

### fMRI Pipeline (PTSD)
- **fMRIPrep** — standard fMRI preprocessing (Docker container, 4400 citations)
- **Nilearn** — fMRI ML: GLM, connectivity, decoding, parcellation

### Benchmarking & Evaluation
- **MOABB** — standardized BCI benchmarking across 60+ datasets

### Deployment & Production
- **BentoML** — model serving (supports mixed model types: SVM + CNN + LightGBM)
- **MNE-LSL** — real-time EEG streaming (JOSS 2025, LSL integration with MNE)
- **NannyML** — post-deployment drift detection + performance estimation
- **Docker** — containerization for MLflow server + model serving

### Explainability
- **captum** — Integrated Gradients + DeepLIFT (validated for EEG DL)
- **pytorch-grad-cam** — spatial topography heatmaps for CNN models

## Build Sequence

| Milestone | Status | What |
|-----------|--------|------|
| M1: Foundation | ✅ Done | Shared pipeline, features, evaluation, MLflow, tests |
| M2: Encoding classifier | ✅ Done | PEERS data, encoding features, SVM/EEGNet/ShallowNet |
| M3: Addiction classifier | ✅ Done | UCI data, addiction features, SVM/RF/EEGNet/DSCnet |
| M4: Ecosystem refactor | 🔲 Next | Replace custom code with mne-icalabel, autoreject, pycrostates, specparam, antropy |
| M5: Deployment & serving | 🔲 | BentoML API, MNE-LSL streaming, NannyML monitoring, Docker |
| M6: PTSD + TMR | 🔲 | fMRIPrep+Nilearn for PTSD, YASA for sleep/TMR, selfEEG for cross-subject |

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

## Build vs Reuse Rule

Before writing any new module, check if a mature package already does it:

1. Search PyPI and GitHub for existing solutions
2. Check: actively maintained? (commits in last 6 months)
3. Check: >100 GitHub stars, published docs, peer-reviewed paper?
4. If mature package exists → wrap with thin adapter (MLflow logging, config)
5. If no mature package → build it, document WHY in module docstring

### Use Library (DO NOT rebuild)

| Need | Use This | NOT This |
|------|----------|----------|
| ICA component labeling | `mne-icalabel` (ICLabel, 7 types) | manual `find_bads_eog`/`find_bads_muscle` |
| Bad epoch rejection | `autoreject` (learned thresholds) | manual amplitude thresholds |
| Microstate analysis | `pycrostates` (JOSS 2022) | custom k-means implementation |
| Sleep staging | `YASA` (eLife 2021, 381 cites) | building classifier from scratch |
| Spindle/SO detection | `YASA.spindles_detect()` / `sw_detect()` | custom detectors |
| Spectral analysis | `specparam` (Nature Neuro 2020) | raw Welch PSD only |
| Entropy features | `antropy` (numba-accelerated) | custom entropy functions |
| BCI benchmarking | `MOABB` (60+ datasets) | manual LOSO benchmark code |
| fMRI preprocessing | `fMRIPrep` (Docker, 4400 cites) | custom fMRI pipeline |
| fMRI ML | `Nilearn` (JOSS-reviewed) | custom fMRI analysis |
| Real-time streaming | `MNE-LSL` (JOSS 2025) | custom LSL bindings |
| Model serving | `BentoML` (mixed model types) | FastAPI+TorchServe |
| Drift detection | `NannyML` (CBPE, 1800 stars) | custom monitoring |
| EEG augmentation | `TorchEEG` transforms | custom augmentation |
| SSL pre-training | `selfEEG` (SimCLR/BYOL for EEG) | training from scratch |
| DL explainability | `captum` + `pytorch-grad-cam` | saliency maps (they fail validity tests) |

### Skip (stale or superseded)

| Package | Why Skip |
|---------|----------|
| `mne-features` | Last release 2021, stale. Use `antropy` + `mne-connectivity` instead |
| `pylsl` | Superseded by `MNE-LSL` |
| TorchServe alone | PyTorch-only, can't serve sklearn/LightGBM |

### Our Value-Add (what we build custom)

- Classifier-specific feature combinations and training pipelines
- MLflow experiment tracking integration across all classifiers
- Subject-based cross-validation and demographic fairness evaluation
- Training/inference CLI orchestration
- Multi-classifier shared architecture (the unified system is the novelty)
- EEG signal quality monitoring wrapper (no mature EEG-specific SQI library exists — publishable contribution)

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
