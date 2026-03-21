# Neuro State Monitor — Requirements

## Vision

A unified EEG-based system with four pluggable classifiers detecting brain states (craving, reconsolidation window, encoding quality, sleep reactivation) sharing a common preprocessing pipeline, MLOps infrastructure, and deployment architecture. Nobody has built this as an integrated open-source system.

---

## Milestone 1: Foundation — Shared Infrastructure

### 1. Project Setup
- Poetry project with dependencies: mne, mne-bids, braindecode, scikit-learn, mlflow, pynwb, neurokit2, structlog, pytest, black, isort
- pyproject.toml with proper metadata, Python 3.10+
- .gitignore for data/, models/, .env, __pycache__, .mlflow/, artifacts/

### 2. Shared Preprocessing Pipeline
- `shared/preprocessing/pipeline.py` — MNE-based EEG preprocessing:
  - Load raw EEG (various formats: .set, .edf, .fif, .csv)
  - Band-pass filter (configurable, default 0.5-45 Hz)
  - Notch filter (50/60 Hz line noise)
  - ICA artifact rejection (eye blinks, muscle)
  - Re-referencing (average reference)
  - Epoching (configurable event markers)
  - Bad channel interpolation
- `shared/preprocessing/bids.py` — BIDS format conversion utilities
- All preprocessing steps logged to MLflow as parameters

### 3. Shared Feature Extraction
- `shared/features/spectral.py` — Power spectral density (alpha, beta, theta, delta, gamma bands)
- `shared/features/connectivity.py` — Phase locking value, coherence, beta connectivity
- `shared/features/erp.py` — Event-related potentials (P300, N400, etc.)
- `shared/features/microstate.py` — EEG microstate extraction
- Feature extraction configs as YAML files

### 4. MLflow Integration
- `infrastructure/docker-compose.yml` — MLflow tracking server + artifact store
- `experiments/base_config.yaml` — default experiment configuration
- Auto-logging: preprocessing params, features used, model hyperparams, metrics
- Model registry with metadata schema: dataset_accession, preprocessing_version, participant_count

### 5. Evaluation Framework
- `shared/evaluation/metrics.py` — accuracy, AUC, F1, cross-subject generalization score
- `shared/evaluation/cross_validation.py` — leave-one-subject-out CV, k-fold with subject grouping
- `shared/evaluation/fairness.py` — demographic subgroup analysis (age, gender where available)
- Benchmark comparison utilities (compare against published baselines)

### 6. Data Management
- `data/README.md` — download instructions for each dataset with checksums
- `data/download_uci.py` — script to download and extract UCI EEG database
- `data/download_peers.py` — script to download PEERS from OpenNeuro
- DVC configuration for data versioning (optional, for larger datasets later)

### 7. Testing Foundation
- `tests/test_preprocessing.py` — unit tests for preprocessing pipeline
- `tests/test_features.py` — unit tests for feature extraction
- `tests/conftest.py` — shared fixtures, synthetic EEG data generators
- CI via GitHub Actions: lint (black, isort) + test on push

---

## Milestone 2: Encoding Classifier (Goal 3 — Start Here)

### 8. PEERS Data Loader
- `classifiers/encoding/data.py` — load PEERS dataset from OpenNeuro BIDS format
- Parse events: encoding trials, recall results, item metadata
- Label each trial: successfully_recalled (1) or forgotten (0)
- Train/validation/test split by subject (not by trial — prevent data leakage)

### 9. Encoding Feature Pipeline
- `classifiers/encoding/features.py` — encoding-specific features:
  - Alpha/beta desynchronization (10-30 Hz power decrease = encoding success)
  - Theta power increase (frontal/hippocampal regions)
  - Inter-electrode correlation patterns
  - Time windows: 0-600ms post-stimulus onset (from Duan et al. 2025)

### 10. Encoding Classifier Models
- `classifiers/encoding/models.py` — multiple architectures:
  - Baseline: SVM on spectral features (reproduce published results)
  - CNN: Braindecode ShallowFBCSPNet or EEGNet
  - Target: beat AUC 0.68 (EEG-ITNet baseline from literature)
- All models registered in MLflow with full metadata

### 11. Encoding Experiment Runner
- `classifiers/encoding/train.py` — end-to-end training script:
  - Load data → preprocess → extract features → train → evaluate → log to MLflow
  - Configurable via YAML
  - Cross-subject validation
- `classifiers/encoding/predict.py` — inference on new EEG data

### 12. Encoding Exploration Notebooks
- `notebooks/encoding/01_data_exploration.ipynb` — PEERS dataset overview, signal visualization
- `notebooks/encoding/02_feature_analysis.ipynb` — which features discriminate encoding success
- `notebooks/encoding/03_model_comparison.ipynb` — SVM vs CNN vs EEGNet results

---

## Milestone 3: Addiction Classifier (Goal 1)

### 13. UCI EEG Data Loader
- `classifiers/addiction/data.py` — load UCI EEG alcohol dataset
- Parse: alcoholic vs control labels, 64 channels, 256Hz
- Handle the dataset's specific format (trial structure, stimulus conditions)

### 14. Addiction Feature Pipeline
- `classifiers/addiction/features.py` — addiction-specific features:
  - Resting beta power (elevated in addiction)
  - Alpha band connectivity patterns
  - P300 amplitude (reduced in addiction)
  - Beta connectivity between electrodes (from Tian et al. 2024)

### 15. Addiction Classifier Models
- `classifiers/addiction/models.py`:
  - Baseline: reproduce published CNN ~95% on UCI (binary classification)
  - Advanced: beta connectivity features for craving prediction (Tian 2024 approach)
  - DSCnet reproduction (Wu 2025, dual-stream CNN)
- Benchmark table comparing all approaches

### 16. Addiction Experiment Runner + Notebooks
- Same pattern as encoding: train.py, predict.py, 3 notebooks
- Cross-subject generalization analysis (critical for clinical validity)

---

## Milestone 4: Deployment & Serving

### 17. Model Serving API
- `shared/serving/api.py` — REST API for real-time inference
  - POST /predict — accept EEG epoch, return state classification + confidence
  - GET /model — current model metadata
  - GET /health — health check
- BentoML or FastAPI + TorchServe

### 18. Docker Deployment
- `infrastructure/Dockerfile` — containerized inference server
- `infrastructure/docker-compose.yml` — full stack: MLflow + API + monitoring
- Environment variable configuration for model selection

### 19. Monitoring
- `shared/monitoring/drift.py` — signal drift detection (EEG amplitude distribution shift)
- `shared/monitoring/quality.py` — signal quality metrics (electrode impedance proxies, noise levels)
- Prometheus metrics endpoint for Grafana dashboards

---

## Milestone 5: PTSD Classifier + TMR (Later)

### 20-24. Trauma Classifier
- ENIGMA data loader (rsfMRI, switches to Nilearn)
- PTSD vs control classification (reproduce 75% AUC)
- Fear conditioning EEG integration
- Reconsolidation window biomarker extraction (theta/P300)

### 25-30. TMR Orchestration
- Sleep stage classifier from EEG
- Memory content decoder (reproduce Wang 2019)
- TMR cue delivery API
- Valence-aware cue selection
- End-to-end sleep TMR pipeline

---

## Success Criteria

### Milestone 1
- Running MLflow server with experiment tracking
- Preprocessing pipeline handles UCI and PEERS data formats
- Feature extraction produces reproducible outputs
- Tests pass, CI green

### Milestone 2
- Encoding classifier trained on PEERS
- AUC ≥ 0.68 (matching EEG-ITNet baseline)
- Cross-subject validation results logged
- At least one notebook with clear visualizations

### Milestone 3
- Addiction classifier trained on UCI EEG
- Reproduces published ~95% accuracy baseline
- Benchmark comparison table against DSCnet, Tian et al.
- Paper draft started

### Milestone 4
- Real-time inference API running in Docker
- < 100ms latency for single epoch classification
- Monitoring dashboard showing signal quality metrics
