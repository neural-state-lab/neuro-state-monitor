# Datasets

Raw data is **never committed** to this repository. Use the download scripts below.

## Phase 1 Datasets

### UCI EEG Database (Alcohol Study)

- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/121/eeg+database)
- **Subjects:** 122 (alcoholic vs control)
- **Channels:** 64, 256 Hz
- **Format:** CSV-like (custom), converted during loading
- **Size:** ~700 MB

```bash
python data/download_uci.py
```

### PEERS (Penn Electrophysiology of Encoding and Retrieval Study)

- **Source:** [OpenNeuro ds004106](https://openneuro.org/datasets/ds004106)
- **Task:** Free recall with EEG
- **Format:** BIDS
- **Requires:** `openneuro-py` or `datalad`

```bash
# Install downloader first
pip install openneuro-py

# Download
python data/download_peers.py
```

## Phase 2 Datasets (apply for access)

### ABCD Study
- **Source:** [NIMH Data Archive](https://nda.nih.gov/abcd)
- **Subjects:** ~10,000 adolescents, fMRI
- **Access:** Requires NDA account and data use agreement

### ENIGMA-PGC PTSD
- **Source:** [ENIGMA Consortium](https://enigma.ini.usc.edu/ongoing/enigma-ptsd/)
- **Subjects:** ~7,000 multi-site
- **Access:** Consortium request required

## Phase 3 Datasets

### Wang et al. 2019 TMR EEG
- **Source:** Requestable from authors
- **Task:** Sleep TMR paradigm

### CRCNS Hippocampus
- **Source:** [crcns.org](https://crcns.org)
- **Type:** Rodent electrophysiology

## Verification

After downloading, verify dataset integrity:

```bash
python -c "from data.download_uci import verify_download; verify_download()"
python -c "from data.download_peers import verify_peers; verify_peers()"
```
