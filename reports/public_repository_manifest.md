# Public Repository File Manifest & Classification

This document classifies every file and directory in the **Disease Diagnosis Prediction** repository to ensure safe public release on GitHub.

---

## 1. Classification Overview

| Category | Description | Policy |
| :--- | :--- | :--- |
| **PUBLIC — SAFE TO INCLUDE** | Application source code, tests, notebooks, reports, model artifacts, configuration, and documentation containing no proprietary data or secrets. | Commit & Publish |
| **LOCAL ONLY — DO NOT PUBLISH** | Raw dataset CSV with unverified redistribution rights, virtual environment files, OS caches, temporary logs, and local agent scratch files. | Excluded via `.gitignore` |
| **REVIEW BEFORE PUBLISHING** | Licensing file requiring explicit user open-source license selection prior to public release. | User Review Required |

---

## 2. Directory & File Manifest

### PUBLIC — SAFE TO INCLUDE

- **`app/`**: Microservice code (`app/main.py`, `app/schemas.py`, `app/__init__.py`).
- **`src/`**: Machine learning modular codebase (`config.py`, `data_loader.py`, `preprocessing.py`, `train.py`, `evaluate.py`, `model_analysis.py`, `advanced_evaluation.py`, `model_selection.py`, `model_persistence.py`, `predict.py`).
- **`tests/`**: Automated pytest test suite (142 unit, integration, and API tests).
- **`notebooks/`**: 9 step-by-step Jupyter notebooks covering EDA, training, tuning, evaluation, and persistence.
- **`reports/`**: 33-section final technical report, metrics summary CSVs, system architecture diagrams, visual figures, and portfolio markdown guides.
- **`models/`**: Persisted model pipeline (`final_model.joblib`, ~54.5 KB) and metadata (`model_metadata.json`).
- **`data/raw/.gitkeep`**: Empty directory placeholder preserving raw data folder structure in Git.
- **`data/processed/.gitkeep`**: Empty directory placeholder preserving processed data folder structure in Git.
- **`Dockerfile`**: Container build recipe (`python:3.11-slim`).
- **`docker-compose.yml`**: Microservice orchestration config.
- **`.dockerignore`**: Docker build exclusion rules.
- **`.gitignore`**: Git repository exclusion rules.
- **`requirements.txt`**: Python dependency pins.
- **`README.md`**: Master project documentation.

---

### LOCAL ONLY — DO NOT PUBLISH (Excluded by `.gitignore`)

- **`data/raw/heart_disease.csv`**: Raw Cleveland / UCI dataset file (~62.7 KB, 918 observations). Excluded because redistribution permissions for the exact CSV file are unverified. Users acquire the raw CSV directly from the official UCI ML Repository.
- **`data/processed/*.csv`**: Any locally generated preprocessed CSV exports.
- **`.venv/`**, **`venv/`**, **`env/`**: Python virtual environment directories.
- **`__pycache__/`**, **`*.pyc`**: Python bytecode files.
- **`.pytest_cache/`**, **`.mypy_cache/`**, **`.ruff_cache/`**: Test and linter execution caches.
- **`scratch/`**: Temporary scripts used during development.
- **`.gemini/`**, **`.agents/`**: Local AI agent execution cache files.
- **`.vscode/`**, **`.idea/`**: Local IDE workspace settings.
- **`.DS_Store`**, **`Thumbs.db`**: Operating system temporary files.
- **`*.log`**: Local execution log files.

---

### REVIEW BEFORE PUBLISHING

- **`LICENSE`**: Currently empty. The project owner must select and insert an appropriate open-source license (e.g., MIT License, Apache License 2.0) before publishing publicly on GitHub.
