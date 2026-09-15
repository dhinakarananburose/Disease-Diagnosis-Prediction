# Phase 15.2 — License Finalization & Pre-Publication Status Report

## 1. License Selected
**MIT License** ([`LICENSE`](file:///c:/projects/Disease-Diagnosis-Prediction/LICENSE))

## 2. Copyright Holder
**Copyright (c) 2026 Dhinakaran Anburose**

## 3. Scope of License
The MIT License applies to:
- Project source code in `src/` and `app/`
- FastAPI REST microservice implementation
- Dockerfile & Docker Compose deployment configuration
- Automated test suite in `tests/`
- Original Jupyter notebooks in `notebooks/`
- Technical documentation reports in `reports/`

> **Note**: The MIT License does **NOT** apply to third-party data or clinical datasets.

## 4. Dataset Licensing & Redistribution Status
- **Status**: **Local-Only (Excluded from Git)**.
- The raw clinical dataset (`data/raw/heart_disease.csv`) originates from the Cleveland / UCI Heart Disease dataset ($N=918$).
- Because redistribution rights for the exact local CSV format were not independently established, the file is **intentionally excluded from public Git commits**.

## 5. Dataset Exclusion Status
- **`.gitignore` Exclusion**: `data/raw/*.csv` is active in `.gitignore`.
- **Placeholder Tracking**: `data/raw/.gitkeep` and `data/processed/.gitkeep` preserve folder structure in Git.
- **Local File Preservation**: `data/raw/heart_disease.csv` remains **intact locally** (MD5: `13c9cfee54ce2b1552ef7d787a5d8be9`, 918 observations).

## 6. Model Publication Status
- **Recommendation**: **Retain in Repository (Direct Commit)**.
- `models/final_model.joblib`: ~54.5 KB (well below GitHub's 50 MB threshold). Git LFS is **not required**.
- `models/model_metadata.json`: Serialized metadata and hyperparameters intact.

## 7. Security Audit Results
- **Result**: **Clean / No Secrets Found**.
- Repeated codebase search confirmed zero exposure of API keys, credentials, tokens, passwords, private keys, or `.env` files.

## 8. Test Result
- `pytest -q`: **142 passed, 0 failed** (in 39.14s).

## 9. Integrity Hashes Verified
- **Model Artifact SHA-256**: `16081825890dc9a3fae9c8fd2216272379ee7ac0f7712072f983a0bb897a1568`
- **Model Metadata SHA-256**: `5ec6e595a0c7d5d0dc899611b1f3f3480c1792564c2113e6e39478329467c836`
- **Raw Dataset MD5**: `13c9cfee54ce2b1552ef7d787a5d8be9` (918 observations)

## 10. Git Status
- **Git Repository Status**: **Not initialized**.
- No `git init`, `git add`, `git commit`, `git remote add`, or `git push` commands were executed.

## 11. Remaining Manual Actions Prior to Public Release
1. Capture visual screenshots following [`reports/github_screenshot_plan.md`](file:///c:/projects/Disease-Diagnosis-Prediction/reports/github_screenshot_plan.md).
2. Initialize local Git repository (`git init`), stage files (`git add .`), create initial commit (`git commit`), and push to GitHub manually when ready.
