# GitHub Public Publication Pre-Flight Checklist

Before making the repository public or sharing with external reviewers, verify each item on this checklist:

---

### Security & Privacy
- [x] **Secrets Audited**: No API keys, passwords, cloud credentials, access tokens, connection strings, or `.env` files present.
- [x] **Gitignore Configured**: `.gitignore` excludes `data/raw/heart_disease.csv`, Python caches, virtual environments, IDE files, OS metadata, scratch scripts, and temporary logs.
- [x] **No Personal PII**: No sensitive personal data or private identity credentials hardcoded in source files.

---

### Data & Model Artifacts
- [x] **Large File Audit Passed**: No files exceed GitHub's 50 MB threshold (`final_model.joblib` is ~54.5 KB).
- [x] **Raw Dataset Excluded**: `data/raw/heart_disease.csv` excluded from Git tracking via `.gitignore` to prevent unauthorized public redistribution.
- [x] **Dataset Setup Documented**: `README.md` explicitly documents how users acquire the Cleveland / UCI Heart Disease dataset directly from the UCI ML Repository and place it at `data/raw/heart_disease.csv`.
- [x] **Model Artifact Reviewed**: `models/final_model.joblib` (~54.5 KB) retained for out-of-the-box API deployment without Git LFS.
- [x] **Model Persistence Verified**: `models/final_model.joblib` and `models/model_metadata.json` SHA-256 hashes verified and intact.

---

### Code & Documentation Quality
- [x] **LICENSE Reviewed**: `LICENSE` file reviewed; status documented as *License decision required before public publication*.
- [x] **README Professionalism**: `README.md` includes project title, held-out test metrics, project directory tree, local setup, dataset acquisition instructions, API endpoint schemas, cURL examples, Docker deployment commands, limitations, and non-clinical disclaimer.
- [x] **Architecture Diagram Available**: `reports/figures/system_architecture.png` generated and linked in documentation.
- [x] **Public Manifest Available**: `reports/public_repository_manifest.md` classifies all files into PUBLIC, LOCAL ONLY, and REVIEW.
- [x] **Portfolio & Resume Guides Included**: Created `reports/github_description.md`, `reports/portfolio_project_description.md`, `reports/resume_project_entry.md`, and `reports/linkedin_project_description.md`.
- [x] **Visual Screenshot Plan Drafted**: `reports/github_screenshot_plan.md` created mapping 9 visual capture targets.

---

### Testing & Integrity
- [x] **Automated Test Suite Passing**: `pytest -q` verified passing all 142 tests cleanly (0 failures).
- [x] **Model SHA-256 Hash**: `16081825890dc9a3fae9c8fd2216272379ee7ac0f7712072f983a0bb897a1568`
- [x] **Metadata SHA-256 Hash**: `5ec6e595a0c7d5d0dc899611b1f3f3480c1792564c2113e6e39478329467c836`
- [x] **Raw CSV Hash Verified**: `13c9cfee54ce2b1552ef7d787a5d8be9` (918 rows, preserved locally)

---

### Repository State Safeguards
- [x] **Git Status Reviewed**: `.gitignore` rules verified via `git status`.
- [x] **No Remote Push Performed**: No `git push` command executed.
- [x] **No Automatic Remote Creation**: GitHub repository creation left for manual user execution.
