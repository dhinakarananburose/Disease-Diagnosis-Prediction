"""
Docker Integration and Configuration Test Suite (Phase 13).
Validates Dockerfile structure, .dockerignore settings, docker-compose.yml configuration,
container working directory path resolutions, and dependency completeness.
"""

import pytest
from pathlib import Path

from src.config import PROJECT_ROOT, MODELS_DIR, DATASET_PATH


def test_dockerfile_exists_and_valid_structure():
    """1. Test that Dockerfile exists, uses a slim Python base image, exposes 8000, and configures non-root user."""
    dockerfile_path = PROJECT_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist in project root."
    assert dockerfile_path.stat().st_size > 0

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:" in content
    assert "slim" in content
    assert "EXPOSE 8000" in content
    assert "COPY app/ ./app/" in content
    assert "COPY src/ ./src/" in content
    assert "COPY models/ ./models/" in content
    assert "useradd" in content
    assert "USER appuser" in content
    assert 'CMD ["uvicorn", "app.main:app"' in content
    assert "--reload" not in content  # Production container should not use --reload


def test_dockerignore_exists_and_preserves_required_artifacts():
    """2. Test that .dockerignore exists, excludes dev files, but preserves app/, src/, and models/."""
    dockerignore_path = PROJECT_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist in project root."
    assert dockerignore_path.stat().st_size > 0

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    assert ".git" in lines
    assert ".venv" in lines
    assert "notebooks/" in lines or "notebooks" in lines
    assert "reports/" in lines or "reports" in lines

    # Ensure required runtime folders are NOT excluded
    assert "app/" not in lines and "app" not in lines
    assert "src/" not in lines and "src" not in lines
    assert "models/" not in lines and "models" not in lines
    assert "requirements.txt" not in lines


def test_docker_compose_exists_and_structure():
    """3. Test that docker-compose.yml exists, configures api service, maps port 8000:8000, and uses image disease-diagnosis-api."""
    compose_path = PROJECT_ROOT / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist in project root."
    assert compose_path.stat().st_size > 0

    content = compose_path.read_text(encoding="utf-8")
    assert "services:" in content
    assert "api:" in content
    assert "disease-diagnosis-api" in content
    assert "8000:8000" in content
    assert "context: ." in content


def test_container_working_directory_path_resolution():
    """4. Test that working directory /app allows src.config.MODELS_DIR to resolve correctly to models/final_model.joblib."""
    assert MODELS_DIR.exists()
    assert (MODELS_DIR / "final_model.joblib").exists()
    assert (MODELS_DIR / "model_metadata.json").exists()


def test_docker_runtime_dependencies():
    """5. Test that requirements.txt contains all required runtime dependencies for Docker container execution."""
    req_path = PROJECT_ROOT / "requirements.txt"
    assert req_path.exists()
    content = req_path.read_text(encoding="utf-8")

    required_pkgs = ["pandas", "numpy", "scikit-learn", "joblib", "fastapi", "uvicorn", "httpx"]
    for pkg in required_pkgs:
        assert pkg in content, f"Missing required dependency '{pkg}' in requirements.txt"
