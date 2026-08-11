"""
Path utilities for robust project-root-relative paths.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory (where config.yaml lives)."""
    # This file is at src/utils/paths.py, so go up 2 levels
    return Path(__file__).parent.parent.parent


def get_config_path() -> Path:
    """Return path to config.yaml."""
    return get_project_root() / "config.yaml"


def get_data_raw_path() -> Path:
    """Return path to data/raw directory."""
    return get_project_root() / "data" / "raw"


def get_data_processed_path() -> Path:
    """Return path to data/processed directory."""
    return get_project_root() / "data" / "processed"


def get_models_path() -> Path:
    """Return path to models directory."""
    return get_project_root() / "models"


def get_figures_path() -> Path:
    """Return path to reports/figures directory."""
    return get_project_root() / "reports" / "figures"


def load_config() -> dict:
    """Load config.yaml from project root."""
    import yaml
    with open(get_config_path(), 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    print(f"Project root: {get_project_root()}")
    print(f"Config: {get_config_path()}")
    print(f"Data raw: {get_data_raw_path()}")
    print(f"Models: {get_models_path()}")
    print(f"Figures: {get_figures_path()}")