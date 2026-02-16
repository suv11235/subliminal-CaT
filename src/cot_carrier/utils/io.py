"""I/O utilities for data persistence and configuration management."""

import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def save_jsonl(items: List[Dict], path: str, mode: str = "w") -> None:
    """
    Save items to JSONL file.

    Args:
        items: List of dictionaries to save
        path: Output path
        mode: File mode ("w" for write, "a" for append)
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode) as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    """
    Load items from JSONL file.

    Args:
        path: Input path

    Returns:
        List of dictionaries
    """
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def create_run_dir(exp_name: str, base_dir: str = "outputs/runs") -> Path:
    """
    Create timestamped run directory.

    Args:
        exp_name: Experiment name
        base_dir: Base directory for runs

    Returns:
        Path to created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{timestamp}_{exp_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (run_dir / "cache").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    logger.info(f"Created run directory: {run_dir}")
    return run_dir


def save_config_snapshot(config: Dict[str, Any], run_dir: Path) -> None:
    """
    Save resolved configuration snapshot.

    Args:
        config: Configuration dictionary
        run_dir: Run directory
    """
    config_path = run_dir / "config_resolved.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config snapshot to {config_path}")


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML config with optional overrides.

    Args:
        path: Path to YAML config file
        overrides: Dictionary of values to override

    Returns:
        Merged configuration dictionary
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if overrides:
        _deep_update(config, overrides)

    return config


def _deep_update(base: Dict, updates: Dict) -> None:
    """
    Recursively update nested dict.

    Args:
        base: Base dictionary to update
        updates: Updates to apply
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Output path
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
