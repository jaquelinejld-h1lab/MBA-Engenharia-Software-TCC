"""Project-relative path resolution.

Every path in the project is derived from ``PROJECT_ROOT``, which is computed
from this file's location. No module may hardcode an absolute path.
"""

from __future__ import annotations

from pathlib import Path

# src/paths.py -> src/ -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_FILE: Path = CONFIG_DIR / "config.yaml"


def resolve(relative: str | Path) -> Path:
    """Resolve a project-relative path against ``PROJECT_ROOT``.

    Parameters
    ----------
    relative : str or Path
        Path relative to the project root. An absolute path is returned
        unchanged so that callers can override locations via environment
        variables without breaking the API.

    Returns
    -------
    Path
        Absolute path.
    """
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path
