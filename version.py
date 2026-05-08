"""Single source of truth for the JARVIS version — read from pyproject.toml."""

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent / "pyproject.toml"

with _PYPROJECT.open("rb") as f:
    __version__: str = tomllib.load(f)["project"]["version"]
