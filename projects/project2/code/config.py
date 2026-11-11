# config.py

import tomllib
from pathlib import Path


def load_config(path=Path(__file__).parent / "config.toml") -> dict:
    """
    Load configurations from a TOML file.
    Path(__file__).parent finds the directory of the current file from any folder, and then we append "config.toml" to it.

    Parameters
    ----------
    path : Path        Path to the TOML configuration file.

    Returns
    -------
    dict               Dictionary containing the configurations.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)
