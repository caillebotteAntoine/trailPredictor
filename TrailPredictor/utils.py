"""Project-wide data directory paths.

The data directory is, in order of precedence:

1. the ``TRAILPREDICTOR_DATA_DIR`` environment variable, when set;
2. a ``data`` folder next to the executable, for the packaged desktop app
   (PyInstaller sets ``sys.frozen``);
3. the ``data`` folder of the repository, when running from the sources.
"""

import os
import sys
from collections.abc import Mapping
from pathlib import Path

DATA_DIR_ENV = "TRAILPREDICTOR_DATA_DIR"
DATA_DIR_NAME = "data"
SOURCE_ROOT = Path(__file__).parent.parent


def resolve_data_dir(
    environ: Mapping[str, str] = os.environ,
    frozen: bool = getattr(sys, "frozen", False),
    executable: Path = Path(sys.executable),
) -> Path:
    """Return the directory of the Strava data and of the computed files.

    Parameters
    ----------
    environ : Mapping[str, str], optional
        Environment variables. Default is ``os.environ``.
    frozen : bool, optional
        Whether the code runs from a PyInstaller executable. Default is
        ``sys.frozen`` (False when running from the sources).
    executable : pathlib.Path, optional
        Path of the running executable. Default is ``sys.executable``.

    Returns
    -------
    pathlib.Path
        ``$TRAILPREDICTOR_DATA_DIR`` (``~`` expanded) when set and not
        empty, else ``data`` next to the executable when `frozen`, else
        ``data`` at the root of the repository.

    Examples
    --------
    >>> resolve_data_dir({DATA_DIR_ENV: "D:/strava"}).as_posix()
    'D:/strava'
    """
    configured = environ.get(DATA_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    if frozen:
        return executable.parent / DATA_DIR_NAME
    return SOURCE_ROOT / DATA_DIR_NAME


DATA_DIR = resolve_data_dir()
COMPUTED_DIR = DATA_DIR / "computed"
MODELS_DIR = COMPUTED_DIR / "models"
