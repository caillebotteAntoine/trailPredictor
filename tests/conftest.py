"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "activity_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "segment": [1000, 2000, 3000, 4000, 5000, 6000, 1000, 2000, 3000, 4000],
            "distance": [1000.0] * 10,
            "altitude": [100, 110, 105, 115, 115, 100, 200, 200, 210, 205],
        }
    )
