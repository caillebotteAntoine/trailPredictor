"""Content-based hash of a DataFrame, stable across a CSV round trip."""

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from TrailPredictor.modeling.config import KEY_LENGTH

# Decimals kept on the float columns before hashing: absorbs the last-bit
# differences introduced by writing / reading a CSV file.
FLOAT_DECIMALS = 9


def _canonical_column(column: pd.Series) -> pd.Series:
    """Return the column with numeric values as rounded, canonical float64."""
    if pd.api.types.is_bool_dtype(column) or not pd.api.types.is_numeric_dtype(column):
        return column
    values = np.round(
        column.to_numpy(dtype=np.float64, na_value=np.nan), FLOAT_DECIMALS
    )
    # "+ 0.0" turns -0.0 into 0.0, and np.nan replaces any NaN bit pattern:
    # both would otherwise hash differently from their canonical value
    canonical = np.where(np.isnan(values), np.nan, values + 0.0)
    return pd.Series(canonical, index=column.index, name=column.name)


def hash_dataframe(data: pd.DataFrame, sort_by: Sequence[str] = ()) -> str:
    """Return a short hash identifying the content of a DataFrame.

    The hash depends only on the column names and values: the column order,
    the index and (when `sort_by` identifies the rows) the row order are
    ignored. Numeric columns are compared as float64 rounded to
    ``FLOAT_DECIMALS`` decimals, so the same data read back from a CSV file
    gets the same hash.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to hash. It is not modified.
    sort_by : sequence of str, optional
        Columns identifying a row, used to sort the rows before hashing.
        Default is empty, meaning the row order is part of the hash.

    Returns
    -------
    str
        First ``KEY_LENGTH`` hexadecimal characters of the SHA-256 hash.

    Raises
    ------
    KeyError
        If a column of `sort_by` is missing.

    See Also
    --------
    pandas.util.hash_pandas_object : Hash of each row.
    TrailPredictor.modeling.config.ModelConfig.key : Hash of a configuration.

    Notes
    -----
    The row hashes come from ``pandas.util.hash_pandas_object``, whose values
    are not guaranteed across pandas versions: upgrading pandas may change
    every hash. Two values on either side of a rounding boundary (e.g. 0.5e-9
    plus or minus one ulp) also hash differently; this is very unlikely in
    measured data.

    Examples
    --------
    >>> left = pd.DataFrame({"a": [1, 2], "b": [0.1, 0.2]})
    >>> right = pd.DataFrame({"b": [0.2, 0.1], "a": [2.0, 1.0]})
    >>> hash_dataframe(left, sort_by=["a"]) == hash_dataframe(right, sort_by=["a"])
    True
    """
    canonical = (
        data.sort_values(list(sort_by), kind="stable")
        .reset_index(drop=True)
        .sort_index(axis=1)
        .apply(_canonical_column)
    )
    row_hashes = pd.util.hash_pandas_object(canonical, index=False)
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, canonical.columns)).encode())
    digest.update(row_hashes.to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()[:KEY_LENGTH]
