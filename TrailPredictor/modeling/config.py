"""Validated, immutable configurations of the models."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field

KEY_LENGTH = 12  # hexadecimal characters of the configuration hash
MIN_SPLITS = 2  # GroupKFold needs at least two folds


class ModelConfig(BaseModel):
    """Configuration shared by every model.

    The configuration is immutable and rejects unknown fields, so a typo in a
    field name fails at construction instead of being silently ignored.
    Subclasses add the fields of their model.

    Parameters
    ----------
    seed : int, optional
        Seed of every random step of the model. Default is 42.
    n_splits : int, optional
        Number of ``GroupKFold`` folds (groups are the activities), at least 2.
        Default is 5.

    Raises
    ------
    pydantic.ValidationError
        If a field is unknown or invalid.

    Examples
    --------
    >>> ModelConfig(n_splits=3).n_splits
    3
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 42
    n_splits: int = Field(default=5, ge=MIN_SPLITS)

    @property
    def key(self) -> str:
        """Return a short hash identifying the configuration.

        The hash is computed on the JSON dump of the fields (in declaration
        order), so two equal configurations share the same key.

        Returns
        -------
        str
            First ``KEY_LENGTH`` hexadecimal characters of the SHA-256 hash.
        """
        payload = self.model_dump_json().encode()
        return hashlib.sha256(payload).hexdigest()[:KEY_LENGTH]
