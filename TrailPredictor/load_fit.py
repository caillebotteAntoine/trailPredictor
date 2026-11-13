"""Low-level reading of FIT files."""

import gzip
import io
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fitdecode

ProgressCallback = Callable[[float, str], None]


def message_fields(
    frame: fitdecode.FitDataMessage,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Extract the non-null fields of a FIT data message.

    Parameters
    ----------
    frame : fitdecode.FitDataMessage
        Decoded data message.

    Returns
    -------
    values : dict[str, Any]
        Value of each field, fields without a value being omitted.
    units : dict[str, str]
        Unit of each field that has a value and a unit.
    """
    values: dict[str, Any] = {}
    units: dict[str, str] = {}
    for field in frame.fields:
        if field.value is None:
            continue
        values[field.name] = field.value
        if field.units:
            units[field.name] = field.units
    return values, units


def load_fit_file(
    file_path: Path,
    progress_cb: ProgressCallback | None = None,
    update_every: int = 500,
):
    """Load a FIT file and optionally report parsing progress.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to a ``.fit`` or gzip-compressed ``.fit.gz`` file.
    progress_cb : ProgressCallback, optional
        Called with ``(fraction, message)`` while parsing, ``fraction`` being
        in ``[0, 1]``.
    update_every : int, optional
        Number of frames between two progress reports. Default is 500.

    Returns
    -------
    content : dict[str, list[dict]]
        Messages by type (``"record"``, ``"lap"``, ...), each message being
        a dict of its non-null fields. ``"record"`` messages sharing a
        timestamp are merged.
    units : dict[str, dict[str, str]]
        Unit of each field, by message type.
    """
    raw = file_path.read_bytes()
    if raw[:2] == b"\x1f\x8b":  # gzip
        raw = gzip.decompress(raw)

    total = len(raw)
    buffer = io.BytesIO(raw)
    content: dict[str, list[dict]] = defaultdict(list)
    units: dict[str, dict[str, str]] = defaultdict(dict)

    with fitdecode.FitReader(buffer) as fit:
        for i, frame in enumerate(fit):
            if progress_cb and i % update_every == 0:
                progress_cb(
                    buffer.tell() / total if total else 1.0,
                    "Loading FIT file...",
                )
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            row, field_units = message_fields(frame)
            if field_units:
                units[frame.name].update(field_units)
            content[frame.name].append(row)

    # merge all rows of records according to their timestamp
    record_merge: dict[Any, dict] = {}
    for i, row in enumerate(content["record"]):
        if row["timestamp"] in record_merge:
            record_merge[row["timestamp"]].update(row)
        else:
            record_merge[row["timestamp"]] = row

    content["record"] = list(record_merge.values())

    if progress_cb:
        progress_cb(1.0, "FIT file loaded")

    return content, units
