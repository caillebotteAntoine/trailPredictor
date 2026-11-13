"""Exhaustive inspection of ONE .fit (or .fit.gz) file to see what it contains.

Command line usage::

    python inspect_fit.py activities/123456.fit.gz
    python inspect_fit.py activities/123456.fit.gz --export-csv output_123456

Notebook usage::

    from inspect_fit import load_messages, inspect_fit
    report = inspect_fit("activities/123456.fit.gz")
    report["record"].head()          # raw DataFrame of the recorded points
    report["fields_record"]          # field profile of 'record'

The script prints:

1. Inventory of the message types (record, lap, session, event, ...).
2. For each type: fields, units, fill rate, example, min / max.
3. File / device / sport / session information.
4. Time series diagnostics (sampling rate, pauses, GPS, recomputed vs
   session elevation gain, available sensors).
5. Laps and events (timer start / stop).
"""

from __future__ import annotations

import argparse
import gzip
import io
from collections import defaultdict
from pathlib import Path

import fitdecode
import pandas as pd

SEMICIRCLE_TO_DEG = 180.0 / 2**31

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 45)


# --------------------------------------------------------------------------- #
# Lecture
# --------------------------------------------------------------------------- #
def load_messages(
    path: str | Path,
) -> tuple[dict[str, list[dict]], dict[str, dict[str, str]]]:
    """Read every message of a FIT file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.fit`` or gzip-compressed ``.fit.gz`` file.

    Returns
    -------
    messages : dict[str, list[dict]]
        Messages by type, each message being a dict of its fields. Fields
        without a value (``None``) are omitted.
    units : dict[str, dict[str, str]]
        Unit of each field, by message type.

    See Also
    --------
    TrailPredictor.load_fit.load_fit_file : Same reading with progress report.
    """
    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":  # gzip
        raw = gzip.decompress(raw)

    messages: dict[str, list[dict]] = defaultdict(list)
    units: dict[str, dict[str, str]] = defaultdict(dict)

    with fitdecode.FitReader(io.BytesIO(raw)) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            row = {}
            for f in frame.fields:
                if f.value is None:
                    continue
                row[f.name] = f.value
                if f.units:
                    units[frame.name][f.name] = f.units
            messages[frame.name].append(row)
    return dict(messages), dict(units)


# --------------------------------------------------------------------------- #
# Profilage des champs
# --------------------------------------------------------------------------- #
def _short(value, width: int = 38) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def profile_fields(df: pd.DataFrame, units: dict[str, str]) -> pd.DataFrame:
    """Profile each field of a message type.

    Parameters
    ----------
    df : pandas.DataFrame
        Messages of one type, one column per field.
    units : dict[str, str]
        Unit of each field.

    Returns
    -------
    pandas.DataFrame
        One row per field: unit, fill count and rate, type, number of unique
        values, min / mean / max and an example value.
    """
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        example = s.iloc[0] if len(s) else None
        row = {
            "champ": col,
            "unité": units.get(col, ""),
            "remplis": len(s),
            "%": round(100 * len(s) / n, 1) if n else 0.0,
            "type": type(example).__name__,
            "n_uniques": s.astype(str).nunique(),
            "min": None,
            "moyenne": None,
            "max": None,
            "exemple": _short(example),
        }
        if pd.api.types.is_bool_dtype(s):
            pass
        elif pd.api.types.is_numeric_dtype(s):
            row["min"], row["moyenne"], row["max"] = (
                round(float(s.min()), 3),
                round(float(s.mean()), 3),
                round(float(s.max()), 3),
            )
        elif pd.api.types.is_datetime64_any_dtype(s):
            row["min"], row["max"] = str(s.min()), str(s.max())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Diagnostics de la série temporelle
# --------------------------------------------------------------------------- #
def diagnose_records(rec: pd.DataFrame) -> None:
    """Print diagnostics on the ``record`` time series.

    Parameters
    ----------
    rec : pandas.DataFrame
        ``record`` messages, one column per field.

    Notes
    -----
    Covers the sampling rate and gaps, distance and speed, altitude and
    smoothed elevation gain, GPS coverage, available sensors and
    non-standard fields.
    """
    if rec.empty:
        print("  (aucun message 'record' dans ce fichier)")
        return

    cols = set(rec.columns)
    print(f"  Points enregistrés : {len(rec)}")

    # --- Temps et fréquence d'échantillonnage
    if "timestamp" in cols:
        ts = pd.to_datetime(rec["timestamp"])
        dt = ts.diff().dt.total_seconds().dropna()
        total = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
        print(f"  Début / fin        : {ts.iloc[0]}  ->  {ts.iloc[-1]}")
        print(f"  Durée écoulée      : {total / 3600:.2f} h ({total:.0f} s)")
        if len(dt):
            print(
                f"  Pas d'échantillonnage : médiane {dt.median():.1f} s | "
                f"min {dt.min():.1f} s | max {dt.max():.1f} s"
            )
            print(
                f"  Pas == 1 s         : {100 * (dt == 1).mean():.1f} % des intervalles"
            )
            gaps = dt[dt > 5]
            print(
                f"  Trous > 5 s (pauses / perte de signal) : {len(gaps)}"
                + (
                    f" (total {gaps.sum():.0f} s, plus long {gaps.max():.0f} s)"
                    if len(gaps)
                    else ""
                )
            )

    # --- Distance / vitesse
    if "distance" in cols:
        print(f"  Distance finale    : {rec['distance'].max() / 1000:.2f} km")
    speed_col = (
        "enhanced_speed"
        if "enhanced_speed" in cols
        else ("speed" if "speed" in cols else None)
    )
    if speed_col:
        v = rec[speed_col].dropna()
        moving = v[v > 0.5]
        if len(moving):
            print(
                f"  Vitesse            : moy. en mouvement {moving.mean() * 3.6:.1f} km/h "
                f"(allure {1000 / moving.mean() / 60:.2f} min/km) | max {v.max() * 3.6:.1f} km/h"
            )
        print(
            f"  Temps à l'arrêt (< 0.5 m/s) : {100 * (v <= 0.5).mean():.1f} % des points"
        )

    # --- Altitude et D+
    alt_col = (
        "enhanced_altitude"
        if "enhanced_altitude" in cols
        else ("altitude" if "altitude" in cols else None)
    )
    if alt_col:
        alt = rec[alt_col].dropna()
        smooth = alt.rolling(5, center=True, min_periods=1).median()
        d = smooth.diff().dropna()
        print(f"  Altitude           : min {alt.min():.0f} m | max {alt.max():.0f} m")
        print(
            f"  D+ recalculé (lissé) : {d[d > 0].sum():.0f} m | D- : {-d[d < 0].sum():.0f} m"
        )

    # --- GPS
    if {"position_lat", "position_long"} <= cols:
        lat = rec["position_lat"].dropna() * SEMICIRCLE_TO_DEG
        lon = rec["position_long"].dropna() * SEMICIRCLE_TO_DEG
        print(
            f"  GPS                : {100 * len(lat) / len(rec):.1f} % des points "
            f"| lat [{lat.min():.4f}, {lat.max():.4f}] | lon [{lon.min():.4f}, {lon.max():.4f}]"
        )
    else:
        print("  GPS                : absent")

    # --- Capteurs
    sensors = {
        "heart_rate": "FC",
        "cadence": "Cadence",
        "power": "Puissance",
        "temperature": "Température",
        "vertical_oscillation": "Oscillation verticale",
        "stance_time": "Temps de contact au sol",
        "step_length": "Longueur de foulée",
        "respiration_rate": "Fréquence respiratoire",
        "grade": "Pente",
        "left_right_balance": "Équilibre G/D",
        "gps_accuracy": "Précision GPS",
    }
    found = [(label, name) for name, label in sensors.items() if name in cols]
    print(
        "  Capteurs / métriques présents : "
        + (
            ", ".join(
                f"{label} ({100 * rec[name].notna().mean():.0f} %)"
                for label, name in found
            )
            or "aucun"
        )
    )
    if "heart_rate" in cols:
        hr = rec["heart_rate"].dropna()
        print(
            f"  FC                 : moy {hr.mean():.0f} | min {hr.min():.0f} | max {hr.max():.0f} bpm"
        )

    # --- Champs non standards (potentiellement propriétaires ou développeur)
    unknown = [c for c in rec.columns if c.startswith("unknown")]
    extras = [
        c
        for c in rec.columns
        if c not in sensors
        and c
        not in {
            "timestamp",
            "distance",
            "speed",
            "enhanced_speed",
            "altitude",
            "enhanced_altitude",
            "position_lat",
            "position_long",
        }
        and not c.startswith("unknown")
    ]
    if unknown:
        print(f"  Champs 'unknown_*' dans record : {unknown}")
    if extras:
        print(f"  Autres champs dans record : {extras}")


# --------------------------------------------------------------------------- #
# Rapport complet
# --------------------------------------------------------------------------- #
def _print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _print_single_message(name: str, rows: list[dict], units: dict[str, str]) -> None:
    """Print single messages (file_id, session, sport, ...) in plain text."""
    for i, row in enumerate(rows):
        if len(rows) > 1:
            print(f"  -- {name} #{i + 1}")
        for k, v in row.items():
            u = f" {units.get(k, '')}" if units.get(k) else ""
            print(f"  {k:<32} {_short(v, 60)}{u}")


def inspect_fit(
    path: str | Path, max_laps: int = 40, export_csv: str | Path | None = None
) -> dict:
    """Print a full report of a FIT file and return its content.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.fit`` or ``.fit.gz`` file.
    max_laps : int, optional
        Maximum number of laps printed. Default is 40.
    export_csv : str or pathlib.Path, optional
        Directory where each message type and its field profile are
        exported as CSV.

    Returns
    -------
    dict
        One DataFrame per message type (e.g. ``report["record"]``) and its
        field profile under ``"fields_<type>"``.

    See Also
    --------
    load_messages : Reads the messages.
    profile_fields : Builds the field profiles.
    """
    path = Path(path)
    messages, units = load_messages(path)
    frames = {name: pd.DataFrame(rows) for name, rows in messages.items()}

    # 1. Inventaire
    _print_header(f"1. INVENTAIRE DES MESSAGES — {path.name}")
    inv = pd.DataFrame(
        [
            {"message": n, "nombre": len(df), "nb_champs": df.shape[1]}
            for n, df in frames.items()
        ]
    ).sort_values("nombre", ascending=False)
    print(inv.to_string(index=False))

    # 2. Fichier / appareil / sport / session
    _print_header("2. IDENTITÉ DU FICHIER, APPAREIL, SPORT, SESSION")
    for name in ("file_id", "file_creator", "sport", "session"):
        if name in messages:
            print(f"\n[{name}]")
            _print_single_message(name, messages[name], units.get(name, {}))
    if "device_info" in frames:
        print("\n[device_info]  (appareil et capteurs connectés)")
        keep = [
            c
            for c in (
                "device_index",
                "device_type",
                "manufacturer",
                "product_name",
                "product",
                "serial_number",
                "software_version",
                "battery_status",
                "antplus_device_type",
                "source_type",
            )
            if c in frames["device_info"]
        ]
        print(frames["device_info"][keep].to_string(index=False))

    # 3. Description des champs de chaque type de message volumineux
    _print_header("3. CHAMPS PAR TYPE DE MESSAGE")
    fields = {}
    for name, df in frames.items():
        fields[name] = profile_fields(df, units.get(name, {}))
        if len(df) > 1 or name in ("record", "lap"):
            print(f"\n[{name}]  {len(df)} message(s)")
            print(fields[name].to_string(index=False))

    # 4. Diagnostics sur record
    _print_header("4. DIAGNOSTICS SUR LA SÉRIE TEMPORELLE (record)")
    diagnose_records(frames.get("record", pd.DataFrame()))

    # 5. Tours et événements
    _print_header("5. TOURS (lap)")
    if "lap" in frames:
        lap = frames["lap"]
        keep = [
            c
            for c in (
                "start_time",
                "total_elapsed_time",
                "total_distance",
                "total_ascent",
                "total_descent",
                "avg_heart_rate",
                "max_heart_rate",
                "enhanced_avg_speed",
                "avg_speed",
                "avg_cadence",
                "avg_power",
                "lap_trigger",
                "intensity",
            )
            if c in lap
        ]
        print(lap[keep].head(max_laps).to_string())
        if len(lap) > max_laps:
            print(f"  ... {len(lap) - max_laps} tours supplémentaires non affichés")
    else:
        print("  (aucun tour)")

    _print_header("6. ÉVÉNEMENTS (event)")
    if "event" in frames:
        ev = frames["event"]
        group = [c for c in ("event", "event_type") if c in ev]
        if group:
            print(
                ev.groupby(group)
                .size()
                .rename("nombre")
                .reset_index()
                .to_string(index=False)
            )
    else:
        print("  (aucun événement)")

    # Autres messages non détaillés ci-dessus
    other = [
        n
        for n in frames
        if n
        not in {
            "record",
            "lap",
            "session",
            "event",
            "file_id",
            "file_creator",
            "sport",
            "device_info",
        }
    ]
    if other:
        _print_header("7. AUTRES TYPES DE MESSAGES (résumé)")
        print(", ".join(f"{n} ({len(frames[n])})" for n in other))
        print("-> utilise report['<nom>'] ou report['fields_<nom>'] pour les examiner.")

    # Export CSV optionnel
    if export_csv:
        out = Path(export_csv)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in frames.items():
            df.to_csv(out / f"{name}.csv", index=False)
            fields[name].to_csv(out / f"_champs_{name}.csv", index=False)
        print(f"\nCSV exportés dans : {out.resolve()}")

    report = dict(frames)
    report.update({f"fields_{k}": v for k, v in fields.items()})
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("fit_file", help="Chemin du fichier .fit ou .fit.gz")
    ap.add_argument(
        "--export-csv",
        metavar="DOSSIER",
        help="Exporte chaque type de message (et son profil de champs) en CSV",
    )
    ap.add_argument("--max-laps", type=int, default=40)
    args = ap.parse_args()
    inspect_fit(args.fit_file, max_laps=args.max_laps, export_csv=args.export_csv)
