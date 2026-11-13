#!/usr/bin/env python3
"""
reconstitution_fit.py : complète une activité dont le début a été perdu et écrit un .fit.

  FIT_END   = ton activité de fin, réellement enregistrée (end_activity.fit)
  GPX_TRACE = la trace complète prévue, sans heures (trace.gpx)

Usage : éditer la section CONFIGURATION, puis  python reconstitution_fit.py
Dépendances : pip install gpxpy fit-tool fitparse numpy pandas

Principe
  1. Lecture du FIT : points, altitude, vitesse, distance, FC/cadence si présentes, et
     fréquence d'échantillonnage (pas médian entre deux enregistrements).
  2. Raccord : le début du FIT est localisé sur la trace (point le plus proche). La partie
     manquante = trace du départ jusqu'à ce point, en ne gardant que ses points de passage.
  3. Heures : ANCHORS (kilomètre -> heure) calent le temps ; entre deux ancres, le temps est
     interpolé linéairement avec la distance (aucun modèle d'allure). La fin de la partie
     manquante est calée sur le premier enregistrement du FIT (moins un pas).
  4. La partie manquante est générée à la même fréquence que le FIT, le long de la trace
     (interpolation linéaire entre points de passage, aucune dérive ajoutée).
  5. Les données réelles du FIT sont conservées telles quelles. La partie reconstituée ne
     contient que position, altitude (trace), distance et heure : aucune vitesse, FC,
     cadence, température ou puissance n'est prédite.
  6. Écriture du FIT fusionné (pause timer si trou > MAX_GAP_S), relecture de contrôle, et
     EXPORT_CSV avec une colonne `synthetic` à exclure de ton jeu d'entraînement.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import gpxpy
import numpy as np
import pandas as pd
from fitparse import FitFile as FitReader

from fit_tool.fit_file import FitFile
from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.activity_message import ActivityMessage
from fit_tool.profile.messages.event_message import EventMessage
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.lap_message import LapMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.session_message import SessionMessage
from fit_tool.profile.profile_type import (
    Activity,
    Event,
    EventType,
    FileType,
    Manufacturer,
    Sport,
    SubSport,
)

# =========================================================================== #
# CONFIGURATION : à modifier
# =========================================================================== #
FIT_END = "data/end_activity.fit"  # activité de fin (réelle)
GPX_TRACE = "data/trace.gpx"  # trace complète (sans heures)
OUTPUT_FIT = "data/activite_reconstituee.fit"

DATE = None  # "AAAA-MM-JJ" du départ ; None = jour du début du FIT
TZ = "Europe/Paris"  # fuseau des heures d'ancres

# (kilomètre, heure) de la partie manquante. 0 = départ obligatoire. Le kilomètre de raccord
# (début du FIT) n'a pas besoin d'être listé : son heure est celle du FIT.
ANCHORS = [(0, "8h54"), (10, "11h01"), (17, "12h16"), (20, "12h50")]
MISSING_KM = 25.0  # distance réelle de la partie manquante (montre) ou None
JOIN_KM = None  # forcer le km (sur la trace) du raccord ; None = auto
JOIN_TOL_M = 100.0  # tolérance de détection du raccord (m)

SPORT = None  # None = repris du FIT ; trail | running | hiking | walking | cycling

ELEV_BLEND_KM = 2.0  # lisse l'écart d'altitude trace/FIT sur les derniers km

STEP_S = None  # pas d'échantillonnage (s) ; None = déduit du FIT
MAX_GAP_S = 60.0  # trou au-delà duquel on insère une pause timer
EXPORT_CSV = "controle.csv"  # ou None
# =========================================================================== #

SPORTS = {
    "trail": (Sport.RUNNING, SubSport.TRAIL),
    "running": (Sport.RUNNING, SubSport.GENERIC),
    "hiking": (Sport.HIKING, SubSport.GENERIC),
    "walking": (Sport.WALKING, SubSport.GENERIC),
    "cycling": (Sport.CYCLING, SubSport.GENERIC),
}
FIELDS = ("hr", "cad", "temp", "pwr")
SEMI = 180.0 / 2**31


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def die(msg: str):
    sys.exit(f"Erreur : {msg}")


def anchor_epoch(tstr: str, date: str) -> float:
    """'8h54', '08:54', '8:54:30' (jour = date) ou date-heure ISO complète."""
    s = str(tstr).strip().lower()
    if "-" in s:
        dt = datetime.fromisoformat(s.replace("z", "+00:00"))
        return (dt if dt.tzinfo else dt.replace(tzinfo=ZoneInfo(TZ))).timestamp()
    s = s.replace("h", ":").rstrip(":")
    parts = s.split(":")
    try:
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        sec = int(parts[2]) if len(parts) > 2 and parts[2] else 0
    except ValueError:
        die(f"heure d'ancre illisible : {tstr!r} (attendu 8h54, 08:54 ou 08:54:30).")
    base = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=ZoneInfo(TZ))
    return (base + timedelta(hours=h, minutes=m, seconds=sec)).timestamp()


def fmt_pace(sec_per_km: float) -> str:
    if not np.isfinite(sec_per_km) or sec_per_km <= 0:
        return "n/a"
    m, s = divmod(int(round(sec_per_km)), 60)
    return f"{m}:{s:02d}/km"


def fmt_hms(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def haversine_segments(lat, lon) -> np.ndarray:
    r = 6371008.8
    la, lo = np.radians(lat), np.radians(lon)
    a = (
        np.sin(np.diff(la) / 2) ** 2
        + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    )
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, k - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def interp_valid(x_new, x, y) -> np.ndarray:
    m = ~np.isnan(y)
    if m.sum() == 0:
        return np.full(len(x_new), np.nan)
    if m.sum() == 1:
        return np.full(len(x_new), y[m][0])
    return np.interp(x_new, x[m], y[m])


# --------------------------------------------------------------------------- #
# Lecture du FIT de fin
# --------------------------------------------------------------------------- #
def load_fit(path: str) -> dict:
    try:
        ff = FitReader(path)
        rows: dict = {}
        for m in ff.get_messages("record"):
            v = m.get_values()
            ts = v.get("timestamp")
            if ts is not None:  # un instant peut être réparti sur plusieurs messages
                rows.setdefault(ts, {}).update(
                    {k: x for k, x in v.items() if x is not None}
                )
        sport = sub = None
        for m in ff.get_messages("session"):
            v = m.get_values()
            sport, sub = v.get("sport"), v.get("sub_sport")
    except FileNotFoundError:
        die(f"fichier introuvable : {path}")
    if len(rows) < 2:
        die(f"{path} contient moins de 2 enregistrements.")
    ks = sorted(rows)

    def col(*names, scale=1.0):
        out = np.full(len(ks), np.nan)
        for i, k in enumerate(ks):
            for n in names:
                x = rows[k].get(n)
                if isinstance(x, (int, float)):
                    out[i] = x * scale
                    break
        return out

    d = {
        "t": np.array(
            [
                (k if k.tzinfo else k.replace(tzinfo=timezone.utc)).timestamp()
                for k in ks
            ]
        ),
        "lat": col("position_lat", scale=SEMI),
        "lon": col("position_long", scale=SEMI),
        "ele": col("enhanced_altitude", "altitude"),
        "dist": col("distance"),
        "speed": col("enhanced_speed", "speed"),
        "hr": col("heart_rate"),
        "cad": col("cadence"),
        "temp": col("temperature"),
        "pwr": col("power"),
    }
    ok = ~np.isnan(d["lat"]) & ~np.isnan(d["lon"])
    if not ok.all():
        print(f"FIT : {(~ok).sum()} enregistrement(s) sans position ignoré(s).")
        d = {k: v[ok] for k, v in d.items()}
    if len(d["t"]) < 2:
        die("le FIT ne contient pas de positions GPS exploitables.")

    if np.isnan(d["dist"]).all():
        d["dist"] = np.r_[0.0, np.cumsum(haversine_segments(d["lat"], d["lon"]))]
    else:
        d["dist"] = interp_valid(d["t"], d["t"], d["dist"])
    d["dist"] = d["dist"] - d["dist"][0]
    if np.isnan(d["speed"]).any():
        v = np.clip(np.gradient(d["dist"], d["t"]), 0, None)
        d["speed"] = np.where(np.isnan(d["speed"]), v, d["speed"])

    dts = np.diff(d["t"])
    step = float(np.median(dts[dts > 0]))
    d["step"] = STEP_S if STEP_S else step
    if sub == "trail":
        d["sport"] = "trail"
    elif sport in SPORTS:
        d["sport"] = sport
    else:
        d["sport"] = "running"
    d["gaps"] = int((dts > MAX_GAP_S).sum())
    return d


# --------------------------------------------------------------------------- #
# Lecture de la trace GPX et raccord
# --------------------------------------------------------------------------- #
def load_trace(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
    except FileNotFoundError:
        die(f"fichier introuvable : {path}")
    raw = [p for tr in gpx.tracks for seg in tr.segments for p in seg.points]
    if not raw:
        raw = [p for rte in gpx.routes for p in rte.points]
    if len(raw) < 2:
        die(f"{path} contient moins de 2 points (ni <trk> ni <rte> exploitable).")
    return {
        "lat": np.array([p.latitude for p in raw], float),
        "lon": np.array([p.longitude for p in raw], float),
        "ele": np.array(
            [np.nan if p.elevation is None else p.elevation for p in raw], float
        ),
    }


def find_join(tr: dict, p0: tuple) -> int:
    cum = np.r_[0.0, np.cumsum(haversine_segments(tr["lat"], tr["lon"]))]
    dm = np.hypot(
        (tr["lat"] - p0[0]) * 111320.0,
        (tr["lon"] - p0[1]) * 111320.0 * np.cos(np.radians(p0[0])),
    )
    if JOIN_KM is not None:
        idx = int(np.argmin(np.abs(cum - JOIN_KM * 1000.0)))
        print(
            f"Raccord forcé : point {idx}, km {cum[idx] / 1000:.2f} de la trace ({dm[idx]:.0f} m du début du FIT)."
        )
        return idx
    pad = np.r_[np.inf, dm, np.inf]
    cand = [
        i
        for i in range(len(dm))
        if pad[i + 1] < pad[i]
        and pad[i + 1] <= pad[i + 2]
        and dm[i] < JOIN_TOL_M
        and i >= 1
    ]
    if not cand:
        die(
            f"le début du FIT est à {dm.min():.0f} m de la trace (tolérance {JOIN_TOL_M:.0f} m) : "
            "augmente JOIN_TOL_M ou fixe JOIN_KM."
        )
    if len(cand) > 1:
        ref = (MISSING_KM or float(ANCHORS[-1][0]) if ANCHORS else 0.0) * 1000.0
        print(
            "Plusieurs passages proches du début du FIT : "
            + ", ".join(f"km {cum[i] / 1000:.2f} ({dm[i]:.0f} m)" for i in cand)
        )
        idx = min(cand, key=lambda i: abs(cum[i] - ref))
        print(
            f"  -> retenu : km {cum[idx] / 1000:.2f} (le plus proche de {ref / 1000:g} km ; sinon fixe JOIN_KM)."
        )
    else:
        idx = cand[0]
    print(
        f"Raccord : point {idx} de la trace, km {cum[idx] / 1000:.2f}, à {dm[idx]:.0f} m du début du FIT."
    )
    return idx


# --------------------------------------------------------------------------- #
# Partie manquante : trace + ancres -> enregistrements à la fréquence du FIT
# --------------------------------------------------------------------------- #
def build_missing(tr: dict, idx: int, fit: dict) -> dict:
    step = fit["step"]
    lat, lon, ele = tr["lat"][: idx + 1], tr["lon"][: idx + 1], tr["ele"][: idx + 1]
    keep = np.r_[
        True, haversine_segments(lat, lon) > 0.01
    ]  # points de passage distincts
    lat, lon, ele = lat[keep], lon[keep], ele[keep]
    seg_raw = haversine_segments(lat, lon)
    dist_raw = np.r_[0.0, np.cumsum(seg_raw)]
    scale = MISSING_KM * 1000.0 / dist_raw[-1] if MISSING_KM else 1.0
    dist = dist_raw * scale
    print(
        f"Partie manquante : {len(lat)} points de passage, {dist_raw[-1] / 1000:.2f} km sur la trace "
        f"-> {dist[-1] / 1000:.2f} km (x{scale:.4f})."
    )

    # ancres (distance, heure)
    t_end = fit["t"][0] - step
    date = DATE or datetime.fromtimestamp(fit["t"][0], ZoneInfo(TZ)).strftime(
        "%Y-%m-%d"
    )
    parsed, prev = [], None
    for km, tstr in sorted(ANCHORS, key=lambda a: a[0]):
        ep = anchor_epoch(tstr, date)
        while prev is not None and ep < prev:  # passage de minuit
            ep += 86400
        parsed.append((float(km), ep))
        prev = ep
    if not parsed or parsed[0][0] > 0.05:
        die("ANCHORS doit contenir le départ, par exemple (0, '8h54').")
    if parsed[0][1] > t_end:  # départ la veille du début du FIT
        parsed = [(k, e - 86400) for k, e in parsed]
    a_d, a_t = [], []
    for km, ep in parsed:
        if km * 1000.0 >= dist[-1] - 50:
            gap = (fit["t"][0] - ep) / 60
            print(
                f"Note : l'ancre {km:g} km correspond au raccord ; heure du FIT utilisée "
                f"(écart avec ton ancre : {gap:+.1f} min)."
            )
            continue
        a_d.append(km * 1000.0)
        a_t.append(ep)
    a_d.append(dist[-1])
    a_t.append(t_end)
    a_d, a_t = np.array(a_d), np.array(a_t)
    if (np.diff(a_d) <= 0).any() or (np.diff(a_t) <= 0).any():
        bad = [
            f"{a_d[i] / 1000:g} km -> {a_d[i + 1] / 1000:g} km"
            for i in range(len(a_d) - 1)
            if a_t[i + 1] <= a_t[i] or a_d[i + 1] <= a_d[i]
        ]
        die(
            "heures d'ancres non croissantes avec le kilomètre ("
            + ", ".join(bad)
            + ") : "
            "vérifie ANCHORS, DATE et le début du FIT."
        )

    t_wp = np.interp(dist, a_d, a_t)  # heure à chaque point de passage (linéaire)

    n = int(np.floor((t_end - a_t[0]) / step + 1e-9)) + 1
    ts = a_t[0] + step * np.arange(n)
    ds = np.interp(ts, t_wp, dist)
    es = interp_valid(ds, dist, ele)
    off = fit["ele"][0] - ele[-1]
    if ELEV_BLEND_KM > 0 and np.isfinite(off):
        ramp = np.clip(
            (ds - (dist[-1] - ELEV_BLEND_KM * 1000.0)) / (ELEV_BLEND_KM * 1000.0), 0, 1
        )
        es = es + off * ramp
        print(
            f"Altitude : écart trace/FIT de {off:+.1f} m au raccord, lissé sur les {ELEV_BLEND_KM:g} derniers km."
        )
    nan = np.full(n, np.nan)
    rec = {
        "t": ts,
        "lat": np.interp(ds, dist, lat),
        "lon": np.interp(ds, dist, lon),
        "ele": es,
        "dist": ds,
        "speed": nan.copy(),  # pas de vitesse prédite
        "hr": nan.copy(),
        "cad": nan.copy(),
        "temp": nan.copy(),
        "pwr": nan.copy(),
    }

    print_coherence(a_d, a_t)
    return rec


def print_coherence(a_d, a_t):
    print("Cohérence entre ancres (la dernière = début du FIT) :")
    print("  tronçon (km)      durée      allure moy.")
    for i in range(len(a_d) - 1):
        dk = (a_d[i + 1] - a_d[i]) / 1000.0
        dt_ = a_t[i + 1] - a_t[i]
        print(
            f"  {a_d[i] / 1000:5.1f} -> {a_d[i + 1] / 1000:5.1f}   {fmt_hms(dt_)}   "
            f"{fmt_pace(dt_ / dk):>10}"
        )


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def merge_records(miss: dict, fit: dict) -> dict:
    rec = {
        k: np.concatenate([miss[k], fit[k]])
        for k in ("t", "lat", "lon", "ele", "speed", *FIELDS)
    }
    rec["dist"] = np.concatenate([miss["dist"], fit["dist"] + miss["dist"][-1]])
    rec["synthetic"] = np.r_[
        np.ones(len(miss["t"]), bool), np.zeros(len(fit["t"]), bool)
    ]
    rec["block"] = np.r_[0, np.cumsum(np.diff(rec["t"]) > MAX_GAP_S)]
    return rec


# --------------------------------------------------------------------------- #
# Écriture FIT
# --------------------------------------------------------------------------- #
def ascent_descent(ele: np.ndarray, thr: float = 3.0):
    e = ele[~np.isnan(ele)]
    if len(e) < 2:
        return None, None
    e = moving_avg(e, 15)  # le bruit d'altitude GPS gonfle sinon le D+
    gain = loss = 0.0
    ref = e[0]
    for x in e[1:]:
        if x - ref >= thr:
            gain += x - ref
            ref = x
        elif ref - x >= thr:
            loss += ref - x
            ref = x
    return gain, loss


def ms(x: float) -> int:
    return int(round(x * 1000))


def build_fit(rec: dict, sport_key: str, out_path: str) -> dict:
    sport, sub_sport = SPORTS[sport_key]
    t = rec["t"]
    t0, t1 = float(t[0]), float(t[-1])
    blocks = rec["block"]
    timer = sum(
        float(t[blocks == b][-1] - t[blocks == b][0]) for b in np.unique(blocks)
    )
    elapsed = t1 - t0
    dist_total = float(rec["dist"][-1] - rec["dist"][0])

    builder = FitFileBuilder(auto_define=True, min_string_size=50)
    fid = FileIdMessage()
    fid.type = FileType.ACTIVITY
    fid.manufacturer = Manufacturer.DEVELOPMENT.value
    fid.product = 0
    fid.time_created = ms(t0)
    fid.serial_number = 0x12345678
    builder.add(fid)

    def event(ts, etype):
        e = EventMessage()
        e.event = Event.TIMER
        e.event_type = etype
        e.event_group = 0
        e.timestamp = ms(ts)
        return e

    last_block = None
    for i in range(len(t)):
        if blocks[i] != last_block:
            if last_block is not None:
                builder.add(event(t[i - 1], EventType.STOP_ALL))
            builder.add(event(t[i], EventType.START))
            last_block = blocks[i]
        r = RecordMessage()
        r.timestamp = ms(t[i])
        r.position_lat = float(rec["lat"][i])
        r.position_long = float(rec["lon"][i])
        r.distance = float(rec["dist"][i] - rec["dist"][0])
        if not np.isnan(rec["speed"][i]):
            r.speed = float(rec["speed"][i])
        if not np.isnan(rec["ele"][i]):
            r.altitude = float(rec["ele"][i])
        if not np.isnan(rec["hr"][i]):
            r.heart_rate = int(np.clip(round(rec["hr"][i]), 1, 255))
        if not np.isnan(rec["cad"][i]):
            r.cadence = int(np.clip(round(rec["cad"][i]), 0, 255))
        if not np.isnan(rec["temp"][i]):
            r.temperature = int(np.clip(round(rec["temp"][i]), -127, 127))
        if not np.isnan(rec["pwr"][i]):
            r.power = int(np.clip(round(rec["pwr"][i]), 0, 65535))
        builder.add(r)
    builder.add(event(t[-1], EventType.STOP_ALL))

    gain, loss = ascent_descent(rec["ele"])
    hr_valid = rec["hr"][~np.isnan(rec["hr"])]
    cad_valid = rec["cad"][~np.isnan(rec["cad"])]

    lap = LapMessage()
    lap.message_index = 0
    lap.timestamp = ms(t1)
    lap.start_time = ms(t0)
    lap.total_elapsed_time = elapsed
    lap.total_timer_time = timer
    lap.total_distance = dist_total
    lap.sport = sport
    lap.sub_sport = sub_sport
    lap.event = Event.LAP
    lap.event_type = EventType.STOP
    builder.add(lap)

    ses = SessionMessage()
    ses.message_index = 0
    ses.timestamp = ms(t1)
    ses.start_time = ms(t0)
    ses.total_elapsed_time = elapsed
    ses.total_timer_time = timer
    ses.total_distance = dist_total
    ses.sport = sport
    ses.sub_sport = sub_sport
    ses.first_lap_index = 0
    ses.num_laps = 1
    ses.event = Event.SESSION
    ses.event_type = EventType.STOP
    ses.avg_speed = dist_total / timer if timer > 0 else 0.0
    if not np.isnan(rec["speed"]).all():
        ses.max_speed = float(np.nanmax(rec["speed"]))
    if gain is not None:
        ses.total_ascent = int(round(gain))
        ses.total_descent = int(round(loss))
    if len(hr_valid):
        ses.avg_heart_rate = int(round(hr_valid.mean()))
        ses.max_heart_rate = int(round(hr_valid.max()))
    if len(cad_valid):
        ses.avg_cadence = int(round(cad_valid.mean()))
        ses.max_cadence = int(round(cad_valid.max()))
    builder.add(ses)

    act = ActivityMessage()
    act.timestamp = ms(t1)
    act.total_timer_time = timer
    act.num_sessions = 1
    act.type = Activity.MANUAL
    act.event = Event.ACTIVITY
    act.event_type = EventType.STOP
    builder.add(act)

    builder.build().to_file(out_path)
    return {
        "elapsed": elapsed,
        "timer": timer,
        "dist": dist_total,
        "gain": gain,
        "loss": loss,
    }


def verify(path: str):
    ff = FitFile.from_file(path)
    msgs = [r.message for r in ff.records]
    n_rec = sum(isinstance(m, RecordMessage) for m in msgs)
    n_ses = sum(isinstance(m, SessionMessage) for m in msgs)
    print(f"Relecture OK : {n_rec} enregistrements, {n_ses} session(s).")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    fit = load_fit(FIT_END)
    tr = load_trace(GPX_TRACE)
    t0 = datetime.fromtimestamp(fit["t"][0], ZoneInfo(TZ))
    print(
        f"FIT : {len(fit['t'])} enregistrements, pas d'échantillonnage {fit['step']:g} s, "
        f"début {t0:%Y-%m-%d %H:%M:%S}, {fit['dist'][-1] / 1000:.2f} km, sport {fit['sport']}."
    )
    if fit["gaps"]:
        print(
            f"FIT : {fit['gaps']} pause(s) > {MAX_GAP_S:g} s détectée(s), conservées."
        )

    idx = find_join(tr, (fit["lat"][0], fit["lon"][0]))
    miss = build_missing(tr, idx, fit)
    rec = merge_records(miss, fit)

    sport_key = SPORT or fit["sport"]

    stats = build_fit(rec, sport_key, OUTPUT_FIT)

    if EXPORT_CSV:
        speed = rec["speed"]
        pace = np.where(speed > 0.3, 1000.0 / np.maximum(speed, 1e-9), np.nan)
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(rec["t"], unit="s", utc=True),
                "elapsed_s": rec["t"] - rec["t"][0],
                "lat": rec["lat"],
                "lon": rec["lon"],
                "ele": rec["ele"],
                "distance_m": rec["dist"] - rec["dist"][0],
                "speed_ms": speed,
                "pace_s_per_km": pace,
                "heart_rate": rec["hr"],
                "cadence": rec["cad"],
                "temperature": rec["temp"],
                "power": rec["pwr"],
                "synthetic": rec["synthetic"],
            }
        ).to_csv(EXPORT_CSV, index=False)

    share = 100 * rec["synthetic"].mean()
    print(f"Fichier écrit : {OUTPUT_FIT}")
    print(
        f"  Distance : {stats['dist'] / 1000:.2f} km | durée écoulée : {fmt_hms(stats['elapsed'])} "
        f"| durée en mouvement : {fmt_hms(stats['timer'])} "
        f"| D+ : {'n/a' if stats['gain'] is None else round(stats['gain'])} m"
    )
    print(f"  Enregistrements reconstitués : {share:.0f} %")
    verify(OUTPUT_FIT)
    print(
        "Pense à indiquer dans la description Strava qu'une partie de la trace est reconstituée."
    )


if __name__ == "__main__":
    main()
