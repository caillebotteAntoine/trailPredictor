"""Loading of the Strava activity summary export."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
ACTIVITIES_FILE = DATA_DIR / "activities.csv"

activity_type_map = {
    "Course à pied": "run",
    "Entraînement aux poids": "weight_training",
    "Marche": "walk",
    "Randonnée": "hike",
    "Natation": "swim",
    "Entraînement": "workout",
    "Vélo": "ride",
    "Canoë": "canoeing",
    "Yoga": "yoga",
    "Windsurf": "windsurf",
    "Course à pied virtuelle": "virtual_run",
}

activity_column_map = {
    "ID de l'activité": "id",
    "Date de l'activité": "date",
    "Nom de l'activité": "name",
    "Type d'activité": "type",
    "Description de l'activité": "description",
    "Temps écoulé": "elapsed_time",
    "Distance": "distance_km",
    "Fréquence cardiaque max.": "max_heart_rate",
    "Effort relatif": "relative_effort",
    "Déplacement-transport": "is_commute",
    "Note privée sur les activités": "private_note",
    "Matériel utilisé pour l'activité": "gear",
    "Nom du fichier": "file_name",
    "Poids de l'athlète": "athlete_weight",
    "Poids du vélo": "bike_weight",
    "Temps écoulé.1": "elapsed_time_1",
    "Durée de déplacement": "moving_time",
    "Distance.1": "distance_m",
    "Vitesse max.": "max_speed",
    "Vitesse moyenne": "avg_speed",
    "Dénivelé positif": "elevation_gain",
    "Dénivelé négatif": "elevation_loss",
    "Altitude min.": "min_altitude",
    "Altitude max.": "max_altitude",
    "Pente max.": "max_grade",
    "Pente moyenne": "avg_grade",
    "Pente positive moyenne": "avg_uphill_grade",
    "Pente négative moyenne": "avg_downhill_grade",
    "Cadence max.": "max_cadence",
    "Cadence moyenne": "avg_cadence",
    "Fréquence cardiaque max..1": "max_heart_rate_1",
    "Fréquence cardiaque moyenne": "avg_heart_rate",
    "Puissance max.": "max_power",
    "Puissance moyenne": "avg_power",
    "Calories": "calories",
    "Température max.": "max_temperature",
    "Température moyenne": "avg_temperature",
    "Effort relatif.1": "relative_effort_1",
    "Effort total": "total_effort",
    "Nombre de sorties course à pied": "run_count",
    "Temps de montée": "ascent_time",
    "Temps de descente": "descent_time",
    "Autres temps": "other_time",
    "Effort ressenti": "perceived_effort",
    "Type": "type_na",
    "Heure de début": "start_time",
    "Puissance moyenne pondérée": "weighted_avg_power",
    "Nombre d'échantillons de puissance": "power_sample_count",
    "Utiliser l'Effort ressenti": "use_perceived_effort",
    "Effort relatif ressenti": "perceived_relative_effort",
    "Déplacement-transport.1": "is_commute_1",
    "Poids total soulevé": "total_weight_lifted",
    "À partir du téléchargement": "from_upload",
    "Distance ajustée selon la pente": "grade_adjusted_distance",
    "Heure d'observation de la météo": "weather_observation_time",
    "Conditions météo": "weather_conditions",
    "Température selon les prévisions météo": "forecast_temperature",
    "Température ressentie": "feels_like_temperature",
    "Point de rosée": "dew_point",
    "Humidité": "humidity",
    "Pression atmosphérique": "atmospheric_pressure",
    "Vitesse du vent": "wind_speed",
    "Rafale de vent": "wind_gust",
    "Direction du vent": "wind_direction",
    "Intensité des précipitations": "precipitation_intensity",
    "Heure de lever du soleil": "sunrise_time",
    "Heure de coucher du soleil": "sunset_time",
    "Phase de la lune": "moon_phase",
    "Vélo": "bike",
    "Matériel": "gear",
    "Probabilité de précipitations": "precipitation_probability",
    "Type de précipitations": "precipitation_type",
    "Couverture nuageuse": "cloud_cover",
    "Visibilité selon les prévisions météo": "forecast_visibility",
    "Indice UV": "uv_index",
    "Ozone selon les prévisions météo": "forecast_ozone",
    "Nombre de sauts": "jump_count",
    "Grit total": "total_grit",
    "Flow moyen": "avg_flow",
    "Signalé": "flagged",
    "Vitesse moyenne (temps écoulé)": "avg_speed_elapsed",
    "Distance sur chemin": "trail_distance",
    "Distance récemment découverte": "newly_discovered_distance",
    "Distance sur chemin récemment découverte": "newly_discovered_trail_distance",
    "Nombre d'activités": "count",
    "Nombre total de pas": "total_steps",
    "CO2 économisé": "co2_saved",
    "Longueur de piscine": "pool_length",
    "Charge d’entraînement": "training_load",
    "Intensité": "intensity",
    "Vitesse moyenne ajustée selon la pente": "grade_adjusted_avg_speed",
    "Temps enregistré par le chronomètre": "timer_time",
    "Nombre total de cycles": "total_cycles",
    "Récupération": "is_recovery",
    "Avec mon animal de compagnie": "with_pet",
    "Compétition": "is_race",
    "Sortie longue": "is_long_run",
    "Pour la bonne cause": "is_charity",
    "Avec enfant": "with_child",
    "Distance en descente": "downhill_distance",
    "Nombre total de séries": "total_sets",
    "Nombre total de répétitions": "total_reps",
    "Support": "support",
}

# only float and int columns are mapped, str columns are not mapped
activity_dtypes_map = {
    "id": int,
    "elapsed_time": float,
    "distance_km": float,
    "distance_m": float,
    "moving_time": float,
    "max_heart_rate": float,
    "relative_effort": float,
    "is_commute": bool,
    "athlete_weight": float,
    "bike_weight": float,
}


def load_activities() -> pd.DataFrame:
    """Load the Strava activity summary.

    The French column names and activity types of the export are mapped to
    English identifiers and the numeric columns are converted.

    Returns
    -------
    pandas.DataFrame
        One row per activity.

    Raises
    ------
    FileNotFoundError
        If ``ACTIVITIES_FILE`` does not exist.
    """
    activities = (
        pd.read_csv(ACTIVITIES_FILE, sep=",", decimal=".", encoding="utf-8")
        .rename(columns=activity_column_map)
        .reset_index(drop=True)
    )
    activities["type"] = activities["type"].replace(activity_type_map)

    # conversion
    for col, dtype in activity_dtypes_map.items():
        if dtype == "datetime64[ns]":
            activities[col] = pd.to_datetime(activities[col], errors="coerce")
        elif dtype in ["float64", float]:
            if activities[col].dtype == "str":
                activities[col] = (
                    activities[col]
                    .str.replace(",", ".")
                    .str.replace("\u202f", "", regex=False)
                )  # .astype(float)

            activities[col] = activities[col].astype(dtype)

        else:
            activities[col] = activities[col].astype(dtype)

    if activities["type_na"].isna().all():
        activities = activities.drop(columns=["type_na"])

    return activities
