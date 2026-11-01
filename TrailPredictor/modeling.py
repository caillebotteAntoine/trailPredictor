# %%
# %% [markdown]
# # Modèle de prédiction de l'allure en trail (sklearn)
#
# Script autonome, sans CLI : à coller dans un notebook (chaque `# %%` = une cellule)
# ou à exécuter tel quel. Modifier uniquement la section CONFIG.

# %% CONFIG
DATA_PATH = "dataset.csv"  # chemin du CSV
N_SPLITS = 5  # folds de la validation croisée (groupée par activité)
FINAL_MODEL = (
    None  # None = meilleur MAE allure ; sinon "ridge" / "random_forest" / "hist_gbm"
)
RANDOM_STATE = 42

# %% IMPORTS
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_models() -> dict:
    # Le scaler est dans le Pipeline : moyenne/écart-type calculés sur le train
    # de chaque fold uniquement (équivalent de ton scale.values en R, sans fuite).
    return {
        "ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-2, 3, 30))),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=5,
                        max_features=0.5,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gbm": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=300,
                        max_depth=4,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# %% VALIDATION CROISÉE GROUPÉE PAR ACTIVITÉ
cv = GroupKFold(n_splits=N_SPLITS)
models = make_models()
oof, rows = {}, []
pace_true = 1000.0 / y

for name, pipe in models.items():
    pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv)
    oof[name] = pred
    pace_pred = 1000.0 / np.clip(pred, 0.3, None)
    tot = pd.DataFrame({"g": groups, "real": t, "pred": pace_pred}).groupby("g").sum()
    err = (tot["pred"] - tot["real"]) / tot["real"] * 100
    rows.append(
        {
            "modèle": name,
            "R² vitesse": r2_score(y, pred),
            "MAE vitesse (m/s)": mean_absolute_error(y, pred),
            "MAE allure (s/km)": mean_absolute_error(pace_true, pace_pred),
            "MAE temps total (%)": err.abs().mean(),
            "biais temps total (%)": err.mean(),
        }
    )

metrics = pd.DataFrame(rows).set_index("modèle")
display_fn = globals().get("display", print)
display_fn(metrics.round(3))

best = FINAL_MODEL or metrics["MAE allure (s/km)"].idxmin()
print(f"Modèle retenu : {best}")
pred_best = oof[best]

# %% FIG 2 — comparaison des modèles
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a, col in zip(ax, ["MAE allure (s/km)", "MAE temps total (%)", "R² vitesse"]):
    metrics[col].plot.bar(ax=a, color="#4c78a8", rot=0)
    a.set_title(col)
plt.tight_layout()
plt.show()

# %% FIG 3 — observé vs prédit (hors-échantillon) + résidus
pace_pred_best = 1000.0 / np.clip(pred_best, 0.3, None)
res_s = pace_pred_best - pace_true  # s/km, >0 = prédit trop lent

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].scatter(pace_true / 60, pace_pred_best / 60, s=8, alpha=0.4)
lim = [3, 20]
ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set(
    xlim=lim,
    ylim=lim,
    title=f"Allure observée vs prédite ({best}, hors-échantillon)",
    xlabel="observée (min/km)",
    ylabel="prédite (min/km)",
)
ax[1].hist(res_s, bins=50, color="#54a24b")
ax[1].axvline(0, color="k", lw=1)
ax[1].set(title="Résidus d'allure", xlabel="prédit − observé (s/km)", ylabel="nb de km")
ax[2].scatter(X["drop_net"], res_s, s=8, alpha=0.4, color="#e45756")
ax[2].axhline(0, color="k", lw=1)
ax[2].set(
    title="Résidus vs pente nette",
    xlabel="D+ − D- (m)",
    ylabel="prédit − observé (s/km)",
)
plt.tight_layout()
plt.show()

# %% FIG 4 — erreur sur le temps total par activité
tot = pd.DataFrame({"g": groups, "real": t, "pred": pace_pred_best}).groupby("g").sum()
tot["km"] = groups.groupby(groups).size()
tot["err_pct"] = (tot["pred"] - tot["real"]) / tot["real"] * 100

fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
ax[0].scatter(tot["real"] / 3600, tot["pred"] / 3600, s=tot["km"] * 2, alpha=0.7)
m = [0, tot[["real", "pred"]].max().max() / 3600 * 1.05]
ax[0].plot(m, m, "k--", lw=1)
ax[0].set(
    title="Temps total : observé vs prédit (taille = distance)",
    xlabel="observé (h)",
    ylabel="prédit (h)",
)
ax[1].bar(
    tot.index.astype(str),
    tot["err_pct"],
    color=np.where(tot["err_pct"] > 0, "#e45756", "#4c78a8"),
)
ax[1].axhline(0, color="k", lw=1)
ax[1].set(
    title="Erreur sur le temps total par activité (+ = prédit trop lent)",
    xlabel="activity_id",
    ylabel="%",
)
plt.tight_layout()
plt.show()
display_fn(tot[["km", "err_pct"]].round(1).T)

# %% FIG 5 — profils de quelques activités : allure observée vs prédite + altitude
n_show = 4
# les activités les plus longues, représentatives des courses visées
show_ids = tot.sort_values("km", ascending=False).index[:n_show]
fig, axes = plt.subplots(n_show, 1, figsize=(13, 3.3 * n_show), sharex=False)
for a, aid in zip(np.atleast_1d(axes), show_ids):
    m_ = (groups == aid).values
    km = meta.loc[m_, "segment"] + 1
    a.plot(km, pace_true[m_] / 60, color="k", lw=1.2, label="observée")
    a.plot(km, pace_pred_best[m_] / 60, color="#e45756", lw=1.2, label="prédite")
    a.set(ylabel="allure (min/km)", title=f"Activité {aid} — {len(km)} km")
    a.set_ylim(top=min(20, a.get_ylim()[1]))
    a2 = a.twinx()
    a2.fill_between(km, meta.loc[m_, "altitude"], color="grey", alpha=0.2)
    a2.set_ylabel("altitude (m)")
    a2.grid(False)
    a.legend(loc="upper left")
axes[-1].set_xlabel("km")
plt.tight_layout()
plt.show()

# %% FIG 6 — importance des variables (permutation, hors-échantillon)
imps = []
for tr, te in cv.split(X, y, groups):
    mdl = make_models()[best].fit(X.iloc[tr], y.iloc[tr])
    r = permutation_importance(
        mdl,
        X.iloc[te],
        y.iloc[te],
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
    )
    imps.append(r.importances_mean)
imp = pd.Series(np.mean(imps, axis=0), index=X.columns).sort_values()

fig, ax = plt.subplots(figsize=(8, 7))
imp.tail(20).plot.barh(ax=ax, color="#4c78a8")
ax.set(
    title=f"Importance par permutation ({best}, hors-échantillon)",
    xlabel="hausse de MAE vitesse (m/s) si la variable est mélangée",
)
plt.tight_layout()
plt.show()

# %% MODÈLE FINAL (entraîné sur tout le jeu)
final = make_models()[best].fit(X, y)

# %% FIG 7 — coefficients Ridge (variables standardisées, comparables entre elles)
ridge = make_models()["ridge"].fit(X, y)
coefs = pd.Series(ridge.named_steps["model"].coef_, index=X.columns).sort_values()
top = pd.concat([coefs.head(10), coefs.tail(10)])
fig, ax = plt.subplots(figsize=(8, 6))
top.plot.barh(ax=ax, color=np.where(top > 0, "#54a24b", "#e45756"))
ax.set(
    title=f"Coefficients Ridge (alpha={ridge.named_steps['model'].alpha_:.2g})",
    xlabel="effet sur la vitesse (m/s) pour +1 sd",
)
plt.tight_layout()
plt.show()

# %% FIG 8 — dépendance partielle des principales variables
top_feats = list(imp.sort_values(ascending=False).index[:4])
fig, ax = plt.subplots(1, len(top_feats), figsize=(4.2 * len(top_feats), 3.6))
PartialDependenceDisplay.from_estimator(final, X, top_feats, ax=ax, grid_resolution=30)
fig.suptitle(f"Dépendance partielle de la vitesse (m/s) — {best}", y=1.03)
plt.tight_layout()
plt.show()


# %% PRÉDICTION SUR UN PARCOURS FUTUR
def predict_course(profile: pd.DataFrame, model=final) -> pd.DataFrame:
    """profile : une ligne par km avec 'segment' (0..n-1), 'drop_pos', 'drop_neg'
    (+ 'ravito' 0/1 si le modèle a été entraîné avec)."""
    Xp = build_features(profile).reindex(columns=X.columns, fill_value=0)
    speed = np.clip(model.predict(Xp), 0.3, None)
    res = profile[["segment"]].copy()
    res["km"] = res["segment"] + 1
    res["speed_ms"] = speed
    res["allure"] = [pace_str(s) for s in speed]
    res["time_s"] = SEGMENT_M / speed
    res["time_cum_h"] = res["time_s"].cumsum() / 3600
    return res


# Exemple : parcours fictif de 55 km — à remplacer par le profil D+/D- par km de ta course
rng = np.random.default_rng(0)
n_km = 55
base = 40 + 35 * np.sin(np.linspace(0, 6 * np.pi, n_km)) ** 2
demo = pd.DataFrame(
    {
        "segment": np.arange(n_km),
        "drop_pos": np.clip(base + rng.normal(0, 8, n_km), 0, None).round(),
        "drop_neg": np.clip(base[::-1] * 0.9 + rng.normal(0, 8, n_km), 0, None).round(),
    }
)
plan = predict_course(demo)

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(plan["km"], 1000 / plan["speed_ms"] / 60, color="#e45756")
ax.set(
    xlabel="km",
    ylabel="allure prédite (min/km)",
    title=f"Parcours fictif {n_km} km — temps prédit : {plan['time_cum_h'].iloc[-1]:.2f} h",
)
ax2 = ax.twinx()
ax2.bar(plan["km"], demo["drop_pos"], color="#54a24b", alpha=0.35, label="D+")
ax2.bar(plan["km"], -demo["drop_neg"], color="#4c78a8", alpha=0.35, label="D-")
ax2.set_ylabel("dénivelé par km (m)")
ax2.grid(False)
ax2.legend(loc="upper right")
plt.tight_layout()
plt.show()
display_fn(plan.head(10).round(2))
