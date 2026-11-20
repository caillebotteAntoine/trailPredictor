// Alpine component of the heart-rate zones page.

import { fetchJson, watchJob } from "./api.js";
import {
  PLOTLY_CONFIG, ZONES_CHOICE_SETTING, errorMessage, loadSetting, nextFrame, saveSetting,
} from "./ui.js";

const TARGETS = [
  "pace", "log_pace", "speed", "log_speed",
  "accel_pace", "log_accel_pace", "accel_speed", "log_accel_speed",
];
const RMSE_DECIMALS = 4;

function formatNumber(value) {
  return value === null || value === undefined ? "—" : value.toFixed(RMSE_DECIMALS);
}

document.addEventListener("alpine:init", () => {
  window.Alpine.data("heartRatePage", () => ({
    targets: TARGETS,
    contextLoaded: false,
    dataset: null,
    limits: null,
    form: null,
    mode: "load",
    saved: [],
    selectedKey: "",
    result: null,
    loadingResult: false,
    trialsN: null,
    trialsError: "",
    zonesChoice: "",
    running: false,
    progress: null,
    progressMessage: "",
    notice: "",
    error: "",
    resultToken: 0,

    get selectedSaved() {
      return this.saved.find((result) => result.key === this.selectedKey) ?? null;
    },

    get datasetCaption() {
      const { selection, n_rows: rows } = this.dataset;
      return `${selection.min_distance}–${selection.max_distance} km · segments de `
        + `${selection.segment_width} m · ${rows} lignes (page « Dataset / Features »)`;
    },

    get zoneCountChoices() {
      const { min_zones: low, max_zones: high } = this.limits;
      return Array.from({ length: high - low + 1 }, (_, i) => low + i);
    },

    get metricCards() {
      const result = this.result;
      const best = result.summary.find((row) => row.n_zones === result.best_n_zones);
      const chosen = result.summary.find((row) => row.chosen);
      const cards = [
        { label: "N retenu (1-SE)", value: String(result.n_zones) },
        { label: "Meilleur N", value: String(result.best_n_zones) },
        { label: "Meilleur RMSE", value: this.formatRmse(best.rmse, best.rmse_se) },
        { label: "Seuil 1-SE", value: formatNumber(result.threshold) },
      ];
      if (result.reference) {
        const [rmse, se] = result.reference;
        const delta = chosen.rmse - rmse;
        cards.push({
          label: "RMSE référence",
          value: this.formatRmse(rmse, se),
          delta: `${this.signed(delta)} (N retenu)`,
          // a lower RMSE than the reference is better
          deltaClass: delta <= 0 ? "better" : "worse",
        });
      }
      return cards;
    },

    get zonesText() {
      return this.result.zones.map((zone) => `${zone.label} ${zone.low}–${zone.high}`).join(" · ");
    },

    formatRmse(rmse, se) {
      return `${formatNumber(rmse)} ± ${formatNumber(se)}`;
    },

    signed(value) {
      if (value === null || value === undefined) return "—";
      return `${value >= 0 ? "+" : ""}${value.toFixed(RMSE_DECIMALS)}`;
    },

    resultLabel(result) {
      const counts = result.config.zone_counts.join(", ");
      return `${result.dataset_label} — ${result.target_name} — N=${counts} — `
        + `${result.config.n_trials} essais — ${result.config_key}`;
    },

    async init() {
      this.zonesChoice = loadSetting(ZONES_CHOICE_SETTING, "");
      try {
        const context = await fetchJson("/api/heart-rate/context");
        this.dataset = context.dataset;
        this.limits = context.limits;
        this.form = {
          ...structuredClone(context.defaults),
          target: context.dataset?.selection.target ?? TARGETS[0],
        };
        this.contextLoaded = true;
        await this.refreshSaved();
      } catch (error) {
        this.error = `Impossible de charger la page : ${errorMessage(error)}`;
        return;
      }
      if (this.saved.length) {
        this.selectedKey = this.saved[0].key;
        await this.showResult();
      }
    },

    async refreshSaved() {
      this.saved = await fetchJson("/api/heart-rate/saved");
    },

    async showResult(key = this.selectedKey) {
      const token = ++this.resultToken;
      this.error = "";
      this.result = null;
      this.loadingResult = true;
      try {
        const result = await fetchJson(`/api/heart-rate/result?${new URLSearchParams({ key })}`);
        if (token !== this.resultToken) return;
        this.result = result;
        this.trialsN = result.zone_counts[0] ?? null;
      } catch (error) {
        if (token === this.resultToken) {
          this.error = `Impossible de charger le modèle : ${errorMessage(error)}`;
        }
        return;
      } finally {
        if (token === this.resultToken) this.loadingResult = false;
      }
      if (!this.result.has_completed_trials) return;
      await this.$nextTick(); // the chart containers are created by x-if
      await nextFrame();
      const chart = this.result.best_zones_chart;
      if (chart?.figure) {
        window.Plotly.react("chart-best-zones", chart.figure.data, chart.figure.layout, PLOTLY_CONFIG);
      }
      await this.renderTrialsChart();
    },

    async renderTrialsChart() {
      const key = this.result.saved.key;
      const query = new URLSearchParams({ key, n_zones: this.trialsN });
      try {
        const chart = await fetchJson(`/api/heart-rate/trials-chart?${query}`);
        if (this.result?.saved.key !== key) return;
        this.trialsError = chart.error ?? "";
        const element = document.getElementById("chart-trials");
        if (chart.figure) {
          window.Plotly.react(element, chart.figure.data, chart.figure.layout, PLOTLY_CONFIG);
        } else {
          window.Plotly.purge(element);
        }
      } catch (error) {
        this.trialsError = errorMessage(error);
      }
    },

    useZones() {
      this.zonesChoice = this.result.saved.key;
      saveSetting(ZONES_CHOICE_SETTING, this.zonesChoice);
    },

    async optimize() {
      this.error = "";
      this.notice = "";
      const { target, ...config } = this.form;
      config.zone_counts = [...config.zone_counts].sort((a, b) => a - b);
      this.running = true;
      try {
        const started = await fetchJson("/api/heart-rate/optimize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ target, config }),
        });
        if (!started.already_saved) {
          this.progress = 0;
          this.progressMessage = "Démarrage…";
        }
        await watchJob(started.job_id, (snapshot) => {
          if (!started.already_saved) {
            this.progress = snapshot.progress;
            this.progressMessage = snapshot.message;
          }
        });
        this.notice = started.already_saved
          ? "Résultat déjà sauvegardé pour cette configuration : rechargé."
          : "Optimisation terminée et sauvegardée.";
        await this.refreshSaved();
        this.selectedKey = started.key;
        await this.showResult(started.key);
      } catch (error) {
        this.error = `Impossible de lancer l'optimisation : ${errorMessage(error)}`;
      } finally {
        this.running = false;
        this.progress = null;
      }
    },
  }));
});
