// Alpine component of the dataset page.

import { fetchJson, watchJob } from "./api.js";
import {
  PLOTLY_CONFIG, ZONES_CHOICE_SETTING, errorMessage, loadSetting, nextFrame, saveSetting,
} from "./ui.js";

const REFERENCE_CUTS = "142, 161, 170, 183";
const TARGETS = [
  "pace", "log_pace", "speed", "log_speed",
  "accel_pace", "log_accel_pace", "accel_speed", "log_accel_speed",
];
const GROUP_LABELS = {
  past: "Passé (back)",
  upcoming: "À venir (front)",
  cumulative: "Cumuls",
  ratio: "Ratios",
  other: "Autres",
};
const PRESETS = [
  { value: "top_5", label: "Top 5 (|ρ|)" },
  { value: "top_10", label: "Top 10 (|ρ|)" },
  { value: "top_20", label: "Top 20 (|ρ|)" },
  { value: "positive", label: "Corrélées positivement" },
  { value: "negative", label: "Corrélées négativement" },
  { value: "all", label: "Toutes les features filtrées" },
];
const COLUMNS = [
  { field: "name", label: "Feature" },
  { field: "spearman", label: "ρ", help: "Corrélation de Spearman avec la vitesse (+ : plus rapide)." },
  { field: "missing_pct", label: "NaN %" },
  { field: "group", label: "Groupe" },
];
const DEFAULT_FILTERS = { search: "", groups: [], minAbs: 0, maxMissing: 100 };
const TARGET_CHART_ID = "chart-target";
const MANY_CHARTS = 12;

function chartId(feature) {
  // ids must be valid in CSS selectors: keep letters, digits, "-" and "_"
  return `chart-feature-${feature.replace(/[^\w-]/g, "_")}`;
}

function compareValues(a, b) {
  if (a === b) return 0;
  if (a === null) return 1; // missing values last
  if (b === null) return -1;
  return a < b ? -1 : 1;
}

document.addEventListener("alpine:init", () => {
  window.Alpine.data("datasetPage", () => ({
    targets: TARGETS,
    presets: PRESETS,
    columns: COLUMNS,
    manyCharts: MANY_CHARTS,
    params: { minDistance: 40, maxDistance: 60, segmentWidth: 1000 },
    target: "pace",
    savedZones: [],
    zonesChoice: "",
    typedCuts: REFERENCE_CUTS,
    cutsError: "",
    loaded: false,
    nRows: null,
    progress: null,
    progressMessage: "",
    error: "",
    filters: structuredClone(DEFAULT_FILTERS),
    rows: [],
    ranked: [],
    groups: [],
    nonNumeric: [],
    sort: { field: null, ascending: true },
    preset: "top_5",
    selection: [],
    selectionStarted: false,
    columnsPerRow: 1,
    chartErrors: {},
    chartLoading: {},
    // figure JSON by chart id, valid for `figuresKey`
    figures: {},
    figuresKey: "",
    loadToken: 0,
    featuresToken: 0,

    // cuts of the chosen saved optimisation, or null for the typed cuts
    get savedCuts() {
      if (this.zonesChoice === "") {
        return null;
      }
      return this.savedZones.find((zones) => zones.key === this.zonesChoice)?.cuts ?? [];
    },

    get cutsText() {
      return this.savedCuts === null ? this.typedCuts : this.savedCuts.join(",");
    },

    get chartEntries() {
      return [
        { id: TARGET_CHART_ID, feature: null, title: "Distribution de la cible" },
        ...this.selection.map((feature) => ({
          id: chartId(feature), feature, title: `Cible vs ${feature}`,
        })),
      ];
    },

    get sortedRows() {
      if (this.sort.field === null) {
        return this.rows; // the server order: decreasing |rho|
      }
      const sign = this.sort.ascending ? 1 : -1;
      return [...this.rows].sort(
        (a, b) => sign * compareValues(a[this.sort.field], b[this.sort.field]),
      );
    },

    groupLabel(group) {
      return GROUP_LABELS[group] ?? group;
    },

    zonesLabel(zones) {
      return `N=${zones.cuts.length + 1} · ${zones.target_name} · ${zones.dataset_label} · ${zones.config_key}`;
    },

    sortMarker(field) {
      if (this.sort.field !== field) return "";
      return this.sort.ascending ? "▲" : "▼";
    },

    sortBy(field) {
      if (this.sort.field === field) {
        this.sort.ascending = !this.sort.ascending;
      } else {
        this.sort = { field, ascending: field === "name" || field === "group" };
      }
    },

    paramsQuery() {
      return new URLSearchParams({
        min_distance: this.params.minDistance,
        max_distance: this.params.maxDistance,
        segment_width: this.params.segmentWidth,
        target: this.target,
      });
    },

    filtersQuery() {
      const query = this.paramsQuery();
      query.set("search", this.filters.search);
      query.set("min_abs", this.filters.minAbs);
      query.set("max_missing", this.filters.maxMissing);
      for (const group of this.filters.groups) {
        query.append("groups", group);
      }
      return query;
    },

    async init() {
      try {
        this.savedZones = await fetchJson("/api/dataset/saved-zones");
      } catch (error) {
        console.error(error); // the typed cuts still work
      }
      const saved = loadSetting(ZONES_CHOICE_SETTING, "");
      this.zonesChoice = this.savedZones.some((zones) => zones.key === saved) ? saved : "";
      await this.loadDataset();
    },

    async loadDataset() {
      const token = ++this.loadToken;
      this.error = "";
      this.loaded = false;
      this.progress = 0;
      this.progressMessage = "Démarrage…";
      try {
        const { job_id: jobId } = await fetchJson(
          `/api/dataset/load?${this.paramsQuery()}`, { method: "POST" },
        );
        await watchJob(jobId, (snapshot) => {
          if (token === this.loadToken) {
            this.progress = snapshot.progress;
            this.progressMessage = snapshot.message;
          }
        });
      } catch (error) {
        if (token === this.loadToken) {
          this.error = `Impossible de charger le dataset : ${errorMessage(error)}`;
          this.progress = null;
        }
        return;
      }
      if (token !== this.loadToken) {
        return;
      }
      this.progress = null;
      this.loaded = true;
      await this.refreshFeatures();
      await this.onZonesChange();
    },

    async refreshFeatures() {
      const token = ++this.featuresToken;
      try {
        const view = await fetchJson(`/api/dataset/features?${this.filtersQuery()}`);
        if (token !== this.featuresToken) {
          return;
        }
        this.nRows = view.n_rows;
        this.rows = view.rows;
        this.ranked = view.ranked;
        this.groups = view.groups;
        this.nonNumeric = view.non_numeric;
        // keep the features that still exist (the dataset may have changed);
        // the first time, start with the most correlated feature
        const kept = this.selection.filter((name) => view.ranked.includes(name));
        this.setSelection(this.selectionStarted ? kept : view.ranked.slice(0, 1));
        this.selectionStarted = true;
      } catch (error) {
        if (token === this.featuresToken) {
          this.error = `Impossible de lister les features : ${errorMessage(error)}`;
        }
      }
    },

    async onTargetChange() {
      await this.refreshFeatures();
      await this.onZonesChange();
    },

    async onZonesChange() {
      saveSetting(ZONES_CHOICE_SETTING, this.zonesChoice);
      if (!this.loaded) {
        return;
      }
      try {
        // the server parses and checks the cuts, then shares the dataset
        const cuts = this.savedCuts
          ?? await fetchJson(`/api/dataset/cuts?${new URLSearchParams({ cuts: this.typedCuts })}`);
        const selection = {
          min_distance: this.params.minDistance,
          max_distance: this.params.maxDistance,
          segment_width: this.params.segmentWidth,
          target: this.target,
          cuts,
        };
        await fetchJson("/api/dataset/current", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(selection),
        });
        this.cutsError = "";
      } catch (error) {
        this.cutsError = errorMessage(error);
        return;
      }
      await this.renderCharts();
    },

    resetFilters() {
      this.filters = structuredClone(DEFAULT_FILTERS);
      this.refreshFeatures();
    },

    async applyPreset(add) {
      const query = this.filtersQuery();
      query.set("preset", this.preset);
      try {
        const names = await fetchJson(`/api/dataset/preset?${query}`);
        this.setSelection(add ? [...this.selection, ...names] : names);
      } catch (error) {
        this.error = `Impossible d'appliquer le preset : ${errorMessage(error)}`;
      }
    },

    toggle(name, checked) {
      this.setSelection(
        checked ? [...this.selection, name] : this.selection.filter((n) => n !== name),
      );
    },

    setSelection(names) {
      this.selection = [...new Set(names)];
      this.renderCharts();
    },

    async renderCharts() {
      if (!this.loaded || this.cutsError) {
        return;
      }
      const key = `${this.paramsQuery()}|${this.cutsText}`;
      if (key !== this.figuresKey) {
        this.figures = {};
        this.figuresKey = key;
      }
      await this.$nextTick();
      await nextFrame();
      await Promise.all(this.chartEntries.map((chart) => this.renderChart(chart, key)));
    },

    async renderChart(chart, key) {
      const element = document.getElementById(chart.id);
      if (!element) {
        return; // removed from the selection meanwhile
      }
      if (this.figures[chart.id]) {
        if (!element.data) {
          window.Plotly.react(element, this.figures[chart.id].data, this.figures[chart.id].layout, PLOTLY_CONFIG);
        }
        return;
      }
      this.chartLoading[chart.id] = true;
      const query = this.paramsQuery();
      query.set("cuts", this.cutsText);
      if (chart.feature !== null) {
        query.set("feature", chart.feature);
      }
      try {
        const view = await fetchJson(`/api/dataset/chart?${query}`);
        if (key !== this.figuresKey) {
          return; // the dataset changed meanwhile
        }
        if (view.figure) {
          this.figures[chart.id] = view.figure;
          delete this.chartErrors[chart.id];
          const current = document.getElementById(chart.id);
          if (current) {
            window.Plotly.react(current, view.figure.data, view.figure.layout, PLOTLY_CONFIG);
          }
        } else {
          this.chartErrors[chart.id] = view.error;
        }
      } catch (error) {
        this.chartErrors[chart.id] = errorMessage(error);
      } finally {
        delete this.chartLoading[chart.id];
      }
    },

    async resizeCharts() {
      await this.$nextTick();
      await nextFrame();
      for (const chart of this.chartEntries) {
        const element = document.getElementById(chart.id);
        if (element?.data) {
          window.Plotly.Plots.resize(element);
        }
      }
    },
  }));
});
