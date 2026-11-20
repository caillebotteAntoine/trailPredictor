// Alpine component of the activity page.

import { fetchJson, watchJob } from "./api.js";
import { PLOTLY_CONFIG, errorMessage as apiError, nextFrame } from "./ui.js";

const CHARTS = [
  { name: "altitude", title: "Altitude" },
  { name: "heart_rate", title: "Fréquence cardiaque" },
  { name: "pace", title: "Allure par segment" },
  { name: "heart_rate_zones", title: "Zones cardiaques" },
];
const SECONDS_PER_HOUR = 3600;
const SECONDS_PER_MINUTE = 60;

function pad2(value) {
  return String(value).padStart(2, "0");
}

function formatDuration(seconds) {
  const total = Math.round(seconds);
  const hours = Math.floor(total / SECONDS_PER_HOUR);
  const minutes = Math.floor((total % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE);
  return `${hours}:${pad2(minutes)}:${pad2(total % SECONDS_PER_MINUTE)}`;
}

function formatPace(minutesPerKm) {
  const seconds = Math.round(minutesPerKm * SECONDS_PER_MINUTE);
  return `${Math.floor(seconds / SECONDS_PER_MINUTE)}:${pad2(seconds % SECONDS_PER_MINUTE)} min/km`;
}

const METRIC_FORMATS = [
  { field: "distance_km", label: "Distance", format: (v) => `${v.toFixed(2)} km` },
  { field: "duration_s", label: "Durée", format: formatDuration },
  { field: "elevation_gain_m", label: "Dénivelé positif", format: (v) => `${v.toFixed(0)} m` },
  { field: "average_pace_min_per_km", label: "Allure moyenne", format: formatPace },
  { field: "average_heart_rate_bpm", label: "FC moyenne", format: (v) => `${v.toFixed(0)} bpm` },
];

document.addEventListener("alpine:init", () => {
  window.Alpine.data("activityPage", () => ({
    activities: [],
    order: "desc",
    activityId: null,
    options: {
      segmentWidth: 100,
      dropThreshold: 2.0,
      colorBy: "altitude",
      xAxis: "distance",
      texture: "osm",
      showKmMarkers: true,
    },
    charts: CHARTS,
    progress: null,
    progressMessage: "",
    error: "",
    metrics: null,
    chartErrors: {},
    track: null,
    trackError: "",
    // incremented at each request: the answers of outdated requests are ignored
    requestToken: 0, // loading of an activity
    chartsToken: 0,
    trackToken: 0,
    deck: null,
    deckActivityId: null,

    get sortedActivities() {
      const sorted = [...this.activities].sort(
        (a, b) => (a.distance_km ?? -Infinity) - (b.distance_km ?? -Infinity),
      );
      return this.order === "asc" ? sorted : sorted.reverse();
    },

    get metricCards() {
      if (!this.metrics) {
        return [];
      }
      return METRIC_FORMATS.filter(({ field }) => this.metrics[field] !== null).map(
        ({ field, label, format }) => ({ label, value: format(this.metrics[field]) }),
      );
    },

    activityLabel(activity) {
      const distance = activity.distance_km === null ? "—" : activity.distance_km.toFixed(1);
      const date = activity.date === null ? "date inconnue" : activity.date.slice(0, 10);
      return `${distance} km — ${date} — ${activity.name}`;
    },

    query(extra = {}) {
      const params = new URLSearchParams({
        segment_width: this.options.segmentWidth,
        drop_threshold: this.options.dropThreshold,
        ...extra,
      });
      return params.toString();
    },

    async init() {
      this.$watch("order", () => {
        this.activityId = this.sortedActivities[0]?.id ?? null;
        this.loadActivity();
      });
      try {
        this.activities = await fetchJson("/api/activities");
      } catch (error) {
        this.error = `Impossible de lister les activités : ${apiError(error)}`;
        return;
      }
      if (!this.activities.length) {
        this.error = "Aucune activité de course avec un fichier FIT disponible.";
        return;
      }
      this.activityId = this.sortedActivities[0].id;
      await this.loadActivity();
    },

    async loadActivity() {
      if (this.activityId === null) {
        return;
      }
      const token = ++this.requestToken;
      this.error = "";
      this.progress = 0;
      this.progressMessage = "Démarrage…";
      try {
        const { job_id: jobId } = await fetchJson(
          `/api/activities/${this.activityId}/load?${this.query()}`,
          { method: "POST" },
        );
        await watchJob(jobId, (snapshot) => {
          if (token === this.requestToken) {
            this.progress = snapshot.progress;
            this.progressMessage = snapshot.message;
          }
        });
      } catch (error) {
        if (token === this.requestToken) {
          this.error = `Impossible de charger l'activité : ${apiError(error)}`;
          this.progress = null;
        }
        return;
      }
      if (token === this.requestToken) {
        this.progress = null;
        await this.refreshAll();
      }
    },

    async refreshAll() {
      await Promise.all([this.refreshCharts(), this.refreshTrack()]);
    },

    async refreshCharts() {
      const token = ++this.chartsToken;
      const isCurrent = () => token === this.chartsToken;
      try {
        const view = await fetchJson(
          `/api/activities/${this.activityId}/charts?${this.query({ x_axis: this.options.xAxis })}`,
        );
        if (!isCurrent()) {
          return;
        }
        this.metrics = view.metrics;
        this.chartErrors = {};
        await this.$nextTick(); // the chart containers are shown with the metrics
        await nextFrame();
        for (const { name } of CHARTS) {
          const chart = view.charts[name];
          const element = document.getElementById(`chart-${name}`);
          if (chart.figure) {
            window.Plotly.react(element, chart.figure.data, chart.figure.layout, PLOTLY_CONFIG);
          } else {
            window.Plotly.purge(element);
            this.chartErrors[name] = chart.error;
          }
        }
      } catch (error) {
        if (isCurrent()) {
          this.error = `Impossible d'afficher les graphiques : ${apiError(error)}`;
        }
      }
    },

    async refreshTrack() {
      const token = ++this.trackToken;
      const isCurrent = () => token === this.trackToken;
      const activityId = this.activityId;
      try {
        const track = await fetchJson(
          `/api/activities/${activityId}/track?${this.query({
            color_by: this.options.colorBy,
            texture: this.options.texture,
            show_km_markers: this.options.showKmMarkers,
          })}`,
        );
        if (!isCurrent()) {
          return;
        }
        this.track = track;
        this.trackError = "";
        await this.$nextTick(); // the map container must be visible to be sized
        await nextFrame();
        this.renderDeck(track, activityId);
      } catch (error) {
        if (isCurrent()) {
          this.track = null;
          this.trackError = apiError(error);
        }
      }
    },

    renderDeck(track, activityId) {
      if (this.deck && this.deckActivityId === activityId) {
        // same activity: keep the camera where the user left it
        const { initialViewState, ...rest } = track.deck;
        window.updateDeck(rest, this.deck);
        return;
      }
      this.deck?.finalize();
      this.$refs.trackMap.replaceChildren();
      this.deck = window.createDeck({
        container: this.$refs.trackMap,
        jsonInput: track.deck,
        tooltip: track.tooltip,
      });
      this.deckActivityId = activityId;
    },
  }));
});
