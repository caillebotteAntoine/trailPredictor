// Alpine components of the home page.

import { fetchJson } from "./api.js";

const PAGES = [
  {
    id: "activities",
    title: "Activités",
    description: "Métriques, carte 3D et graphiques d'une sortie.",
    href: "activities.html",
    ready: true,
  },
  {
    id: "dataset",
    title: "Dataset / Features",
    description: "Corrélations et graphiques des features du dataset.",
    href: "dataset.html",
    ready: true,
  },
  {
    id: "heart-rate",
    title: "Zones cardiaques",
    description: "Optimisation et comparaison des zones de fréquence cardiaque.",
    ready: false,
  },
];

const STATUS_LABELS = {
  checking: "Connexion…",
  online: "Serveur connecté",
  offline: "Serveur injoignable",
};

document.addEventListener("alpine:init", () => {
  window.Alpine.data("home", () => ({
    pages: PAGES,
    serverStatus: "checking",

    get statusLabel() {
      return STATUS_LABELS[this.serverStatus];
    },

    async init() {
      try {
        const health = await fetchJson("/api/health");
        this.serverStatus = health.status === "ok" ? "online" : "offline";
      } catch (error) {
        console.error(error);
        this.serverStatus = "offline";
      }
    },
  }));
});
