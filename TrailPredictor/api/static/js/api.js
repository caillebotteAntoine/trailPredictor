// Client of the TrailPredictor HTTP API.

const FINISHED_STATUSES = new Set(["succeeded", "failed"]);

/**
 * Read the error message of a failed API response.
 * @param {Response} response - Response with a non-2xx status.
 * @returns {Promise<string>} The "detail" of FastAPI: its message, or the
 *   field and message of each validation error.
 */
async function errorDetail(response) {
  const body = await response.json().catch(() => null);
  const detail = body?.detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail.map((error) => `${error.loc?.at(-1) ?? ""} : ${error.msg}`).join(" ; ");
  }
  return response.statusText;
}

/**
 * Fetch a JSON resource of the API.
 * @param {string} url - Path of the resource, e.g. "/api/health".
 * @param {RequestInit} [options] - Options passed to fetch.
 * @returns {Promise<any>} Decoded JSON body.
 * @throws {Error} If the response status is not 2xx.
 */
export async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { Accept: "application/json", ...options.headers },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`${await errorDetail(response)} (${response.status})`);
  }
  return response.json();
}

/**
 * Follow the progress of a background job through server-sent events.
 * @param {string} jobId - Identifier returned when the job was submitted.
 * @param {(snapshot: object) => void} onUpdate - Called with every
 *   JobSnapshot: {job_id, status, progress, message, error, version}.
 * @returns {Promise<object>} Last snapshot, once the job succeeded.
 * @throws {Error} If the job failed or the connection was lost.
 */
export function watchJob(jobId, onUpdate) {
  return new Promise((resolve, reject) => {
    const source = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
    source.onmessage = (event) => {
      const snapshot = JSON.parse(event.data);
      onUpdate(snapshot);
      if (!FINISHED_STATUSES.has(snapshot.status)) {
        return;
      }
      source.close();
      if (snapshot.status === "succeeded") {
        resolve(snapshot);
      } else {
        reject(new Error(snapshot.error ?? "La tâche a échoué."));
      }
    };
    source.onerror = () => {
      source.close();
      reject(new Error("Connexion au serveur perdue."));
    };
  });
}
