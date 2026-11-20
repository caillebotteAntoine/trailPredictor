// Helpers shared by the pages.

export const PLOTLY_CONFIG = { responsive: true, displaylogo: false };

/**
 * Return the message of an error, for the user.
 * @param {unknown} error - Caught error.
 * @returns {string} Its message.
 */
export function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Resolve after the next layout: containers just shown by x-show then have
 * their real size (Plotly and deck.gl otherwise fall back to a default size).
 * @returns {Promise<void>}
 */
export function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/**
 * Read a value saved in the browser, ignoring a missing or blocked storage.
 * @param {string} key - Storage key.
 * @param {string} fallback - Value when nothing is saved.
 * @returns {string} Saved or fallback value.
 */
export function loadSetting(key, fallback) {
  try {
    return window.localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

/**
 * Save a value in the browser; a blocked storage only loses the setting.
 * @param {string} key - Storage key.
 * @param {string} value - Value to save.
 */
export function saveSetting(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // private window or blocked storage: the setting is simply not kept
  }
}
