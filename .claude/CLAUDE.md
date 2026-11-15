# CLAUDE.md

Guidance for Claude when working in this repository. Keep this file short and current; when I ask you to, update it when a rule turns out to be wrong or missing.

## Project context
- **Purpose:** process Strava activity data (FIT files) and build a model predicting pace per fixed-width distance segment
- **Stack:** Python 3.12+, pandas, Streamlit, pytest
- **Layout:** source in `TrailPredictor/`, tests in `tests/`, Streamlit entry point `app.py` with pages in `pages/`

## Commands
Use **Poetry** only (no uv/pip).
```bash
poetry install                   # install dependencies
poetry run pytest                # run all tests
poetry run pytest tests/test_x.py -k name   # run a single test
poetry run ruff check . --fix    # lint
poetry run ruff format .         # format
poetry run mypy TrailPredictor   # type-check
poetry run streamlit run app.py  # launch the app
```

## Communication
- Talk to me in **French**. Everything written into the codebase (identifiers, comments, docstrings, commit messages, log messages) is in **English**.
- If a request is ambiguous or has several reasonable designs, state the options and ask before writing large amounts of code.
- When you make a non-obvious decision or trade-off, say so in one or two sentences.

## Workflow
1. **Reuse** existing helpers and test fixtures instead of duplicating them.
2. **Keep diffs focused.** Change only what the task requires; no drive-by refactors or reformatting of untouched code. Suggest broader refactors separately.
3. **Verify.** After changes, run the relevant tests, `ruff` and `mypy`. Do not report a task as done while checks fail.
4. **Never weaken a test to make it pass.** If a test looks wrong, explain why and ask.
5. Don't add new dependencies without asking.

## Git commits
- Follow [gitmoji](https://gitmoji.dev): every subject line starts with **one** gitmoji as the Unicode character (`🐛`, not `:bug:`), then a space and the summary: `🐛 Return NaN for non-positive speeds in speed columns`.
- Subject in English, imperative mood, capitalized, no trailing period, ≤ 72 characters. Add a body (after a blank line) only when the *why* is not obvious.
- Pick the gitmoji matching the main intent of the change; split unrelated changes into separate commits rather than stacking emojis. Common ones here:

  | Emoji | Use for |
  |---|---|
  | ✨ | new feature |
  | 🐛 | bug fix |
  | 🚑️ | critical hotfix |
  | ♻️ | refactor without behaviour change |
  | 🎨 | code structure / formatting |
  | ⚡️ | performance |
  | ✅ | add or update tests |
  | 📝 | documentation / docstrings |
  | 🔧 | configuration (pre-commit, pyproject, linters) |
  | ➕ / ➖ | add / remove a dependency |
  | ⬆️ | upgrade dependencies |
  | 🔥 | remove code or files |
  | 🚚 | move or rename files |
  | 🚧 | work in progress |
  | 🚀 | deploy / release |
  | 🙈 | `.gitignore` |
- Commit only when asked. Never commit unrelated working-tree changes: stage files explicitly, not `git add -A`.

## Code quality
- Modern type hints on every function signature: `int | None`, `list[str]`, `dict[str, float]`; import `Callable`, `Iterable`, `Sequence` from `collections.abc`. No `Optional`, `List`, `Dict`.
- **NumPy-style (numpydoc) docstrings** for every public function and class; skip them for trivial private helpers. Write each parameter as `name : type`, adding shape/units/optional info where relevant. Sections, in this order, only those that apply: one-line summary, extended summary, `Parameters`, `Returns` (or `Yields`), `Raises`, `See Also`, `Notes`, `References`, `Examples`. Each header is underlined with dashes of the same length:
  ```python
  def predict_time(distance_m: float, elevation_gain_m: float = 0.0) -> float:
      """Predict the finishing time of a trail segment.

      Extended summary describing the behaviour in more detail.

      Parameters
      ----------
      distance_m : float
          Horizontal distance, in metres.
      elevation_gain_m : float, optional
          Positive elevation gain, in metres. Default is 0.0.

      Returns
      -------
      float
          Predicted time, in seconds.

      Raises
      ------
      ValueError
          If `distance_m` is negative.

      See Also
      --------
      minetti_cost : Energy cost of running on a slope.

      Notes
      -----
      Mathematical or technical remarks.

      References
      ----------
      .. [1] Author, Title, Publisher, Year.

      Examples
      --------
      >>> predict_time(1000.0)
      300.0
      """
  ```
- `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for module-level constants. Names should describe intent, not type (`daily_revenue`, not `df2`).
- Use `@dataclass` (prefer `frozen=True, slots=True`) or `NamedTuple` for structured data instead of loose dictionaries; use `Enum`/`Literal` for closed sets of values.
- Prefer pure functions; keep I/O (files, network, Streamlit UI) at the edges so core logic stays testable.
- Use `pathlib.Path`, not string paths. No magic numbers: name them as constants or config.

## Errors and logging
- Never use `print()` for diagnostics. Each module declares `logger = logging.getLogger(__name__)`; logging is configured once, at the application entry point only.
- Use lazy formatting: `logger.info("Loaded %d rows", n)`, not f-strings.
- Catch specific exceptions only; no bare `except:` or broad `except Exception` unless it re-raises or is a top-level boundary. Never silently swallow an error.
- Use `logger.exception(...)` inside `except` blocks to keep the stack trace. Raise custom exceptions where callers need to distinguish failure modes.

## Testing
- Design functions to be modular and isolated so they can be tested without Streamlit, network or disk.
- For every piece of business logic you add or change, write or update a `pytest` test in the same change.
- Tests live in the root `tests/` directory and are named `test_<module_name>.py`.
- Use fixtures for shared setup, `@pytest.mark.parametrize` for input variants, `tmp_path` for files, and `pytest.approx` / `pd.testing.assert_frame_equal` for numeric and DataFrame comparisons.
- Tests must be deterministic: fixed seeds, no dependence on the current date or external services (mock them).
- Cover edge cases: empty inputs, missing values, wrong dtypes, boundary values.

## Pandas
- **Vectorize.** No `for` loops over rows, `iterrows()` or `itertuples()` for transformations. Use vectorized operations, `np.where`/`np.select`, `.map()`, `groupby().transform()`. Row-wise `apply(axis=1)` is a last resort and needs a comment explaining why.
- **Explicit indexing.** Select and assign with `.loc[]` / `.iloc[]`; no chained assignment (`df[a][b] = ...`). Don't use `inplace=True`.
- **Method chaining** with `.assign()`, `.query()`, `.pipe()`, `.groupby()` for multi-step transformations, one step per line inside parentheses.
- **Dtypes.** Pass `dtype=`, `parse_dates=` and `usecols=` when reading data; use `category` for low-cardinality strings and nullable dtypes (`Int64`, `string`, `boolean`) where missing values are possible.
- **Validate before critical joins and aggregations:** check missing values and duplicate keys, and use `merge(..., validate="one_to_one" | "many_to_one", indicator=True)` when the relationship matters. Assert expected shapes/columns at function boundaries.
- Functions take and return DataFrames without mutating their input.

## NumPy / numerical code
- **Import:** always `import numpy as np`; no `from numpy import *` or bare `from numpy import array`.
- **Randomness:** use `rng = np.random.default_rng(seed)` passed explicitly as a function argument (`rng: np.random.Generator`); never the global `np.random.seed` / `np.random.rand`.
- **Type hints:** annotate arrays with `numpy.typing` (`npt.NDArray[np.float64]`, `npt.ArrayLike` for inputs that accept lists); document expected shape and units in the `Parameters` section (e.g. `distance_m : ndarray of shape (n,)` / `Distance in metres.`).
- **Vectorize.** No Python `for` loops over array elements; use ufuncs, `np.where`, `np.select`, `np.cumsum`, `np.diff`, boolean masks. `np.vectorize` is not vectorization (it is a loop) — avoid it.
- **Shapes and broadcasting:** make axes explicit (`axis=0`, `keepdims=True`), use `x[:, np.newaxis]` rather than relying on implicit broadcasting, and `assert arr.ndim == 1` / `arr.shape == (n, m)` at function boundaries where it matters.
- **Dtypes:** be explicit at creation (`np.zeros(n, dtype=np.float64)`, `np.asarray(x, dtype=float)`); beware integer division and overflow with integer arrays.
- **No mutation of inputs:** don't modify argument arrays in place; work on `arr.copy()` or return new arrays. Remember that slices are views.
- **Missing values:** use `np.nan` with `np.nanmean` / `np.nansum` etc. deliberately; check with `np.isnan`, never `x == np.nan`. Handle division by zero explicitly (`np.divide(a, b, out=..., where=b != 0)`) rather than silencing warnings with `np.errstate` globally.
- **Float comparisons:** use `np.isclose` / `np.allclose` in code and `np.testing.assert_allclose` or `pytest.approx` in tests; never `==` on floats.
- **Avoid legacy APIs:** no `np.matrix` or `np.random.RandomState` in new code.

## Streamlit
- Use explicit elements (`st.title`, `st.header`, `st.markdown`, `st.dataframe`, `st.metric`) rather than `st.write` for text and headings.
- Keep the UI layer thin: Streamlit scripts call functions from the package; no business logic inside the page script.
- Wrap I/O and external calls (file loading, database, APIs) in `try/except` with specific exceptions; show a short user-facing message with `st.error` and log the details with `logger.exception`.
- Cache expensive work: `@st.cache_data` for serializable data/computations (set `ttl` when the source can change), `@st.cache_resource` for connections and models. Don't cache cheap functions or functions with side effects, and never mutate a cached object in place.
- Initialize `st.session_state` keys explicitly at the top of the script (`if "key" not in st.session_state: ...`); group related inputs in `st.form` to avoid unnecessary reruns.
