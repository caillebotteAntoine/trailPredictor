# PyInstaller build of the desktop application (one folder, no console).
#
# Build:   poetry run pyinstaller TrailPredictor.spec --noconfirm
# Output:  dist/TrailPredictor/TrailPredictor.exe
#
# The Strava data is not bundled: the executable reads the ``data`` folder
# next to it, or the folder of the TRAILPREDICTOR_DATA_DIR environment
# variable (see TrailPredictor/utils.py).
# pylint: skip-file

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = [
    # HTML/CSS/JS front end, served by FastAPI
    ("TrailPredictor/api/static", "TrailPredictor/api/static"),
    # plotly.js (served at /lib/plotly.min.js) and the figure templates
    *collect_data_files("plotly", includes=["package_data/**"]),
    # deck.gl bundle (served at /lib/deckgl.min.js), without its source map
    *collect_data_files(
        "pydeck", includes=["nbextension/static/index.js"], excludes=["**/*.map"]
    ),
]

# uvicorn imports its loops, protocols and lifespan by name ("auto")
hiddenimports = collect_submodules("uvicorn")

# Imported lazily by optional code paths the desktop app never uses (notebook
# display, Streamlit pages), or development tools: not bundled.
excludes = [
    "IPython",
    "ipykernel",
    "ipywidgets",
    "jupyter_client",
    "jupyter_core",
    "nbformat",
    "notebook",
    "streamlit",
    "tkinter",
    "pytest",
    "mypy",
    "pylint",
    "ruff",
    "vulture",
    "deptry",
]

a = Analysis(
    ["desktop.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TrailPredictor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed app: the logs go to <data>/logs/desktop.log
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="TrailPredictor",
)
