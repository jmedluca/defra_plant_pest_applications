"""Plant Pest Prevalence Dashboard.

Purpose
-------
This Shiny application turns annual polygon surveillance records into a
reproducible prevalence-change analysis using the project's two monitoring
methods:

* Change Method — compares the selected first and last survey years.
* Regression Method — fits a prevalence trend across all selected survey years.

Operational data flow
---------------------
1. Load survey polygons, host-coverage polygons and optional SPHN polygons.
2. Validate geometry, CRS, survey year and survey outcome fields.
3. Build an operational host landscape from the public host map plus, when
   enabled, surveyed areas that lie outside that map.
4. Intersect each year's survey footprints with the host landscape and convert
   surveyed area to estimated host counts using the selected host density.
5. Treat hosts in positive survey footprints as infected and hosts in negative
   survey footprints as uninfected.
6. Estimate yearly prevalence, apply the Change Method and Regression Method,
   compare estimated survey coverage with planning references, and export a
   PDF report.

Important interpretation
------------------------
The source files are polygon surveillance records rather than host-by-host
inspection logs.  The resulting prevalence values are therefore decision-
support estimates under explicit assumptions, not direct ground-truth counts.
The assumptions shown in the Results tab and PDF are part of the result.

Code organisation
-----------------
The file is intentionally arranged in the same order as the data flow:
configuration -> input validation -> GIS conversion -> statistical summaries ->
reporting -> UI -> server logic.  Statistical planning/fitting calculations live
in ``plant_pest_backend.py`` so there is a single implementation to audit.
"""

from __future__ import annotations

import base64
from html import escape
import io
import json
import re
from pathlib import Path
from matplotlib.patches import FancyBboxPatch
import sys
import tempfile
import textwrap
from typing import Any

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _open_native_directory_dialog(current_value: str, title: str) -> tuple[str | None, str | None]:
    """Open the operating system's folder chooser for a locally run dashboard.

    The Shiny page itself runs in a browser, and browsers are deliberately not
    allowed to reveal arbitrary local directory paths.  For the packaged/local
    dashboard we therefore ask Python (the local server process) to open the
    native folder chooser.  This works when the dashboard is running on the
    user's own desktop.  If the app is later hosted on a remote server, the
    chooser would open on that server instead, so a browser upload workflow
    should be used for remote deployments.

    Returns
    -------
    (selected_path, error_message)
        ``selected_path`` is ``None`` when the user cancels.  ``error_message``
        is populated only when a native dialog cannot be opened.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        return None, f"The native folder chooser is unavailable on this system: {exc}"

    try:
        initial = Path(str(current_value or "").strip()).expanduser()
        if initial.is_file():
            initial = initial.parent
        if not initial.exists() or not initial.is_dir():
            initial = APP_DIR

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        root.update_idletasks()
        selected = filedialog.askdirectory(
            parent=root,
            title=title,
            initialdir=str(initial),
            mustexist=True,
        )
        root.destroy()
        return (str(Path(selected).resolve()), None) if selected else (None, None)
    except Exception as exc:
        try:
            root.destroy()
        except Exception:
            pass
        return None, f"The folder chooser could not be opened: {exc}"

# The backend below is the frozen production implementation used for the published
# Monte Carlo study.  The dashboard imports its statistical routines rather than
# maintaining independent copies that could drift from the published analysis.
from plant_pest_backend import (
    DEFAULT_ALPHA as BACKEND_DEFAULT_ALPHA,
    DEFAULT_CHANGE_METHOD_CORR as BACKEND_DEFAULT_CHANGE_METHOD_CORR,
    DEFAULT_CHANGE_METHOD_DELTA as BACKEND_DEFAULT_CHANGE_METHOD_DELTA,
    DEFAULT_BASELINE_PREVALENCE_CONF as BACKEND_DEFAULT_BASELINE_PREVALENCE_CONF,
    DEFAULT_BASELINE_PREVALENCE_UPPER as BACKEND_DEFAULT_BASELINE_PREVALENCE_UPPER,
    DEFAULT_BASELINE_PREVALENCE_WIDTH as BACKEND_DEFAULT_BASELINE_PREVALENCE_WIDTH,
    DEFAULT_REGRESSION_METHOD_TARGET as BACKEND_DEFAULT_REGRESSION_METHOD_TARGET,
    compute_n_change_method_cells as _backend_compute_n_change_method_cells,
    compute_n_baseline_prevalence_cells as _backend_compute_n_baseline_prevalence_cells,
    compute_n_total_regression_method_cells as _backend_compute_n_total_regression_method_cells,
    _sim_fit_logistic_counts as _backend_fit_logistic_counts,
    _sim_predict_logistic as _backend_predict_logistic,
    _simx_estimate_design_effect as _backend_estimate_design_effect,
)

def _logistic_irls(t: np.ndarray, x: np.ndarray, n: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper around the frozen backend grouped-binomial fit.

    New dashboard code calls the backend fit directly so that convergence and
    separation diagnostics are retained.  This wrapper remains only for older
    helper code that may still import it.
    """
    del max_iter
    beta, cov = _backend_fit_logistic_counts(
        np.asarray(t, dtype=float),
        np.asarray(x, dtype=float),
        np.asarray(n, dtype=float),
        model_form="logistic",
    )
    return np.asarray(beta, dtype=float), np.asarray(cov, dtype=float)


# Application entry point and shared statistical wrappers.
# The dashboard is intentionally thin: GIS preparation and presentation live
# here, while the statistical formulas/fits are imported from the backend.

# ============================================================================
# Optional Dependencies
# ============================================================================

try:
    import geopandas as gpd
except Exception:  
    gpd = None

try:
    import folium
except Exception:  
    folium = None

try:
    from xhtml2pdf import pisa
except Exception:  
    pisa = None

# ============================================================================
# Default Paths And Settings
# ============================================================================

POLYGON_FILES_DIR = APP_DIR / "polygon_files"
# Packaged default data folders.  They are resolved relative to this Python file,
# so the application does not depend on a developer-specific absolute path.
TARGET_SITES_DIR = POLYGON_FILES_DIR / "survey_data"
LARCH_HOST_DIR = POLYGON_FILES_DIR / "host_coverage_data"
SPHN_DIR = POLYGON_FILES_DIR / "sphn_data"
# Folder paths are intentional: each folder may contain one or several shapefiles.
LARCH_HOST_PATH = LARCH_HOST_DIR
SPHN_PATH = SPHN_DIR

# ============================================================================
# Input data contract
# ============================================================================
# The dashboard is deliberately more flexible than the original DEFRA files,
# but it still needs a small, explicit schema contract.  Keeping these rules in
# one place makes future data-source changes auditable.
#
# Survey data requirements
#   * ESRI shapefile geometry (Polygon or MultiPolygon).
#   * A valid CRS (.prj) so areas/overlays can be calculated in metres.
#   * A recognisable year/date field.
#   * A recognisable outcome/status field, or a field selected in Advanced
#     settings.
#
# Host data requirements
#   * Polygon/MultiPolygon geometry with a valid CRS.
#   * No attribute field is mandatory unless a host-category filter is used.
#
# SPHN data are optional map context and do not block analysis.
SUPPORTED_VECTOR_SUFFIXES = {".shp"}
REQUIRED_SHAPEFILE_SIDECARS = (".dbf", ".shx", ".prj")
SURVEY_YEAR_FIELD_CANDIDATES = (
    "surveyyear", "survyear", "surveyyr", "survyr", "yearsurveyed",
    "createdyr", "createdyear", "surveydate", "survey_date", "visitdate",
    "inspectiondate", "obsdate", "date", "created", "createdat",
)
SURVEY_STATUS_FIELD_CANDIDATES = (
    "site status", "site_status", "site_statu", "survey status", "survey_status",
    "survey outcome", "survey_outcome", "outcome", "result", "survey result",
)
SPHN_YEAR_FIELD_CANDIDATES = ("File_creat", "Compliance", "Remain_in_", "Flight_dat", "Woodland_o")
# Interactive basemap.  OpenStreetMap is deliberately used as the default
# because it does not require a project API key and therefore avoids the
# "API key required" watermark introduced on CARTO raster tiles.  Keep the
# attribution visible in accordance with the OpenStreetMap tile policy.
OSM_RASTER_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)
# These values are only UI fallbacks before data have loaded.  Analysis years are
# replaced with the years detected in the uploaded survey files.
YEAR_MIN = 2017
YEAR_MAX = 2024
# These are the default statistical settings used unless the user changes them.
DEFAULT_ALPHA = BACKEND_DEFAULT_ALPHA
# The production Monte Carlo reference study used 99% power.  Keep that as the
# dashboard starting value while leaving the setting editable for operational use.
DEFAULT_POWER = 0.99
DEFAULT_CHANGE_METHOD_DELTA = BACKEND_DEFAULT_CHANGE_METHOD_DELTA
DEFAULT_CHANGE_METHOD_CORR = BACKEND_DEFAULT_CHANGE_METHOD_CORR
DEFAULT_REGRESSION_METHOD_TARGET = BACKEND_DEFAULT_REGRESSION_METHOD_TARGET
DEFAULT_BASELINE_PREVALENCE_CONF = BACKEND_DEFAULT_BASELINE_PREVALENCE_CONF
DEFAULT_BASELINE_PREVALENCE_WIDTH = BACKEND_DEFAULT_BASELINE_PREVALENCE_WIDTH
DEFAULT_BASELINE_PREVALENCE_UPPER = BACKEND_DEFAULT_BASELINE_PREVALENCE_UPPER
RETRO_BENCHMARK_CHOICES = {
    "srs": "SRS reference (no Design Effect inflation)",
    "mss_no_inflation": "MSS reference (no Design Effect inflation)",
    "mss_pilot": "MSS with approximate Design Effect from survey polygons",
    "mss_fixed": "MSS with user-specified Design Effect",
}


# ============================================================================
# GIS And Map Helpers
# ============================================================================
#
# GIS helpers keep all map work in one place. 
# Web maps need latitude/longitude, but area calculations need projected metres.


def _gis_available() -> bool:
    # The app can still open without geopandas, but shapefile loading needs it.
    return gpd is not None


def _interactive_map_available() -> bool:
    # Folium is used for the interactive map in the Data Viewer tab.
    return folium is not None


def _to_wgs84(gdf: Any | None):
    # Convert data to latitude/longitude so it can be shown on a web map.
    if gdf is None or len(gdf) == 0:
        return None
    if getattr(gdf, "crs", None) is not None:
        try:
            return gdf.to_crs(4326)
        except Exception:
            return gdf
    return gdf


def _to_bng(gdf: Any | None):
    # Convert data to British National Grid so areas and distances are in metres.
    if gdf is None or len(gdf) == 0:
        return None
    if getattr(gdf, "crs", None) is not None:
        try:
            return gdf.to_crs(27700)
        except Exception:
            return gdf
    return gdf


def _bounds_from_gdf(gdf: Any | None) -> list[list[float]] | None:
    # Get the map bounds so the web map can zoom to the loaded polygons.
    if gdf is None or len(gdf) == 0:
        return None
    try:
        xmin, ymin, xmax, ymax = gdf.total_bounds
    except Exception:
        return None
    if not np.all(np.isfinite([xmin, ymin, xmax, ymax])):
        return None
    return [[float(ymin), float(xmin)], [float(ymax), float(xmax)]]


def _serializable_geojson_payload(gdf: Any | None) -> dict[str, Any] | None:
    # Convert a GeoDataFrame into browser-friendly GeoJSON.
    # Dates and unusual values are converted to strings so Folium can display them.
    if gdf is None or len(gdf) == 0:
        return None

    try:
        work = gdf.copy()
        non_geom_cols = [col for col in work.columns if col != "geometry"]
        for col in non_geom_cols:
            if pd.api.types.is_datetime64_any_dtype(work[col]):
                work[col] = work[col].dt.strftime("%Y-%m-%d").fillna("")
            elif pd.api.types.is_numeric_dtype(work[col]) or pd.api.types.is_bool_dtype(work[col]):
                continue
            else:
                work[col] = work[col].fillna("").astype(str)
        return json.loads(work.to_json())
    except Exception:
        return None


def _color_hex_series(values: pd.Series | np.ndarray, cmap_name: str, vmin: float | None = None, vmax: float | None = None, alpha: float = 1.0) -> list[str]:
    # Turn numeric values into colours for map layers.
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if arr.size == 0:
        return []
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((arr - float(vmin)) / max(float(vmax - vmin), 1e-12), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    return [tuple(int(round(255 * c)) for c in cmap(float(x))[:3]) for x in norm]


def _rgb_tuple_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _popup_config_for_layer(gdf: Any | None, preferred_fields: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    if gdf is None:
        return [], []
    fields = [(field, label) for field, label in preferred_fields if field in gdf.columns]
    fields = fields[:5]
    return [field for field, _label in fields], [label for _field, label in fields]


def build_folium_map_html(
    survey_gdf: Any | None,
    host_gdf: Any | None,
    sphn_gdf: Any | None,
    year: int,
    view_mode: str,
    visible_layers: list[str],
) -> str:
    # Build the interactive web map as an HTML string.
    # Shiny displays this inside an iframe.
    if not _interactive_map_available():
        return (
            "<div style='padding:1rem;font-family:sans-serif;'>"
            "Interactive map library is not available in this Python environment."
            "</div>"
        )

    map_obj = folium.Map(
        location=[55.0, -3.5],
        zoom_start=6,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles=OSM_RASTER_TILE_URL,
        attr=OSM_ATTRIBUTION,
        name="OpenStreetMap",
        overlay=False,
        control=False,
        max_zoom=19,
    ).add_to(map_obj)

    host_plot = _to_wgs84(host_gdf)
    survey_plot = _to_wgs84(survey_gdf)
    sphn_plot = _to_wgs84(sphn_gdf)

    bounds_to_fit: list[list[list[float]]] = []

    def add_geojson_layer(
        gdf: Any | None,
        name: str,
        color: str,
        popup_fields: list[str],
        popup_labels: list[str],
        show: bool = True,
    ):
        if gdf is None or len(gdf) == 0:
            return
        try:
            payload = _serializable_geojson_payload(gdf)
            if payload is None:
                return
            folium.GeoJson(
                data=payload,
                name=name,
                style_function=lambda _feature, line_color=color: {
                    "color": line_color,
                    "weight": 2,
                    "fillOpacity": 0.08,
                },
                popup=(
                    folium.GeoJsonPopup(fields=popup_fields, aliases=popup_labels, labels=True, localize=True)
                    if popup_fields
                    else None
                ),
                show=show,
            ).add_to(map_obj)
        except Exception:
            return

    show_survey = "survey" in visible_layers
    show_host = "larch" in visible_layers
    show_sphn = "sphn" in visible_layers

    if show_host:
        host_fields, host_labels = _popup_config_for_layer(
            host_plot,
            [
                ("Shape_Area", "Area"),
                ("Shape_Leng", "Perimeter"),
                ("OBJECTID", "Feature ID"),
            ],
        )
        add_geojson_layer(
            host_plot,
            "Larch coverage",
            "#2ca25f",
            host_fields,
            host_labels,
            show=True,
        )
        bounds = _bounds_from_gdf(host_plot)
        if bounds is not None:
            bounds_to_fit.append(bounds)

    if show_survey:
        survey_fields, survey_labels = _popup_config_for_layer(
            survey_plot,
            [
                ("Site_statu", "Site status"),
                ("Disease", "Disease"),
                ("Species", "Species"),
                ("Species_2", "Species 2"),
                ("Created_yr", "Survey year"),
            ],
        )
        add_geojson_layer(
            survey_plot,
            f"Survey sites {year}",
            "#d95f02",
            survey_fields,
            survey_labels,
            show=True,
        )
        bounds = _bounds_from_gdf(survey_plot)
        if bounds is not None:
            bounds_to_fit.append(bounds)

    if show_sphn:
        sphn_fields, sphn_labels = _popup_config_for_layer(
            sphn_plot,
            [
                ("Action", "Action"),
                ("File_creat", "File created"),
                ("Compliance", "Compliance date"),
                ("Remain_in_", "Remain in force"),
                ("Flight_dat", "Flight date"),
            ],
        )
        add_geojson_layer(
            sphn_plot,
            f"SPHN sites {year}",
            "#756bb1",
            sphn_fields,
            sphn_labels,
            show=True,
        )
        bounds = _bounds_from_gdf(sphn_plot)
        if bounds is not None:
            bounds_to_fit.append(bounds)

    if view_mode == "uk":
        map_obj.fit_bounds([[49.5, -8.5], [59.5, 2.5]])
    elif view_mode == "all_zoom" and bounds_to_fit:
        south = min(b[0][0] for b in bounds_to_fit)
        west = min(b[0][1] for b in bounds_to_fit)
        north = max(b[1][0] for b in bounds_to_fit)
        east = max(b[1][1] for b in bounds_to_fit)
        map_obj.fit_bounds([[south, west], [north, east]])
    else:
        zoom_target = {
            "survey": survey_plot if show_survey else None,
            "larch": host_plot if show_host else None,
            "sphn": sphn_plot if show_sphn else None,
        }.get(view_mode)
        bounds = _bounds_from_gdf(zoom_target)
        if bounds is not None:
            map_obj.fit_bounds(bounds)
        else:
            map_obj.fit_bounds([[49.5, -8.5], [59.5, 2.5]])

    folium.LayerControl(collapsed=False).add_to(map_obj)
    map_html = map_obj.get_root().render()
    # Defensive check: this source file intentionally contains no CARTO basemap.
    # If a CARTO URL/name somehow appears, show a clear diagnostic rather than
    # silently serving tiles with an API-key watermark.
    lowered = map_html.lower()
    if "cartocdn" in lowered or "cartodbpositron" in lowered or "api key required" in lowered:
        return (
            "<div style='padding:1rem;font-family:sans-serif;color:#8a1f11;'>"
            "Unexpected CARTO basemap detected. The running Shiny process is not using the current dashboard file. "
            "Stop the app completely and restart it from the updated plant_pest_dashboard.py."
            "</div>"
        )
    return map_html


# ============================================================================
# Data Loading And Real Survey Conversion
# ============================================================================
#
# This section is where raw GIS files are turned into the tables needed for
# prevalence estimation.  The key idea is:
#   survey polygon area x host density = estimated number of hosts sampled.


def _default_layer_paths() -> dict[str, Path]:
    # Discover every survey shapefile in the packaged survey folder.  Filenames
    # and year ranges are deliberately not hard-coded.
    return {shp.stem: shp for shp in _resolve_shapefile_paths(TARGET_SITES_DIR)}


def _resolve_shapefile_path(path: Path, preferred_stem: str | None = None) -> Path:
    # Compatibility helper: return a direct shapefile or the first suitable file
    # in a folder. New loaders use _resolve_shapefile_paths so folders containing
    # several files are handled correctly.
    if path.is_file():
        return path
    if not path.exists():
        return path
    shp_files = sorted(
        [child for child in path.iterdir() if child.is_file() and child.suffix.lower() in SUPPORTED_VECTOR_SUFFIXES],
        key=lambda x: x.name.lower(),
    )
    if preferred_stem is not None:
        for shp in shp_files:
            if shp.stem.lower() == str(preferred_stem).lower():
                return shp
    return shp_files[0] if shp_files else path


def _resolve_shapefile_paths(path: Path, preferred_stems: list[str] | None = None) -> list[Path]:
    # Accept either one .shp file or a folder containing any number of shapefiles.
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() == ".shp" else []
    if not path.exists() or not path.is_dir():
        return []
    shp_files = sorted(
        [child for child in path.iterdir() if child.is_file() and child.suffix.lower() in SUPPORTED_VECTOR_SUFFIXES],
        key=lambda x: x.name.lower(),
    )
    if not shp_files or not preferred_stems:
        return shp_files
    ordered: list[Path] = []
    used: set[Path] = set()
    preferred = [str(stem).lower() for stem in preferred_stems]
    for stem in preferred:
        for shp in shp_files:
            if shp.stem.lower() == stem and shp not in used:
                ordered.append(shp)
                used.add(shp)
    ordered.extend([shp for shp in shp_files if shp not in used])
    return ordered


def _normalise_column_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _extract_year_values(series: pd.Series) -> pd.Series:
    # Prefer an explicit four-digit year when present, otherwise try a date parse.
    raw = series.copy()
    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_valid = numeric.where((numeric >= 1900) & (numeric <= 2200))
    if numeric_valid.notna().mean() >= 0.5 and numeric_valid.notna().any():
        return numeric_valid.round().astype("Int64")
    extracted = raw.astype(str).str.extract(r"((?:19|20|21)\d{2})", expand=False)
    extracted_numeric = pd.to_numeric(extracted, errors="coerce")
    if extracted_numeric.notna().any():
        return extracted_numeric.astype("Int64")
    try:
        parsed = pd.to_datetime(raw, errors="coerce")
        if parsed.notna().any():
            return parsed.dt.year.astype("Int64")
    except Exception:
        pass
    return pd.Series(pd.array([pd.NA] * len(raw), dtype="Int64"), index=raw.index)


def _detect_survey_year_column(gdf: Any) -> tuple[str | None, pd.Series | None]:
    if gdf is None or len(gdf) == 0:
        return None, None
    cols = [str(col) for col in gdf.columns if str(col) != "geometry"]
    normalised = {_normalise_column_name(col): col for col in cols}
    for key in SURVEY_YEAR_FIELD_CANDIDATES:
        col = normalised.get(_normalise_column_name(key))
        if col is None:
            continue
        years = _extract_year_values(gdf[col])
        if years.notna().any():
            return col, years
    # Controlled fallback for a slightly changed schema: inspect columns whose
    # names strongly imply a survey year/date and accept only plausible years.
    candidates = []
    for col in cols:
        key = _normalise_column_name(col)
        if (
            "year" in key or "date" in key
            or key.startswith(("surveyyea", "surveyea", "surveyyr", "survyear", "survyr", "surveyda", "visitda", "inspect", "obsdate", "created"))
        ):
            years = _extract_year_values(gdf[col])
            score = float(years.notna().mean()) if len(years) else 0.0
            if score > 0:
                candidates.append((score, col, years))
    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1].lower()))
        _score, col, years = candidates[0]
        return col, years
    return None, None


def _extract_survey_year_series(gdf: Any) -> pd.Series | None:
    _col, years = _detect_survey_year_column(gdf)
    return years


def extract_available_survey_years(layers: dict[str, Any]) -> list[int]:
    years: set[int] = set()
    for gdf in (layers or {}).values():
        year_series = _extract_survey_year_series(gdf)
        if year_series is None:
            continue
        for val in pd.to_numeric(year_series, errors="coerce").dropna().astype(int).tolist():
            if 1900 <= int(val) <= 2200:
                years.add(int(val))
    return sorted(years)


def _shapefile_sidecar_checks(shp_path: Path, layer_label: str) -> tuple[list[dict[str, str]], bool]:
    checks: list[dict[str, str]] = []
    fatal = False
    siblings = {child.name.lower() for child in shp_path.parent.iterdir()} if shp_path.parent.exists() else set()

    def sidecar_exists(ext: str) -> bool:
        return f"{shp_path.stem}{ext}".lower() in siblings

    missing_required = [ext for ext in REQUIRED_SHAPEFILE_SIDECARS if not sidecar_exists(ext)]
    if missing_required:
        fatal = True
        checks.append({
            "level": "error",
            "message": f"{layer_label}: {shp_path.name} is missing required shapefile component(s): {', '.join(missing_required)}.",
        })
    if not sidecar_exists(".cpg"):
        checks.append({
            "level": "warning",
            "message": f"{layer_label}: {shp_path.name} has no .cpg file. Text encoding will be inferred and unusual characters may display incorrectly.",
        })
    return checks, fatal


def _prepare_polygon_gdf(gdf: Any, shp_path: Path, layer_label: str) -> tuple[Any | None, list[dict[str, str]], bool]:
    checks: list[dict[str, str]] = []
    fatal = False
    if gdf is None or len(gdf) == 0:
        checks.append({"level": "warning", "message": f"{layer_label}: {shp_path.name} loaded but contains no features."})
        return gdf, checks, fatal
    if "geometry" not in gdf.columns:
        checks.append({"level": "error", "message": f"{layer_label}: {shp_path.name} has no geometry column."})
        return None, checks, True
    if getattr(gdf, "crs", None) is None:
        checks.append({
            "level": "error",
            "message": f"{layer_label}: {shp_path.name} has no recognised coordinate reference system (.prj). Spatial area/overlay results cannot be calculated safely.",
        })
        fatal = True

    work = gdf.copy()
    null_or_empty = work.geometry.isna() | work.geometry.is_empty
    if bool(null_or_empty.any()):
        n_bad = int(null_or_empty.sum())
        work = work.loc[~null_or_empty].copy()
        checks.append({"level": "warning", "message": f"{layer_label}: removed {n_bad} feature(s) with empty geometry from {shp_path.name}."})
    if len(work) == 0:
        checks.append({"level": "error", "message": f"{layer_label}: {shp_path.name} contains no usable geometries."})
        return None, checks, True

    try:
        invalid = ~work.geometry.is_valid
        if bool(invalid.any()):
            n_invalid = int(invalid.sum())
            try:
                if hasattr(work.geometry, "make_valid"):
                    work.loc[invalid, "geometry"] = work.loc[invalid].geometry.make_valid()
                else:
                    work.loc[invalid, "geometry"] = work.loc[invalid].geometry.buffer(0)
                remaining = ~work.geometry.is_valid
                if bool(remaining.any()):
                    fatal = True
                    checks.append({"level": "error", "message": f"{layer_label}: {int(remaining.sum())} invalid geometries in {shp_path.name} could not be repaired."})
                else:
                    checks.append({"level": "warning", "message": f"{layer_label}: repaired {n_invalid} invalid geometries in {shp_path.name}."})
            except Exception as exc:
                fatal = True
                checks.append({"level": "error", "message": f"{layer_label}: invalid geometries in {shp_path.name} could not be repaired ({exc})."})
    except Exception as exc:
        checks.append({"level": "warning", "message": f"{layer_label}: geometry validity could not be checked for {shp_path.name} ({exc})."})

    try:
        polygon_mask = work.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        if not bool(polygon_mask.all()):
            n_other = int((~polygon_mask).sum())
            work = work.loc[polygon_mask].copy()
            checks.append({"level": "warning", "message": f"{layer_label}: ignored {n_other} non-polygon feature(s) in {shp_path.name}."})
        if len(work) == 0:
            checks.append({"level": "error", "message": f"{layer_label}: {shp_path.name} contains no polygon features."})
            return None, checks, True
    except Exception:
        pass
    return work, checks, fatal


def _read_polygon_shapefile(shp_path: Path, layer_label: str) -> tuple[Any | None, list[dict[str, str]], bool]:
    checks, fatal = _shapefile_sidecar_checks(shp_path, layer_label)
    if fatal:
        return None, checks, True
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as exc:
        checks.append({"level": "error", "message": f"{layer_label}: failed to read {shp_path.name}: {exc}"})
        return None, checks, True
    prepared, prep_checks, prep_fatal = _prepare_polygon_gdf(gdf, shp_path, layer_label)
    checks.extend(prep_checks)
    if prepared is not None:
        checks.append({"level": "ok", "message": f"{layer_label}: loaded {len(prepared):,} polygon(s) from {shp_path.name}."})
    return prepared, checks, bool(fatal or prep_fatal)


def _combine_loaded_polygon_files(path: Path, layer_label: str, optional: bool = False) -> dict[str, Any]:
    path = Path(path)
    info: dict[str, Any] = {
        "available_file": False,
        "layer": None,
        "messages": [],
        "checks": [],
        "fatal_messages": [],
        "files_loaded": [],
    }
    if not _gis_available():
        msg = "Geospatial packages are not installed. Install geopandas/shapely/pyogrio/pyproj before loading polygon data."
        info["checks"].append({"level": "error", "message": msg})
        info["fatal_messages"].append(msg)
        info["messages"].append(msg)
        return info
    if not path.exists():
        level = "warning" if optional else "error"
        msg = f"{layer_label} path was not found: {path}"
        info["checks"].append({"level": level, "message": msg})
        if not optional:
            info["fatal_messages"].append(msg)
        info["messages"].append(msg)
        return info
    shp_files = _resolve_shapefile_paths(path)
    if not shp_files:
        level = "warning" if optional else "error"
        msg = f"No .shp files were found for {layer_label.lower()} in: {path}"
        info["checks"].append({"level": level, "message": msg})
        if not optional:
            info["fatal_messages"].append(msg)
        info["messages"].append(msg)
        return info

    parts: list[Any] = []
    base_crs = None
    for shp in shp_files:
        gdf, checks, fatal = _read_polygon_shapefile(shp, layer_label)
        info["checks"].extend(checks)
        if fatal:
            info["fatal_messages"].extend([c["message"] for c in checks if c.get("level") == "error"])
        if gdf is None or len(gdf) == 0:
            continue
        if base_crs is None:
            base_crs = getattr(gdf, "crs", None)
        elif getattr(gdf, "crs", None) is not None and base_crs is not None and gdf.crs != base_crs:
            try:
                gdf = gdf.to_crs(base_crs)
                info["checks"].append({"level": "warning", "message": f"{layer_label}: reprojected {shp.name} to match the other loaded files."})
            except Exception as exc:
                msg = f"{layer_label}: could not reproject {shp.name} to the common CRS ({exc})."
                info["checks"].append({"level": "error", "message": msg})
                info["fatal_messages"].append(msg)
                continue
        gdf = gdf.copy()
        gdf["_source_layer"] = shp.stem
        parts.append(gdf)
        info["files_loaded"].append(str(shp))

    if parts:
        try:
            info["layer"] = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=base_crs)
            info["available_file"] = True
        except Exception as exc:
            msg = f"{layer_label}: the loaded files could not be combined ({exc})."
            info["checks"].append({"level": "error", "message": msg})
            info["fatal_messages"].append(msg)
    elif not optional and not info["fatal_messages"]:
        msg = f"{layer_label}: no usable polygon features were loaded."
        info["checks"].append({"level": "error", "message": msg})
        info["fatal_messages"].append(msg)

    info["messages"] = [c["message"] for c in info["checks"] if c.get("level") in {"warning", "error"}]
    return info


def _build_cell_polygons(cells_df: pd.DataFrame):
    # Legacy helper for old grid outputs.  It is kept for compatibility.
    if not _gis_available():
        return None

    try:
        from shapely.geometry import box
    except Exception:
        return None

    lons = np.sort(cells_df["lon"].unique())
    lats = np.sort(cells_df["lat"].unique())
    dx = float(np.median(np.diff(lons))) if len(lons) > 1 else 0.1
    dy = float(np.median(np.diff(lats))) if len(lats) > 1 else 0.1

    polys = [
        box(lon - dx / 2, lat - dy / 2, lon + dx / 2, lat + dy / 2)
        for lon, lat in zip(cells_df["lon"], cells_df["lat"])
    ]
    return gpd.GeoDataFrame(
        cells_df[["cell_id", "row", "col", "lat", "lon", "in_uk", "land_mask"]].copy(),
        geometry=polys,
        crs="EPSG:4326",
    )


def load_target_sites(layer_paths: dict[str, Path] | Path) -> dict[str, Any]:
    # Load every survey shapefile that can be read safely. A bad file is reported
    # explicitly rather than terminating the Shiny session.
    if isinstance(layer_paths, Path):
        source_path = Path(layer_paths)
        if not source_path.exists():
            msg = f"Survey polygon path was not found: {source_path}"
            return {"available_files": {}, "layers": {}, "messages": [msg], "checks": [{"level": "error", "message": msg}], "fatal_messages": [msg], "files_loaded": []}
        shp_files = _resolve_shapefile_paths(source_path)
        if not shp_files:
            msg = f"No survey .shp files were found in: {source_path}"
            return {"available_files": {}, "layers": {}, "messages": [msg], "checks": [{"level": "error", "message": msg}], "fatal_messages": [msg], "files_loaded": []}
        layer_paths = {shp.stem: shp for shp in shp_files}

    info: dict[str, Any] = {
        "available_files": {name: Path(path).exists() for name, path in layer_paths.items()},
        "layers": {},
        "messages": [],
        "checks": [],
        "fatal_messages": [],
        "files_loaded": [],
    }
    if not _gis_available():
        msg = "Geospatial packages are not installed. Install geopandas/shapely/pyogrio/pyproj before loading polygon data."
        info["checks"].append({"level": "error", "message": msg})
        info["fatal_messages"].append(msg)
        info["messages"].append(msg)
        return info

    for name, raw_path in layer_paths.items():
        path = Path(raw_path)
        if not path.exists():
            msg = f"Survey polygons: file was not found: {path}"
            info["checks"].append({"level": "error", "message": msg})
            info["fatal_messages"].append(msg)
            continue
        gdf, checks, fatal = _read_polygon_shapefile(path, "Survey polygons")
        info["checks"].extend(checks)
        if fatal:
            info["fatal_messages"].extend([c["message"] for c in checks if c.get("level") == "error"])
        if gdf is None or len(gdf) == 0:
            continue

        year_col, year_series = _detect_survey_year_column(gdf)
        if year_col is None or year_series is None or not year_series.notna().any():
            msg = (
                f"Survey polygons: {path.name} has no recognised survey year/date field. "
                "Results cannot use this file until a survey year field is supplied."
            )
            info["checks"].append({"level": "error", "message": msg})
            info["fatal_messages"].append(msg)
        else:
            detected = sorted(pd.to_numeric(year_series, errors="coerce").dropna().astype(int).unique().tolist())
            info["checks"].append({"level": "ok", "message": f"Survey polygons: detected year field '{year_col}' ({', '.join(map(str, detected))})."})

        status_col = _status_column(gdf)
        if status_col is None:
            # Do not block loading solely because automatic status detection failed.
            # Every attribute column is exposed in Advanced settings, so an analyst
            # can select the correct outcome field explicitly after loading.
            msg = (
                f"Survey polygons: {path.name} has no automatically recognised survey outcome/status field. "
                "Choose the correct outcome field in Advanced settings before running the analysis."
            )
            info["checks"].append({"level": "warning", "message": msg})
        else:
            n_status = int(gdf[status_col].dropna().astype(str).nunique())
            info["checks"].append({"level": "ok", "message": f"Survey polygons: detected outcome field '{status_col}' with {n_status} value(s)."})

        info["layers"][name] = gdf
        info["files_loaded"].append(str(path))

    if not info["layers"] and not info["fatal_messages"]:
        msg = "No usable survey polygon layers were loaded."
        info["checks"].append({"level": "error", "message": msg})
        info["fatal_messages"].append(msg)
    info["messages"] = [c["message"] for c in info["checks"] if c.get("level") in {"warning", "error"}]
    return info


def load_host_layer(host_path: Path) -> dict[str, Any]:
    # A host folder may contain several shapefiles; combine all usable files.
    return _combine_loaded_polygon_files(Path(host_path), "Host coverage", optional=False)


def load_sphn_layer(sphn_path: Path) -> dict[str, Any]:
    # SPHN polygons are map context rather than an analysis requirement, so
    # missing/bad SPHN files produce warnings but do not block Results.
    return _combine_loaded_polygon_files(Path(sphn_path), "SPHN polygons", optional=True)


def filter_target_sites_for_year(
    layers: dict[str, Any],
    year: int,
) -> tuple[Any | None, list[str]]:
    # Pull out survey polygons for one year.
    # The app uses this when the Data Viewer year slider changes and when
    # yearly prevalence estimates are calculated.
    if not _gis_available():
        return None, []

    messages: list[str] = []
    year_frames: list[Any] = []
    for layer_name, gdf in layers.items():
        if gdf.empty:
            continue

        df = gdf.copy()
        df["_source_layer"] = layer_name

        survey_year = _extract_survey_year_series(df)
        if survey_year is None:
            messages.append(f"Layer {layer_name} has no recognised year field.")
            continue
        df = df.loc[pd.to_numeric(survey_year, errors="coerce") == int(year)].copy()

        if not df.empty:
            year_frames.append(df)

    if not year_frames:
        return None, messages

    return gpd.GeoDataFrame(pd.concat(year_frames, ignore_index=True), geometry="geometry"), messages


def _extract_year_from_columns(gdf: Any, candidates: list[str]) -> pd.Series | None:
    # Try to find a year from several possible date columns.
    # This makes the app less dependent on one exact DEFRA file format.
    for col in candidates:
        if col not in gdf.columns:
            continue
        series = gdf[col]
        if np.issubdtype(series.dtype, np.number):
            years = pd.to_numeric(series, errors="coerce")
        else:
            years = pd.to_datetime(series, errors="coerce").dt.year
        if years.notna().any():
            return years
    return None


def filter_sphn_for_year(
    sphn_gdf: Any | None,
    year: int,
) -> tuple[Any | None, list[str]]:
    # Pull out SPHN polygons for one selected year for the Data Viewer map.
    if not _gis_available() or sphn_gdf is None or len(sphn_gdf) == 0:
        return None, []

    messages: list[str] = []
    df = sphn_gdf.copy()
    year_series = _extract_year_from_columns(
        df,
        list(SPHN_YEAR_FIELD_CANDIDATES),
    )
    if year_series is None:
        messages.append("SPHN layer has no recognised year or date field.")
        return None, messages

    df = df.loc[year_series == int(year)].copy()
    if df.empty:
        return None, messages
    return df, messages


def _explode_union_geometry(geometry, crs, source_label: str, prefix: str):
    """Return non-overlapping polygon components from a union/difference geometry."""
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    try:
        pieces = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    except Exception:
        pieces = [geometry]
    pieces = [g for g in pieces if g is not None and not g.is_empty and g.geom_type in {"Polygon", "MultiPolygon"}]
    if not pieces:
        return None
    out = gpd.GeoDataFrame({"geometry": pieces}, geometry="geometry", crs=crs)
    out = out.explode(index_parts=False, ignore_index=True)
    out["cluster_id"] = [f"{prefix}_{i}" for i in range(len(out))]
    out["cluster_source"] = source_label
    return out[["geometry", "cluster_id", "cluster_source"]]


def build_real_host_landscape(
    host_gdf: Any | None,
    survey_layers: dict[str, Any] | None = None,
    host_column_name: str | None = None,
    host_values: list[str] | tuple[str, ...] | set[str] | None = None,
    include_unmatched_survey_polygons: bool = True,
):
    """Build a non-overlapping operational host landscape.

    Public host polygons are the primary host evidence.  When requested, the
    portions of DEFRA survey footprints lying outside that public layer are
    added as *inferred host area*.  This reflects the operational assumption
    that a larch-targeted DEFRA survey is evidence that larch was present even
    where the public host map is incomplete.  Only the uncovered geometry is
    added, so host area is never duplicated merely because the two sources
    overlap.
    """
    if not _gis_available():
        return None

    host_proj = _to_bng(host_gdf)
    host_union = None
    crs = getattr(host_proj, "crs", None) if host_proj is not None else None
    host_frame = None
    if host_proj is not None and len(host_proj) > 0:
        host_work = host_proj.copy()
        host_col = str(host_column_name or "").strip()
        if host_col and host_col != "__all__" and host_col in host_work.columns:
            selected_values = {_normalize_status_text(x) for x in (host_values or []) if _normalize_status_text(x)}
            if selected_values:
                keep_mask = host_work[host_col].fillna("").astype(str).map(_normalize_status_text).isin(selected_values)
                host_work = host_work.loc[keep_mask].copy()
        host_work = host_work.loc[host_work.geometry.notna() & ~host_work.geometry.is_empty].copy()
        if len(host_work) > 0:
            try:
                host_union = host_work.geometry.union_all()
            except Exception:
                host_union = host_work.geometry.unary_union
            host_frame = _explode_union_geometry(host_union, host_work.crs, "public_host_polygon", "host")
            crs = host_work.crs

    survey_parts: list[Any] = []
    for gdf in (survey_layers or {}).values():
        if gdf is None or len(gdf) == 0:
            continue
        survey_proj = _to_bng(gdf)
        if survey_proj is None or len(survey_proj) == 0:
            continue
        if crs is not None and getattr(survey_proj, "crs", None) is not None and survey_proj.crs != crs:
            survey_proj = survey_proj.to_crs(crs)
        work = survey_proj.loc[survey_proj.geometry.notna() & ~survey_proj.geometry.is_empty, ["geometry"]].copy()
        if len(work):
            survey_parts.append(work)
            if crs is None:
                crs = survey_proj.crs

    inferred_frame = None
    if include_unmatched_survey_polygons and survey_parts:
        survey_all = gpd.GeoDataFrame(pd.concat(survey_parts, ignore_index=True), geometry="geometry", crs=survey_parts[0].crs)
        try:
            survey_union = survey_all.geometry.union_all()
        except Exception:
            survey_union = survey_all.geometry.unary_union
        inferred_geom = survey_union if host_union is None else survey_union.difference(host_union)
        inferred_frame = _explode_union_geometry(inferred_geom, survey_all.crs, "defra_survey_inferred_host", "survey_inferred")

    frames = [frame for frame in (host_frame, inferred_frame) if frame is not None and len(frame) > 0]
    if not frames:
        return None
    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    combined["cluster_area_km2"] = pd.to_numeric(combined.geometry.area, errors="coerce").fillna(0.0) / 1_000_000.0
    return combined


def allocate_surveys_to_host_clusters(
    host_landscape: Any | None,
    survey_gdf: Any | None,
    host_density_km2: float,
    positive_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    negative_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    status_column_name: str | None = None,
) -> pd.DataFrame:
    """Convert survey footprints into non-overlapping host-area pseudo-counts.

    Operational assumptions:
    * every host inside a surveyed host footprint was inspected; and
    * every host inside a polygon classified positive is treated as infected.

    Areas are based on true survey/host intersections.  Overlapping survey
    polygons are dissolved within each host cluster before areas are measured,
    preventing repeated coverage of the same land from being double counted.
    """
    cols = [
        "cluster_id", "cluster_source", "cluster_area_km2", "cluster_host_count",
        "survey_polygons", "surveyed_area_km2", "infected_polygons", "infected_area_km2",
        "surveyed_hosts_est", "infected_hosts_est", "estimated_prevalence",
    ]
    if not _gis_available() or host_landscape is None or len(host_landscape) == 0 or survey_gdf is None or len(survey_gdf) == 0:
        return pd.DataFrame(columns=cols)

    clusters = _to_bng(host_landscape).copy().reset_index(drop=True)
    surveys = _to_bng(survey_gdf).copy().reset_index(drop=True)
    if clusters is None or surveys is None or len(clusters) == 0 or len(surveys) == 0:
        return pd.DataFrame(columns=cols)
    if getattr(surveys, "crs", None) is not None and getattr(clusters, "crs", None) is not None and surveys.crs != clusters.crs:
        surveys = surveys.to_crs(clusters.crs)

    density = max(float(host_density_km2), 0.0)
    clusters["cluster_area_km2"] = pd.to_numeric(clusters.geometry.area, errors="coerce").fillna(0.0) / 1_000_000.0
    clusters["cluster_host_count"] = np.maximum(0.0, clusters["cluster_area_km2"] * density)
    surveys["_survey_index"] = np.arange(len(surveys), dtype=int)

    status_col = _status_column(surveys, preferred=status_column_name)
    if status_col is not None:
        include_flag, positive_flag = _classify_survey_status(
            surveys[status_col],
            positive_status_values=positive_status_values,
            negative_status_values=negative_status_values,
        )
        surveys = surveys.loc[include_flag].copy()
        surveys["positive_flag"] = positive_flag.loc[include_flag].to_numpy(dtype=bool)
    else:
        surveys["positive_flag"] = False
    if surveys.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, Any]] = []
    # Spatial index keeps this loop practical while unioning coverage correctly.
    sindex = surveys.sindex
    for cluster in clusters.itertuples(index=False):
        geom = cluster.geometry
        if geom is None or geom.is_empty:
            continue
        candidate_idx = list(sindex.query(geom, predicate="intersects"))
        if not candidate_idx:
            continue
        cand = surveys.iloc[candidate_idx].copy()
        cand["_intersection"] = cand.geometry.intersection(geom)
        cand = cand.loc[cand["_intersection"].notna() & ~cand["_intersection"].is_empty].copy()
        if cand.empty:
            continue
        all_geoms = gpd.GeoSeries(cand["_intersection"], crs=clusters.crs)
        try:
            surveyed_union = all_geoms.union_all()
        except Exception:
            surveyed_union = all_geoms.unary_union
        pos = cand.loc[cand["positive_flag"]]
        infected_union = None
        if not pos.empty:
            pos_geoms = gpd.GeoSeries(pos["_intersection"], crs=clusters.crs)
            try:
                infected_union = pos_geoms.union_all()
            except Exception:
                infected_union = pos_geoms.unary_union
        surveyed_area = float(surveyed_union.area) / 1_000_000.0 if surveyed_union is not None else 0.0
        infected_area = float(infected_union.area) / 1_000_000.0 if infected_union is not None and not infected_union.is_empty else 0.0
        cluster_hosts = float(cluster.cluster_host_count)
        surveyed_hosts = min(cluster_hosts, max(0.0, surveyed_area * density))
        infected_hosts = min(surveyed_hosts, max(0.0, infected_area * density))
        rows.append({
            "cluster_id": cluster.cluster_id,
            "cluster_source": cluster.cluster_source,
            "cluster_area_km2": float(cluster.cluster_area_km2),
            "cluster_host_count": cluster_hosts,
            "survey_polygons": int(cand["_survey_index"].nunique()),
            "surveyed_area_km2": surveyed_area,
            "infected_polygons": int(pos["_survey_index"].nunique()),
            "infected_area_km2": infected_area,
            "surveyed_hosts_est": surveyed_hosts,
            "infected_hosts_est": infected_hosts,
            "estimated_prevalence": infected_hosts / surveyed_hosts if surveyed_hosts > 0 else np.nan,
        })
    return pd.DataFrame(rows, columns=cols)


# ============================================================================
# Survey Status, Host Filtering, And Benchmark Settings
# ============================================================================
#
# Different datasets may use different text labels for survey results.  This
# section decides which labels mean "positive", which mean "negative", and which
# should be ignored because they do not clearly state disease presence/absence.


def summarise_target_sites(gdf: Any | None) -> pd.DataFrame:
    # Simple summary used in the Data Viewer.
    if gdf is None or len(gdf) == 0:
        return pd.DataFrame(columns=["metric", "value"])

    if "Site status" in gdf.columns:
        status_col = "Site status"
    elif "Site status " in gdf.columns:
        status_col = "Site status "
    else:
        status_col = None

    rows: list[dict[str, Any]] = [
        {"metric": "Survey polygons", "value": int(len(gdf))},
        {"metric": "Columns", "value": int(len(gdf.columns))},
    ]
    if status_col is not None:
        rows.append({"metric": "Unique site statuses", "value": int(gdf[status_col].dropna().nunique())})

    return pd.DataFrame(rows)


def _status_column(gdf: Any | None, preferred: str | None = None) -> str | None:
    # Find the survey result/status field while tolerating small naming changes.
    if gdf is None:
        return None
    columns = [str(col) for col in gdf.columns if str(col) != "geometry"]
    if preferred is not None:
        pref = str(preferred).strip()
        if pref and pref != "__auto__" and pref in gdf.columns:
            return pref
    normalised = {_normalise_column_name(col): col for col in columns}
    for candidate in SURVEY_STATUS_FIELD_CANDIDATES:
        col = normalised.get(_normalise_column_name(candidate))
        if col is not None:
            return col
    # Controlled fallback: prefer a low-cardinality text field whose name clearly
    # describes status/outcome/result. This handles minor future schema changes
    # without guessing from unrelated fields such as species or disease name.
    candidates: list[tuple[int, str]] = []
    for col in columns:
        key = _normalise_column_name(col)
        looks_like_status = (
            any(token in key for token in ("status", "outcome", "result"))
            or key.startswith(("surveyres", "surveyout", "siteres", "siteout", "survres", "survout"))
        )
        if not looks_like_status:
            continue
        try:
            nunique = int(gdf[col].dropna().astype(str).nunique())
        except Exception:
            continue
        if 1 <= nunique <= 100:
            candidates.append((nunique, col))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1].lower()))
        return candidates[0][1]
    return None


def _extract_status_column_choices_from_layers(layers: dict[str, Any]) -> tuple[list[str], str]:
    seen: dict[str, str] = {}
    auto_default = "__auto__"
    default_selected = auto_default
    for gdf in (layers or {}).values():
        if gdf is None or len(gdf) == 0:
            continue
        for col in gdf.columns:
            if col == "geometry":
                continue
            if col not in seen:
                seen[col] = col
        auto_col = _status_column(gdf)
        if auto_col is not None and default_selected == auto_default:
            default_selected = auto_col
    choices = [auto_default] + sorted(seen.values(), key=lambda x: str(x).lower())
    return choices, default_selected


def _normalize_status_text(value: Any) -> str:
    return str("" if value is None else value).strip().lower()


def _default_positive_status_mask(series: pd.Series) -> pd.Series:
    # Conservative defaults for common positive labels.  Explicit negation is
    # excluded so a future label such as "not infected" is not misclassified.
    vals = series.fillna("").astype(str).str.strip().str.lower()
    positive_tokens = [
        "clear evidence", "confirmed infected", "assumed infected",
        "positive", "disease present", "infection present", "confirmed positive",
    ]
    negative_phrases = ["no evidence", "not infected", "uninfected", "negative", "no disease", "not detected"]
    return vals.apply(lambda x: (not any(tok in x for tok in negative_phrases)) and any(tok in x for tok in positive_tokens))


def _default_negative_status_mask(series: pd.Series) -> pd.Series:
    # Conservative defaults for common negative labels. Ambiguous workflow
    # states (for example awaiting a visit) remain unclassified.
    vals = series.fillna("").astype(str).str.strip().str.lower()
    negative_tokens = [
        "no evidence", "no_evidence", "negative", "not infected",
        "uninfected", "no disease", "not detected", "disease absent",
    ]
    return vals.apply(lambda x: any(tok in x for tok in negative_tokens))


def _retrospective_benchmark_assumptions(
    mode: str,
    *,
    fixed_phi: float = 1.5,
    within_cluster_cap: int = 5,
) -> dict[str, Any]:
    """Return transparent planning assumptions for the retrospective effort benchmark.

    The survey polygons do not record the actual within-area sampling design.  The
    dashboard therefore presents alternative planning references rather than labelling
    one hidden assumption as optimistic/moderate/cautious.  When the pilot-estimated
    option is selected, the Design Effect is an approximation reconstructed from
    polygon coverage and polygon status, not a host-level ICC estimate.
    """
    mode = str(mode)
    cap = max(1, int(within_cluster_cap))
    fixed = max(1.0, float(fixed_phi))
    mapping = {
        "srs": {
            "label": "SRS reference; no Design Effect inflation",
            "sampling_mode": "srs",
            "cluster_inflation_mode": "none",
            "fixed_phi": 1.0,
            "within_cluster_cap": cap,
            "note": "Reference calculation assumes simple random sampling.",
        },
        "mss_no_inflation": {
            "label": "MSS reference; no Design Effect inflation",
            "sampling_mode": "multistage",
            "cluster_inflation_mode": "none",
            "fixed_phi": 1.0,
            "within_cluster_cap": cap,
            "note": "Reference calculation uses the MSS structure without sample-size inflation.",
        },
        "mss_pilot": {
            "label": "MSS with approximate polygon-based Design Effect",
            "sampling_mode": "multistage",
            "cluster_inflation_mode": "pilot",
            "fixed_phi": 1.0,
            "within_cluster_cap": cap,
            "note": (
                "Design Effect is approximated from survey-polygon coverage and polygon status because "
                "host-level inspection records are not available."
            ),
        },
        "mss_fixed": {
            "label": f"MSS with fixed Design Effect {fixed:.2f}",
            "sampling_mode": "multistage",
            "cluster_inflation_mode": "fixed",
            "fixed_phi": fixed,
            "within_cluster_cap": cap,
            "note": "Design Effect is supplied by the user as a planning assumption.",
        },
    }
    return mapping.get(mode, mapping["srs"]).copy()


def _is_positive_status(series: pd.Series, positive_status_values: list[str] | tuple[str, ...] | set[str] | None = None) -> pd.Series:
    if positive_status_values is None:
        return _default_positive_status_mask(series)
    selected = {_normalize_status_text(x) for x in positive_status_values if _normalize_status_text(x)}
    if not selected:
        return pd.Series(False, index=series.index)
    vals = series.fillna("").astype(str).map(_normalize_status_text)
    return vals.isin(selected)


def _classify_survey_status(
    series: pd.Series,
    positive_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    negative_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    # Convert messy status text into two simple flags:
    #   include_flag = use this survey result in the analysis
    #   positive_flag = the included result is disease-positive
    vals = series.fillna("").astype(str).map(_normalize_status_text)
    if positive_status_values is None and negative_status_values is None:
        # Conservative automatic classification: only clearly positive or
        # clearly negative outcomes are analysed. Ambiguous workflow/status
        # values are excluded rather than silently treated as disease-negative.
        positive_flag = _default_positive_status_mask(series)
        negative_flag = _default_negative_status_mask(series)
        include_flag = positive_flag | negative_flag
        return include_flag.astype(bool), positive_flag.astype(bool)
    positive_set = {_normalize_status_text(x) for x in (positive_status_values or []) if _normalize_status_text(x)}
    negative_set = {_normalize_status_text(x) for x in (negative_status_values or []) if _normalize_status_text(x)}
    include_flag = vals.isin(positive_set | negative_set)
    positive_flag = vals.isin(positive_set)
    return include_flag.astype(bool), positive_flag.astype(bool)


def _extract_status_choices_from_layers(
    layers: dict[str, Any],
    status_column_name: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    seen: dict[str, str] = {}
    default_positive: list[str] = []
    default_negative: list[str] = []
    for gdf in (layers or {}).values():
        if gdf is None or len(gdf) == 0:
            continue
        status_col = _status_column(gdf, preferred=status_column_name)
        if status_col is None:
            continue
        raw_vals = gdf[status_col].dropna().astype(str).map(str.strip)
        for raw in raw_vals.tolist():
            if not raw:
                continue
            key = _normalize_status_text(raw)
            if key not in seen:
                seen[key] = raw
    ordered = sorted(seen.values(), key=lambda x: _normalize_status_text(x))
    if ordered:
        pos_mask = _default_positive_status_mask(pd.Series(ordered))
        neg_mask = _default_negative_status_mask(pd.Series(ordered))
        default_positive = [val for val, is_pos in zip(ordered, pos_mask.tolist()) if is_pos]
        default_negative = [val for val, is_neg in zip(ordered, neg_mask.tolist()) if is_neg]
    return ordered, default_positive, default_negative


def _extract_host_column_choices_from_layer(host_gdf: Any | None) -> tuple[list[str], str]:
    all_default = "__all__"
    if host_gdf is None or len(host_gdf) == 0:
        return [all_default], all_default
    cols: list[str] = []
    for col in host_gdf.columns:
        col_str = str(col)
        if col_str == "geometry":
            continue
        series = host_gdf[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
            nunique = int(series.dropna().astype(str).nunique())
            if 1 <= nunique <= 50:
                cols.append(col_str)
    return [all_default] + sorted(cols, key=lambda x: x.lower()), all_default


def _extract_host_value_choices_from_layer(host_gdf: Any | None, host_column_name: str | None = None) -> tuple[list[str], list[str]]:
    if host_gdf is None or len(host_gdf) == 0:
        return [], []
    col = str(host_column_name or "").strip()
    if not col or col == "__all__" or col not in host_gdf.columns:
        return [], []
    raw_vals = host_gdf[col].dropna().astype(str).map(str.strip)
    choices = sorted([val for val in raw_vals.unique().tolist() if val], key=lambda x: x.lower())
    return choices, choices


def _clamp_open01(x: float, eps: float = 1e-8) -> float:
    return float(min(1.0 - eps, max(eps, x)))


def _logit(x: float) -> float:
    x = _clamp_open01(float(x))
    return float(np.log(x / (1.0 - x)))


def _inv_logit(eta: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0))))


def _apply_fpc(n: float, population_size: int | float | None) -> int:
    # Finite population correction.
    # If the required sample is a large share of the total population, this
    # reduces the required sample size because sampling without replacement gives
    # more information than sampling from an infinite population.
    if population_size is None or not np.isfinite(population_size) or population_size <= 0:
        return max(1, int(np.ceil(n)))
    n = max(1.0, float(n))
    population_size = float(population_size)
    if n <= 0.05 * population_size:
        return max(1, int(np.ceil(n)))
    n_adj = (n * population_size) / (population_size + n)
    return max(1, int(np.ceil(n_adj)))


def compute_n_change_method_cells(
    p0: float,
    delta: float = DEFAULT_CHANGE_METHOD_DELTA,
    corr: float = DEFAULT_CHANGE_METHOD_CORR,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    population_size: int | float | None = None,
    design_effect: float = 1.0,
) -> int:
    """Change Method sample size from the frozen production backend."""
    return int(
        _backend_compute_n_change_method_cells(
            p0=float(p0),
            delta=float(delta),
            corr=float(corr),
            alpha=float(alpha),
            power=float(power),
            population_size=population_size,
            design_effect=float(design_effect),
        )
    )


def compute_n_baseline_prevalence_cells(
    p0_upper: float = DEFAULT_BASELINE_PREVALENCE_UPPER,
    conf_level: float = DEFAULT_BASELINE_PREVALENCE_CONF,
    width: float = DEFAULT_BASELINE_PREVALENCE_WIDTH,
    population_size: int | float | None = None,
) -> int:
    """Exact-binomial initial survey size from the frozen production backend."""
    return int(
        _backend_compute_n_baseline_prevalence_cells(
            p0_upper=float(p0_upper),
            conf_level=float(conf_level),
            width=float(width),
            population_size=population_size,
        )
    )


def compute_n_total_regression_method_cells(
    pi0: float,
    pi_target: float,
    alpha: float,
    power: float,
    t_vec: np.ndarray,
    population_size: int | float | None = None,
    phi: float = 1.0,
    initial_n: int | float = 0,
) -> int:
    """Regression Method follow-up sample size from the frozen production backend."""
    return int(
        _backend_compute_n_total_regression_method_cells(
            pi0=float(pi0),
            pi_target=float(pi_target),
            alpha=float(alpha),
            power=float(power),
            t_vec=np.asarray(t_vec, dtype=float),
            population_size=population_size,
            phi=float(phi),
            initial_n=float(initial_n),
        )
    )


# ============================================================================
# Results Tables, Trend Fitting, And Spatial Diagnostics
# ============================================================================


def run_regression_method_real(prevalence_ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_regression_method_real_with_model(prevalence_ts, model_form="logistic")


def run_regression_method_real_with_model(prevalence_ts: pd.DataFrame, model_form: str = "logistic") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the Regression Method to polygon-derived yearly counts using the backend.

    Host-level identifiers are not present in the source survey files, so the dashboard
    uses the backend grouped-binomial fit and its model-based covariance.  The backend
    convergence, separation and conditioning diagnostics are retained and invalid fits
    are not presented as successful estimates.
    """
    if prevalence_ts.empty or len(prevalence_ts) < 2:
        return (
            pd.DataFrame([{"metric": "Regression Method note", "value": "At least two yearly observations are required"}]),
            prevalence_ts.copy(),
        )

    fit = prevalence_ts.copy().sort_values("year").reset_index(drop=True)
    t = fit["year"].to_numpy(dtype=float) - float(fit["year"].min())
    model_form = str(model_form).lower()
    x = fit["x"].to_numpy(dtype=float)
    n = fit["n"].to_numpy(dtype=float)
    beta, cov, diagnostics = _backend_fit_logistic_counts(
        t,
        x,
        n,
        model_form=model_form,
        return_diagnostics=True,
    )
    fit_valid = bool(diagnostics.get("fit_valid", False))

    fit["model_pi_hat"] = np.nan
    fit["model_ci_low"] = np.nan
    fit["model_ci_high"] = np.nan

    if fit_valid:
        pred = np.asarray(_backend_predict_logistic(beta, t, model_form=model_form), dtype=float)
        fit["model_pi_hat"] = pred
        if model_form == "logistic":
            feature = t
        elif model_form == "fp2":
            feature = np.power(t + 1.0, 2.0)
        elif model_form == "fp3":
            feature = np.power(t + 1.0, 3.0)
        else:
            feature = t
        z = 1.96
        vec = np.column_stack([np.ones_like(feature), feature])
        var_eta = np.einsum("ij,jk,ik->i", vec, np.asarray(cov, dtype=float), vec)
        se_eta = np.sqrt(np.clip(var_eta, 0.0, None))
        eta = np.log(np.clip(pred, 1e-12, 1 - 1e-12) / np.clip(1.0 - pred, 1e-12, 1.0))
        fit["model_ci_low"] = 1.0 / (1.0 + np.exp(-np.clip(eta - z * se_eta, -30.0, 30.0)))
        fit["model_ci_high"] = 1.0 / (1.0 + np.exp(-np.clip(eta + z * se_eta, -30.0, 30.0)))
        slope = float(beta[1])
        slope_se = float(np.sqrt(max(float(cov[1, 1]), 0.0)))
        slope_low = slope - z * slope_se
        slope_high = slope + z * slope_se
        delta = float(pred[-1] - pred[0])
        declining = bool(delta < 0.0)
    else:
        slope = slope_low = slope_high = delta = np.nan
        declining = False

    summary = pd.DataFrame(
        [
            {"metric": "Model form", "value": model_form},
            {"metric": "Regression Method yearly points", "value": int(len(fit))},
            {"metric": "Fit valid", "value": fit_valid},
            {"metric": "Fit converged", "value": bool(diagnostics.get("converged", False))},
            {"metric": "Fit reason", "value": str(diagnostics.get("reason", "unknown"))},
            {"metric": "Separation suspected", "value": bool(diagnostics.get("separation_suspected", False))},
            {"metric": "Information-matrix condition number", "value": float(diagnostics.get("condition_number", np.nan))},
            {"metric": "Covariance basis", "value": "Model-based grouped-binomial; host IDs unavailable in polygon survey data"},
            {"metric": "Modelled prevalence start", "value": round(float(fit["model_pi_hat"].iloc[0]), 4) if fit_valid else np.nan},
            {"metric": "Modelled prevalence end", "value": round(float(fit["model_pi_hat"].iloc[-1]), 4) if fit_valid else np.nan},
            {"metric": "Modelled change over timeframe", "value": round(float(delta), 4) if fit_valid else np.nan},
            {"metric": "Trend slope (logit scale)", "value": round(float(slope), 4) if fit_valid else np.nan},
            {"metric": "Trend slope 95% CI low", "value": round(float(slope_low), 4) if fit_valid else np.nan},
            {"metric": "Trend slope 95% CI high", "value": round(float(slope_high), 4) if fit_valid else np.nan},
            {"metric": "Declining trend", "value": bool(declining) if fit_valid else np.nan},
        ]
    )
    return summary, fit


def build_quick_results_year_summary(layers: dict[str, Any], year_min: int = YEAR_MIN, year_max: int = YEAR_MAX) -> pd.DataFrame:
    # Older shortcut kept for compatibility. The main dashboard now normally
    # uses build_quick_results_host_summary because it includes host polygons.
    return build_quick_results_host_summary(layers, None, host_density_km2=2500.0, year_min=year_min, year_max=year_max)


def build_quick_results_host_summary(
    layers: dict[str, Any],
    host_gdf: Any | None,
    host_density_km2: float,
    positive_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    negative_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    status_column_name: str | None = None,
    host_column_name: str | None = None,
    host_values: list[str] | tuple[str, ...] | set[str] | None = None,
    include_unmatched_survey_polygons: bool = True,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
    years: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    # Build the yearly prevalence table used by the Results tab.
    # Each year is handled separately, then survey area is converted into
    # estimated sampled hosts and estimated infected hosts.
    host_landscape = build_real_host_landscape(
        host_gdf,
        layers,
        host_column_name=host_column_name,
        host_values=host_values,
        include_unmatched_survey_polygons=include_unmatched_survey_polygons,
    )
    rows: list[dict[str, Any]] = []
    host_density_km2 = max(float(host_density_km2), 0.0)
    if years is None:
        analysis_years = list(range(int(year_min), int(year_max) + 1))
    else:
        analysis_years = sorted({int(y) for y in years})
    for year in analysis_years:
        # Pull out only the survey polygons that belong to this year.
        gdf_year, _ = filter_target_sites_for_year(layers, year)
        if gdf_year is None or len(gdf_year) == 0:
            rows.append(
                {
                    "year": year,
                    "survey_polygons": 0,
                    "surveyed_area_km2": 0.0,
                    "infected_polygons": 0,
                    "infected_area_km2": 0.0,
                    "surveyed_hosts_est": 0.0,
                    "infected_hosts_est": 0.0,
                    "estimated_prevalence": np.nan,
                }
            )
            continue

        status_col = _status_column(gdf_year, preferred=status_column_name)
        survey_polygon_count = int(len(gdf_year))
        infected_polygon_count = 0
        if status_col is not None and status_col in gdf_year.columns:
            include_flag, positive_flag = _classify_survey_status(
                gdf_year[status_col],
                positive_status_values=positive_status_values,
                negative_status_values=negative_status_values,
            )
            survey_polygon_count = int(include_flag.sum())
            infected_polygon_count = int((include_flag & positive_flag).sum())

        cluster_alloc = allocate_surveys_to_host_clusters(
            host_landscape=host_landscape,
            survey_gdf=gdf_year,
            host_density_km2=host_density_km2,
            positive_status_values=positive_status_values,
            negative_status_values=negative_status_values,
            status_column_name=status_column_name,
        )
        # Add up all host-polygon summaries to get one prevalence estimate for the year.
        surveyed_area_km2 = float(pd.to_numeric(cluster_alloc["surveyed_area_km2"], errors="coerce").fillna(0.0).sum()) if not cluster_alloc.empty else 0.0
        infected_area_km2 = float(pd.to_numeric(cluster_alloc["infected_area_km2"], errors="coerce").fillna(0.0).sum()) if not cluster_alloc.empty else 0.0
        surveyed_hosts_est = float(pd.to_numeric(cluster_alloc["surveyed_hosts_est"], errors="coerce").fillna(0.0).sum()) if not cluster_alloc.empty else 0.0
        infected_hosts_est = float(pd.to_numeric(cluster_alloc["infected_hosts_est"], errors="coerce").fillna(0.0).sum()) if not cluster_alloc.empty else 0.0
        prevalence = infected_hosts_est / surveyed_hosts_est if surveyed_hosts_est > 0 else np.nan
        rows.append(
            {
                "year": year,
                "survey_polygons": survey_polygon_count,
                "surveyed_area_km2": surveyed_area_km2,
                "infected_polygons": infected_polygon_count,
                "infected_area_km2": infected_area_km2,
                "surveyed_hosts_est": surveyed_hosts_est,
                "infected_hosts_est": infected_hosts_est,
                "estimated_prevalence": prevalence,
            }
        )
    return pd.DataFrame(rows)


def plot_quick_results_trend(prevalence_ts: pd.DataFrame, regression_method_fit: pd.DataFrame, year_before: int, year_after: int):
    """Create the principal report/dashboard prevalence figure.

    The styling is intentionally restrained so the figure remains legible when
    embedded in the PDF at A4 size.  Raw yearly estimates remain visually
    distinct from the fitted Regression Method and the two-point Change Method.
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.25))
    if prevalence_ts.empty:
        ax.text(0.5, 0.5, "Generate results to view prevalence estimates.", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    years = pd.to_numeric(prevalence_ts["year"], errors="coerce")
    obs = pd.to_numeric(prevalence_ts["pi_hat"], errors="coerce")
    valid_obs = np.isfinite(years) & np.isfinite(obs)

    # Light horizontal guides only; remove visual clutter around the data.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.scatter(
        years[valid_obs], obs[valid_obs],
        s=42, zorder=4, label="Yearly survey estimate"
    )

    if regression_method_fit is not None and "model_pi_hat" in regression_method_fit.columns:
        model_year = pd.to_numeric(regression_method_fit.get("year"), errors="coerce")
        model_vals = pd.to_numeric(regression_method_fit["model_pi_hat"], errors="coerce")
        valid_model = np.isfinite(model_year) & np.isfinite(model_vals)
        if int(valid_model.sum()) >= 2:
            ax.plot(
                model_year[valid_model], model_vals[valid_model],
                linewidth=2.4, label="Regression Method", zorder=3
            )

    before_row = prevalence_ts.loc[pd.to_numeric(prevalence_ts["year"], errors="coerce") == int(year_before)]
    after_row = prevalence_ts.loc[pd.to_numeric(prevalence_ts["year"], errors="coerce") == int(year_after)]
    if not before_row.empty and not after_row.empty:
        ax.plot(
            [year_before, year_after],
            [float(before_row["pi_hat"].iloc[0]), float(after_row["pi_hat"].iloc[0])],
            linewidth=2.1, linestyle="--", marker="s", markersize=5,
            label="Change Method", zorder=2
        )

    # Prevalence is intrinsically bounded; always show the full scale in the
    # report so apparently dramatic movements are not created by axis cropping.
    ax.set_ylim(0.0, 1.0)
    if valid_obs.any():
        xmin, xmax = int(np.nanmin(years[valid_obs])), int(np.nanmax(years[valid_obs]))
        ax.set_xlim(xmin - 0.35, xmax + 0.35)
        ax.set_xticks(sorted(set(years[valid_obs].astype(int).tolist())))
    ax.set_ylabel("Estimated prevalence")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _pos: f"{100*y:.0f}%"))
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=8.5)
    fig.tight_layout(pad=0.8)
    return fig

def _concat_survey_layers(layers: dict[str, Any]) -> Any | None:
    if not _gis_available():
        return None
    frames: list[Any] = []
    for gdf in (layers or {}).values():
        if gdf is None or len(gdf) == 0:
            continue
        work = _to_bng(gdf)
        if work is None or len(work) == 0:
            continue
        work = work.loc[work.geometry.notna()].copy()
        work = work.loc[~work.geometry.is_empty].copy()
        if len(work) > 0:
            frames.append(work)
    if not frames:
        return None
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)


def _nearest_neighbour_distance_ratio(points_xy: np.ndarray, scale_m: float) -> float:
    if points_xy.shape[0] < 2 or not np.isfinite(scale_m) or scale_m <= 0:
        return np.nan
    dx = points_xy[:, 0][:, None] - points_xy[:, 0][None, :]
    dy = points_xy[:, 1][:, None] - points_xy[:, 1][None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return np.nan
    return float(np.median(nearest) / scale_m)


def build_spatial_survey_diagnostics(
    layers: dict[str, Any],
    host_gdf: Any | None,
    host_column_name: str | None = None,
    host_values: list[str] | tuple[str, ...] | set[str] | None = None,
    include_unmatched_survey_polygons: bool = True,
) -> pd.DataFrame:
    # Build simple measures that describe the shape of the survey footprint.
    # These diagnostics are not prevalence estimates.  They are warning signs
    # about whether the survey data looks broad/random or narrow/targeted.
    cols = ["metric", "value"]
    if not _gis_available():
        return pd.DataFrame(cols=cols)
    host_landscape = build_real_host_landscape(
        host_gdf,
        layers,
        host_column_name=host_column_name,
        host_values=host_values,
        include_unmatched_survey_polygons=include_unmatched_survey_polygons,
    )
    survey_all = _concat_survey_layers(layers)
    if host_landscape is None or len(host_landscape) == 0 or survey_all is None or len(survey_all) == 0:
        return pd.DataFrame(cols=cols)

    clusters = _to_bng(host_landscape).copy().reset_index(drop=True)
    surveys = _to_bng(survey_all).copy().reset_index(drop=True)
    if clusters is None or surveys is None or len(clusters) == 0 or len(surveys) == 0:
        return pd.DataFrame(cols=cols)

    clusters["cluster_area_km2"] = pd.to_numeric(clusters.geometry.area, errors="coerce").fillna(0.0) / 1_000_000.0
    clusters["_cluster_index"] = np.arange(len(clusters), dtype=int)
    clusters["cluster_scale_m"] = np.sqrt(np.maximum(clusters["cluster_area_km2"].to_numpy(dtype=float), 1e-12)) * 1000.0

    surveys["_survey_index"] = np.arange(len(surveys), dtype=int)
    surveys["survey_area_km2"] = pd.to_numeric(surveys.geometry.area, errors="coerce").fillna(0.0) / 1_000_000.0
    centroids = surveys.geometry.centroid
    surveys["survey_cx"] = pd.to_numeric(centroids.x, errors="coerce")
    surveys["survey_cy"] = pd.to_numeric(centroids.y, errors="coerce")

    joined = gpd.sjoin(
        # Link each survey polygon to the host polygon it intersects.
        surveys[["_survey_index", "survey_area_km2", "survey_cx", "survey_cy", "geometry"]].copy(),
        clusters[["_cluster_index", "cluster_id", "cluster_source", "cluster_area_km2", "cluster_scale_m", "geometry"]].copy(),
        how="left",
        predicate="intersects",
    )
    if joined.empty:
        return pd.DataFrame(cols=cols)
    joined = joined.merge(
        clusters[["_cluster_index", "geometry"]].rename(columns={"geometry": "cluster_geometry"}),
        on="_cluster_index",
        how="left",
    )
    joined["intersect_area_km2"] = joined.apply(
        lambda row: float(row.geometry.intersection(row.cluster_geometry).area) / 1_000_000.0
        if row.geometry is not None and row.cluster_geometry is not None
        else 0.0,
        axis=1,
    )
    joined = joined.sort_values(["_survey_index", "intersect_area_km2"], ascending=[True, False])
    # Keep only the strongest host-polygon match for each survey polygon.
    joined = joined.drop_duplicates(subset=["_survey_index"], keep="first").copy()
    if joined.empty:
        return pd.DataFrame(cols=cols)

    cluster_summary = (
        # Summarise survey concentration within each host polygon.
        joined.groupby("cluster_id", as_index=False)
        .agg(
            cluster_source=("cluster_source", "first"),
            cluster_area_km2=("cluster_area_km2", "first"),
            cluster_scale_m=("cluster_scale_m", "first"),
            survey_polygons=("_survey_index", "nunique"),
            surveyed_area_km2=("survey_area_km2", "sum"),
            largest_patch_area_km2=("survey_area_km2", "max"),
        )
        .copy()
    )
    cluster_summary["surveyed_share"] = np.where(
        cluster_summary["cluster_area_km2"] > 0,
        np.minimum(1.0, cluster_summary["surveyed_area_km2"] / cluster_summary["cluster_area_km2"]),
        np.nan,
    )
    cluster_summary["largest_patch_share"] = np.where(
        cluster_summary["surveyed_area_km2"] > 0,
        np.minimum(1.0, cluster_summary["largest_patch_area_km2"] / cluster_summary["surveyed_area_km2"]),
        np.nan,
    )

    spacing_rows: list[dict[str, Any]] = []
    for cluster_id, grp in joined.groupby("cluster_id"):
        points_xy = grp.loc[:, ["survey_cx", "survey_cy"]].to_numpy(dtype=float)
        scale_m = float(pd.to_numeric(grp["cluster_scale_m"], errors="coerce").iloc[0])
        spacing_rows.append(
            {
                "cluster_id": cluster_id,
                "nn_spacing_ratio": _nearest_neighbour_distance_ratio(points_xy, scale_m),
            }
        )
    spacing_df = pd.DataFrame(spacing_rows)
    cluster_summary = cluster_summary.merge(spacing_df, on="cluster_id", how="left")

    total_clusters = int(len(clusters))
    surveyed_clusters = int(len(cluster_summary))
    coverage_prop = float(surveyed_clusters / max(total_clusters, 1))
    total_host_area = float(pd.to_numeric(clusters["cluster_area_km2"], errors="coerce").fillna(0.0).sum())
    covered_host_area = float(pd.to_numeric(cluster_summary["cluster_area_km2"], errors="coerce").fillna(0.0).sum())
    covered_host_area_prop = covered_host_area / max(total_host_area, 1e-12)
    total_survey_polygons = int(len(joined))
    median_polygons_per_cluster = float(pd.to_numeric(cluster_summary["survey_polygons"], errors="coerce").median())

    weights = pd.to_numeric(cluster_summary["surveyed_area_km2"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if weights.size > 0 and float(weights.sum()) > 0:
        # Effective clusters is lower when most survey effort is concentrated in
        # only a few host polygons.
        weights_sorted = np.sort(weights)[::-1]
        top_n = max(1, int(np.ceil(0.10 * len(weights_sorted))))
        top10_share = float(weights_sorted[:top_n].sum() / max(weights_sorted.sum(), 1e-12))
        effective_clusters = float((weights.sum() ** 2) / max(np.sum(weights**2), 1e-12))
        effective_prop = effective_clusters / max(float(surveyed_clusters), 1.0)
    else:
        top10_share = np.nan
        effective_clusters = np.nan
        effective_prop = np.nan

    median_surveyed_share = float(pd.to_numeric(cluster_summary["surveyed_share"], errors="coerce").median())
    median_largest_patch_share = float(pd.to_numeric(cluster_summary["largest_patch_share"], errors="coerce").median())
    median_spacing_ratio = float(pd.to_numeric(cluster_summary["nn_spacing_ratio"], errors="coerce").dropna().median()) if cluster_summary["nn_spacing_ratio"].notna().any() else np.nan

    return pd.DataFrame(
        [
            {"metric": "Host polygons available", "value": total_clusters},
            {"metric": "Host polygons surveyed", "value": surveyed_clusters},
            {"metric": "Host polygon coverage proportion", "value": coverage_prop},
            {"metric": "Host area covered proportion", "value": covered_host_area_prop},
            {"metric": "Survey polygons analysed", "value": total_survey_polygons},
            {"metric": "Median survey polygons per surveyed host polygon", "value": median_polygons_per_cluster},
            {"metric": "Top 10% host polygons share of surveyed footprint", "value": top10_share},
            {"metric": "Effective surveyed host polygons", "value": effective_clusters},
            {"metric": "Effective surveyed host polygon proportion", "value": effective_prop},
            {"metric": "Median surveyed share within surveyed host polygons", "value": median_surveyed_share},
            {"metric": "Median largest patch share within surveyed host polygons", "value": median_largest_patch_share},
            {"metric": "Median nearest-neighbour spacing ratio within surveyed host polygons", "value": median_spacing_ratio},
        ]
    )


# ============================================================================
# Report Text, Formatting, And PDF Creation
# ============================================================================
#
# This section turns the numerical outputs into a concise three-page PDF:
# 1) result summary, 2) survey evidence/spatial coverage, 3) planning and audit
# record.  The report uses the same result tables as the interactive UI.


def _format_report_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def _series_lookup_value(df: pd.DataFrame, key_col: str, key: str, value_col: str = "value") -> Any:
    if df is None or df.empty or key_col not in df.columns or value_col not in df.columns:
        return None
    row = df.loc[df[key_col] == key, value_col]
    if row.empty:
        return None
    return row.iloc[0]


def _method_metric_value(df: pd.DataFrame, method: str, metric: str) -> Any:
    if df is None or df.empty:
        return None
    row = df.loc[(df["method"] == method) & (df["metric"] == metric), "value"]
    if row.empty:
        return None
    return row.iloc[0]


def _spatial_metric_value(df: pd.DataFrame, metric: str) -> Any:
    return _series_lookup_value(df, "metric", metric)


def _spatial_band(value: float | None, low: float, high: float, labels: tuple[str, str, str]) -> str | None:
    try:
        val = float(value)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    if val < low:
        return labels[0]
    if val < high:
        return labels[1]
    return labels[2]


def build_spatial_survey_narrative(spatial_df: pd.DataFrame) -> list[str]:
    if spatial_df is None or spatial_df.empty:
        return ["No spatial survey-pattern summary was available for the selected dataset."]

    coverage_prop = _spatial_metric_value(spatial_df, "Host polygon coverage proportion")
    top10_share = _spatial_metric_value(spatial_df, "Top 10% host polygons share of surveyed footprint")
    effective_prop = _spatial_metric_value(spatial_df, "Effective surveyed host polygon proportion")
    median_patch_share = _spatial_metric_value(spatial_df, "Median largest patch share within surveyed host polygons")
    median_spacing_ratio = _spatial_metric_value(spatial_df, "Median nearest-neighbour spacing ratio within surveyed host polygons")
    median_surveyed_share = _spatial_metric_value(spatial_df, "Median surveyed share within surveyed host polygons")
    surveyed_clusters = _spatial_metric_value(spatial_df, "Host polygons surveyed")
    total_clusters = _spatial_metric_value(spatial_df, "Host polygons available")

    coverage_label = _spatial_band(coverage_prop, 0.15, 0.40, ("narrow", "moderate", "broad"))
    concentration_label = _spatial_band(top10_share, 0.45, 0.65, ("broadly distributed", "moderately concentrated", "highly concentrated"))

    within_cluster_label = "unclear"
    try:
        patch_share = float(median_patch_share)
    except Exception:
        patch_share = np.nan
    try:
        spacing_ratio = float(median_spacing_ratio)
    except Exception:
        spacing_ratio = np.nan
    if np.isfinite(patch_share) or np.isfinite(spacing_ratio):
        if (np.isfinite(patch_share) and patch_share >= 0.75) or (np.isfinite(spacing_ratio) and spacing_ratio < 0.20):
            within_cluster_label = "strongly clustered"
        elif (np.isfinite(patch_share) and patch_share >= 0.50) or (np.isfinite(spacing_ratio) and spacing_ratio < 0.40):
            within_cluster_label = "moderately clustered"
        else:
            within_cluster_label = "relatively dispersed"

    lines: list[str] = []
    if surveyed_clusters is not None and total_clusters is not None and coverage_prop is not None:
        lines.append(
            f"Surveys fell within {_format_report_value(surveyed_clusters, digits=0)} of {_format_report_value(total_clusters, digits=0)} host polygons "
            f"({_format_percent(coverage_prop)} of those available), which indicates {coverage_label or 'unclear'} geographic coverage."
        )
    if top10_share is not None and effective_prop is not None:
        lines.append(
            f"Survey effort was {concentration_label or 'of unclear concentration'} across host polygons: "
            f"the top 10% of surveyed host polygons accounted for {_format_percent(top10_share)} of surveyed area, "
            f"and the equivalent number of evenly surveyed host polygons was {_format_report_value(_spatial_metric_value(spatial_df, 'Effective surveyed host polygons'), digits=1)} "
            f"({_format_percent(effective_prop)} of those surveyed)."
        )
    if median_surveyed_share is not None or median_patch_share is not None:
        lines.append(
            f"Within host polygons, survey placement was {within_cluster_label}: the median surveyed share of a host polygon was "
            f"{_format_percent(median_surveyed_share)}, and the median share of surveyed area contained in the largest local patch was "
            f"{_format_percent(median_patch_share)}."
        )

    targeted_score = 0
    try:
        if float(coverage_prop) < 0.15:
            targeted_score += 1
    except Exception:
        pass
    try:
        if float(top10_share) >= 0.65:
            targeted_score += 1
    except Exception:
        pass
    try:
        if float(effective_prop) < 0.35:
            targeted_score += 1
    except Exception:
        pass
    try:
        if float(median_patch_share) >= 0.75:
            targeted_score += 1
    except Exception:
        pass
    try:
        if np.isfinite(float(median_spacing_ratio)) and float(median_spacing_ratio) < 0.20:
            targeted_score += 1
    except Exception:
        pass

    if targeted_score >= 2:
        lines.append(
            "Overall, the survey pattern looks more like targeted detection surveying than broad, representative prevalence sampling."
        )
    else:
        lines.append(
            "Overall, the survey pattern shows some concentration, but not enough on its own to rule out broader surveillance coverage."
        )
    return lines


def _direction_label(change: Any, tolerance: float = 1e-12) -> str | None:
    try:
        change_val = float(change)
    except Exception:
        return None
    if not np.isfinite(change_val):
        return None
    if change_val < -tolerance:
        return "decrease"
    if change_val > tolerance:
        return "increase"
    return "no_clear_change"


def _format_percent(value: Any, digits: int = 1) -> str:
    try:
        val = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{100.0 * val:.{digits}f}%"


def _format_percentage_points(value: Any, digits: int = 1) -> str:
    try:
        val = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(val):
        return "NA"
    sign = "+" if val > 0 else ""
    return f"{sign}{100.0 * val:.{digits}f}%"


def _format_signed_percent(value: Any, digits: int = 1, suffix: str = "%") -> str:
    try:
        val = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(val):
        return "NA"
    sign = "+" if val > 0 else ""
    return f"{sign}{100.0 * val:.{digits}f}{suffix}"


def _overall_trend_label(change_method: Any, regression_change: Any) -> str:
    change_dir = _direction_label(change_method)
    regression_dir = _direction_label(regression_change)
    if change_dir == "increase" and regression_dir == "increase":
        return "Increasing"
    if change_dir == "decrease" and regression_dir == "decrease":
        return "Decreasing"
    if change_dir == "no_clear_change" and regression_dir == "no_clear_change":
        return "Stable"
    if change_dir is None and regression_dir is None:
        return "Unclear"
    return "Mixed"


def _method_agreement_label(change_method: Any, regression_change: Any) -> str:
    change_dir = _direction_label(change_method)
    regression_dir = _direction_label(regression_change)
    if change_dir is None or regression_dir is None:
        return "insufficient evidence"
    if change_dir == regression_dir:
        return "method agreement"
    return "method disagreement"


def _latest_reported_prevalence(prevalence_ts_df: pd.DataFrame, regression_fit_df: pd.DataFrame) -> tuple[int | None, float | None]:
    if regression_fit_df is not None and not regression_fit_df.empty and "model_pi_hat" in regression_fit_df.columns:
        model_vals = pd.to_numeric(regression_fit_df["model_pi_hat"], errors="coerce")
        valid = regression_fit_df.loc[np.isfinite(model_vals)].copy()
        if not valid.empty:
            row = valid.iloc[-1]
            return int(row["year"]), float(row["model_pi_hat"])
    if prevalence_ts_df is not None and not prevalence_ts_df.empty:
        obs_vals = pd.to_numeric(prevalence_ts_df["pi_hat"], errors="coerce")
        valid = prevalence_ts_df.loc[np.isfinite(obs_vals)].copy()
        if not valid.empty:
            row = valid.iloc[-1]
            return int(row["year"]), float(row["pi_hat"])
    return None, None


def _trend_change_per_year(regression_fit_df: pd.DataFrame) -> float | None:
    if regression_fit_df is None or regression_fit_df.empty or "year" not in regression_fit_df.columns or "model_pi_hat" not in regression_fit_df.columns:
        return None
    years = pd.to_numeric(regression_fit_df["year"], errors="coerce").to_numpy(dtype=float)
    prev = pd.to_numeric(regression_fit_df["model_pi_hat"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(years) & np.isfinite(prev)
    if mask.sum() < 2:
        return None
    years = years[mask]
    prev = prev[mask]
    span = float(years[-1] - years[0])
    if not np.isfinite(span) or abs(span) < 1e-12:
        return None
    return float((prev[-1] - prev[0]) / span)


def _trend_projection_sentence(
    latest_year: int | None,
    latest_prev: float | None,
    annual_change: float | None,
    target_prevalence: float,
) -> str | None:
    # Convert the fitted yearly change into a plain-English extrapolation sentence.
    # This is a simple straight-line extrapolation of fitted prevalence, not a disease forecast.
    try:
        year_now = int(latest_year)
        prev_now = float(latest_prev)
        change_per_year = float(annual_change)
        target = float(target_prevalence)
    except Exception:
        return None
    if not np.isfinite(prev_now) or not np.isfinite(change_per_year) or not np.isfinite(target):
        return None
    if abs(change_per_year) < 1e-12:
        return "The fitted trend is broadly stable, so there is no clear extrapolated milestone year."
    if change_per_year < 0:
        if prev_now <= target:
            return f"The fitted prevalence is already below the design prevalence of {_format_percent(target)}; no future crossing is projected."
        years_to_target = (prev_now - target) / abs(change_per_year)
        if not np.isfinite(years_to_target) or years_to_target < 0:
            return None
        projected_year = int(np.ceil(year_now + years_to_target))
        return f"If the fitted trend continued unchanged, prevalence would fall below the design prevalence of {_format_percent(target)} around {projected_year}. This is an extrapolation, not a disease forecast."
    if prev_now >= 1.0:
        return "On the current trend, prevalence is already at or above 100%."
    years_to_full = (1.0 - prev_now) / change_per_year
    if not np.isfinite(years_to_full) or years_to_full < 0:
        return None
    projected_year = int(np.ceil(year_now + years_to_full))
    return f"If the fitted trend continued unchanged, the extrapolation would reach 100% around {projected_year}. This is not a disease forecast."


def build_core_analysis_assumptions(
    host_density_km2: float,
    include_unmatched_survey_polygons: bool = True,
    regression_model: str = "logistic",
    status_column_name: str | None = None,
    positive_statuses: list[str] | tuple[str, ...] | None = None,
    negative_statuses: list[str] | tuple[str, ...] | None = None,
    benchmark_label: str | None = None,
) -> list[str]:
    """Return the explicit assumptions that define an operational dashboard run.

    These are deliberately written in decision-support language.  They describe
    assumptions made by the dashboard and limitations of the source polygons;
    they are not statements that these assumptions are biologically true.
    """
    try:
        density_text = f"{float(host_density_km2):,.1f} hosts per km²"
    except Exception:
        density_text = "the selected host-density value"
    model_labels = {
        "logistic": "logit-linear",
        "fp2": "fractional-polynomial power 2",
        "fp3": "fractional-polynomial power 3",
    }
    model_text = model_labels.get(str(regression_model).lower(), str(regression_model))
    status_text = "automatically detected survey outcome field" if not status_column_name or status_column_name == "__auto__" else f"survey outcome field '{status_column_name}'"
    pos = ", ".join(str(x) for x in (positive_statuses or [])) or "the selected positive outcome values"
    neg = ", ".join(str(x) for x in (negative_statuses or [])) or "the selected negative outcome values"
    host_extension = (
        "DEFRA survey footprint outside the public host map is added as inferred host area because the public host layer is treated as incomplete. Only the uncovered part is added."
        if include_unmatched_survey_polygons
        else "Survey footprint outside the public host map is excluded from the operational host landscape."
    )
    benchmark_text = str(benchmark_label or "the selected planning benchmark")
    return [
        "Every host estimated to lie inside the usable surveyed host footprint is treated as surveyed.",
        "Every host inside the host-covered part of a polygon classified positive is treated as infected; a selected negative polygon contributes surveyed hosts but no infected hosts.",
        f"Host numbers are estimated from mapped area using {density_text}; they are not observed host counts, and the selected density is applied uniformly unless the user changes it.",
        host_extension,
        "Survey/host intersections are used, and overlapping survey footprints are dissolved before area is converted to host counts so the same ground is not counted repeatedly within a yearly estimate.",
        f"Survey outcomes are interpreted from the {status_text}. Positive values used in this run: {pos}. Negative values used in this run: {neg}. Other outcome values are ignored.",
        "No correction is made for imperfect detection or diagnostic sensitivity; the recorded polygon classification is taken at face value.",
        "The historical survey may be targeted toward high-risk or known affected locations rather than being a representative probability sample. The resulting prevalence estimate can therefore differ systematically from prevalence across the wider host population.",
        "The operational host landscape is an approximation. The public host layer may omit hosts, while survey-inferred host area may include some non-host space.",
        "Year-to-year comparisons assume that changes in survey targeting, polygon construction, recording practice and surveillance intensity do not create changes that are mistaken for biological prevalence change.",
        "SPHN polygons are map context only and do not directly enter the prevalence calculation.",
        "The Change Method compares the selected first and last survey years only. Intermediate years do not contribute to its prevalence-change estimate, and the operational planning benchmark uses zero endpoint correlation.",
        f"The Regression Method uses a {model_text} model in this run. Its fitted change and final prevalence depend on that model being an adequate description of the temporal trajectory.",
        f"Sampling-effort figures are planning references under {benchmark_text}. They do not prove that the historical field survey followed that design or was adequately representative. Regression Method allocations are designed to detect a prespecified trend, not to guarantee precise annual prevalence estimates.",
        "Regression confidence intervals and model covariance do not propagate uncertainty in host density, host-map completeness, polygon interpretation, outcome classification or preferential site selection.",
        "An estimated temporal trend is an association in the survey record. It does not by itself prove that control measures caused the observed change.",
        "The dashboard therefore provides reproducible decision-support estimates under explicit assumptions; it does not recover factual ground-truth prevalence for the whole host population.",
    ]



def build_concise_analysis_assumptions(
    host_density_km2: float,
    include_unmatched_survey_polygons: bool = True,
    regression_model: str = "logistic",
) -> list[str]:
    """Return a short management-facing summary of the assumptions that matter most."""
    try:
        density_text = f"{float(host_density_km2):,.1f} hosts per km²"
    except Exception:
        density_text = "the selected host-density value"
    model_labels = {
        "logistic": "logit-linear",
        "fp2": "fractional-polynomial power 2",
        "fp3": "fractional-polynomial power 3",
    }
    model_text = model_labels.get(str(regression_model).lower(), str(regression_model))
    host_text = (
        "Surveyed areas outside the public host map are treated as additional host habitat because the public layer may be incomplete."
        if include_unmatched_survey_polygons
        else "Surveyed areas outside the public host map are excluded from the host population."
    )
    return [
        "Hosts inside a surveyed host footprint are treated as surveyed; in a positive polygon, all hosts in that footprint are treated as infected.",
        f"Host numbers are estimated from mapped area using {density_text}, rather than observed tree counts.",
        host_text,
        "Only selected positive and negative survey outcomes are analysed; overlapping survey footprints are dissolved and no detection-sensitivity correction is applied.",
        "Historical surveys may be targeted rather than representative, and changes in survey practice between years can affect the estimated trend.",
        f"The Change Method uses the first and last selected years; the Regression Method uses a {model_text} trend. Planning sample sizes and model confidence intervals do not capture all mapping, host-density or targeting uncertainty.",
    ]

def build_results_report_caveats(
    effort_df: pd.DataFrame,
    spatial_diagnostics_df: pd.DataFrame,
    regression_change: Any,
    change_method: Any,
    host_density_km2: float,
    baseline_observed_prevalence: float | None = None,
    baseline_planning_prevalence: float | None = None,
) -> list[str]:
    # Build the "points to keep in mind" section for the PDF.
    # These are practical warnings about sampling effort, survey bias, and assumptions.
    caveats: list[str] = [
        f"The survey files contain polygons rather than host-level inspection counts. Every host within a surveyed footprint is treated as surveyed, with host numbers estimated from mapped area using {_format_report_value(host_density_km2, digits=1)} hosts per km^2.",
        "A positive DEFRA survey polygon is assumed to mean that every host within that surveyed footprint is infected. This is an operational interpretation of the polygon status, not a host-by-host infection count.",
        "The operational host landscape combines public mapped larch coverage with portions of DEFRA larch survey footprints outside that map. Those survey-only areas are treated as evidence of larch presence because the public host layer may be incomplete.",
    ]
    try:
        if (
            baseline_observed_prevalence is not None and baseline_planning_prevalence is not None
            and np.isfinite(float(baseline_observed_prevalence)) and np.isfinite(float(baseline_planning_prevalence))
            and float(baseline_observed_prevalence) > float(baseline_planning_prevalence)
        ):
            caveats.append(
                f"Observed baseline prevalence ({_format_percent(float(baseline_observed_prevalence))}) exceeded the "
                f"baseline planning prevalence ({_format_percent(float(baseline_planning_prevalence))}) used to size the initial survey. "
                "If that value was not a defensible upper planning value before surveying, rerun with a higher baseline planning prevalence."
            )
    except Exception:
        pass

    if effort_df is not None and not effort_df.empty:
        planning = _planning_benchmark_summary(effort_df)
        if planning["baseline"].get("status") == "not_met":
            caveats.append("Estimated host coverage was below the initial-prevalence precision requirement.")
        if planning["change_method"].get("status") == "not_met":
            years = ", ".join(str(y) for y in planning["change_method"].get("years_below", []))
            caveats.append(f"Estimated host coverage was below the Change Method per-survey requirement in {years}.")
        if planning["regression_method"].get("status") == "not_met":
            years = ", ".join(str(y) for y in planning["regression_method"].get("years_below", []))
            caveats.append(f"Estimated host coverage was below the Regression Method trend-detection allocation in {years}.")

    coverage_prop = _spatial_metric_value(spatial_diagnostics_df, "Host polygon coverage proportion")
    top10_share = _spatial_metric_value(spatial_diagnostics_df, "Top 10% host polygons share of surveyed footprint")
    median_patch_share = _spatial_metric_value(spatial_diagnostics_df, "Median largest patch share within surveyed host polygons")
    effective_prop = _spatial_metric_value(spatial_diagnostics_df, "Effective surveyed host polygon proportion")
    try:
        # These thresholds flag survey patterns that look targeted rather than random.
        targeted_flag = (
            (float(coverage_prop) < 0.15)
            or (float(top10_share) >= 0.65)
            or (float(median_patch_share) >= 0.75)
            or (float(effective_prop) < 0.35)
        )
    except Exception:
        targeted_flag = False
    if targeted_flag:
        caveats.append(
            "The survey footprint looks concentrated and targeted rather than broadly representative, so estimated prevalence may be higher than the true prevalence across the wider host landscape."
        )

    if _method_agreement_label(change_method, regression_change) == "method disagreement":
        caveats.append(
            "The two methods do not tell exactly the same story, so the overall trend should be interpreted cautiously."
        )

    return caveats


def build_results_report_narrative(
    method_df: pd.DataFrame,
    year_summary_df: pd.DataFrame,
    regression_summary_df: pd.DataFrame,
    spatial_diagnostics_df: pd.DataFrame,
    before_year: int,
    after_year: int,
) -> list[str]:
    # Build a short text summary of the epidemic trend.
    # This lets the PDF explain the results without expecting the reader to read model tables.
    lines: list[str] = []
    change_method = _method_metric_value(method_df, "Change Method", "Estimated change")
    regression_change = _method_metric_value(method_df, "Regression Method", "Estimated change")
    # Compare the direction of change estimated by the two methods.
    change_dir = _direction_label(change_method)
    regression_dir = _direction_label(regression_change)

    if change_dir == "decrease" and regression_dir == "decrease":
        lines.append(f"Estimated prevalence was lower in {after_year} than in {before_year}, and both methods indicate a decline.")
    elif change_dir == "increase" and regression_dir == "increase":
        lines.append(f"Estimated prevalence was higher in {after_year} than in {before_year}, and both methods indicate an increase.")
    elif change_dir == "no_clear_change" and regression_dir == "no_clear_change":
        lines.append(f"The results do not show a clear overall change in disease prevalence between {before_year} and {after_year}.")
    elif change_dir is None and regression_dir is None:
        lines.append("The current results are not sufficient to state an overall trend.")
    elif change_dir == regression_dir and change_dir is not None:
        lines.append(f"Both methods point to the same overall direction of change between {before_year} and {after_year}.")
    else:
        lines.append(f"The two methods do not fully agree on the direction of change between {before_year} and {after_year}.")

    if change_dir is not None:
        lines.append(f"Estimated change from the Change Method: {_format_percentage_points(change_method)}.")
    if regression_dir is not None:
        lines.append(f"Estimated change from the Regression Method: {_format_percentage_points(regression_change)}.")

    slope = _series_lookup_value(regression_summary_df, "metric", "Trend slope (logit scale)")
    declining = _series_lookup_value(regression_summary_df, "metric", "Declining trend")
    if slope is not None:
        trend_text = "falling" if bool(declining) else "rising or flat"
        lines.append(f"The fitted trend line has a slope of {_format_report_value(slope)}, which is consistent with prevalence {trend_text} over time.")

    if year_summary_df is not None and not year_summary_df.empty:
        observed = year_summary_df.loc[year_summary_df["survey_polygons"] > 0].copy()
        if not observed.empty:
            total_polygons = int(pd.to_numeric(observed["survey_polygons"], errors="coerce").fillna(0).sum())
            total_hosts = float(pd.to_numeric(observed["surveyed_hosts_est"], errors="coerce").fillna(0.0).sum())
            lines.append(
                f"The analysed survey record covers {len(observed)} years with survey data, {total_polygons} survey polygons, "
                f"and an estimated {_format_report_value(total_hosts, digits=1)} surveyed hosts."
            )
            peak_row = observed.loc[pd.to_numeric(observed["estimated_prevalence"], errors="coerce").idxmax()]
            lines.append(
                f"The highest yearly estimated prevalence in the summary is {_format_percent(peak_row['estimated_prevalence'])} "
                f"in {int(peak_row['year'])}."
            )

    lines.append("")
    lines.extend(build_spatial_survey_narrative(spatial_diagnostics_df))

    return lines


def _draw_report_card(ax, x: float, y: float, w: float, h: float, title: str, value: str, subtitle: str = "", facecolor: str = "#eef5f2", edgecolor: str = "#d6e5dd"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    title_lines = textwrap.wrap(title, width=26)[:2] or [title]
    ty = y + h - 0.14 * h
    for line in title_lines:
        ax.text(x + 0.07 * w, ty, line, ha="left", va="top", fontsize=10.0, color="#436257", transform=ax.transAxes)
        ty -= 0.085 * h
    ax.text(x + 0.07 * w, y + 0.44 * h, value, ha="left", va="center", fontsize=19, fontweight="bold", color="#17352d", transform=ax.transAxes)
    if subtitle:
        wrapped = textwrap.wrap(subtitle, width=34)
        sy = y + 0.14 * h
        for line in wrapped[:2]:
            ax.text(x + 0.07 * w, sy, line, ha="left", va="bottom", fontsize=9.0, color="#51665f", transform=ax.transAxes)
            sy -= 0.06 * h


def _draw_report_dual_card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    left_label: str,
    left_value: str,
    right_label: str,
    right_value: str,
    facecolor: str = "#eef5f2",
    edgecolor: str = "#d6e5dd",
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    title_lines = textwrap.wrap(title, width=24)[:2] or [title]
    ty = y + h - 0.14 * h
    for line in title_lines:
        ax.text(x + 0.07 * w, ty, line, ha="left", va="top", fontsize=9.8, color="#436257", transform=ax.transAxes)
        ty -= 0.08 * h
    ax.text(x + 0.07 * w, y + 0.56 * h, left_label, ha="left", va="bottom", fontsize=9.2, color="#51665f", transform=ax.transAxes)
    ax.text(x + 0.07 * w, y + 0.42 * h, left_value, ha="left", va="center", fontsize=16, fontweight="bold", color="#17352d", transform=ax.transAxes)
    ax.plot([x + 0.07 * w, x + 0.93 * w], [y + 0.28 * h, y + 0.28 * h], color="#d7e7de", linewidth=1.0, transform=ax.transAxes, clip_on=False)
    ax.text(x + 0.07 * w, y + 0.15 * h, right_label, ha="left", va="bottom", fontsize=9.2, color="#51665f", transform=ax.transAxes)
    ax.text(x + 0.07 * w, y + 0.05 * h, right_value, ha="left", va="bottom", fontsize=16, fontweight="bold", color="#17352d", transform=ax.transAxes)


def _draw_report_front_page(
    pdf: PdfPages,
    year_summary_df: pd.DataFrame,
    method_df: pd.DataFrame,
    regression_summary_df: pd.DataFrame,
    regression_fit_df: pd.DataFrame,
    prevalence_ts_df: pd.DataFrame,
    effort_df: pd.DataFrame,
    spatial_diagnostics_df: pd.DataFrame,
    before_year: int,
    after_year: int,
    host_density_km2: float,
    target_prevalence: float = DEFAULT_REGRESSION_METHOD_TARGET,
):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    change_method = _method_metric_value(method_df, "Change Method", "Estimated change")
    change_final = _method_metric_value(method_df, "Change Method", "Final prevalence")
    regression_change = _method_metric_value(method_df, "Regression Method", "Estimated change")
    regression_final = _method_metric_value(method_df, "Regression Method", "Final prevalence")
    trend_label = _overall_trend_label(change_method, regression_change)
    agreement_label = _method_agreement_label(change_method, regression_change)
    latest_year, latest_prev = _latest_reported_prevalence(prevalence_ts_df, regression_fit_df)
    annual_change = _trend_change_per_year(regression_fit_df)
    trend_projection = _trend_projection_sentence(
        latest_year=latest_year,
        latest_prev=latest_prev,
        annual_change=annual_change,
        target_prevalence=target_prevalence,
    )
    survey_years = 0
    survey_polygons = 0
    surveyed_hosts = np.nan
    if year_summary_df is not None and not year_summary_df.empty:
        observed = year_summary_df.loc[pd.to_numeric(year_summary_df["survey_polygons"], errors="coerce").fillna(0) > 0].copy()
        if not observed.empty:
            survey_years = int(len(observed))
            survey_polygons = int(pd.to_numeric(observed["survey_polygons"], errors="coerce").fillna(0).sum())
            surveyed_hosts = float(pd.to_numeric(observed["surveyed_hosts_est"], errors="coerce").fillna(0.0).sum())

    coverage_prop = _spatial_metric_value(spatial_diagnostics_df, "Host polygon coverage proportion")
    caveats = build_results_report_caveats(
        effort_df=effort_df,
        spatial_diagnostics_df=spatial_diagnostics_df,
        regression_change=regression_change,
        change_method=change_method,
        host_density_km2=host_density_km2,
    )

    if agreement_label == "method agreement":
        agreement_text = "The two methods point in the same overall direction."
    elif agreement_label == "method disagreement":
        agreement_text = "The two methods do not fully agree, so the overall trend should be treated with caution."
    else:
        agreement_text = "There is not enough information to judge whether the two methods agree."

    overview_lines = [
        f"The available survey record suggests that prevalence was {trend_label.lower()} between {before_year} and {after_year}.",
        agreement_text,
    ]
    if latest_year is not None and latest_prev is not None:
        overview_lines.append(f"The latest estimated prevalence is {_format_percent(latest_prev)} in {latest_year}.")
    if trend_projection:
        overview_lines.append(trend_projection)
    if survey_years > 0:
        overview_lines.append(
            f"The dataset used here contains survey information for {survey_years} years, {survey_polygons:,} survey polygons, and an estimated {_format_report_value(surveyed_hosts, digits=1)} surveyed hosts."
        )

    page_width = 0.84
    page_left = (1.0 - page_width) / 2.0
    page_right = page_left + page_width

    header_box = FancyBboxPatch(
        (page_left, 0.915),
        page_width,
        0.07,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0,
        facecolor="#eef5f2",
        transform=ax.transAxes,
    )
    ax.add_patch(header_box)
    ax.text(page_left + 0.03, 0.965, "Plant Pest Prevalence Report", ha="left", va="top", fontsize=26, fontweight="bold", color="#17352d", transform=ax.transAxes)
    ax.text(page_left + 0.03, 0.918, f"Report period: {before_year} to {after_year}", ha="left", va="top", fontsize=11.5, color="#567066", transform=ax.transAxes)

    overview_box = FancyBboxPatch(
        (page_left, 0.70),
        page_width,
        0.14,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        facecolor="#f8fbf9",
        edgecolor="#e2ece7",
        transform=ax.transAxes,
    )
    ax.add_patch(overview_box)
    ax.text(page_left + 0.02, 0.84, "Overview", ha="left", va="top", fontsize=16, fontweight="bold", color="#17352d", transform=ax.transAxes)
    y = 0.81
    for line in overview_lines:
        wrapped = textwrap.wrap(line, width=94)
        for part in wrapped:
            ax.text(page_left + 0.025, y, u"\u2022 " + part, ha="left", va="top", fontsize=10.8, color="#22332d", transform=ax.transAxes)
            y -= 0.022
        y -= 0.003

    ax.text(page_left, 0.655, "Key figures", ha="left", va="top", fontsize=16, fontweight="bold", color="#17352d", transform=ax.transAxes)

    top_y = 0.425
    card_h = 0.17
    card_gap = 0.03
    card_w = (page_width - 2 * card_gap) / 3.0
    card_x1 = page_left
    card_x2 = page_left + card_w + card_gap
    card_x3 = page_right - card_w
    _draw_report_dual_card(
        ax, card_x1, top_y, card_w, card_h,
        "Estimated change in prevalence",
        "Change Method",
        _format_percentage_points(change_method),
        "Regression Method",
        _format_percentage_points(regression_change),
    )
    _draw_report_card(
        ax, card_x2, top_y, card_w, card_h,
        "Prevalence trend",
        _format_percentage_points(annual_change),
        "Average yearly change from the fitted trend",
    )
    _draw_report_dual_card(
        ax, card_x3, top_y, card_w, card_h,
        "Estimated final prevalence",
        "Change Method",
        _format_percent(change_final),
        "Regression Method",
        _format_percent(regression_final),
    )

    ax.text(page_left, 0.335, "Points To Keep In Mind", ha="left", va="top", fontsize=16, fontweight="bold", color="#17352d", transform=ax.transAxes)
    caveat_box = FancyBboxPatch(
        (page_left, 0.10),
        page_width,
        0.16,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.0,
        facecolor="#faf7f1",
        edgecolor="#e8dcc5",
        transform=ax.transAxes,
    )
    ax.add_patch(caveat_box)
    cy = 0.305
    for line in caveats[:5]:
        wrapped = textwrap.wrap(line, width=92)
        for part in wrapped:
            ax.text(page_left + 0.025, cy, u"\u2022 " + part, ha="left", va="top", fontsize=10.2, color="#3c3122", transform=ax.transAxes)
            cy -= 0.023
        cy -= 0.003

    footer = []
    try:
        coverage_ok = coverage_prop is not None and np.isfinite(float(coverage_prop))
    except Exception:
        coverage_ok = False
    if coverage_ok:
        footer.append(f"Host polygons touched by surveys: {_format_percent(coverage_prop)}")
    footer.append(f"Host density assumption: {_format_report_value(host_density_km2, digits=1)} hosts per km^2")
    ax.text(page_left, 0.075, "   |   ".join(footer), ha="left", va="bottom", fontsize=9.4, color="#6c7b75", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_report_text_page(pdf: PdfPages, title: str, lines: list[str], wrap_width: int = 110):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.05, 0.97, title, ha="left", va="top", fontsize=16, fontweight="bold", transform=ax.transAxes)
    y = 0.92
    line_step = 0.028
    for raw_line in lines:
        wrapped = textwrap.wrap(str(raw_line), width=wrap_width) or [""]
        for line in wrapped:
            ax.text(0.05, y, line, ha="left", va="top", fontsize=10.5, transform=ax.transAxes)
            y -= line_step
            if y < 0.06:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                ax.text(0.05, 0.97, f"{title} (continued)", ha="left", va="top", fontsize=16, fontweight="bold", transform=ax.transAxes)
                y = 0.92
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_report_table_pages(pdf: PdfPages, title: str, df: pd.DataFrame, rows_per_page: int = 24):
    if df is None or df.empty:
        _draw_report_text_page(pdf, title, ["No data available for this section."])
        return

    work = df.copy()
    for col in work.columns:
        work[col] = work[col].apply(_format_report_value)

    n_pages = int(np.ceil(len(work) / rows_per_page))
    for page_idx in range(n_pages):
        page_df = work.iloc[page_idx * rows_per_page : (page_idx + 1) * rows_per_page]
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        suffix = f" (page {page_idx + 1} of {n_pages})" if n_pages > 1 else ""
        ax.text(0.05, 0.97, f"{title}{suffix}", ha="left", va="top", fontsize=16, fontweight="bold", transform=ax.transAxes)
        table = ax.table(
            cellText=page_df.values.tolist(),
            colLabels=[str(col).replace("_", " ") for col in page_df.columns],
            loc="upper left",
            cellLoc="left",
            colLoc="left",
            bbox=[0.05, 0.05, 0.90, 0.86],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _figure_to_data_uri(fig, dpi: int = 180) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _html_bullet_list(items: list[str], klass: str = "bullets") -> str:
    rows = "".join(
        f'<div class="bullet-row"><span class="bullet-mark">&#8226;</span> {escape(str(item))}</div>'
        for item in items
        if str(item).strip()
    )
    return f'<div class="{klass}">{rows}</div>'


def _html_report_table(
    headers: list[str],
    rows: list[list[Any]],
    empty_message: str,
    widths: list[int] | None = None,
    extra_class: str = "",
) -> str:
    if not rows:
        return f'<div class="table-empty">{escape(empty_message)}</div>'
    if widths is None or len(widths) != len(headers):
        widths = [int(100 / max(len(headers), 1))] * len(headers)
    header_html = "".join(
        f'<th width="{int(width)}%">{escape(str(header))}</th>'
        for header, width in zip(headers, widths)
    )
    body_html = "".join(
        "<tr>"
        + "".join(
            f'<td width="{int(width)}%">{escape(str(value))}</td>'
            for value, width in zip(row, widths)
        )
        + "</tr>"
        for row in rows
    )
    table_class = f"report-table {extra_class}".strip()
    return f'<table class="{table_class}"><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>'


def _yearly_report_table_html(year_summary_df: pd.DataFrame) -> str:
    headers = ["Year", "Survey polygons used", "Positive polygons", "Estimated hosts surveyed", "Estimated prevalence"]
    if year_summary_df is None or year_summary_df.empty:
        return _html_report_table(headers, [], "No yearly survey summary is available.")
    work = year_summary_df.copy()
    rows: list[list[Any]] = []
    for row in work.itertuples(index=False):
        values = row._asdict()
        rows.append(
            [
                int(values.get("year")) if pd.notna(values.get("year")) else "",
                int(round(float(values.get("survey_polygons", 0)))) if pd.notna(values.get("survey_polygons")) else "",
                int(round(float(values.get("infected_polygons", 0)))) if pd.notna(values.get("infected_polygons")) else "",
                _format_report_value(values.get("surveyed_hosts_est"), digits=1),
                _format_percent(values.get("estimated_prevalence")),
            ]
        )
    return _html_report_table(
        headers,
        rows,
        "No yearly survey summary is available.",
        widths=[10, 19, 19, 29, 23],
    )


def _effort_report_table_html(effort_df: pd.DataFrame) -> str:
    """Render baseline precision, Change Method and Regression Method planning separately."""
    headers = [
        "Year", "Estimated hosts covered", "Observed prevalence",
        "Initial prevalence precision", "Change Method per-survey requirement",
        "Regression Method follow-up allocation",
    ]
    if effort_df is None or effort_df.empty:
        return _html_report_table(headers, [], "No sampling-effort summary is available.")
    rows: list[list[Any]] = []
    for row in effort_df.itertuples(index=False):
        values = row._asdict()
        def host_count(value: Any) -> str:
            return "" if value is None or pd.isna(value) else f"{int(round(float(value))):,}"
        rows.append([
            int(values.get("year")) if pd.notna(values.get("year")) else "",
            host_count(values.get("observed_hosts")),
            _format_percent(values.get("observed_prevalence")),
            host_count(values.get("initial_prevalence_required_hosts")),
            host_count(values.get("change_method_required_hosts")),
            host_count(values.get("regression_required_hosts")),
        ])
    return _html_report_table(headers, rows, "No sampling-effort summary is available.", widths=[7,16,15,20,21,21], extra_class="effort-table")



def _build_report_overview_lines(
    method_df: pd.DataFrame,
    prevalence_ts_df: pd.DataFrame,
    regression_fit_df: pd.DataFrame,
    effort_df: pd.DataFrame,
    before_year: int,
    after_year: int,
    survey_years: int,
    survey_polygons: int,
    surveyed_hosts: float,
    target_prevalence: float,
) -> tuple[list[str], str]:
    """Create the concise interpretation shown in Results and on PDF page 1.

    The final prevalence estimates are reported separately for the two methods.
    This avoids describing the Regression Method fitted endpoint as though it
    were the single factual prevalence for the final year.
    """
    change_value = _method_metric_value(method_df, "Change Method", "Estimated change")
    regression_change = _method_metric_value(method_df, "Regression Method", "Estimated change")
    change_final = _method_metric_value(method_df, "Change Method", "Final prevalence")
    regression_final = _method_metric_value(method_df, "Regression Method", "Final prevalence")
    trend_label = _overall_trend_label(change_value, regression_change)
    annual_change = _trend_change_per_year(regression_fit_df)

    overview_lines: list[str] = [
        f"The two methods indicate an overall {trend_label.lower()} pattern between {before_year} and {after_year}."
        if trend_label not in {"Mixed", "Unclear"}
        else f"The overall pattern between {before_year} and {after_year} is {trend_label.lower()} across the two methods.",
    ]

    agreement = _method_agreement_label(change_value, regression_change)
    if agreement == "method agreement":
        overview_lines.append("Both methods point in the same direction, although their fitted final prevalence estimates need not be identical.")
    elif agreement == "method disagreement":
        overview_lines.append("The methods do not point in the same direction, so the result should be interpreted cautiously.")
    else:
        overview_lines.append("There is not enough information to assess agreement between the methods.")

    if change_final is not None or regression_final is not None:
        pieces = []
        try:
            if change_final is not None and np.isfinite(float(change_final)):
                pieces.append(f"Change Method {_format_percent(change_final)}")
        except Exception:
            pass
        try:
            if regression_final is not None and np.isfinite(float(regression_final)):
                pieces.append(f"Regression Method {_format_percent(regression_final)}")
        except Exception:
            pass
        if pieces:
            overview_lines.append(f"Estimated final prevalence in {after_year}: " + "; ".join(pieces) + ".")

    benchmark_lines = _planning_benchmark_interpretation(effort_df)
    overview_lines.extend(benchmark_lines[:3])

    # A fitted-trend extrapolation is only a descriptive extension of the
    # Regression Method curve; keep it separate from the final-year estimates.
    latest_year, latest_prev = _latest_reported_prevalence(prevalence_ts_df, regression_fit_df)
    trend_projection = _trend_projection_sentence(
        latest_year=latest_year,
        latest_prev=latest_prev,
        annual_change=annual_change,
        target_prevalence=target_prevalence,
    )
    if trend_projection:
        overview_lines.append(trend_projection)

    metadata = ""
    if survey_years > 0:
        metadata = (
            f"Dataset used: {survey_years} survey years, {survey_polygons:,} survey polygons, "
            f"and an estimated {_format_report_value(surveyed_hosts, digits=1)} hosts covered by those survey polygons under the full-coverage assumption."
        )
    return overview_lines, metadata



def _effort_status_lines(effort_df: pd.DataFrame) -> list[str]:
    """Describe observed effort without overstating Regression Method's purpose.

    Change Method values are equal per-survey planning requirements. Regression Method values are
    allocations designed to power a prespecified downward *trend test*; they are
    not minimum sample sizes for precise annual or final prevalence estimates.
    """
    if effort_df is None or effort_df.empty:
        return ["No sampling-effort comparison is available for the current results."]
    work = effort_df.copy()
    for col in ("observed_hosts", "change_method_required_hosts", "regression_required_hosts"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    lines: list[str] = []

    comp = work.loc[work["change_method_required_hosts"].notna() & work["observed_hosts"].notna(), ["year", "observed_hosts", "change_method_required_hosts"]].copy()
    if not comp.empty:
        below = comp.loc[comp["observed_hosts"] < comp["change_method_required_hosts"], "year"].astype(int).tolist()
        meets = comp.loc[comp["observed_hosts"] >= comp["change_method_required_hosts"], "year"].astype(int).tolist()
        if below:
            lines.append(f"Change Method: estimated host coverage was below the planning benchmark in {', '.join(str(y) for y in below)}.")
        if meets:
            lines.append(f"Change Method: estimated host coverage met or exceeded the planning benchmark in {', '.join(str(y) for y in meets)}.")

    comp = work.loc[work["regression_required_hosts"].notna() & work["observed_hosts"].notna(), ["year", "observed_hosts", "regression_required_hosts"]].copy()
    if not comp.empty:
        below = comp.loc[comp["observed_hosts"] < comp["regression_required_hosts"], "year"].astype(int).tolist()
        meets = comp.loc[comp["observed_hosts"] >= comp["regression_required_hosts"], "year"].astype(int).tolist()
        if below:
            lines.append(f"Regression Method trend-detection allocation was higher than estimated coverage in {', '.join(str(y) for y in below)}.")
        if meets:
            lines.append(f"Estimated coverage exceeded the Regression Method trend-detection allocation in {', '.join(str(y) for y in meets)}. This does not establish adequate precision for yearly or final prevalence estimates.")
    return lines or ["No sampling-effort comparison is available for the current results."]


def _planning_benchmark_summary(effort_df: pd.DataFrame) -> dict[str, Any]:
    """Summarise whether historical effort met each distinct planning criterion."""
    result: dict[str, Any] = {
        "baseline": {"status": "unavailable", "years_below": [], "years_met": []},
        "change_method": {"status": "unavailable", "years_below": [], "years_met": []},
        "regression_method": {"status": "unavailable", "years_below": [], "years_met": []},
    }
    if effort_df is None or effort_df.empty:
        return result
    work = effort_df.copy()
    for col in ("year", "observed_hosts", "initial_prevalence_required_hosts", "change_method_required_hosts", "regression_required_hosts"):
        work[col] = pd.to_numeric(work.get(col), errors="coerce")

    base_rows = work.loc[work["initial_prevalence_required_hosts"].notna() & work["observed_hosts"].notna()].copy()
    if not base_rows.empty:
        base = base_rows.sort_values("year").iloc[0]
        required = float(base["initial_prevalence_required_hosts"])
        observed = float(base["observed_hosts"])
        year = int(base["year"])
        result["baseline"] = {
            "status": "met" if observed >= required else "not_met",
            "years_below": [] if observed >= required else [year],
            "years_met": [year] if observed >= required else [],
            "observed": observed, "required": required,
        }

    change_rows = work.loc[work["change_method_required_hosts"].notna() & work["observed_hosts"].notna()].copy()
    if not change_rows.empty:
        below = change_rows.loc[change_rows["observed_hosts"] < change_rows["change_method_required_hosts"], "year"].astype(int).tolist()
        met = change_rows.loc[change_rows["observed_hosts"] >= change_rows["change_method_required_hosts"], "year"].astype(int).tolist()
        result["change_method"] = {
            "status": "met" if not below else "not_met", "years_below": below, "years_met": met,
            "required_per_survey": float(change_rows["change_method_required_hosts"].dropna().iloc[0]),
        }

    regression_rows = work.loc[work["regression_required_hosts"].notna() & work["observed_hosts"].notna()].copy()
    if not regression_rows.empty:
        below = regression_rows.loc[regression_rows["observed_hosts"] < regression_rows["regression_required_hosts"], "year"].astype(int).tolist()
        met = regression_rows.loc[regression_rows["observed_hosts"] >= regression_rows["regression_required_hosts"], "year"].astype(int).tolist()
        result["regression_method"] = {
            "status": "met" if not below else "not_met", "years_below": below, "years_met": met,
            "required_per_round": float(regression_rows["regression_required_hosts"].dropna().iloc[0]),
            "required_total_followup": float(regression_rows["regression_required_hosts"].fillna(0).sum()),
            "observed_total_followup": float(regression_rows["observed_hosts"].fillna(0).sum()),
        }
    return result


def _planning_benchmark_interpretation(effort_df: pd.DataFrame) -> list[str]:
    """Return concise method-specific planning statements for Results and PDF."""
    summary = _planning_benchmark_summary(effort_df)
    lines: list[str] = []
    baseline = summary["baseline"]
    if baseline.get("status") in {"met", "not_met"}:
        status = "met" if baseline["status"] == "met" else "not met"
        lines.append(
            f"Initial prevalence precision requirement: {int(round(baseline['required'])):,} hosts; "
            f"historical baseline effort {int(round(baseline['observed'])):,} — {status}."
        )
    change = summary["change_method"]
    if change.get("status") in {"met", "not_met"}:
        required = int(round(float(change["required_per_survey"])))
        if change["status"] == "met":
            lines.append(f"Change Method: {required:,} hosts were required at each compared survey, and both historical surveys met that requirement.")
        else:
            years = ", ".join(str(y) for y in change.get("years_below", []))
            lines.append(f"Change Method: {required:,} hosts were required at each compared survey; the requirement was not met in {years}.")
    regression = summary["regression_method"]
    if regression.get("status") in {"met", "not_met"}:
        per_round = int(round(float(regression["required_per_round"])))
        total = int(round(float(regression["required_total_followup"])))
        if regression["status"] == "met":
            lines.append(f"Regression Method: {total:,} follow-up hosts were required in total, allocated as about {per_round:,} per annual round; every historical follow-up round met that allocation.")
        else:
            years = ", ".join(str(y) for y in regression.get("years_below", []))
            lines.append(f"Regression Method: {total:,} follow-up hosts were required in total, allocated as about {per_round:,} per annual round; the allocation was not met in {years}.")
    if lines:
        lines.append("Meeting a numerical planning requirement does not by itself establish representativeness or remove the other survey and modelling assumptions.")
    return lines



def _spatial_report_table_html(spatial_diagnostics_df: pd.DataFrame) -> str:
    headers = ["Survey coverage diagnostic", "Value"]
    if spatial_diagnostics_df is None or spatial_diagnostics_df.empty:
        return _html_report_table(headers, [], "No spatial diagnostics are available.")
    wanted = [
        "Host polygons available",
        "Host polygons surveyed",
        "Host polygon coverage proportion",
        "Host area covered proportion",
        "Top 10% host polygons share of surveyed footprint",
        "Effective surveyed host polygon proportion",
    ]
    rows = []
    for metric in wanted:
        value = _spatial_metric_value(spatial_diagnostics_df, metric)
        if value is None:
            continue
        if "proportion" in metric.lower() or "share" in metric.lower():
            formatted = _format_percent(value)
        else:
            formatted = _format_report_value(value, digits=1)
        rows.append([metric, formatted])
    return _html_report_table(headers, rows, "No spatial diagnostics are available.", widths=[72, 28])


def _regression_diagnostics_report_table_html(regression_summary_df: pd.DataFrame) -> str:
    headers = ["Regression diagnostic", "Value"]
    if regression_summary_df is None or regression_summary_df.empty:
        return _html_report_table(headers, [], "No Regression Method diagnostics are available.")
    wanted = [
        "Model form", "Fit valid", "Fit converged", "Fit reason", "Separation suspected",
        "Information-matrix condition number", "Covariance basis",
    ]
    rows = []
    for metric in wanted:
        value = _series_lookup_value(regression_summary_df, "metric", metric)
        if value is None:
            continue
        if metric == "Information-matrix condition number":
            try:
                value = "NA" if not np.isfinite(float(value)) else f"{float(value):.3g}"
            except Exception:
                pass
        rows.append([metric, _format_report_value(value)])
    return _html_report_table(headers, rows, "No Regression Method diagnostics are available.", widths=[48, 52])


def _settings_report_table_html(
    regression_model: str,
    host_density_km2: float,
    change_method_settings: dict[str, Any],
    regression_settings: dict[str, Any],
    benchmark_label: str | None,
) -> str:
    model_label = {"logistic": "Logit-linear", "fp2": "Fractional polynomial 2", "fp3": "Fractional polynomial 3"}.get(str(regression_model), str(regression_model))
    rows = [
        ["Host density assumption", f"{_format_report_value(host_density_km2, digits=1)} hosts per km^2"],
        ["Sampling benchmark", str(benchmark_label or "Not specified")],
        ["Baseline confidence level", _format_percent(change_method_settings.get("baseline_confidence"))],
        ["Baseline CI total width", _format_percent(change_method_settings.get("baseline_ci_width"))],
        ["Baseline planning prevalence", _format_percent(change_method_settings.get("baseline_planning_prevalence"))],
        ["Significance level (alpha)", _format_report_value(change_method_settings.get("alpha"))],
        ["Power", _format_report_value(change_method_settings.get("power"))],
        ["Change Method detectable change", _format_percentage_points(change_method_settings.get("delta"))],
        ["Change Method planning correlation", "0"],
        ["Regression model", model_label],
        ["Regression design prevalence", _format_percent(regression_settings.get("target"))],
    ]
    return _html_report_table(["Analysis setting", "Selected value"], rows, "No analysis settings are available.", widths=[48, 52])


def _build_results_report_html(
    year_summary_df: pd.DataFrame,
    method_df: pd.DataFrame,
    regression_summary_df: pd.DataFrame,
    regression_fit_df: pd.DataFrame,
    prevalence_ts_df: pd.DataFrame,
    effort_df: pd.DataFrame,
    spatial_diagnostics_df: pd.DataFrame,
    before_year: int,
    after_year: int,
    host_density_km2: float,
    target_prevalence: float,
    trend_plot_uri: str,
    effort_plot_uri: str,
    regression_model: str,
    change_method_settings: dict[str, Any],
    regression_settings: dict[str, Any],
    benchmark_label: str | None = None,
    analysis_assumptions: list[str] | None = None,
) -> str:
    """Build a concise three-page HTML report used for the PDF export."""
    del effort_plot_uri

    change_method = _method_metric_value(method_df, "Change Method", "Estimated change")
    change_final = _method_metric_value(method_df, "Change Method", "Final prevalence")
    regression_change = _method_metric_value(method_df, "Regression Method", "Estimated change")
    regression_final = _method_metric_value(method_df, "Regression Method", "Final prevalence")
    annual_change = _trend_change_per_year(regression_fit_df)

    survey_years = 0
    survey_polygons = 0
    surveyed_hosts = np.nan
    if year_summary_df is not None and not year_summary_df.empty:
        observed = year_summary_df.loc[pd.to_numeric(year_summary_df["survey_polygons"], errors="coerce").fillna(0) > 0].copy()
        if not observed.empty:
            survey_years = int(len(observed))
            survey_polygons = int(pd.to_numeric(observed["survey_polygons"], errors="coerce").fillna(0).sum())
            surveyed_hosts = float(pd.to_numeric(observed["surveyed_hosts_est"], errors="coerce").fillna(0.0).sum())

    overview_lines, overview_meta = _build_report_overview_lines(
        method_df=method_df,
        prevalence_ts_df=prevalence_ts_df,
        regression_fit_df=regression_fit_df,
        effort_df=effort_df,
        before_year=before_year,
        after_year=after_year,
        survey_years=survey_years,
        survey_polygons=survey_polygons,
        surveyed_hosts=surveyed_hosts,
        target_prevalence=target_prevalence,
    )
    baseline_observed_for_warning = np.nan
    try:
        _base_warn = year_summary_df.loc[pd.to_numeric(year_summary_df["year"], errors="coerce") == int(before_year)]
        if not _base_warn.empty:
            baseline_observed_for_warning = float(pd.to_numeric(_base_warn["estimated_prevalence"], errors="coerce").iloc[0])
    except Exception:
        baseline_observed_for_warning = np.nan
    caveats = build_results_report_caveats(
        effort_df=effort_df,
        spatial_diagnostics_df=spatial_diagnostics_df,
        regression_change=regression_change,
        change_method=change_method,
        host_density_km2=host_density_km2,
        baseline_observed_prevalence=baseline_observed_for_warning,
        baseline_planning_prevalence=change_method_settings.get("baseline_planning_prevalence"),
    )

    regression_method_design_note = ""
    planning_reference_note = ""
    try:
        baseline_rows = year_summary_df.loc[pd.to_numeric(year_summary_df["year"], errors="coerce") == int(before_year)]
        baseline_required = effort_df.loc[pd.to_numeric(effort_df["year"], errors="coerce") == int(before_year), "baseline_required_hosts"]
        change_required = effort_df.loc[pd.to_numeric(effort_df["year"], errors="coerce") == int(after_year), "change_method_required_hosts"]
        if not baseline_rows.empty:
            p0_design_report = float(pd.to_numeric(baseline_rows["estimated_prevalence"], errors="coerce").iloc[0])
            target_design_report = float(regression_settings.get("target", target_prevalence))
            follow = effort_df.loc[pd.to_numeric(effort_df["year"], errors="coerce") > int(before_year), "regression_required_hosts"]
            follow = pd.to_numeric(follow, errors="coerce").dropna()
            years_report = int(after_year) - int(before_year)
            baseline_required_text = "-"
            change_required_text = "-"
            if not baseline_required.empty:
                _v = pd.to_numeric(baseline_required, errors="coerce").dropna()
                if len(_v):
                    baseline_required_text = f"{int(round(float(_v.iloc[0]))):,}"
            if not change_required.empty:
                _v = pd.to_numeric(change_required, errors="coerce").dropna()
                if len(_v):
                    change_required_text = f"{int(round(float(_v.iloc[0]))):,}"
            baseline_conf_text = _format_percent(change_method_settings.get("baseline_confidence"))
            baseline_width_text = _format_percent(change_method_settings.get("baseline_ci_width"))
            baseline_prev_text = _format_percent(change_method_settings.get("baseline_planning_prevalence"))
            infer_conf_text = _format_percent(1.0 - float(change_method_settings.get("alpha")))
            power_text = _format_percent(change_method_settings.get("power"))
            delta_text = _format_percent(change_method_settings.get("delta"))
            planning_reference_note = (
                f"Initial prevalence precision sizes the first survey at about {baseline_required_text} hosts to estimate the unknown starting prevalence with {baseline_conf_text} confidence, assuming a planning prevalence of {baseline_prev_text} and a total confidence-interval width of {baseline_width_text}. "
                f"The Change Method recommends about {change_required_text} hosts at both {before_year} and {after_year} to detect an absolute change of {delta_text} with {infer_conf_text} confidence and {power_text} power."
            )
            if len(follow):
                per_round_report = int(round(float(follow.iloc[0])))
                total_followup_report = int(round(float(follow.sum())))
                regression_method_design_note = (
                    f"The Regression Method recommends {total_followup_report:,} follow-up hosts in total to detect a decline from {_format_percent(p0_design_report)} to "
                    f"{_format_percent(target_design_report)} over {years_report} years with {infer_conf_text} confidence and {power_text} power, split evenly as about "
                    f"{per_round_report:,} hosts per annual follow-up round. This is a trend-detection allocation, not a guarantee of precise annual prevalence estimates."
                )
                if regression_change is not None and np.isfinite(float(regression_change)) and float(regression_change) >= 0:
                    regression_method_design_note += " The fitted historical trend is not declining towards that target, so this is a counterfactual planning reference."
            planning_reference_note = planning_reference_note + (" " + regression_method_design_note if regression_method_design_note else "")
    except Exception:
        regression_method_design_note = ""
        planning_reference_note = ""

    yearly_table_html = _yearly_report_table_html(year_summary_df)
    effort_table_html = _effort_report_table_html(effort_df)
    settings_table_html = _settings_report_table_html(
        regression_model=regression_model,
        host_density_km2=host_density_km2,
        change_method_settings=change_method_settings,
        regression_settings=regression_settings,
        benchmark_label=benchmark_label,
    )

    headline = overview_lines[0] if overview_lines else f"Survey evidence from {before_year} to {after_year}."

    host_cov = _spatial_metric_value(spatial_diagnostics_df, "Host polygon coverage proportion")
    host_area_cov = _spatial_metric_value(spatial_diagnostics_df, "Host area covered proportion")
    top10 = _spatial_metric_value(spatial_diagnostics_df, "Top 10% host polygons share of surveyed footprint")
    effective = _spatial_metric_value(spatial_diagnostics_df, "Effective surveyed host polygon proportion")

    def metric_pct(value: Any) -> str:
        try:
            return _format_percent(float(value)) if np.isfinite(float(value)) else "-"
        except Exception:
            return "-"

    run_assumptions = list(analysis_assumptions or [])
    include_survey_host = not (len(run_assumptions) > 3 and "excluded" in str(run_assumptions[3]).lower())
    concise_assumptions = build_concise_analysis_assumptions(
        host_density_km2=host_density_km2,
        include_unmatched_survey_polygons=include_survey_host,
        regression_model=regression_model,
    )
    status_line = next(
        (x for x in run_assumptions if str(x).startswith("Survey outcomes are interpreted")),
        "Positive and negative survey outcomes were mapped using the selected classification rules; other outcomes were excluded.",
    )

    def assumption_columns(items: list[str]) -> str:
        mid = (len(items) + 1) // 2
        left, right = items[:mid], items[mid:]
        left_html = "".join(f"<div class='assumption-item'><b>{i+1}.</b> {escape(str(v))}</div>" for i, v in enumerate(left))
        right_html = "".join(f"<div class='assumption-item'><b>{mid+i+1}.</b> {escape(str(v))}</div>" for i, v in enumerate(right))
        return f"<table class='two-col assumption-columns' role='presentation'><tr><td>{left_html}</td><td>{right_html}</td></tr></table>"

    assumption_html = assumption_columns(concise_assumptions)
    summary_bullets = overview_lines[1:5] if len(overview_lines) > 1 else overview_lines
    data_caveats = caveats[3:] if len(caveats) > 3 else []

    spatial_summary = []
    if np.isfinite(host_cov) if isinstance(host_cov, (int, float, np.floating)) else False:
        spatial_summary.append(f"Survey coverage reached {metric_pct(host_cov)} of mapped host polygons")
    if np.isfinite(host_area_cov) if isinstance(host_area_cov, (int, float, np.floating)) else False:
        spatial_summary.append(f"{metric_pct(host_area_cov)} of mapped host area was represented")
    if np.isfinite(top10) if isinstance(top10, (int, float, np.floating)) else False:
        spatial_summary.append(f"the top 10% of surveyed host polygons contained {metric_pct(top10)} of the survey footprint")
    spatial_summary_text = "; ".join(spatial_summary) + "." if spatial_summary else "Spatial coverage diagnostics were not available for this dataset."

    def reg_value(metric: str) -> str:
        if regression_summary_df is None or regression_summary_df.empty:
            return "-"
        work = regression_summary_df.loc[regression_summary_df["metric"].astype(str) == str(metric), "value"]
        if work.empty:
            return "-"
        value = work.iloc[0]
        try:
            if metric == "Information-matrix condition number":
                return "-" if not np.isfinite(float(value)) else f"{float(value):.3g}"
        except Exception:
            pass
        return str(value)

    diag_summary_parts = []
    model_form = reg_value("Model form")
    fit_valid = reg_value("Fit valid")
    fit_conv = reg_value("Fit converged")
    separation = reg_value("Separation suspected")
    if model_form != "-":
        diag_summary_parts.append(f"Model form: {model_form}")
    if fit_valid != "-":
        diag_summary_parts.append(f"Fit valid: {fit_valid}")
    if fit_conv != "-":
        diag_summary_parts.append(f"Converged: {fit_conv}")
    if separation != "-":
        diag_summary_parts.append(f"Separation suspected: {separation}")
    diag_summary = " · ".join(diag_summary_parts)

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    @page {{ size:A4; margin:12mm 14mm 12mm 14mm; }}
    body {{ font-family:Arial,Helvetica,sans-serif; color:#293a34; margin:0; font-size:8.7pt; line-height:1.35; }}
    .page {{ width:100%; box-sizing:border-box; }}
    .page-break {{ page-break-before:always; }}
    .rule {{ height:3px; background:#1f5a49; margin:0 0 10px; }}
    h1 {{ color:#173d32; font-size:21pt; line-height:1.08; margin:0 0 4px; }}
    h2 {{ color:#173d32; font-size:13.2pt; line-height:1.15; margin:12px 0 6px; }}
    h3 {{ color:#2e5146; font-size:9pt; margin:0 0 4px; }}
    .period {{ color:#6d7d78; font-size:8.4pt; margin-bottom:8px; }}
    .headline {{ background:#eef5f2; border-left:3px solid #2d725c; padding:8px 10px; margin:7px 0 11px; color:#1e4539; font-weight:700; font-size:10pt; }}
    .trend-figure {{ width:170mm; height:auto; display:block; margin:2px auto 10px; }}
    .kpis {{ width:100%; border-collapse:separate; border-spacing:8px 0; table-layout:fixed; margin:4px 0 12px; }}
    .kpis td {{ width:33.33%; border:1px solid #e1e8e4; background:#fafcfb; padding:9px 10px; vertical-align:top; border-radius:6px; }}
    .kpi-title {{ color:#6a7b75; font-size:7pt; font-weight:700; text-transform:uppercase; letter-spacing:.3px; }}
    .kpi-label {{ color:#7d8b86; font-size:6.9pt; margin-top:4px; }}
    .kpi-value {{ color:#173d32; font-size:14pt; font-weight:800; line-height:1.05; margin-top:2px; }}
    .kpi-value-small {{ color:#173d32; font-size:11.4pt; font-weight:800; line-height:1.05; margin-top:1px; }}
    .text-block {{ margin-top:2px; }}
    .bullets {{ margin:3px 0 0; }}
    .bullet-row {{ margin:0 0 4px; padding:0; border:none; }}
    .bullet-mark {{ color:#2d725c; font-weight:700; display:inline-block; width:12px; }}
    .meta {{ margin-top:8px; color:#71817b; font-size:7.4pt; }}
    .report-table {{ width:100%; border-collapse:collapse; table-layout:fixed; font-size:7.3pt; margin-top:5px; }}
    .report-table th {{ background:#edf2f0; color:#315247; border-bottom:1px solid #bdcec7; padding:5px 5px; text-align:left; font-weight:700; word-wrap:break-word; -pdf-keep-in-frame-mode:shrink; }}
    .report-table td {{ border-bottom:1px solid #e4e9e7; padding:5px 5px; color:#3e4e49; word-wrap:break-word; -pdf-keep-in-frame-mode:shrink; }}
    .metric-strip {{ width:100%; border-collapse:separate; border-spacing:7px 0; table-layout:fixed; margin:6px 0 8px; }}
    .metric-strip td {{ width:25%; border:1px solid #e1e8e4; background:#fafcfb; padding:8px; vertical-align:top; border-radius:6px; }}
    .metric-strip .v {{ color:#173d32; font-size:12.2pt; font-weight:800; margin-bottom:2px; }}
    .metric-strip .l {{ color:#71817b; font-size:6.8pt; line-height:1.15; }}
    .soft-card {{ border:1px solid #e4ebe7; background:#fbfcfb; padding:8px 10px; margin:6px 0 0; border-radius:6px; }}
    .note {{ border-left:3px solid #b37b39; background:#fcf8f1; padding:8px 10px; margin:8px 0 0; color:#514534; }}
    .section-note {{ color:#687a73; font-size:7.35pt; margin:0 0 5px; }}
    .two-col {{ width:100%; border-collapse:separate; border-spacing:12px 0; table-layout:fixed; }}
    .two-col td {{ width:50%; vertical-align:top; padding:0; }}
    .assumption-columns {{ margin-top:2px; }}
    .assumption-item {{ margin:0 0 7px; color:#4a5b55; font-size:7.45pt; line-height:1.3; }}
    .small-note {{ color:#70807a; font-size:7.1pt; margin-top:8px; }}
    .footer {{ color:#788782; font-size:6.9pt; margin-top:10px; border-top:1px solid #e4e9e7; padding-top:5px; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="rule"></div>
    <h1>Plant Pest Prevalence Report</h1>
    <div class="period">Analysis period: {escape(str(before_year))} - {escape(str(after_year))}</div>
    <div class="headline">{escape(headline)}</div>
    <h2>Prevalence through time</h2>
    <img class="trend-figure" src="{trend_plot_uri}" alt="Estimated prevalence over time" />
    <table class="kpis" role="presentation"><tr>
      <td><div class="kpi-title">Estimated change</div><div class="kpi-label">Change Method</div><div class="kpi-value-small">{escape(_format_signed_percent(change_method))}</div><div class="kpi-label">Regression Method</div><div class="kpi-value-small">{escape(_format_signed_percent(regression_change))}</div></td>
      <td><div class="kpi-title">Average fitted trend</div><div class="kpi-value">{escape(_format_signed_percent(annual_change))}</div><div class="kpi-label">percentage-point change per year</div></td>
      <td><div class="kpi-title">Final prevalence</div><div class="kpi-label">Change Method</div><div class="kpi-value-small">{escape(_format_percent(change_final))}</div><div class="kpi-label">Regression Method</div><div class="kpi-value-small">{escape(_format_percent(regression_final))}</div></td>
    </tr></table>
    <h2>Interpretation</h2>
    <div class="text-block">{_html_bullet_list(summary_bullets)}{("<div class='meta'>" + escape(overview_meta) + "</div>") if overview_meta else ""}</div>
    <div class="footer">Decision-support estimates derived from the loaded survey polygons and the assumptions summarised in this report.</div>
  </div>

  <div class="page-break"></div>
  <div class="page">
    <div class="rule"></div>
    <h1 style="font-size:17pt;">Survey evidence</h1>
    <div class="section-note">Yearly polygon-derived coverage and prevalence estimates used by both methods.</div>
    {yearly_table_html}
    <h2>Spatial coverage</h2>
    <table class="metric-strip" role="presentation"><tr>
      <td><div class="v">{escape(metric_pct(host_cov))}</div><div class="l">Host polygons reached</div></td>
      <td><div class="v">{escape(metric_pct(host_area_cov))}</div><div class="l">Host area represented</div></td>
      <td><div class="v">{escape(metric_pct(top10))}</div><div class="l">Survey footprint in top 10% of surveyed host polygons</div></td>
      <td><div class="v">{escape(metric_pct(effective))}</div><div class="l">Effective surveyed host polygon proportion</div></td>
    </tr></table>
    <div class="soft-card">{escape(spatial_summary_text)}</div>
    {('<h2>Warnings</h2>' + _html_bullet_list(data_caveats[:2])) if data_caveats else ''}
    <div class="footer">Spatial coverage describes the survey footprint; it does not establish that the historical survey was representative of the full host population.</div>
  </div>

  <div class="page-break"></div>
  <div class="page">
    <div class="rule"></div>
    <h1 style="font-size:17pt;">Planning and analysis record</h1>
    <h2>Planning references</h2>
    <div class="section-note">Three different planning questions are shown: baseline precision, Change Method change detection, and Regression Method trend detection.</div>
    {effort_table_html}
    <div class="note"><b>How the planning references were calculated.</b> {escape(planning_reference_note) if planning_reference_note else 'Initial prevalence precision sizes the first survey. The Change Method sizes both compared surveys equally to detect the selected absolute change. The Regression Method sizes the total follow-up programme and splits it across the annual follow-up rounds.'}</div>
    <table class="two-col" role="presentation" style="margin-top:10px;"><tr>
      <td>
        <h2 style="margin-top:0;">Analysis settings</h2>
        {settings_table_html}
        {('<div class="small-note">' + escape(diag_summary) + '.</div>') if diag_summary else ''}
      </td>
      <td>
        <h2 style="margin-top:0;">Key assumptions and limitations</h2>
        {assumption_html}
        <div class="small-note"><b>Outcome classification.</b> {escape(status_line)}</div>
      </td>
    </tr></table>
    <div class="footer">The estimates are not ground-truth prevalence. Planning references do not prove historical survey adequacy, and model uncertainty does not include all uncertainty introduced by polygon interpretation, host mapping or survey targeting.</div>
  </div>
</body></html>'''


def _write_results_report_pdf_matplotlib(
    pdf_path: Path,
    year_summary_df: pd.DataFrame,
    method_df: pd.DataFrame,
    regression_summary_df: pd.DataFrame,
    regression_fit_df: pd.DataFrame,
    prevalence_ts_df: pd.DataFrame,
    effort_df: pd.DataFrame,
    spatial_diagnostics_df: pd.DataFrame,
    before_year: int,
    after_year: int,
    regression_model: str,
    host_density_km2: float,
    change_method_settings: dict[str, Any],
    regression_settings: dict[str, Any],
    benchmark_label: str | None = None,
):
    # Backup PDF writer. It uses matplotlib only, so the report can still be
    # produced if the HTML-to-PDF package is not installed correctly.
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        _draw_report_front_page(
            pdf=pdf,
            year_summary_df=year_summary_df,
            method_df=method_df,
            regression_summary_df=regression_summary_df,
            regression_fit_df=regression_fit_df,
            prevalence_ts_df=prevalence_ts_df,
            effort_df=effort_df,
            spatial_diagnostics_df=spatial_diagnostics_df,
            before_year=before_year,
            after_year=after_year,
            host_density_km2=host_density_km2,
            target_prevalence=float(regression_settings.get("target", DEFAULT_REGRESSION_METHOD_TARGET)),
        )

        fig = plot_quick_results_trend(prevalence_ts_df, regression_fit_df, before_year, after_year)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        effort_plot = _plot_sampling_effort_combined_figure(effort_df, benchmark_label=benchmark_label)
        pdf.savefig(effort_plot, bbox_inches="tight")
        plt.close(effort_plot)


def write_results_report_pdf(
    pdf_path: Path,
    year_summary_df: pd.DataFrame,
    method_df: pd.DataFrame,
    regression_summary_df: pd.DataFrame,
    regression_fit_df: pd.DataFrame,
    prevalence_ts_df: pd.DataFrame,
    effort_df: pd.DataFrame,
    spatial_diagnostics_df: pd.DataFrame,
    before_year: int,
    after_year: int,
    regression_model: str,
    host_density_km2: float,
    change_method_settings: dict[str, Any],
    regression_settings: dict[str, Any],
    benchmark_label: str | None = None,
    analysis_assumptions: list[str] | None = None,
):
    # Main PDF writer. The preferred route is HTML/CSS because it gives cleaner
    # page layout. If that is unavailable, we fall back to matplotlib pages.
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pisa is None:
        _write_results_report_pdf_matplotlib(
            pdf_path=pdf_path,
            year_summary_df=year_summary_df,
            method_df=method_df,
            regression_summary_df=regression_summary_df,
            regression_fit_df=regression_fit_df,
            prevalence_ts_df=prevalence_ts_df,
            effort_df=effort_df,
            spatial_diagnostics_df=spatial_diagnostics_df,
            before_year=before_year,
            after_year=after_year,
            regression_model=regression_model,
            host_density_km2=host_density_km2,
            change_method_settings=change_method_settings,
            regression_settings=regression_settings,
            benchmark_label=benchmark_label,
        )
        return

    trend_fig = plot_quick_results_trend(prevalence_ts_df, regression_fit_df, before_year, after_year)
    try:
        html_text = _build_results_report_html(
            year_summary_df=year_summary_df,
            method_df=method_df,
            regression_summary_df=regression_summary_df,
            regression_fit_df=regression_fit_df,
            prevalence_ts_df=prevalence_ts_df,
            effort_df=effort_df,
            spatial_diagnostics_df=spatial_diagnostics_df,
            before_year=before_year,
            after_year=after_year,
            host_density_km2=host_density_km2,
            target_prevalence=float(regression_settings.get("target", DEFAULT_REGRESSION_METHOD_TARGET)),
            trend_plot_uri=_figure_to_data_uri(trend_fig),
            effort_plot_uri="",
            regression_model=regression_model,
            change_method_settings=change_method_settings,
            regression_settings=regression_settings,
            benchmark_label=benchmark_label,
            analysis_assumptions=analysis_assumptions,
        )
        with pdf_path.open("wb") as pdf_file:
            render_result = pisa.CreatePDF(
                src=html_text,
                dest=pdf_file,
                encoding="utf-8",
                path=str(APP_DIR),
            )
        if render_result.err:
            raise RuntimeError("The HTML report could not be converted to PDF.")
    finally:
        plt.close(trend_fig)


def build_quick_results_method_summary(
    prevalence_ts: pd.DataFrame,
    regression_summary_df: pd.DataFrame,
    regression_fit_df: pd.DataFrame,
    before_year: int,
    after_year: int,
) -> pd.DataFrame:
    before_row = prevalence_ts.loc[prevalence_ts["year"] == before_year]
    after_row = prevalence_ts.loc[prevalence_ts["year"] == after_year]
    rows: list[dict[str, Any]] = []
    regression_valid = False
    if not regression_fit_df.empty and "model_pi_hat" in regression_fit_df.columns:
        model_vals = pd.to_numeric(regression_fit_df["model_pi_hat"], errors="coerce")
        regression_valid = bool(np.isfinite(model_vals).sum() >= 2)
    if regression_valid:
        slope_val = _series_lookup_value(regression_summary_df, "metric", "Trend slope (logit scale)")
        rows.extend(
            [
                {"method": "Regression Method", "metric": "Slope", "value": slope_val},
                {"method": "Regression Method", "metric": "Final prevalence", "value": round(float(model_vals.iloc[-1]), 4)},
                {"method": "Regression Method", "metric": "Estimated change", "value": round(float(model_vals.iloc[-1] - model_vals.iloc[0]), 4)},
            ]
        )
    else:
        fit_reason = _series_lookup_value(regression_summary_df, "metric", "Fit reason")
        note = "Regression fit was not valid under the backend diagnostics"
        if fit_reason not in (None, ""):
            note += f" ({fit_reason})"
        rows.append({"method": "Regression Method", "metric": "Note", "value": note})
    if before_row.empty or after_row.empty:
        rows.append({"method": "Change Method", "metric": "Note", "value": "Selected years need observed survey data"})
    else:
        rows.extend(
            [
                {"method": "Change Method", "metric": "Final prevalence", "value": round(float(after_row["pi_hat"].iloc[0]), 4)},
                {"method": "Change Method", "metric": "Estimated change", "value": round(float(after_row["pi_hat"].iloc[0] - before_row["pi_hat"].iloc[0]), 4)},
            ]
        )
    return pd.DataFrame(rows)


def _plot_sampling_effort_combined_figure(effort_df: pd.DataFrame, benchmark_label: str | None = None):
    """Create a compact sampling-adequacy figure for the fallback PDF route."""
    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    if effort_df.empty:
        ax.text(0.5, 0.5, "No sampling-effort data available.", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([]); fig.tight_layout(); return fig
    years = effort_df["year"].astype(int).tolist()
    x = np.arange(len(years), dtype=float); width = 0.19
    observed = pd.to_numeric(effort_df["observed_prop_population"], errors="coerce").to_numpy(dtype=float)
    baseline_req = pd.to_numeric(effort_df.get("initial_prevalence_required_prop_population"), errors="coerce").to_numpy(dtype=float)
    change_req = pd.to_numeric(effort_df["change_method_required_prop_population"], errors="coerce").to_numpy(dtype=float)
    reg_req = pd.to_numeric(effort_df["regression_required_prop_population"], errors="coerce").to_numpy(dtype=float)
    ax.bar(x-1.5*width, observed, width=width, label="Estimated historical effort")
    ax.bar(x-0.5*width, baseline_req, width=width, label="Initial prevalence precision")
    ax.bar(x+0.5*width, change_req, width=width, label="Change Method per-survey")
    ax.bar(x+1.5*width, reg_req, width=width, label="Regression Method follow-up")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8); ax.set_ylabel("Share of host population")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _pos: f"{100*y:.1f}%"))
    ax.set_xticks(x); ax.set_xticklabels([str(year) for year in years]); ax.legend(loc="upper left", frameon=False, fontsize=7.6, ncol=2)
    if benchmark_label:
        ax.text(1.0, -0.17, str(benchmark_label), ha="right", va="top", fontsize=7.4, color="#66736f", transform=ax.transAxes)
    fig.tight_layout(pad=0.8); return fig

def summarise_prevalence_timeseries_from_hosts(year_summary_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["year", "n", "x", "pi_hat", "ci_low", "ci_high"]
    if year_summary_df is None or year_summary_df.empty:
        return pd.DataFrame(columns=cols)

    work = year_summary_df.copy()
    work = work[(work["surveyed_hosts_est"] > 0) & np.isfinite(work["surveyed_hosts_est"])].copy()
    if work.empty:
        return pd.DataFrame(columns=cols)

    out = work[["year", "surveyed_hosts_est", "infected_hosts_est", "estimated_prevalence"]].copy()
    out = out.rename(columns={"surveyed_hosts_est": "n", "infected_hosts_est": "x", "estimated_prevalence": "pi_hat"})
    out["ci_low"] = np.nan
    out["ci_high"] = np.nan
    return out[cols].sort_values("year").reset_index(drop=True)


def build_sampling_effort_summary_from_hosts(
    year_summary_df: pd.DataFrame,
    year_before: int,
    year_after: int,
    change_method_delta: float,
    change_method_corr: float,
    planning_alpha: float,
    planning_power: float,
    regression_method_target: float,
    baseline_confidence: float = DEFAULT_BASELINE_PREVALENCE_CONF,
    baseline_ci_width: float = DEFAULT_BASELINE_PREVALENCE_WIDTH,
    baseline_planning_prevalence: float = DEFAULT_BASELINE_PREVALENCE_UPPER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy combined effort helper retained for compatibility.

    It now uses the same configurable baseline-precision, Change Method and
    Regression Method calculations as the main workflow under an SRS reference.
    The main Results workflow uses the more detailed method-specific helpers below.
    """
    summary_cols = [
        "year",
        "observed_hosts",
        "observed_infected_hosts",
        "observed_prevalence",
        "regression_method_required_hosts",
        "regression_method_observed_ratio",
        "change_method_required_hosts",
    ]
    if year_summary_df is None or year_summary_df.empty:
        empty = pd.DataFrame(columns=summary_cols)
        return empty, pd.DataFrame(columns=["year", "series", "n_hosts"])

    effort = year_summary_df[["year", "surveyed_hosts_est", "infected_hosts_est", "estimated_prevalence"]].copy()
    effort = effort.rename(
        columns={
            "surveyed_hosts_est": "observed_hosts",
            "infected_hosts_est": "observed_infected_hosts",
            "estimated_prevalence": "observed_prevalence",
        }
    ).sort_values("year")
    effort["regression_method_required_hosts"] = np.nan

    before_row = effort.loc[effort["year"] == year_before]
    after_row = effort.loc[effort["year"] == year_after]

    if not before_row.empty:
        pi0 = float(before_row["observed_prevalence"].iloc[0])
        observed_initial = float(pd.to_numeric(before_row["observed_hosts"], errors="coerce").iloc[0])
        observed_initial = max(0.0, observed_initial) if np.isfinite(observed_initial) else 0.0
        n_initial = compute_n_baseline_prevalence_cells(
            p0_upper=float(baseline_planning_prevalence),
            conf_level=float(baseline_confidence),
            width=float(baseline_ci_width),
            population_size=None,
        )
        effort.loc[effort["year"] == year_before, "regression_method_required_hosts"] = n_initial

        followup_years = effort.loc[
            (effort["year"] > year_before)
            & (effort["year"] <= year_after)
            & (pd.to_numeric(effort["observed_hosts"], errors="coerce") > 0),
            "year",
        ].to_numpy(dtype=float)
        if followup_years.size:
            t_vec = followup_years - float(year_before)
            n_total = compute_n_total_regression_method_cells(
                pi0=pi0,
                pi_target=regression_method_target,
                alpha=planning_alpha,
                power=planning_power,
                t_vec=t_vec,
                population_size=None,
                phi=1.0,
                initial_n=float(n_initial),
            )
            n_per_round = int(np.ceil(n_total / len(t_vec)))
            effort.loc[effort["year"].isin(followup_years.tolist()), "regression_method_required_hosts"] = n_per_round

    effort["regression_method_observed_ratio"] = effort["observed_hosts"] / effort["regression_method_required_hosts"].replace({0: np.nan})

    change_method_required = np.nan
    if not before_row.empty and not after_row.empty:
        p0 = float(before_row["observed_prevalence"].iloc[0])
        change_method_required = compute_n_change_method_cells(
            p0=p0,
            delta=change_method_delta,
            corr=change_method_corr,
            alpha=planning_alpha,
            power=planning_power,
            population_size=None,
        )
    effort["change_method_required_hosts"] = np.nan
    if np.isfinite(change_method_required):
        effort.loc[effort["year"].isin([year_before, year_after]), "change_method_required_hosts"] = change_method_required

    chart_df = pd.concat(
        [
            effort[["year", "observed_hosts"]].rename(columns={"observed_hosts": "n_hosts"}).assign(series="Estimated hosts covered"),
            effort[["year", "regression_method_required_hosts"]].rename(columns={"regression_method_required_hosts": "n_hosts"}).assign(series="Regression Method recommended"),
            effort.loc[effort["year"].isin([year_before, year_after]), ["year", "change_method_required_hosts"]]
            .rename(columns={"change_method_required_hosts": "n_hosts"})
            .assign(series="Change Method recommended"),
        ],
        ignore_index=True,
    )
    chart_df = chart_df.dropna(subset=["n_hosts"]).copy()
    chart_df["n_hosts"] = chart_df["n_hosts"].astype(float)
    return effort[summary_cols], chart_df


def estimate_total_host_population(
    host_gdf: Any | None,
    host_density_km2: float,
    layers: dict[str, Any] | None = None,
    host_column_name: str | None = None,
    host_values: list[str] | tuple[str, ...] | set[str] | None = None,
    include_unmatched_survey_polygons: bool = True,
) -> float:
    host_landscape = build_real_host_landscape(
        host_gdf,
        layers or {},
        host_column_name=host_column_name,
        host_values=host_values,
        include_unmatched_survey_polygons=include_unmatched_survey_polygons,
    )
    if host_landscape is None or len(host_landscape) == 0:
        return float("nan")
    try:
        total_area_km2 = float(pd.to_numeric(host_landscape.geometry.area, errors="coerce").fillna(0.0).sum()) / 1_000_000.0
        return total_area_km2 * max(float(host_density_km2), 0.0)
    except Exception:
        return float("nan")


def plot_sampling_effort_bars(chart_df: pd.DataFrame, year_before: int, year_after: int):
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    if chart_df.empty:
        ax.set_title("Observed vs recommended sampling effort")
        ax.text(0.5, 0.5, "No sampling-effort data available.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    years = sorted(chart_df["year"].astype(int).unique())
    x = np.arange(len(years), dtype=float)
    width = 0.24
    year_to_x = {year: xpos for year, xpos in zip(years, x)}

    observed = chart_df[chart_df["series"] == "Observed"].set_index("year")["n_cells"]
    regression_method = chart_df[chart_df["series"] == "Regression Method recommended"].set_index("year")["n_cells"]
    change_method = chart_df[chart_df["series"] == "Change Method paired requirement"].set_index("year")["n_cells"]

    ax.bar(
        x - width,
        [float(observed.get(year, np.nan)) for year in years],
        width=width,
        color="#7f7f7f",
        label="Observed",
    )
    ax.bar(
        x,
        [float(regression_method.get(year, np.nan)) for year in years],
        width=width,
        color="#2b8cbe",
        label="Regression Method recommended",
    )

    c_years = [year_before, year_after]
    c_x = [year_to_x[year] + width for year in c_years if year in year_to_x and year in change_method.index]
    c_vals = [float(change_method.loc[year]) for year in c_years if year in year_to_x and year in change_method.index]
    if c_x:
        ax.bar(
            c_x,
            c_vals,
            width=width,
            color="#de2d26",
            alpha=0.85,
            label="Change Method paired requirement",
        )

    ax.set_title("Observed vs recommended sampling effort")
    ax.set_xlabel("Year")
    ax.set_ylabel("Eligible surveyed cells")
    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in years])
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


# ============================================================================
# Dashboard UI
# ============================================================================


def render_grid(df: pd.DataFrame):
    return render.DataGrid(df, width="100%")


def _settings_field(label: str, hint: str, control: Any) -> Any:
    return ui.div(
        ui.tags.div(label, class_="form-label fw-semibold"),
        ui.p(hint, class_="text-muted", style="font-size:0.92rem;margin-top:-0.15rem;margin-bottom:0.45rem;"),
        control,
        style="margin-bottom:0.85rem;",
    )


def _settings_cog_button(input_id: str) -> Any:
    return ui.input_action_button(
        input_id,
        "Advanced settings",
        class_="btn-outline-secondary",
    )


def _results_advanced_settings_panel() -> Any:
    """Advanced controls kept out of the main results workflow."""
    return ui.div(
        ui.h5("Survey interpretation", class_="settings-section-title"),
        _settings_field(
            "Survey status column",
            "Choose the field that records the survey outcome.",
            ui.input_select("quick_status_column", "", choices={"__auto__": "Auto-detect"}, selected="__auto__"),
        ),
        _settings_field(
            "Positive outcomes",
            "Outcomes treated as disease present.",
            ui.input_checkbox_group("quick_positive_statuses", "", choices={}, selected=[]),
        ),
        _settings_field(
            "Negative outcomes",
            "Outcomes treated as disease absent. Unticked values are ignored.",
            ui.input_checkbox_group("quick_negative_statuses", "", choices={}, selected=[]),
        ),
        ui.hr(),
        ui.h5("Host landscape", class_="settings-section-title"),
        _settings_field(
            "Host category column",
            "Optional filter when the host layer contains several categories.",
            ui.input_select("quick_host_column", "", choices={"__all__": "Use all host polygons"}, selected="__all__"),
        ),
        _settings_field(
            "Host categories to include",
            "Select the categories retained in the host landscape.",
            ui.input_checkbox_group("quick_host_values", "", choices={}, selected=[]),
        ),
        _settings_field(
            "Survey-only host areas",
            "Include surveyed areas outside the public host map as inferred host area.",
            ui.input_checkbox("quick_include_unmatched_surveys", "Include survey-only host areas", value=True),
        ),
        _settings_field(
            "Host density (hosts per km²)",
            "Used to convert mapped host area into estimated host counts.",
            ui.input_numeric("quick_host_density_km2", "", value=2500.0, min=0.0, step=100.0),
        ),
        ui.hr(),
        ui.h5("Initial prevalence survey", class_="settings-section-title"),
        _settings_field(
            "Baseline confidence level",
            "Confidence level used to size the first survey for estimating the unknown starting prevalence.",
            ui.input_numeric("baseline_confidence", "", value=DEFAULT_BASELINE_PREVALENCE_CONF, min=0.80, max=0.999, step=0.01),
        ),
        _settings_field(
            "Desired total confidence-interval width",
            "Maximum total width of the baseline prevalence interval; 0.025 means 2.5 percentage points.",
            ui.input_numeric("baseline_ci_width", "", value=DEFAULT_BASELINE_PREVALENCE_WIDTH, min=0.001, max=0.5, step=0.005),
        ),
        _settings_field(
            "Baseline planning prevalence",
            "Conservative prevalence used before the first survey is available. It should be a defensible upper planning value for the pest population.",
            ui.input_numeric("baseline_planning_prevalence", "", value=DEFAULT_BASELINE_PREVALENCE_UPPER, min=0.001, max=0.99, step=0.01),
        ),
        ui.hr(),
        ui.h5("Change and trend detection", class_="settings-section-title"),
        ui.p(
            "Baseline confidence controls precision of the starting estimate. Significance level and power control how much evidence is required to detect the selected change or trend.",
            class_="settings-help",
        ),
        _settings_field(
            "Sampling benchmark",
            "Reference sampling design used when comparing historical effort with the calculated requirements.",
            ui.input_select("shared_effort_benchmark", "", choices=RETRO_BENCHMARK_CHOICES, selected="srs"),
        ),
        _settings_field(
            "Hosts per location visit for MSS",
            "Used only for MSS design-effect calculations.",
            ui.input_numeric("shared_within_cluster_cap", "", value=5, min=1, step=1),
        ),
        _settings_field(
            "Fixed Design Effect",
            "Used only when the fixed Design Effect benchmark is selected.",
            ui.input_numeric("shared_fixed_phi", "", value=1.5, min=1.0, step=0.1),
        ),
        _settings_field(
            "Significance level (alpha)",
            "Evidence threshold for the Change Method and Regression Method tests. Alpha 0.05 corresponds to the usual 5% significance threshold.",
            ui.input_numeric("shared_alpha", "", value=DEFAULT_ALPHA, min=0.001, max=0.2, step=0.005),
        ),
        _settings_field(
            "Power",
            "Probability of detecting the specified change or trend when it is truly present.",
            ui.input_numeric("shared_power", "", value=DEFAULT_POWER, min=0.5, max=0.999, step=0.01),
        ),
        ui.hr(),
        ui.h5("Change Method", class_="settings-section-title"),
        _settings_field(
            "First survey year",
            "Starting year for the two-time-point comparison.",
            ui.input_select("change_method_before_year", "", choices={str(y): str(y) for y in range(YEAR_MIN, YEAR_MAX)}, selected="2017"),
        ),
        _settings_field(
            "Last survey year",
            "Final year for the two-time-point comparison.",
            ui.input_select("change_method_after_year", "", choices={str(y): str(y) for y in range(YEAR_MIN + 1, YEAR_MAX + 1)}, selected="2024"),
        ),
        _settings_field(
            "Minimum change to detect",
            "Smallest absolute prevalence change used in the Change Method planning benchmark.",
            ui.input_numeric("change_method_delta", "", value=DEFAULT_CHANGE_METHOD_DELTA, min=0.001, max=0.5, step=0.005),
        ),
        ui.hr(),
        ui.h5("Regression Method", class_="settings-section-title"),
        _settings_field(
            "Initial survey year",
            "First year included in the fitted trend.",
            ui.input_select("regression_before_year", "", choices={str(y): str(y) for y in range(YEAR_MIN, YEAR_MAX)}, selected="2017"),
        ),
        _settings_field(
            "Final survey year",
            "Last year included in the fitted trend.",
            ui.input_select("regression_after_year", "", choices={str(y): str(y) for y in range(YEAR_MIN + 1, YEAR_MAX + 1)}, selected="2024"),
        ),
        _settings_field(
            "Regression model",
            "Logit-linear is the primary model; alternatives are sensitivity analyses.",
            ui.input_select(
                "quick_regression_model", "",
                choices={"logistic": "Logit-linear", "fp2": "Fractional polynomial 2", "fp3": "Fractional polynomial 3"},
                selected="logistic",
            ),
        ),
        _settings_field(
            "Target prevalence",
            "Target used by the Regression Method trend-detection planning calculation.",
            ui.input_numeric("regression_target", "", value=DEFAULT_REGRESSION_METHOD_TARGET, min=0.0001, max=0.5, step=0.001),
        ),
        class_="advanced-settings-panel",
    )


def _guide_assumption_details() -> Any:
    """Full assumptions remain available without dominating the guide page."""
    return ui.tags.details(
        ui.tags.summary("View full assumptions and limitations"),
        ui.div(
            ui.h4("From polygons to prevalence"),
            ui.tags.ul(
                ui.tags.li("Hosts inside the usable surveyed host footprint are treated as surveyed."),
                ui.tags.li("A positive survey polygon is treated as infected throughout its host-covered footprint; a negative polygon contributes surveyed hosts but no infected hosts."),
                ui.tags.li("Host counts are estimated from mapped area and the selected host density rather than counted directly."),
                ui.tags.li("Survey footprint outside the public host map is added as inferred host area by default because the public layer is treated as incomplete."),
                ui.tags.li("Overlapping survey coverage is dissolved so the same ground is not counted repeatedly within a yearly estimate."),
                ui.tags.li("Only outcomes classified as positive or negative enter the calculation; other status values are ignored."),
                ui.tags.li("Recorded survey status is taken at face value; no correction for imperfect detection sensitivity is applied."),
            ),
            ui.h4("What the survey can represent"),
            ui.tags.ul(
                ui.tags.li("Historical surveys may be targeted towards high-risk or known affected locations and may not represent the wider host population."),
                ui.tags.li("The operational host landscape is approximate: the public map can omit hosts and survey-inferred host area can include some non-host space."),
                ui.tags.li("Year-to-year comparisons assume that changes in targeting, polygon construction and recording practice are not mistaken for biological change."),
                ui.tags.li("SPHN polygons are map context only and do not directly enter the prevalence calculation."),
            ),
            ui.h4("Statistical interpretation"),
            ui.tags.ul(
                ui.tags.li("The Change Method compares the selected first and last survey years only."),
                ui.tags.li("The Regression Method is model-dependent; the default analysis uses a logit-linear trend."),
                ui.tags.li("Change Method and Regression Method sample sizes are planning references, not proof that the historical survey was representative or sufficiently precise."),
                ui.tags.li("Model confidence intervals do not include uncertainty in host density, host-map completeness, polygon interpretation or targeted site selection."),
                ui.tags.li("An observed temporal trend does not by itself establish that control measures caused the change."),
            ),
            class_="guide-details-body",
        ),
        class_="guide-details",
    )


app_ui = ui.page_navbar(
    ui.nav_panel(
        "Home",
        ui.div(
            ui.div(
                ui.h1("Plant Pest Prevalence Dashboard", class_="page-title"),
                ui.p(
                    "Apply two monitoring methods to annual plant-pest survey polygons. Load and validate the source data, inspect survey coverage on the map, estimate prevalence change over time, and export a PDF report of the results.",
                    class_="page-lead",
                ),
                ui.div(
                    ui.tags.strong("Change Method"),
                    ui.tags.span(" compares estimated prevalence in the first and last selected years."),
                    ui.tags.span("  ·  "),
                    ui.tags.strong("Regression Method"),
                    ui.tags.span(" uses all selected survey years to estimate the overall trend."),
                    class_="method-inline",
                ),
                class_="page-heading home-heading",
            ),
            ui.div(
                ui.h3("How to use the dashboard", class_="home-section-title"),
                ui.p("Work through the three tabs in order for a standard analysis.", class_="compact-copy home-guide-copy"),
                class_="home-guide-heading",
            ),
            ui.div(
                ui.div(
                    ui.div("1", class_="guide-step-number"),
                    ui.div(ui.div("Home", class_="nav-card-title"), ui.div("Choose the data locations, load the files and review the automated data checks.")),
                    class_="nav-card guide-card",
                ),
                ui.div(
                    ui.div("2", class_="guide-step-number"),
                    ui.div(ui.div("Data Viewer", class_="nav-card-title"), ui.div("Inspect survey sites, host coverage and optional SPHN context for each year.")),
                    class_="nav-card guide-card",
                ),
                ui.div(
                    ui.div("3", class_="guide-step-number"),
                    ui.div(ui.div("Results", class_="nav-card-title"), ui.div("Run both methods, review the interpretation and key metrics, then export the PDF report.")),
                    class_="nav-card guide-card",
                ),
                class_="nav-card-grid",
            ),
            ui.tags.details(
                ui.tags.summary("Data Loader"),
                ui.div(
                    ui.p("The default application folders are filled in automatically. Change these locations only when analysing another dataset.", class_="compact-copy"),
                    ui.div(
                        ui.div(ui.tags.div("Survey polygons", class_="field-label"), ui.input_text("home_target_sites_dir", "", value=str(TARGET_SITES_DIR)), class_="loader-field"),
                        ui.div(ui.tags.div("Host coverage", class_="field-label"), ui.input_text("home_host_layer_path", "", value=str(LARCH_HOST_DIR)), class_="loader-field"),
                        ui.div(ui.tags.div("SPHN polygons (optional)", class_="field-label"), ui.input_text("home_sphn_layer_path", "", value=str(SPHN_DIR)), class_="loader-field"),
                        class_="loader-grid",
                    ),
                    # These hidden action inputs are triggered when a path box is clicked.
                    # The server then opens the operating system's native folder chooser.
                    ui.input_action_button("btn_browse_survey_dir", "Browse survey folder", style="display:none;"),
                    ui.input_action_button("btn_browse_host_dir", "Browse host folder", style="display:none;"),
                    ui.input_action_button("btn_browse_sphn_dir", "Browse SPHN folder", style="display:none;"),
                    ui.input_action_button("btn_home_load_data", "Load data", class_="btn-primary load-button"),
                    ui.output_ui("home_data_status_panel"),
                    class_="loader-body",
                ),
                class_="data-loader-details",
            ),
            class_="app-page home-page",
        ),
    ),
    ui.nav_panel(
        "Data Viewer",
        ui.div(
            ui.div(
                ui.h2("Data Viewer", class_="page-title page-title-small"),
                ui.p("Inspect the loaded survey, host and optional SPHN layers before interpreting the results.", class_="page-lead"),
                class_="page-heading compact-heading",
            ),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_slider("year_selected_interactive", "Year", min=YEAR_MIN, max=YEAR_MAX, value=2024, step=1, sep=""),
                    ui.input_select(
                        "view_mode_interactive", "View",
                        choices={"uk": "Whole UK", "all_zoom": "Loaded polygons", "survey": "Survey sites", "larch": "Host distribution", "sphn": "SPHN sites"},
                        selected="all_zoom",
                    ),
                    ui.input_checkbox_group(
                        "visible_layers_interactive", "Layers",
                        choices={"survey": "Survey sites", "larch": "Host distribution", "sphn": "SPHN sites"},
                        selected=["survey"],
                    ),
                    width="245px",
                ),
                ui.card(ui.output_ui("survey_map_interactive_trial"), class_="map-card"),
            ),
            class_="app-page data-page",
        ),
    ),
    ui.nav_panel(
        "Results",
        ui.div(
            ui.div(
                ui.div(
                    ui.h2("Results", class_="page-title page-title-small"),
                ),
                ui.div(
                    ui.input_action_button("btn_quick_results", "Run analysis", class_="btn-primary"),
                    ui.popover(
                        _settings_cog_button("btn_results_settings"),
                        _results_advanced_settings_panel(),
                        title="Analysis settings",
                        placement="bottom",
                        options={"customClass": "results-settings-popover"},
                    ),
                    ui.download_button("download_results_pdf", "Download PDF", class_="btn-outline-secondary"),
                    class_="results-actions",
                ),
                class_="results-toolbar",
            ),
            ui.output_ui("results_status_panel"),
            ui.card(
                ui.output_plot("regression_method_plot", height="430px"),
                class_="clean-card chart-card results-chart-card",
            ),
            ui.output_ui("results_key_figures_panel"),
            ui.card(
                ui.card_header("Interpretation"),
                ui.output_ui("results_overview_panel"),
                class_="clean-card interpretation-card",
            ),
            ui.tags.details(
                ui.tags.summary("Warnings"),
                ui.div(ui.output_ui("results_caveats_panel"), class_="details-body"),
                class_="results-details",
            ),
            ui.tags.details(
                ui.tags.summary("Assumptions and limitations"),
                ui.div(ui.output_ui("results_assumptions_panel"), class_="details-body"),
                class_="results-details",
            ),
            class_="app-page results-page",
        ),
    ),
    title="Plant Pest Prevalence Dashboard",
    id="top_nav",
    header=ui.tags.head(
        ui.tags.style(
            r'''
            :root { --ink:#23342e; --muted:#687873; --green:#1f5a49; --green-soft:#eef5f2; --line:#dfe7e3; --paper:#ffffff; --amber:#8a6122; --amber-soft:#fbf6ea; }
            body { background:#f7f9f8; color:var(--ink); }
            .navbar { background:#fff !important; border-bottom:1px solid #e6ece9; box-shadow:none !important; }
            .app-page { max-width:1460px; margin:0 auto; padding:1.35rem 1.5rem 2.25rem; }
            .page-heading { margin:0 0 1rem; }
            .home-heading { max-width:980px; }
            .compact-heading { margin-bottom:.8rem; }
                        .page-title { margin:0; color:#173d32; font-size:2rem; line-height:1.12; font-weight:720; }
            .page-title-small { font-size:1.58rem; }
            .page-lead { color:var(--muted); margin:.42rem 0 0; max-width:58rem; font-size:.96rem; line-height:1.48; }
            .method-inline { color:#52665f; margin-top:.62rem; font-size:.88rem; }
            .home-guide-heading { margin:.15rem 0 .5rem; }
            .home-section-title { color:#24493d; font-size:1rem; margin:0 0 .18rem; font-weight:720; }
            .home-guide-copy { margin-bottom:0; }
            .clean-card { border:1px solid var(--line) !important; border-radius:11px !important; box-shadow:0 1px 2px rgba(23,61,50,.03) !important; background:#fff; }
            .card-header { background:#fff !important; border-bottom:1px solid #edf1ef !important; color:#314b42; font-weight:680; padding:.68rem .9rem !important; }
            .nav-card-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem; margin:0 0 .9rem; }
            .nav-card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:.78rem .85rem; color:#667771; font-size:.83rem; line-height:1.35; }
            .guide-card { display:flex; gap:.65rem; align-items:flex-start; min-height:76px; }
            .guide-step-number { width:1.55rem; height:1.55rem; border-radius:50%; background:var(--green-soft); color:var(--green); display:flex; align-items:center; justify-content:center; font-weight:800; font-size:.76rem; flex:0 0 auto; }
            .nav-card-title { color:#24493d; font-weight:750; margin-bottom:.18rem; font-size:.92rem; }
            .data-loader-details { background:#fff; border:1px solid var(--line); border-radius:11px; padding:.2rem .85rem .75rem; margin-bottom:.8rem; }
            .data-loader-details > summary { cursor:pointer; list-style:none; padding:.72rem 0 .58rem; color:#24493d; font-weight:740; font-size:.98rem; }
            .data-loader-details > summary::-webkit-details-marker { display:none; }
            .data-loader-details > summary:after { content:'▾'; float:right; color:#7a8b85; font-size:.85rem; }
            .data-loader-details:not([open]) > summary:after { content:'▸'; }
            .loader-body { border-top:1px solid #edf1ef; padding-top:.72rem; }
            .loader-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem; margin:.65rem 0 .55rem; }
            .loader-field { min-width:0; }
            .field-label { color:#52675f; font-size:.8rem; font-weight:680; margin-bottom:.22rem; }
            .compact-copy { margin:0; color:#6c7b76; font-size:.84rem; }
            .load-button { margin-top:.08rem; }
            .data-check-panel { background:#f9fbfa; border:1px solid #e5ebe8; border-radius:9px; padding:.7rem .78rem; margin-top:.75rem; }
            .data-check-header { display:flex; justify-content:space-between; align-items:center; gap:.7rem; margin-bottom:.45rem; }
            .data-check-title { font-weight:720; color:#24493d; }
            .data-check-counts { color:#778781; font-size:.78rem; }
            .data-readiness { border-radius:8px; padding:.48rem .62rem; margin:.25rem 0 .48rem; font-size:.82rem; }
            .data-readiness-ok { background:#edf7f1; border:1px solid #d3e8da; color:#275e43; }
            .data-readiness-error { background:#fff3f1; border:1px solid #efd0ca; color:#82382d; }
            .data-source-line { color:#6d7d78; font-size:.77rem; margin:.2rem 0 .45rem; }
            .data-check-row { display:flex; gap:.45rem; align-items:flex-start; margin:.3rem 0; }
            .data-check-badge { min-width:3.4rem; padding:.08rem .32rem; border-radius:999px; text-align:center; font-size:.62rem; font-weight:750; }
            .data-check-ok { background:#e4f3ea; color:#23633f; } .data-check-warning { background:#fff1d7; color:#87560b; } .data-check-error { background:#fde6e1; color:#913a2c; }
            .data-check-message { font-size:.78rem; color:#596a64; line-height:1.35; }
            .all-checks-details { border-top:1px solid #edf1ef; margin-top:.55rem; padding-top:.4rem; }
            .all-checks-details summary { cursor:pointer; color:#567068; font-size:.78rem; font-weight:650; }
            .data-page .bslib-sidebar-layout { gap:.8rem; }
            .map-card { border:1px solid var(--line) !important; border-radius:11px !important; overflow:hidden; box-shadow:none !important; }
            .results-toolbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:.72rem; }
            .results-actions { display:flex; gap:.42rem; align-items:center; flex-wrap:wrap; }
            .results-status { padding:.5rem .65rem; border-radius:8px; font-size:.82rem; margin:0 0 .65rem; }
            .results-status-error { background:#fff3f1; border:1px solid #efd0ca; color:#81382d; }
            .results-status-neutral { background:#f2f5f3; border:1px solid #e2e8e5; color:#60716b; }
            .chart-card { margin-bottom:1rem; padding:.35rem .45rem .05rem; }
            .results-chart-card .card-body { padding:.3rem .5rem .15rem !important; }
            .interpretation-card { margin-bottom:1rem; }
            .results-kpi-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.85rem; margin:0 0 1rem; }
            .results-kpi-card { border:1px solid var(--line); border-radius:10px; padding:.72rem .82rem; background:#fff; }
            .results-kpi-title { font-size:.77rem; font-weight:680; color:#597069; margin-bottom:.38rem; }
            .results-kpi-label { font-size:.7rem; color:#85918d; margin-top:.24rem; }
            .results-kpi-value { font-size:1.24rem; line-height:1.05; font-weight:760; color:#183f34; }
            .results-kpi-value-secondary { font-size:1.04rem; } .results-kpi-value-large { font-size:1.62rem; margin-top:.22rem; }
            .results-kpi-caption { font-size:.71rem; color:#7a8984; margin-top:.32rem; line-height:1.3; }
            .results-summary-line { margin:0 0 .6rem; color:#485a54; line-height:1.5; font-size:.88rem; }
            .results-summary-meta { border-top:1px solid #edf1ef; margin-top:.65rem; padding-top:.6rem; color:#75847f; font-size:.75rem; }
            .results-details { margin-top:.85rem; background:#fff; border:1px solid var(--line); border-radius:10px; padding:.6rem .9rem; }
            .results-details > summary { cursor:pointer; color:#315e50; font-weight:690; padding:.22rem 0; }
            .details-body { padding:.45rem 0 .1rem; border-top:1px solid #edf1ef; margin-top:.35rem; }
            .results-caveat-list { margin:0; padding-left:1.05rem; } .results-caveat-list li { margin-bottom:.38rem; color:#53655f; line-height:1.4; font-size:.82rem; }
            .results-empty { color:#798883; font-style:italic; }
            .results-assumption-intro { color:#6a7b75; font-size:.8rem; margin:0 0 .45rem; }
            .run-assumption-list { padding-left:1.15rem; margin:0; }
            .run-assumption-list li { margin-bottom:.42rem; line-height:1.38; color:#586a63; font-size:.8rem; }
            .settings-section-title { color:#24493d; margin-top:.2rem; }
            .advanced-settings-panel { min-width:44rem; max-width:72rem; }
            .results-settings-popover { max-width:min(90vw,1100px) !important; width:max-content !important; }
            .popover .popover-body { max-height:76vh; overflow-y:auto; }
            @media (max-width: 1000px) { .loader-grid,.nav-card-grid { grid-template-columns:1fr; } .results-toolbar { align-items:flex-start; flex-direction:column; } }
            @media (max-width: 800px) { .results-kpi-grid { grid-template-columns:1fr; } .app-page { padding:1rem; } }
            #home_target_sites_dir, #home_host_layer_path, #home_sphn_layer_path { cursor:pointer; background-color:#fff; }
            '''
        ),
        ui.tags.script(
            r'''
            // Clicking a Data Loader path field asks the local Python process to
            // open the native directory picker. Event delegation keeps this working
            // if Shiny redraws any surrounding UI.
            document.addEventListener('click', function(event) {
              const browseButtons = {
                home_target_sites_dir: 'btn_browse_survey_dir',
                home_host_layer_path: 'btn_browse_host_dir',
                home_sphn_layer_path: 'btn_browse_sphn_dir'
              };
              const buttonId = browseButtons[event.target && event.target.id];
              if (!buttonId) return;
              const button = document.getElementById(buttonId);
              if (button) {
                event.preventDefault();
                event.target.blur();
                button.click();
              }
            });
            document.addEventListener('DOMContentLoaded', function() {
              ['home_target_sites_dir','home_host_layer_path','home_sphn_layer_path'].forEach(function(id) {
                const el = document.getElementById(id);
                if (el) el.title = 'Click to choose a folder';
              });
            });
            '''
        )
    ),
)

# ============================================================================
# Shiny Server Logic
# ============================================================================


def server(input, output, session):
    # These reactive values are the app's memory.
    # They store loaded map layers and the latest results so different tabs can
    # use the same data without reloading everything.
    rv_layer_info_interactive = reactive.Value({"available_files": {}, "layers": {}, "messages": []})
    rv_host_info_interactive = reactive.Value({"available_file": False, "layer": None, "messages": []})
    rv_sphn_info_interactive = reactive.Value({"available_file": False, "layer": None, "messages": []})
    rv_quick_layer_info = reactive.Value({"available_files": {}, "layers": {}, "messages": []})
    rv_quick_host_info = reactive.Value({"available_file": False, "layer": None, "messages": []})
    rv_startup_load_done = reactive.Value(False)
    rv_results_bundle = reactive.Value(None)
    rv_results_error = reactive.Value("")

    def _empty_layer_info(message: str | None = None) -> dict[str, Any]:
        # Standard empty result for a folder of survey layers.
        out = {"available_files": {}, "layers": {}, "messages": [], "checks": [], "fatal_messages": [], "files_loaded": []}
        if message:
            out["messages"].append(message)
        return out

    def _empty_single_layer_info(message: str | None = None) -> dict[str, Any]:
        # Standard empty result for one optional polygon layer.
        out = {"available_file": False, "layer": None, "messages": [], "checks": [], "fatal_messages": [], "files_loaded": []}
        if message:
            out["messages"].append(message)
        return out

    def _empty_results_bundle() -> dict[str, Any]:
        # Empty tables shown before the user presses Generate Results.
        return {
            "year_summary_df": pd.DataFrame(),
            "prevalence_ts_df": pd.DataFrame(),
            "regression_summary_df": pd.DataFrame(),
            "regression_fit_df": pd.DataFrame(),
            "method_df": pd.DataFrame(),
            "change_method_effort_df": pd.DataFrame(),
            "regression_effort_df": pd.DataFrame(),
            "spatial_diagnostics_df": pd.DataFrame(),
            "total_host_population": np.nan,
            "benchmark_label": "",
            "benchmark_note": "",
            "change_method_benchmark_phi": np.nan,
            "regression_benchmark_phi": np.nan,
            "change_method_benchmark_rho": np.nan,
            "regression_benchmark_rho": np.nan,
            "benchmark_phi_estimator": "",
        }

    def _show_wait_notification(message: str = "Please wait...") -> str:
        # Show a small message while a slow GIS or modelling step is running.
        return ui.notification_show(
            message,
            duration=None,
            close_button=False,
            type="message",
            session=session,
        )

    def _sync_quick_results_choices(layer_info: dict[str, Any], host_info: dict[str, Any]) -> None:
        # After data is loaded, update the settings menu with real column names
        # and real survey status values from the user's files.
        status_col_choices, status_col_selected = _extract_status_column_choices_from_layers(layer_info.get("layers", {}))
        ui.update_select(
            "quick_status_column",
            choices={choice: ("Auto-detect" if choice == "__auto__" else choice) for choice in status_col_choices},
            selected=status_col_selected,
            session=session,
        )
        choices, selected_pos, selected_neg = _extract_status_choices_from_layers(
            layer_info.get("layers", {}),
            status_column_name=status_col_selected,
        )
        ui.update_checkbox_group(
            "quick_positive_statuses",
            choices={val: val for val in choices},
            selected=selected_pos,
            session=session,
        )
        ui.update_checkbox_group(
            "quick_negative_statuses",
            choices={val: val for val in choices},
            selected=selected_neg,
            session=session,
        )

        host_col_choices, host_col_selected = _extract_host_column_choices_from_layer(host_info.get("layer"))
        ui.update_select(
            "quick_host_column",
            choices={choice: ("Use all host polygons" if choice == "__all__" else choice) for choice in host_col_choices},
            selected=host_col_selected,
            session=session,
        )
        host_value_choices, host_value_selected = _extract_host_value_choices_from_layer(host_info.get("layer"), host_col_selected)
        ui.update_checkbox_group(
            "quick_host_values",
            choices={val: val for val in host_value_choices},
            selected=host_value_selected,
            session=session,
        )

        years = extract_available_survey_years(layer_info.get("layers", {}))
        if years:
            year_choices = {str(year): str(year) for year in years}
            first_year = str(min(years))
            last_year = str(max(years))
            for input_id, selected in (
                ("change_method_before_year", first_year),
                ("change_method_after_year", last_year),
                ("regression_before_year", first_year),
                ("regression_after_year", last_year),
            ):
                ui.update_select(input_id, choices=year_choices, selected=selected, session=session)

    def _load_shared_data_sources(
        target_dir_text: str,
        host_path_text: str,
        sphn_path_text: str,
    ) -> list[str]:
        # Load all data paths from the Home page once, then share the same data
        # with the Data Viewer and Results tabs. Unexpected loader failures are
        # converted into user-facing messages instead of terminating the app.
        messages: list[str] = []

        if target_dir_text:
            try:
                layer_info = load_target_sites(Path(target_dir_text).expanduser())
            except Exception as exc:
                layer_info = _empty_layer_info(f"Survey polygons could not be loaded: {exc}")
                layer_info["fatal_messages"].append(str(exc))
                layer_info["checks"].append({"level": "error", "message": f"Survey polygons could not be loaded: {exc}"})
        else:
            layer_info = _empty_layer_info("Survey layer path was left blank, so survey polygons were not loaded.")
            layer_info["fatal_messages"].append("Survey polygon path is blank.")
            layer_info["checks"].append({"level": "error", "message": "Survey polygon path is blank."})
        rv_layer_info_interactive.set(layer_info)
        rv_quick_layer_info.set(layer_info)
        messages.extend(layer_info.get("messages", []))

        years = extract_available_survey_years(layer_info.get("layers", {}))
        if years:
            ui.update_slider(
                "year_selected_interactive",
                min=min(years),
                max=max(years),
                value=max(years),
                session=session,
            )

        if host_path_text:
            try:
                host_info = load_host_layer(Path(host_path_text).expanduser())
            except Exception as exc:
                host_info = _empty_single_layer_info(f"Host coverage could not be loaded: {exc}")
                host_info["fatal_messages"].append(str(exc))
                host_info["checks"].append({"level": "error", "message": f"Host coverage could not be loaded: {exc}"})
        else:
            host_info = _empty_single_layer_info("Host layer path was left blank, so the host layer was not loaded.")
            host_info["fatal_messages"].append("Host coverage path is blank.")
            host_info["checks"].append({"level": "error", "message": "Host coverage path is blank."})
        rv_host_info_interactive.set(host_info)
        rv_quick_host_info.set(host_info)
        messages.extend(host_info.get("messages", []))

        if sphn_path_text:
            try:
                sphn_info = load_sphn_layer(Path(sphn_path_text).expanduser())
            except Exception as exc:
                sphn_info = _empty_single_layer_info(f"SPHN polygons could not be loaded: {exc}")
                sphn_info["checks"].append({"level": "warning", "message": f"SPHN polygons could not be loaded: {exc}"})
        else:
            sphn_info = _empty_single_layer_info("SPHN path was left blank. The map will work without the optional SPHN layer.")
            sphn_info["checks"].append({"level": "warning", "message": "SPHN path is blank; the optional SPHN map layer will be unavailable."})
        rv_sphn_info_interactive.set(sphn_info)
        messages.extend(sphn_info.get("messages", []))

        _sync_quick_results_choices(layer_info, host_info)
        return messages

    @reactive.calc
    def results_bundle_selected():
        # Always return a results bundle, even before results have been generated.
        # This prevents plots and tables from crashing on the first page load.
        bundle = rv_results_bundle.get()
        return _empty_results_bundle() if bundle is None else bundle

    @reactive.calc
    def prevalence_ts_selected():
        return results_bundle_selected()["prevalence_ts_df"]

    @reactive.calc
    def regression_summary_selected():
        return results_bundle_selected()["regression_summary_df"]

    @reactive.calc
    def regression_fit_selected():
        return results_bundle_selected()["regression_fit_df"]

    @reactive.calc
    def method_summary_selected():
        return results_bundle_selected()["method_df"]

    @reactive.calc
    def quick_year_summary_selected():
        return results_bundle_selected()["year_summary_df"]

    @reactive.calc
    def quick_total_host_population_selected():
        return results_bundle_selected()["total_host_population"]

    @reactive.calc
    def benchmark_assumptions_selected():
        return _retrospective_benchmark_assumptions(
            str(input.shared_effort_benchmark()),
            fixed_phi=float(input.shared_fixed_phi()),
            within_cluster_cap=int(input.shared_within_cluster_cap()),
        )

    @reactive.calc
    def change_method_effort_selected():
        return results_bundle_selected()["change_method_effort_df"]

    @reactive.calc
    def regression_effort_selected():
        return results_bundle_selected()["regression_effort_df"]

    def _choose_data_directory(input_id: str, title: str) -> None:
        """Open a local native folder picker and write the choice back to Shiny."""
        current = str(getattr(input, input_id)() or "").strip()
        selected, error = _open_native_directory_dialog(current, title)
        if error:
            ui.notification_show(
                error + " You can still type or paste the path manually.",
                duration=10,
                type="warning",
                session=session,
            )
            return
        if selected:
            ui.update_text(input_id, value=selected, session=session)

    @reactive.effect
    @reactive.event(input.btn_browse_survey_dir)
    def _browse_survey_directory():
        _choose_data_directory("home_target_sites_dir", "Choose survey polygon folder")

    @reactive.effect
    @reactive.event(input.btn_browse_host_dir)
    def _browse_host_directory():
        _choose_data_directory("home_host_layer_path", "Choose host coverage folder")

    @reactive.effect
    @reactive.event(input.btn_browse_sphn_dir)
    def _browse_sphn_directory():
        _choose_data_directory("home_sphn_layer_path", "Choose SPHN polygon folder")

    @reactive.effect
    def _autoload_home_data_once():
        # Try to load the default files when the app starts.
        # If paths are wrong, the app keeps running and shows a warning instead.
        if rv_startup_load_done.get():
            return
        target_dir_text = str(input.home_target_sites_dir()).strip()
        host_path_text = str(input.home_host_layer_path()).strip()
        sphn_path_text = str(input.home_sphn_layer_path()).strip()
        _load_shared_data_sources(target_dir_text, host_path_text, sphn_path_text)
        rv_startup_load_done.set(True)

    @reactive.effect
    @reactive.event(input.btn_home_load_data)
    def _load_home_data_on_click():
        # Reload map data when the user changes paths on the Home page.
        notif_id = _show_wait_notification()
        target_dir_text = str(input.home_target_sites_dir()).strip()
        host_path_text = str(input.home_host_layer_path()).strip()
        sphn_path_text = str(input.home_sphn_layer_path()).strip()
        try:
            _load_shared_data_sources(target_dir_text, host_path_text, sphn_path_text)
            rv_results_bundle.set(None)
            rv_results_error.set("")
        except Exception as exc:
            rv_results_error.set(f"Data loading failed unexpectedly: {exc}")
            ui.notification_show(f"Data loading failed: {exc}", duration=10, type="error", session=session)
        finally:
            ui.notification_remove(notif_id, session=session)

    @reactive.effect
    @reactive.event(input.btn_quick_results)
    def _load_quick_results_inputs():
        # Main Results button. Validate the loaded data first, then convert the
        # polygon record into yearly prevalence, fitted methods and benchmarks.
        notif_id = _show_wait_notification()
        try:
            layer_info = rv_layer_info_interactive.get()
            host_info = rv_host_info_interactive.get()
            rv_quick_layer_info.set(layer_info)
            rv_quick_host_info.set(host_info)
            layers = layer_info.get("layers", {})
            host_gdf = host_info.get("layer")
            available_years = extract_available_survey_years(layers)

            blocking: list[str] = []
            blocking.extend([str(x) for x in layer_info.get("fatal_messages", []) if str(x).strip()])
            blocking.extend([str(x) for x in host_info.get("fatal_messages", []) if str(x).strip()])
            if not layers:
                blocking.append("No usable survey polygon files are loaded.")
            if host_gdf is None or len(host_gdf) == 0:
                blocking.append("No usable host coverage polygons are loaded.")
            if len(available_years) < 2:
                blocking.append("At least two recognised survey years are required to estimate change and fit a trend.")

            host_density = float(input.quick_host_density_km2())
            if not np.isfinite(host_density) or host_density <= 0:
                blocking.append("Host density must be greater than zero.")

            baseline_confidence = float(input.baseline_confidence())
            baseline_ci_width = float(input.baseline_ci_width())
            baseline_planning_prevalence = float(input.baseline_planning_prevalence())
            if not 0 < baseline_confidence < 1:
                blocking.append("Baseline confidence level must be between 0 and 1.")
            if not 0 < baseline_ci_width < 1:
                blocking.append("Baseline confidence-interval width must be between 0 and 1.")
            if not 0 < baseline_planning_prevalence < 1:
                blocking.append("Baseline planning prevalence must be between 0 and 1.")

            positive_statuses = list(input.quick_positive_statuses())
            negative_statuses = list(input.quick_negative_statuses())
            overlap_status = set(positive_statuses) & set(negative_statuses)
            if overlap_status:
                blocking.append("The same survey outcome cannot be classified as both positive and negative.")
            if not positive_statuses and not negative_statuses:
                blocking.append("No survey outcomes have been classified. Open Advanced settings in Results and choose which outcomes count as positive and/or negative.")

            try:
                change_before = int(input.change_method_before_year())
                change_after = int(input.change_method_after_year())
                regression_before = int(input.regression_before_year())
                regression_after = int(input.regression_after_year())
            except Exception:
                blocking.append("The analysis years could not be read from the Advanced results settings.")
                change_before = change_after = regression_before = regression_after = 0
            if change_before >= change_after:
                blocking.append("The Change Method first survey year must be earlier than its last survey year.")
            if regression_before >= regression_after:
                blocking.append("The Regression Method initial survey year must be earlier than its final survey year.")
            if available_years:
                for label, year in (("Change first", change_before), ("Change last", change_after), ("Regression initial", regression_before), ("Regression final", regression_after)):
                    if year not in available_years:
                        blocking.append(f"{label} year {year} is not present in the loaded survey data.")

            if blocking:
                # De-duplicate while preserving order so DEFRA sees a concise list
                # of what needs fixing rather than a raw Python exception.
                seen = set()
                blocking = [msg for msg in blocking if not (msg in seen or seen.add(msg))]
                message = "Results were not generated:\n- " + "\n- ".join(blocking)
                rv_results_error.set(message)
                rv_results_bundle.set(None)
                ui.notification_show("Results were not generated. See the Results page for the data checks that need attention.", duration=9, type="error", session=session)
                return

            rv_results_error.set("")
            status_column_name = str(input.quick_status_column())
            host_column_name = str(input.quick_host_column())
            host_values = list(input.quick_host_values())
            include_unmatched = bool(input.quick_include_unmatched_surveys())

            year_summary_df = build_quick_results_host_summary(
                layers,
                host_gdf,
                host_density_km2=host_density,
                positive_status_values=positive_statuses,
                negative_status_values=negative_statuses,
                status_column_name=status_column_name,
                host_column_name=host_column_name,
                host_values=host_values,
                include_unmatched_survey_polygons=include_unmatched,
                years=available_years,
            )
            prevalence_ts_df = summarise_prevalence_timeseries_from_hosts(year_summary_df)
            if prevalence_ts_df.empty or len(prevalence_ts_df) < 2:
                raise ValueError("The loaded polygons did not produce at least two yearly prevalence estimates. Check survey outcomes, host overlap and host density.")

            regression_input_df = prevalence_ts_df.loc[
                (prevalence_ts_df["year"] >= regression_before) &
                (prevalence_ts_df["year"] <= regression_after)
            ].copy()
            if len(regression_input_df) < 2:
                raise ValueError("The selected Regression Method period contains fewer than two survey years with usable data.")
            regression_summary_df, regression_fit_df = run_regression_method_real_with_model(
                regression_input_df,
                model_form=str(input.quick_regression_model()),
            )
            method_df = build_quick_results_method_summary(
                prevalence_ts_df,
                regression_summary_df,
                regression_fit_df,
                change_before,
                change_after,
            )

            total_host_population = estimate_total_host_population(
                host_gdf,
                host_density_km2=host_density,
                layers=layers,
                host_column_name=host_column_name,
                host_values=host_values,
                include_unmatched_survey_polygons=include_unmatched,
            )
            bench = benchmark_assumptions_selected()
            change_phi_info = _resolve_quick_results_phi(
                layers=layers, host_gdf=host_gdf, pilot_year=change_before,
                host_density_km2=host_density, sampling_mode=str(bench["sampling_mode"]),
                cluster_inflation_mode=str(bench["cluster_inflation_mode"]),
                within_cluster_cap=int(bench["within_cluster_cap"]), fixed_phi=float(bench["fixed_phi"]),
                positive_status_values=positive_statuses, negative_status_values=negative_statuses,
                status_column_name=status_column_name, host_column_name=host_column_name, host_values=host_values,
                include_unmatched_survey_polygons=include_unmatched,
            )
            regression_phi_info = _resolve_quick_results_phi(
                layers=layers, host_gdf=host_gdf, pilot_year=regression_before,
                host_density_km2=host_density, sampling_mode=str(bench["sampling_mode"]),
                cluster_inflation_mode=str(bench["cluster_inflation_mode"]),
                within_cluster_cap=int(bench["within_cluster_cap"]), fixed_phi=float(bench["fixed_phi"]),
                positive_status_values=positive_statuses, negative_status_values=negative_statuses,
                status_column_name=status_column_name, host_column_name=host_column_name, host_values=host_values,
                include_unmatched_survey_polygons=include_unmatched,
            )
            benchmark_label = str(bench["label"])
            if str(bench["cluster_inflation_mode"]) != "none":
                change_phi = float(change_phi_info.get("phi", 1.0))
                regression_phi = float(regression_phi_info.get("phi", 1.0))
                if abs(change_phi - regression_phi) < 1e-12:
                    benchmark_label += f"; Design Effect {change_phi:.2f}"
                else:
                    benchmark_label += f"; Change DE {change_phi:.2f}, Regression DE {regression_phi:.2f}"

            change_method_effort_df = build_change_method_effort_from_hosts(
                year_summary_df, year_before=change_before, year_after=change_after, pilot_year=change_before,
                delta=float(input.change_method_delta()), corr=0.0, alpha=float(input.shared_alpha()),
                power=float(input.shared_power()), total_population_size=total_host_population, layers=layers,
                host_gdf=host_gdf, host_density_km2=host_density, sampling_mode=str(bench["sampling_mode"]),
                cluster_inflation_mode=str(bench["cluster_inflation_mode"]),
                within_cluster_cap=int(bench["within_cluster_cap"]), fixed_phi=float(bench["fixed_phi"]),
                positive_status_values=positive_statuses, negative_status_values=negative_statuses,
                status_column_name=status_column_name, host_column_name=host_column_name, host_values=host_values,
                include_unmatched_survey_polygons=include_unmatched,
            )
            regression_effort_df = build_regression_effort_from_hosts(
                year_summary_df, year_before=regression_before, year_after=regression_after,
                alpha=float(input.shared_alpha()), power=float(input.shared_power()),
                regression_method_target=float(input.regression_target()),
                baseline_confidence=float(input.baseline_confidence()),
                baseline_ci_width=float(input.baseline_ci_width()),
                baseline_planning_prevalence=float(input.baseline_planning_prevalence()),
                total_population_size=total_host_population,
                layers=layers, host_gdf=host_gdf, host_density_km2=host_density, pilot_year=regression_before,
                sampling_mode=str(bench["sampling_mode"]), cluster_inflation_mode=str(bench["cluster_inflation_mode"]),
                within_cluster_cap=int(bench["within_cluster_cap"]), fixed_phi=float(bench["fixed_phi"]),
                positive_status_values=positive_statuses, negative_status_values=negative_statuses,
                status_column_name=status_column_name, host_column_name=host_column_name, host_values=host_values,
                include_unmatched_survey_polygons=include_unmatched,
            )
            spatial_diagnostics_df = build_spatial_survey_diagnostics(
                layers, host_gdf, host_column_name=host_column_name, host_values=host_values,
                include_unmatched_survey_polygons=include_unmatched,
            )
            rv_results_bundle.set({
                "year_summary_df": year_summary_df,
                "prevalence_ts_df": prevalence_ts_df,
                "regression_summary_df": regression_summary_df,
                "regression_fit_df": regression_fit_df,
                "method_df": method_df,
                "change_method_effort_df": change_method_effort_df,
                "regression_effort_df": regression_effort_df,
                "spatial_diagnostics_df": spatial_diagnostics_df,
                "total_host_population": total_host_population,
                "benchmark_label": benchmark_label,
                "benchmark_note": str(bench.get("note", "")),
                "change_method_benchmark_phi": float(change_phi_info.get("phi", np.nan)),
                "regression_benchmark_phi": float(regression_phi_info.get("phi", np.nan)),
                "change_method_benchmark_rho": float(change_phi_info.get("rho", np.nan)),
                "regression_benchmark_rho": float(regression_phi_info.get("rho", np.nan)),
                "benchmark_phi_estimator": str(change_phi_info.get("estimator", "")),
                "available_years": available_years,
                "change_before_year": change_before,
                "change_after_year": change_after,
                "regression_before_year": regression_before,
                "regression_after_year": regression_after,
                "regression_model": str(input.quick_regression_model()),
                "host_density_km2": host_density,
                "alpha": float(input.shared_alpha()),
                "power": float(input.shared_power()),
                "baseline_confidence": float(input.baseline_confidence()),
                "baseline_ci_width": float(input.baseline_ci_width()),
                "baseline_planning_prevalence": float(input.baseline_planning_prevalence()),
                "change_delta": float(input.change_method_delta()),
                "regression_target": float(input.regression_target()),
                "sampling_mode": str(bench["sampling_mode"]),
                "status_column_name": status_column_name,
                "positive_statuses": list(positive_statuses),
                "negative_statuses": list(negative_statuses),
                "host_column_name": host_column_name,
                "host_values": list(host_values),
                "include_unmatched_survey_polygons": bool(include_unmatched),
            })
            ui.notification_show("Results generated. Open the Results page to review the estimates, assumptions and warnings.", duration=6, type="message", session=session)
        except Exception as exc:
            rv_results_bundle.set(None)
            rv_results_error.set(f"Results could not be generated: {exc}")
            ui.notification_show(f"Results could not be generated: {exc}", duration=10, type="error", session=session)
        finally:
            ui.notification_remove(notif_id, session=session)

    @reactive.effect
    def _update_positive_status_choices_for_column():
        # If the user changes the survey-status column, rebuild the positive and
        # negative checkbox lists using values from that column.
        layer_info = rv_quick_layer_info.get()
        if not layer_info.get("layers"):
            layer_info = rv_layer_info_interactive.get()
        layers = layer_info.get("layers", {})
        status_col = str(input.quick_status_column())
        choices, selected_pos, selected_neg = _extract_status_choices_from_layers(layers, status_column_name=status_col)
        ui.update_checkbox_group(
            "quick_positive_statuses",
            choices={val: val for val in choices},
            selected=selected_pos,
            session=session,
        )
        ui.update_checkbox_group(
            "quick_negative_statuses",
            choices={val: val for val in choices},
            selected=selected_neg,
            session=session,
        )

    @reactive.effect
    def _update_host_value_choices_for_column():
        # If the user selects a host-filter column, show the values available in
        # that column so they can choose which polygons count as host habitat.
        host_info = rv_quick_host_info.get()
        if host_info.get("layer") is None:
            host_info = rv_host_info_interactive.get()
        host_layer = host_info.get("layer")
        host_col = str(input.quick_host_column())
        choices, selected = _extract_host_value_choices_from_layer(host_layer, host_column_name=host_col)
        ui.update_checkbox_group(
            "quick_host_values",
            choices={val: val for val in choices},
            selected=selected,
            session=session,
        )

    @output
    @render.ui
    def home_data_status_panel():
        layer_info = rv_layer_info_interactive.get()
        host_info = rv_host_info_interactive.get()
        sphn_info = rv_sphn_info_interactive.get()

        years = extract_available_survey_years(layer_info.get("layers", {}))
        fatal_messages = [str(x) for x in layer_info.get("fatal_messages", [])] + [str(x) for x in host_info.get("fatal_messages", [])]
        ready = bool(layer_info.get("layers")) and host_info.get("layer") is not None and len(years) >= 2 and not fatal_messages

        checks: list[dict[str, str]] = []
        checks.extend(layer_info.get("checks", []))
        checks.extend(host_info.get("checks", []))
        checks.extend(sphn_info.get("checks", []))
        if years:
            checks.append({"level": "ok", "message": f"Survey years available: {', '.join(str(y) for y in years)}."})
        elif layer_info.get("layers"):
            checks.append({"level": "error", "message": "No survey years could be recognised in the loaded survey polygons."})
        if len(years) == 1:
            checks.append({"level": "error", "message": "Only one survey year was detected; at least two are required for change/trend analysis."})

        seen = set()
        clean_checks: list[dict[str, str]] = []
        for check in checks:
            level = str(check.get("level", "warning"))
            msg = str(check.get("message", "")).strip()
            key = (level, msg)
            if not msg or key in seen:
                continue
            seen.add(key)
            clean_checks.append({"level": level, "message": msg})

        counts = {"ok": 0, "warning": 0, "error": 0}
        for check in clean_checks:
            counts[str(check.get("level", "warning"))] = counts.get(str(check.get("level", "warning")), 0) + 1

        labels = {"ok": "OK", "warning": "Warning", "error": "Error"}
        def row(check: dict[str, str]):
            level = str(check.get("level", "warning"))
            return ui.div(
                ui.tags.span(labels.get(level, "Note"), class_=f"data-check-badge data-check-{level}"),
                ui.tags.span(str(check.get("message", "")), class_="data-check-message"),
                class_="data-check-row",
            )

        visible_checks = [c for c in clean_checks if str(c.get("level", "")) in {"warning", "error"}]
        visible_rows = [row(c) for c in visible_checks]
        all_rows = [row(c) for c in clean_checks]

        readiness = ui.div(
            ui.tags.strong("Ready for analysis" if ready else "Action needed"),
            ui.tags.span(" Required survey and host data passed the checks." if ready else " Resolve the errors below before running the analysis."),
            class_="data-readiness data-readiness-ok" if ready else "data-readiness data-readiness-error",
        )
        source_line = (
            f"Survey files: {len(layer_info.get('files_loaded', []))} · "
            f"Host files: {len(host_info.get('files_loaded', []))} · "
            f"SPHN files: {len(sphn_info.get('files_loaded', []))} · "
            f"Years: {len(years)}"
        )

        return ui.div(
            ui.div(
                ui.div("Data checks", class_="data-check-title"),
                ui.div(f"{counts.get('ok', 0)} passed · {counts.get('warning', 0)} warnings · {counts.get('error', 0)} errors", class_="data-check-counts"),
                class_="data-check-header",
            ),
            readiness,
            ui.div(source_line, class_="data-source-line"),
            *visible_rows,
            ui.tags.details(
                ui.tags.summary("View all checks"),
                ui.div(*all_rows),
                class_="all-checks-details",
            ) if all_rows else ui.div(),
            class_="data-check-panel",
        )


    @output
    @render.ui
    def survey_map_interactive_trial():
        year = int(input.year_selected_interactive())
        layers = rv_layer_info_interactive.get().get("layers", {})
        survey_gdf_year, _ = filter_target_sites_for_year(layers, year)
        host_gdf = rv_host_info_interactive.get().get("layer")
        sphn_gdf_year, _ = filter_sphn_for_year(rv_sphn_info_interactive.get().get("layer"), year)
        html = build_folium_map_html(
            survey_gdf=survey_gdf_year,
            host_gdf=host_gdf,
            sphn_gdf=sphn_gdf_year,
            year=year,
            view_mode=str(input.view_mode_interactive()),
            visible_layers=list(input.visible_layers_interactive()),
        )
        return ui.tags.iframe(
            srcdoc=html,
            style="width:100%;height:720px;border:0;border-radius:0.5rem;background:#fff;",
        )

    @output
    @render.data_frame
    def quick_results_summary_table():
        out = quick_year_summary_selected().copy()
        if out.empty:
            return render_grid(pd.DataFrame(columns=["year", "survey_polygons", "surveyed_area_km2", "infected_polygons", "infected_area_km2", "surveyed_hosts_est", "infected_hosts_est", "estimated_prevalence"]))
        for col in ("surveyed_area_km2", "infected_area_km2", "surveyed_hosts_est", "infected_hosts_est"):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(1)
        out["estimated_prevalence"] = out["estimated_prevalence"].apply(_format_percent)
        out = out.rename(
            columns={
                "year": "Year",
                "survey_polygons": "Survey polygons used",
                "surveyed_area_km2": "Surveyed area (km^2)",
                "infected_polygons": "Positive survey polygons",
                "infected_area_km2": "Positive surveyed area (km^2)",
                "surveyed_hosts_est": "Estimated hosts covered / treated as surveyed",
                "infected_hosts_est": "Estimated infected hosts under full-coverage assumption",
                "estimated_prevalence": "Estimated prevalence",
            }
        )
        return render_grid(out)

    @output
    @render.data_frame
    def quick_results_method_table():
        out = method_summary_selected().copy()
        if out.empty:
            return render_grid(out)
        out["method"] = out["method"].replace(
            {
                "Change Method": "Change Method",
                "Regression Method": "Regression Method",
            }
        )
        out["metric"] = out["metric"].replace(
            {
                "Final prevalence": "Estimated final prevalence",
                "Estimated change": "Estimated change",
                "Slope": "Trend slope",
                "Note": "Note",
            }
        )
        def _format_method_value(row: pd.Series) -> Any:
            metric = str(row.get("metric", ""))
            value = row.get("value")
            if metric in {"Estimated final prevalence", "Estimated change"}:
                return _format_percent(value) if metric == "Estimated final prevalence" else _format_percentage_points(value)
            if metric == "Trend slope":
                return _format_report_value(value)
            return value
        out["value"] = out.apply(_format_method_value, axis=1)
        out = out.rename(columns={"method": "Method", "metric": "Measure", "value": "Result"})
        return render_grid(out)

    @output
    @render.data_frame
    def regression_fit_diagnostics_table():
        out = regression_summary_selected().copy()
        if out.empty:
            return render_grid(pd.DataFrame(columns=["Diagnostic", "Value"]))
        keep = [
            "Model form",
            "Fit valid",
            "Fit converged",
            "Fit reason",
            "Separation suspected",
            "Information-matrix condition number",
            "Covariance basis",
        ]
        out = out.loc[out["metric"].isin(keep), ["metric", "value"]].copy()
        if out.empty:
            return render_grid(pd.DataFrame(columns=["Diagnostic", "Value"]))
        def _format_diag(row: pd.Series) -> Any:
            value = row.get("value")
            if str(row.get("metric")) == "Information-matrix condition number":
                try:
                    return "NA" if not np.isfinite(float(value)) else f"{float(value):.3g}"
                except Exception:
                    return value
            return value
        out["value"] = out.apply(_format_diag, axis=1)
        out = out.rename(columns={"metric": "Diagnostic", "value": "Value"})
        return render_grid(out)

    @output
    @render.data_frame
    def change_method_sampling_effort_table():
        out = change_method_effort_selected().copy()
        if out.empty:
            return render_grid(pd.DataFrame(columns=["year", "observed_hosts", "observed_prop_population", "required_hosts", "required_prop_population", "observed_prevalence"]))
        for col in ("observed_prop_population", "required_prop_population", "observed_prevalence"):
            out[col] = out[col].astype(float).round(4)
        for col in ("observed_hosts", "required_hosts"):
            out[col] = out[col].apply(lambda x: "" if pd.isna(x) else int(round(float(x))))
        return render_grid(out)

    @output
    @render.data_frame
    def regression_sampling_effort_table():
        out = regression_effort_selected().copy()
        if out.empty:
            return render_grid(pd.DataFrame(columns=["year", "observed_hosts", "observed_prop_population", "required_hosts", "required_prop_population", "observed_prevalence"]))
        for col in ("observed_prop_population", "required_prop_population", "observed_prevalence"):
            out[col] = out[col].astype(float).round(4)
        for col in ("observed_hosts", "required_hosts"):
            out[col] = out[col].apply(lambda x: "" if pd.isna(x) else int(round(float(x))))
        return render_grid(out)

    def _build_combined_sampling_effort_table() -> pd.DataFrame:
        """Join historical effort with three distinct planning quantities."""
        change_df = change_method_effort_selected().copy()
        reg_df = regression_effort_selected().copy()
        bundle = results_bundle_selected()
        columns = [
            "year", "observed_hosts", "observed_prop_population", "observed_prevalence",
            "initial_prevalence_required_hosts", "initial_prevalence_required_prop_population",
            "change_method_required_hosts", "change_method_required_prop_population",
            "regression_required_hosts", "regression_required_prop_population",
            "benchmark_assumption",
        ]
        if change_df.empty and reg_df.empty:
            return pd.DataFrame(columns=columns)

        if not reg_df.empty:
            merged = reg_df.rename(columns={
                "required_hosts": "regression_required_hosts",
                "required_prop_population": "regression_required_prop_population",
            })[["year", "observed_hosts", "observed_prop_population", "observed_prevalence", "regression_required_hosts", "regression_required_prop_population"]].copy()
        else:
            merged = change_df[["year", "observed_hosts", "observed_prop_population", "observed_prevalence"]].copy()
            merged["regression_required_hosts"] = np.nan
            merged["regression_required_prop_population"] = np.nan

        merged["initial_prevalence_required_hosts"] = np.nan
        merged["initial_prevalence_required_prop_population"] = np.nan
        if not merged.empty:
            baseline_candidates = pd.to_numeric(merged.loc[merged["regression_required_hosts"].notna(), "year"], errors="coerce").dropna()
            if not baseline_candidates.empty:
                baseline_year = int(baseline_candidates.min())
                baseline_mask = pd.to_numeric(merged["year"], errors="coerce") == baseline_year
                merged.loc[baseline_mask, "initial_prevalence_required_hosts"] = merged.loc[baseline_mask, "regression_required_hosts"]
                merged.loc[baseline_mask, "initial_prevalence_required_prop_population"] = merged.loc[baseline_mask, "regression_required_prop_population"]
                merged.loc[baseline_mask, "regression_required_hosts"] = np.nan
                merged.loc[baseline_mask, "regression_required_prop_population"] = np.nan

        if not change_df.empty:
            change_keep = change_df.rename(columns={
                "required_hosts": "change_method_required_hosts",
                "required_prop_population": "change_method_required_prop_population",
            })[["year", "change_method_required_hosts", "change_method_required_prop_population"]]
            merged = merged.merge(change_keep, on="year", how="outer")
        else:
            merged["change_method_required_hosts"] = np.nan
            merged["change_method_required_prop_population"] = np.nan

        merged = merged.sort_values("year").reset_index(drop=True)
        merged["benchmark_assumption"] = str(bundle.get("benchmark_label", ""))
        for col in columns:
            if col not in merged.columns:
                merged[col] = np.nan if col != "benchmark_assumption" else str(bundle.get("benchmark_label", ""))
        return merged[columns]

    @output
    @render.ui
    def home_analysis_status_panel():
        error = str(rv_results_error.get() or "").strip()
        bundle = rv_results_bundle.get()
        if error:
            return ui.div(
                ui.tags.strong("Analysis could not be generated."),
                ui.tags.span(" Open Results to see what needs attention."),
                class_="home-analysis-state home-analysis-error",
            )
        if bundle is None:
            return ui.div(
                "No analysis has been generated yet. The standard button uses the current default interpretation and statistical settings.",
                class_="home-analysis-state home-analysis-neutral",
            )
        return ui.div(
            ui.tags.strong("Results are ready."),
            ui.tags.span(" Open the Results page to review the estimates, assumptions and data-specific warnings."),
            class_="home-analysis-state home-analysis-ok",
        )

    @output
    @render.ui
    def results_assumptions_panel():
        bundle = rv_results_bundle.get()
        if bundle is None:
            return ui.div(
                ui.p("Generate results to record the assumptions used for the current analysis.", class_="results-empty"),
                ui.p("Run the analysis to record the assumptions used for this result."),
            )
        assumptions = build_concise_analysis_assumptions(
            host_density_km2=float(bundle.get("host_density_km2", input.quick_host_density_km2())),
            include_unmatched_survey_polygons=bool(bundle.get("include_unmatched_survey_polygons", True)),
            regression_model=str(bundle.get("regression_model", input.quick_regression_model())),
        )
        return ui.div(
            ui.p("Key assumptions used to interpret this result:", class_="results-assumption-intro"),
            ui.tags.ul(*[ui.tags.li(item) for item in assumptions], class_="run-assumption-list"),
        )

    @output
    @render.ui
    def results_status_panel():
        error = str(rv_results_error.get() or "").strip()
        if error:
            lines = [line.strip().lstrip("- ") for line in error.splitlines() if line.strip() and not line.lower().startswith("results were not generated")]
            return ui.div(
                ui.tags.strong("Results not available"),
                ui.tags.ul(*[ui.tags.li(line) for line in lines]) if lines else ui.p(error),
                class_="results-status results-status-error",
            )
        bundle = rv_results_bundle.get()
        if bundle is None:
            return ui.div()
        return ui.div()

    @output
    @render.ui
    def results_key_figures_panel():
        method_df = method_summary_selected()
        if method_df is None or method_df.empty:
            return ui.div("Generate results to view the key figures.", class_="results-empty")
        change_method = _method_metric_value(method_df, "Change Method", "Estimated change")
        regression_change = _method_metric_value(method_df, "Regression Method", "Estimated change")
        change_final = _method_metric_value(method_df, "Change Method", "Final prevalence")
        regression_final = _method_metric_value(method_df, "Regression Method", "Final prevalence")
        annual_change = _trend_change_per_year(regression_fit_selected())

        def kpi(title: str, body: Any):
            return ui.div(ui.div(title, class_="results-kpi-title"), body, class_="results-kpi-card")

        change_body = ui.div(
            ui.div("Change Method", class_="results-kpi-label"),
            ui.div(_format_signed_percent(change_method), class_="results-kpi-value"),
            ui.div("Regression Method", class_="results-kpi-label"),
            ui.div(_format_signed_percent(regression_change), class_="results-kpi-value results-kpi-value-secondary"),
        )
        trend_body = ui.div(
            ui.div(_format_signed_percent(annual_change), class_="results-kpi-value results-kpi-value-large"),
            ui.div("average change per year from the fitted Regression Method", class_="results-kpi-caption"),
        )
        final_body = ui.div(
            ui.div("Change Method", class_="results-kpi-label"),
            ui.div(_format_percent(change_final), class_="results-kpi-value"),
            ui.div("Regression Method", class_="results-kpi-label"),
            ui.div(_format_percent(regression_final), class_="results-kpi-value results-kpi-value-secondary"),
        )
        return ui.div(
            kpi("Estimated change in prevalence", change_body),
            kpi("Prevalence trend", trend_body),
            kpi("Estimated final prevalence", final_body),
            class_="results-kpi-grid",
        )

    @output
    @render.ui
    def results_overview_panel():
        bundle = results_bundle_selected()
        year_summary_df = quick_year_summary_selected()
        if year_summary_df is None or year_summary_df.empty:
            return ui.div("Generate results to view the interpretation summary.", class_="results-empty")
        observed = year_summary_df.loc[pd.to_numeric(year_summary_df["survey_polygons"], errors="coerce").fillna(0) > 0].copy()
        survey_years = int(len(observed))
        survey_polygons = int(pd.to_numeric(observed.get("survey_polygons", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not observed.empty else 0
        surveyed_hosts = float(pd.to_numeric(observed.get("surveyed_hosts_est", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not observed.empty else np.nan
        before_year = int(bundle.get("change_before_year", min(observed["year"]) if not observed.empty else YEAR_MIN))
        after_year = int(bundle.get("change_after_year", max(observed["year"]) if not observed.empty else YEAR_MAX))
        lines, metadata = _build_report_overview_lines(
            method_df=method_summary_selected(),
            prevalence_ts_df=prevalence_ts_selected(),
            regression_fit_df=regression_fit_selected(),
            effort_df=_build_combined_sampling_effort_table(),
            before_year=before_year,
            after_year=after_year,
            survey_years=survey_years, survey_polygons=survey_polygons, surveyed_hosts=surveyed_hosts,
            target_prevalence=float(bundle.get("regression_target", input.regression_target())),
        )
        return ui.div(
            *[ui.p(line, class_="results-summary-line") for line in lines[:6]],
            ui.p(metadata, class_="results-summary-meta") if metadata else ui.div(),
        )

    @output
    @render.ui
    def results_caveats_panel():
        bundle = results_bundle_selected()
        method_df = method_summary_selected()
        if method_df is None or method_df.empty:
            return ui.div("Generate results to view data-specific caveats.", class_="results-empty")
        baseline_observed = np.nan
        try:
            baseline_year = int(bundle.get("change_before_year", input.change_method_before_year()))
            baseline_rows = quick_year_summary_selected().loc[
                pd.to_numeric(quick_year_summary_selected()["year"], errors="coerce") == baseline_year
            ]
            if not baseline_rows.empty:
                baseline_observed = float(pd.to_numeric(baseline_rows["estimated_prevalence"], errors="coerce").iloc[0])
        except Exception:
            baseline_observed = np.nan
        caveats = build_results_report_caveats(
            effort_df=_build_combined_sampling_effort_table(),
            spatial_diagnostics_df=bundle.get("spatial_diagnostics_df", pd.DataFrame()),
            regression_change=_method_metric_value(method_df, "Regression Method", "Estimated change"),
            change_method=_method_metric_value(method_df, "Change Method", "Estimated change"),
            host_density_km2=float(bundle.get("host_density_km2", input.quick_host_density_km2())),
            baseline_observed_prevalence=baseline_observed,
            baseline_planning_prevalence=float(bundle.get("baseline_planning_prevalence", input.baseline_planning_prevalence())),
        )
        data_caveats = caveats[3:] if len(caveats) > 3 else []
        if not data_caveats:
            return ui.p("No additional warnings were triggered by the current diagnostics.", class_="results-summary-line")
        return ui.tags.ul(*[ui.tags.li(item) for item in data_caveats[:3]], class_="results-caveat-list")

    @output
    @render.data_frame
    def sampling_effort_table_combined():
        # Display the combined effort table with friendly names and percentages.
        out = _build_combined_sampling_effort_table().copy()
        if out.empty:
            return render_grid(out)
        for col in ("observed_prop_population", "initial_prevalence_required_prop_population", "change_method_required_prop_population", "regression_required_prop_population", "observed_prevalence"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        for col in ("observed_hosts", "initial_prevalence_required_hosts", "change_method_required_hosts", "regression_required_hosts"):
            out[col] = out[col].apply(lambda x: "" if pd.isna(x) else int(round(float(x))))
        out["observed_prop_population"] = out["observed_prop_population"].apply(_format_percent)
        out["initial_prevalence_required_prop_population"] = out["initial_prevalence_required_prop_population"].apply(_format_percent)
        out["change_method_required_prop_population"] = out["change_method_required_prop_population"].apply(_format_percent)
        out["regression_required_prop_population"] = out["regression_required_prop_population"].apply(_format_percent)
        out["observed_prevalence"] = out["observed_prevalence"].apply(_format_percent)
        out = out.rename(
            columns={
                "year": "Year",
                "observed_hosts": "Estimated hosts covered by survey polygons",
                "observed_prop_population": "Estimated covered share of host population",
                "observed_prevalence": "Observed prevalence",
                "initial_prevalence_required_hosts": "Initial prevalence precision requirement",
                "initial_prevalence_required_prop_population": "Initial prevalence precision share",
                "change_method_required_hosts": "Change Method per-survey requirement",
                "change_method_required_prop_population": "Change Method required share",
                "regression_required_hosts": "Regression Method follow-up allocation",
                "regression_required_prop_population": "Regression Method follow-up share",
                "benchmark_assumption": "Benchmark assumption",
            }
        )
        return render_grid(out)

    @output
    @render.plot
    def regression_method_plot():
        # Main results plot: yearly prevalence estimates plus the fitted trend.
        return plot_quick_results_trend(
            prevalence_ts_selected(),
            regression_fit_selected(),
            int(input.change_method_before_year()),
            int(input.change_method_after_year()),
        )

    def _plot_effort_props(effort_df: pd.DataFrame, title: str):
        # Simple bar chart comparing actual survey effort to the indicative benchmark.
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        if effort_df.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No sampling-effort data available.", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return fig
        years = effort_df["year"].astype(int).tolist()
        x = np.arange(len(years), dtype=float)
        width = 0.34
        ax.bar(x - width / 2, effort_df["observed_prop_population"].astype(float), width=width, color="#7f7f7f", label="Survey polygons")
        ax.bar(x + width / 2, effort_df["required_prop_population"].astype(float), width=width, color="#2b8cbe", label="Required")
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Proportion of total host population")
        ax.set_xticks(x)
        ax.set_xticklabels([str(year) for year in years])
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        return fig

    @output
    @render.plot
    def change_method_sampling_effort_plot():
        return _plot_effort_props(change_method_effort_selected(), "Estimated host coverage vs planning reference: Change Method")

    @output
    @render.plot
    def regression_sampling_effort_plot():
        return _plot_effort_props(regression_effort_selected(), "Estimated host coverage vs planning reference: Regression Method")

    @output
    @render.plot
    def sampling_effort_plot_combined():
        return _plot_sampling_effort_combined_figure(
            _build_combined_sampling_effort_table(),
            benchmark_label=str(results_bundle_selected().get("benchmark_label", "")),
        )

    @output
    @render.download(filename="plant_pest_prevalence_report.pdf", media_type="application/pdf")
    def download_results_pdf():
        # Create the printable PDF from the same tables and plots shown in the app.
        notif_id = _show_wait_notification()
        bundle = results_bundle_selected()
        try:
            year_summary_df = quick_year_summary_selected().copy()
            prevalence_ts_df = prevalence_ts_selected().copy()
            regression_summary_df = regression_summary_selected().copy()
            regression_fit_df = regression_fit_selected().copy()
            method_df = method_summary_selected().copy()
            effort_df = _build_combined_sampling_effort_table().copy()
            spatial_diagnostics_df = bundle["spatial_diagnostics_df"].copy()
            before_year = int(bundle.get("change_before_year", input.change_method_before_year()))
            after_year = int(bundle.get("change_after_year", input.change_method_after_year()))
            regression_model = str(bundle.get("regression_model", input.quick_regression_model()))

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                write_results_report_pdf(
                    pdf_path=tmp_path,
                    year_summary_df=year_summary_df,
                    method_df=method_df,
                    regression_summary_df=regression_summary_df,
                    regression_fit_df=regression_fit_df,
                    prevalence_ts_df=prevalence_ts_df,
                    effort_df=effort_df,
                    spatial_diagnostics_df=spatial_diagnostics_df,
                    before_year=before_year,
                    after_year=after_year,
                    regression_model=regression_model,
                    host_density_km2=float(input.quick_host_density_km2()),
                    change_method_settings={
                        "sampling_mode": str(benchmark_assumptions_selected()["sampling_mode"]),
                        "overlap_type": "cross_sectional_benchmark",
                        "baseline_confidence": float(input.baseline_confidence()),
                        "baseline_ci_width": float(input.baseline_ci_width()),
                        "baseline_planning_prevalence": float(input.baseline_planning_prevalence()),
                        "alpha": float(input.shared_alpha()),
                        "power": float(input.shared_power()),
                        "delta": float(input.change_method_delta()),
                    },
                    regression_settings={
                        "sampling_mode": str(benchmark_assumptions_selected()["sampling_mode"]),
                        "alpha": float(input.shared_alpha()),
                        "power": float(input.shared_power()),
                        "target": float(input.regression_target()),
                    },
                    benchmark_label=str(bundle.get("benchmark_label", "")),
                    analysis_assumptions=build_core_analysis_assumptions(
                        host_density_km2=float(bundle.get("host_density_km2", input.quick_host_density_km2())),
                        include_unmatched_survey_polygons=bool(bundle.get("include_unmatched_survey_polygons", True)),
                        regression_model=str(bundle.get("regression_model", input.quick_regression_model())),
                        status_column_name=str(bundle.get("status_column_name", input.quick_status_column())),
                        positive_statuses=list(bundle.get("positive_statuses", [])),
                        negative_statuses=list(bundle.get("negative_statuses", [])),
                        benchmark_label=str(bundle.get("benchmark_label", "")),
                    ),
                )
                with tmp_path.open("rb") as f:
                    yield f.read()
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        finally:
            ui.notification_remove(notif_id, session=session)


def compute_n_total_regression_method_hosts(
    pi0: float,
    piD: float,
    alpha: float,
    power: float,
    t_vec: list[int] | np.ndarray,
    population_size: int | float | None = None,
    phi: float = 1.0,
    initial_n: int | float = 0,
) -> dict[str, float]:
    """Compatibility helper backed by the frozen Regression Method calculation."""
    t_arr = np.asarray(t_vec, dtype=float)
    if t_arr.size == 0:
        t_arr = np.array([1.0], dtype=float)
    n_total = _backend_compute_n_total_regression_method_cells(
        pi0=float(pi0),
        pi_target=float(piD),
        alpha=float(alpha),
        power=float(power),
        t_vec=t_arr,
        population_size=population_size,
        phi=float(phi),
        initial_n=float(initial_n),
    )
    t_end = float(np.max(t_arr))
    beta_star = (_logit(_clamp_open01(float(piD))) - _logit(_clamp_open01(float(pi0)))) / max(t_end, 1e-8)
    return {"n_total": float(n_total), "beta_star": float(beta_star), "V": float("nan")}


def _estimate_design_effect_from_binary_cluster_sample(sampled_df: pd.DataFrame, planned_m: int | None = None) -> dict[str, float]:
    """Apply the frozen backend Design Effect estimator to reconstructed polygon data.

    The reconstruction is necessary because the real survey files contain polygons
    rather than host-level outcomes.  The estimator itself is not reimplemented here.
    """
    if sampled_df is None or sampled_df.empty:
        return _backend_estimate_design_effect(pd.DataFrame(), planned_m=planned_m)
    work = sampled_df.copy()
    if "cluster_id" not in work.columns and "cell_id" in work.columns:
        work = work.rename(columns={"cell_id": "cluster_id"})
    if "cluster_id" not in work.columns or "detected" not in work.columns:
        return _backend_estimate_design_effect(pd.DataFrame(), planned_m=planned_m)
    return _backend_estimate_design_effect(work[["cluster_id", "detected"]].copy(), planned_m=planned_m)


def _build_pilot_sampled_hosts_from_year(
    layers: dict[str, Any],
    host_gdf: Any | None,
    year: int,
    host_density_km2: float,
    within_cluster_cap: int,
    positive_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    negative_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    status_column_name: str | None = None,
    host_column_name: str | None = None,
    host_values: list[str] | tuple[str, ...] | set[str] | None = None,
    include_unmatched_survey_polygons: bool = True,
) -> pd.DataFrame:
    gdf_year, _ = filter_target_sites_for_year(layers, int(year))
    if gdf_year is None or len(gdf_year) == 0:
        return pd.DataFrame(columns=["cell_id", "detected"])
    host_landscape = build_real_host_landscape(
        host_gdf,
        layers,
        host_column_name=host_column_name,
        host_values=host_values,
        include_unmatched_survey_polygons=include_unmatched_survey_polygons,
    )
    cluster_alloc = allocate_surveys_to_host_clusters(
        host_landscape=host_landscape,
        survey_gdf=gdf_year,
        host_density_km2=float(host_density_km2),
        positive_status_values=positive_status_values,
        negative_status_values=negative_status_values,
        status_column_name=status_column_name,
    )
    if cluster_alloc.empty:
        return pd.DataFrame(columns=["cell_id", "detected"])
    rows: list[pd.DataFrame] = []
    for row in cluster_alloc.itertuples():
        cluster_size = min(max(1, int(within_cluster_cap)), max(0, int(round(float(row.surveyed_hosts_est)))))
        if cluster_size <= 0:
            continue
        pos_n = min(cluster_size, max(0, int(round(float(row.infected_hosts_est) / max(float(row.surveyed_hosts_est), 1.0) * cluster_size))))
        detected = np.array([1] * pos_n + [0] * (cluster_size - pos_n), dtype=int)
        rows.append(pd.DataFrame({"cell_id": [str(row.cluster_id)] * cluster_size, "detected": detected}))
    if not rows:
        return pd.DataFrame(columns=["cell_id", "detected"])
    return pd.concat(rows, ignore_index=True)


def _resolve_quick_results_phi(
    layers: dict[str, Any],
    host_gdf: Any | None,
    pilot_year: int,
    host_density_km2: float,
    sampling_mode: str,
    cluster_inflation_mode: str,
    within_cluster_cap: int,
    fixed_phi: float,
    positive_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    negative_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    status_column_name: str | None = None,
    host_column_name: str | None = None,
    host_values: list[str] | tuple[str, ...] | set[str] | None = None,
    include_unmatched_survey_polygons: bool = True,
) -> dict[str, float]:
    sampling_mode = str(sampling_mode)
    cluster_inflation_mode = str(cluster_inflation_mode)
    if sampling_mode == "srs" or cluster_inflation_mode == "none":
        return {"phi": 1.0, "rho": 0.0, "m": 1.0 if sampling_mode == "srs" else float(within_cluster_cap), "m_observed": 1.0, "estimator": "none"}
    if cluster_inflation_mode == "fixed":
        phi = max(1.0, float(fixed_phi))
        return {"phi": phi, "rho": np.nan, "m": float(within_cluster_cap), "m_observed": float(within_cluster_cap), "estimator": "fixed"}
    pilot_sampled = _build_pilot_sampled_hosts_from_year(
        layers=layers,
        host_gdf=host_gdf,
        year=int(pilot_year),
        host_density_km2=float(host_density_km2),
        within_cluster_cap=int(within_cluster_cap),
        positive_status_values=positive_status_values,
        negative_status_values=negative_status_values,
        status_column_name=status_column_name,
        host_column_name=host_column_name,
        host_values=host_values,
        include_unmatched_survey_polygons=include_unmatched_survey_polygons,
    )
    return _estimate_design_effect_from_binary_cluster_sample(pilot_sampled, planned_m=int(within_cluster_cap))


def build_change_method_effort_from_hosts(
    year_summary_df: pd.DataFrame,
    year_before: int,
    year_after: int,
    pilot_year: int,
    delta: float,
    corr: float,
    alpha: float,
    power: float,
    total_population_size: int | float | None = None,
    layers: dict[str, Any] | None = None,
    host_gdf: Any | None = None,
    host_density_km2: float = 2500.0,
    sampling_mode: str = "srs",
    cluster_inflation_mode: str = "none",
    within_cluster_cap: int = 5,
    fixed_phi: float = 1.0,
    positive_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    negative_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    status_column_name: str | None = None,
    host_column_name: str | None = None,
    host_values: list[str] | tuple[str, ...] | set[str] | None = None,
    include_unmatched_survey_polygons: bool = True,
) -> pd.DataFrame:
    """Compare historical effort with the Change Method equal-n requirement.

    The Change Method planning formula sizes the variance of the difference
    between two prevalence estimates under an equal-sample-size assumption.
    Once the baseline prevalence has been observed, the calculated requirement
    therefore applies to both the first and final compared surveys.
    """
    cols = ["year", "observed_hosts", "observed_prop_population", "required_hosts", "required_prop_population", "observed_prevalence"]
    if year_summary_df is None or year_summary_df.empty:
        return pd.DataFrame(columns=cols)
    work = year_summary_df.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    effort = work.loc[
        work["year"].isin([int(year_before), int(year_after)]),
        ["year", "surveyed_hosts_est", "estimated_prevalence"],
    ].copy()
    if effort.empty:
        return pd.DataFrame(columns=cols)
    effort = effort.rename(columns={"surveyed_hosts_est": "observed_hosts", "estimated_prevalence": "observed_prevalence"})
    total_pop = float(total_population_size) if total_population_size is not None and np.isfinite(total_population_size) and total_population_size > 0 else float("nan")
    phi_info = _resolve_quick_results_phi(
        layers=layers or {}, host_gdf=host_gdf, pilot_year=int(pilot_year),
        host_density_km2=float(host_density_km2), sampling_mode=str(sampling_mode),
        cluster_inflation_mode=str(cluster_inflation_mode), within_cluster_cap=int(within_cluster_cap),
        fixed_phi=float(fixed_phi), positive_status_values=positive_status_values,
        negative_status_values=negative_status_values, status_column_name=status_column_name,
        host_column_name=host_column_name, host_values=host_values,
        include_unmatched_survey_polygons=include_unmatched_survey_polygons,
    )
    before_row = effort.loc[effort["year"] == int(year_before)]
    required_per_survey = np.nan
    if not before_row.empty:
        p0 = pd.to_numeric(before_row["observed_prevalence"], errors="coerce").iloc[0]
        if np.isfinite(p0):
            required_per_survey = compute_n_change_method_cells(
                p0=float(p0), delta=float(delta), corr=float(corr), alpha=float(alpha), power=float(power),
                population_size=total_pop if np.isfinite(total_pop) else None,
                design_effect=float(phi_info["phi"]),
            )
    effort["required_hosts"] = np.nan
    if np.isfinite(required_per_survey):
        effort.loc[effort["year"].isin([int(year_before), int(year_after)]), "required_hosts"] = float(required_per_survey)
    effort["observed_prop_population"] = effort["observed_hosts"] / total_pop if np.isfinite(total_pop) and total_pop > 0 else np.nan
    effort["required_prop_population"] = effort["required_hosts"] / total_pop if np.isfinite(total_pop) and total_pop > 0 else np.nan
    return effort[cols]

def build_regression_effort_from_hosts(
    year_summary_df: pd.DataFrame,
    year_before: int,
    year_after: int,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    regression_method_target: float = DEFAULT_REGRESSION_METHOD_TARGET,
    baseline_confidence: float = DEFAULT_BASELINE_PREVALENCE_CONF,
    baseline_ci_width: float = DEFAULT_BASELINE_PREVALENCE_WIDTH,
    baseline_planning_prevalence: float = DEFAULT_BASELINE_PREVALENCE_UPPER,
    total_population_size: int | float | None = None,
    layers: dict[str, Any] | None = None,
    host_gdf: Any | None = None,
    host_density_km2: float = 2500.0,
    pilot_year: int | None = None,
    sampling_mode: str = "srs",
    cluster_inflation_mode: str = "none",
    within_cluster_cap: int = 5,
    fixed_phi: float = 1.0,
    positive_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    negative_status_values: list[str] | tuple[str, ...] | set[str] | None = None,
    status_column_name: str | None = None,
    host_column_name: str | None = None,
    host_values: list[str] | tuple[str, ...] | set[str] | None = None,
    include_unmatched_survey_polygons: bool = True,
) -> pd.DataFrame:
    """Compare historical effort with the Regression Method planning sequence.

    The original method uses two stages. First, the time-zero survey is sized
    for precision of the unknown initial prevalence. Second, the observed
    baseline prevalence and chosen design prevalence define a target negative
    slope; the required *total* follow-up sample is then divided equally across
    the scheduled annual follow-up rounds.
    """
    cols = ["year", "observed_hosts", "observed_prop_population", "required_hosts", "required_prop_population", "observed_prevalence"]
    if year_summary_df is None or year_summary_df.empty:
        return pd.DataFrame(columns=cols)

    raw = year_summary_df.copy()
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw = raw.loc[
        raw["year"].between(float(year_before), float(year_after), inclusive="both"),
        ["year", "surveyed_hosts_est", "estimated_prevalence"],
    ].copy()
    raw = raw.rename(columns={"surveyed_hosts_est": "observed_hosts", "estimated_prevalence": "observed_prevalence"})

    # Keep the full scheduled annual sequence. If a historical year is missing,
    # it appears as zero effort rather than being silently removed from planning.
    schedule = pd.DataFrame({"year": np.arange(int(year_before), int(year_after) + 1, dtype=int)})
    work = schedule.merge(raw, on="year", how="left").sort_values("year")
    work["observed_hosts"] = pd.to_numeric(work["observed_hosts"], errors="coerce").fillna(0.0)
    work["observed_prevalence"] = pd.to_numeric(work["observed_prevalence"], errors="coerce")

    total_pop = float(total_population_size) if total_population_size is not None and np.isfinite(total_population_size) and total_population_size > 0 else float("nan")
    baseline_year = int(year_before)
    p0_row = work.loc[work["year"] == baseline_year]
    if p0_row.empty or not np.isfinite(p0_row["observed_prevalence"].iloc[0]):
        return pd.DataFrame(columns=cols)
    p0_hat = float(p0_row["observed_prevalence"].iloc[0])
    design_pilot_year = int(pilot_year if pilot_year is not None else baseline_year)
    phi_info = _resolve_quick_results_phi(
        layers=layers or {}, host_gdf=host_gdf, pilot_year=design_pilot_year,
        host_density_km2=float(host_density_km2), sampling_mode=str(sampling_mode),
        cluster_inflation_mode=str(cluster_inflation_mode), within_cluster_cap=int(within_cluster_cap),
        fixed_phi=float(fixed_phi), positive_status_values=positive_status_values,
        negative_status_values=negative_status_values, status_column_name=status_column_name,
        host_column_name=host_column_name, host_values=host_values,
        include_unmatched_survey_polygons=include_unmatched_survey_polygons,
    )

    out = work.copy()
    out["required_hosts"] = np.nan
    out["observed_prop_population"] = out["observed_hosts"] / total_pop if np.isfinite(total_pop) and total_pop > 0 else np.nan

    baseline_required = compute_n_baseline_prevalence_cells(
        p0_upper=float(baseline_planning_prevalence),
        conf_level=float(baseline_confidence),
        width=float(baseline_ci_width),
        population_size=total_pop if np.isfinite(total_pop) else None,
    )
    out.loc[out["year"] == baseline_year, "required_hosts"] = float(baseline_required)

    followup_years = out.loc[out["year"] > baseline_year, "year"].astype(float)
    if not followup_years.empty:
        t_vec = followup_years.to_numpy(dtype=float) - float(baseline_year)
        try:
            total_followup = compute_n_total_regression_method_cells(
                pi0=p0_hat, pi_target=float(regression_method_target), alpha=float(alpha), power=float(power),
                t_vec=t_vec, population_size=total_pop if np.isfinite(total_pop) else None,
                phi=float(phi_info["phi"]), initial_n=float(baseline_required),
            )
        except ValueError:
            total_followup = np.nan
        if np.isfinite(total_followup):
            per_round = max(1, int(np.ceil(float(total_followup) / len(t_vec))))
            out.loc[out["year"].isin(followup_years.astype(int).tolist()), "required_hosts"] = float(per_round)

    out["required_prop_population"] = out["required_hosts"] / total_pop if np.isfinite(total_pop) and total_pop > 0 else np.nan
    return out[cols]


app = App(app_ui, server)
