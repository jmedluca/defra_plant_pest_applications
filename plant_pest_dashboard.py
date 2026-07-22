"""Main plant pest dashboard for real survey data.

This app is for analysing real polygon survey data:

1. loads survey polygons and host coverage polygons,
2. converts surveyed area into an estimated number of hosts sampled,
3. converts positive survey outcomes into estimated infected hosts,
4. estimates prevalence for each year,
5. fits the Change Method and Regression Method, and
6. writes a PDF report.
"""

from __future__ import annotations

import base64
from html import escape
import io
import json
from pathlib import Path
from matplotlib.patches import FancyBboxPatch
import sys
from statistics import NormalDist
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

def _logistic_irls(t: np.ndarray, x: np.ndarray, n: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones_like(t, dtype=float), t.astype(float)])
    y = x.astype(float)
    n = n.astype(float)

    p0 = np.clip(y / np.maximum(n, 1.0), 1e-6, 1.0 - 1e-6)
    eta = np.log(p0 / (1.0 - p0))
    beta = np.linalg.lstsq(X, eta, rcond=None)[0]

    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        w = n * p * (1.0 - p)
        z = eta + (y - n * p) / np.maximum(w, 1e-6)

        XtW = X.T * w
        XtWX = XtW @ X
        XtWz = XtW @ z
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    w = n * p * (1.0 - p)
    XtW = X.T * w
    XtWX = XtW @ X
    cov = np.linalg.pinv(XtWX)
    return beta, cov

# This is the main DEFRA dashboard app.
# It loads real survey polygons and host coverage polygons, turns them into
# prevalence estimates over time, fits the two methods, and creates the PDF.

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

try:
    from scipy.spatial import cKDTree
except Exception:  
    cKDTree = None

# ============================================================================
# Default Paths And Settings
# ============================================================================

DEFRA_DATA_DIR = APP_DIR / "DEFRA_data"
# Default data folders inside the packaged application.
# Users can change these from the Home page if their files are elsewhere.
TARGET_SITES_DIR = DEFRA_DATA_DIR / "Survey_data"
LARCH_HOST_DIR = DEFRA_DATA_DIR / "Host_coverage_data"
LARCH_HOST_PATH = LARCH_HOST_DIR / "larch_coverage.shp"
SPHN_DIR = DEFRA_DATA_DIR / "SPHN_data"
SPHN_PATH = SPHN_DIR / "P_ramorum_SPHN_extent.shp"
LARCH_AREA_CSV_PATH = LARCH_HOST_DIR / "larch_coverage_area.csv"
YEAR_MIN = 2017
YEAR_MAX = 2024
# These are the default statistical settings used unless the user changes them.
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_APPENDIX_C_DELTA = 0.03
DEFAULT_APPENDIX_C_CORR = 0.0
DEFAULT_APPENDIX_E_TARGET = 0.005
DEFAULT_APPENDIX_E_INITIAL_CONF = 0.95
DEFAULT_APPENDIX_E_INITIAL_WIDTH = 0.025
DEFAULT_APPENDIX_E_INITIAL_UPPER = 0.1
RETRO_BENCHMARK_CHOICES = {
    "optimistic": "Optimistic benchmark",
    "moderate": "Moderate clustering benchmark",
    "cautious": "Cautious clustering benchmark",
}
SIM_DEFAULT_TOTAL_HOSTS = 60000
SIM_DEFAULT_N_GROUPS = 10000
SIM_DEFAULT_WITHIN_CLUSTER_CAP = 5
SIM_DEFAULT_SEED = 123
SIMX_DEFAULT_CELL_AREA_KM2 = 1.0
SIMX_DEFAULT_HOST_DENSITY_KM2 = 2500.0
SIMX_DEFAULT_SURVEY_HOSTS_PER_ROUND = 600
SIMX_DEFAULT_PSU_BLOCK_SIDE = 4
SIMX_DEFAULT_BIASED_MULTIPLIER = 1.0
SIMX_DEFAULT_TOTAL_CELLS = 5000
try:
    # Pick a default host area for the simulator/dashboard examples.
    # If the cached area file is not present, try reading the host polygons.
    if LARCH_AREA_CSV_PATH.exists():
        SIMX_DEFAULT_TOTAL_AREA_KM2 = round(
            float(pd.to_numeric(pd.read_csv(LARCH_AREA_CSV_PATH)["Area"], errors="coerce").fillna(0.0).sum()) / 100.0
        )
    elif gpd is not None and LARCH_HOST_PATH.exists():
        _host_area_gdf = gpd.read_file(LARCH_HOST_PATH)
        _host_area_gdf = _host_area_gdf.to_crs(27700) if getattr(_host_area_gdf, "crs", None) else _host_area_gdf
        SIMX_DEFAULT_TOTAL_AREA_KM2 = round(float(pd.to_numeric(_host_area_gdf.geometry.area, errors="coerce").fillna(0.0).sum()) / 1_000_000.0)
    else:
        SIMX_DEFAULT_TOTAL_AREA_KM2 = 250.0
except Exception:
    SIMX_DEFAULT_TOTAL_AREA_KM2 = 250.0

SIM_CURVE_CHOICES = {
    "logit_linear": "Logit-linear",
    "linear": "Linear",
    "logistic": "Logistic",
}
SIM_HOST_CLUSTERING_CHOICES = {
    "diffuse": "Diffuse",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}
SIM_INFECTION_CLUSTERING_CHOICES = {
    "random": "Random",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}
SIM_SURVEY_TARGETING_CHOICES = {
    "random": "Random",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}
SIM_OVERLAP_CHOICES = {
    "cross_sectional": "Cross-sectional",
    "rotating_panel": "Rotating panel",
    "longitudinal": "Longitudinal",
}
SIM_METRIC_CHOICES = {
    "appendix_e_final": "Appendix E final prevalence",
    "appendix_c_final": "Appendix C final prevalence",
    "appendix_e_change": "Appendix E change",
    "appendix_c_change": "Appendix C change",
    "appendix_e_slope": "Appendix E slope",
}
SIM_VARY_FACTOR_CHOICES = {
    "curve_type": "Curve family",
    "p_end": "Final true prevalence",
    "prevalence_shift": "Prevalence shift",
    "host_clustering_level": "Host clustering",
    "infection_clustering_level": "Infection clustering",
    "survey_targeting_level": "Preferential surveying",
    "overlap_type": "Overlap type",
}
SIM_VARY_LEVELS: dict[str, list[Any]] = {
    "curve_type": ["logit_linear", "linear", "logistic"],
    "p_end": [0.0010, 0.0050, 0.0150, 0.0333, 0.0516, 0.0616, 0.0656],
    "prevalence_shift": [-0.004, 0.0, 0.004, 0.012, 0.022],
    "host_clustering_level": ["diffuse", "low", "medium", "high"],
    "infection_clustering_level": ["random", "low", "medium", "high"],
    "survey_targeting_level": ["random", "low", "medium", "high"],
    "overlap_type": ["cross_sectional", "rotating_panel", "longitudinal"],
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
        tiles="CartoDB positron",
        control_scale=True,
    )

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
    return map_obj.get_root().render()


# ============================================================================
# Data Loading And Real Survey Conversion
# ============================================================================
#
# This section is where raw GIS files are turned into the tables needed for
# prevalence estimation.  The key idea is:
#   survey polygon area x host density = estimated number of hosts sampled.


def _default_layer_paths() -> dict[str, Path]:
    # Default survey shapefiles bundled with the app.
    return {
        "2017_2020": TARGET_SITES_DIR / "Target_sites_2017_2020.shp",
        "2021_2024": TARGET_SITES_DIR / "Target_sites_2021_2024.shp",
    }


def _resolve_shapefile_path(path: Path, preferred_stem: str | None = None) -> Path:
    # Accept either a direct .shp file or a folder containing shapefiles.
    if path.is_file():
        return path
    if not path.exists():
        return path

    shp_files = sorted(path.glob("*.shp"))
    if preferred_stem is not None:
        for shp in shp_files:
            if shp.stem == preferred_stem:
                return shp
    return shp_files[0] if shp_files else path


def _resolve_shapefile_paths(path: Path, preferred_stems: list[str] | None = None) -> list[Path]:
    # Accept a folder containing several shapefiles and return them in a stable order.
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    shp_files = sorted(path.glob("*.shp"))
    if not shp_files:
        return []
    if preferred_stems:
        ordered: list[Path] = []
        used: set[Path] = set()
        for stem in preferred_stems:
            for shp in shp_files:
                if shp.stem == stem and shp not in used:
                    ordered.append(shp)
                    used.add(shp)
        for shp in shp_files:
            if shp not in used:
                ordered.append(shp)
        return ordered
    return shp_files


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
    # Load the survey polygon files.
    # These are the real field survey areas and their recorded outcomes.
    if isinstance(layer_paths, Path):
        shp_files = _resolve_shapefile_paths(layer_paths, preferred_stems=["Target_sites_2017_2020", "Target_sites_2021_2024"])
        layer_paths = {shp.stem: shp for shp in shp_files}
    info: dict[str, Any] = {
        "available_files": {name: path.exists() for name, path in layer_paths.items()},
        "layers": {},
        "messages": [],
    }

    if not _gis_available():
        info["messages"].append(
            "Geospatial packages are not installed in the shiny Python environment. "
            "Install geopandas/shapely/pyogrio/pyproj to enable polygon overlay."
        )
        return info

    for name, path in layer_paths.items():
        if not path.exists():
            info["messages"].append(f"Missing layer: {path}")
            continue
        gdf = gpd.read_file(path)
        info["layers"][name] = gdf

    return info


def _extract_survey_year_series(gdf: Any) -> pd.Series | None:
    # Try several possible year columns because different shapefiles may use
    # slightly different field names.
    if gdf is None or len(gdf) == 0:
        return None
    for col in ("Survey year", "Created_yr"):
        if col in gdf.columns:
            years = pd.to_numeric(gdf[col], errors="coerce")
            if years.notna().any():
                return years.astype("Int64")
    if "Survey_yea" in gdf.columns:
        series = gdf["Survey_yea"].astype(str).str.extract(r"(\d{4})", expand=False)
        years = pd.to_numeric(series, errors="coerce")
        if years.notna().any():
            return years.astype("Int64")
    years = _extract_year_from_columns(gdf, ["SurveyDate", "Survey_Date", "Date", "Created", "CreatedAt"])
    if years is not None and years.notna().any():
        return years.astype("Int64")
    return None


def extract_available_survey_years(layers: dict[str, Any]) -> list[int]:
    # Get the year range for the Data Viewer slider from the loaded survey files.
    years: set[int] = set()
    for gdf in (layers or {}).values():
        year_series = _extract_survey_year_series(gdf)
        if year_series is None:
            continue
        for val in pd.to_numeric(year_series, errors="coerce").dropna().astype(int).tolist():
            years.add(int(val))
    return sorted(years)


def load_host_layer(host_path: Path) -> dict[str, Any]:
    # Load the host coverage polygons used as the prevalence denominator.
    host_path = _resolve_shapefile_path(host_path, preferred_stem=LARCH_HOST_PATH.stem)
    info: dict[str, Any] = {
        "available_file": host_path.exists(),
        "layer": None,
        "messages": [],
    }

    if not _gis_available():
        info["messages"].append(
            "Geospatial packages are not installed in the shiny Python environment. "
            "Install geopandas/shapely/pyogrio/pyproj to enable host-layer overlay."
        )
        return info

    if not host_path.exists():
        info["messages"].append(f"Missing host layer: {host_path}")
        return info

    try:
        info["layer"] = gpd.read_file(host_path)
    except Exception as exc:
        info["messages"].append(f"Failed to load host layer: {exc}")

    return info


def load_sphn_layer(sphn_path: Path) -> dict[str, Any]:
    # Load SPHN polygons for map context if the file is available.
    sphn_path = _resolve_shapefile_path(sphn_path, preferred_stem=SPHN_PATH.stem)
    info: dict[str, Any] = {
        "available_file": sphn_path.exists(),
        "layer": None,
        "messages": [],
    }

    if not _gis_available():
        info["messages"].append(
            "Geospatial packages are not installed in the shiny Python environment. "
            "Install geopandas/shapely/pyogrio/pyproj to enable SPHN overlay."
        )
        return info

    if not sphn_path.exists():
        info["messages"].append(f"Missing SPHN layer: {sphn_path}")
        return info

    try:
        info["layer"] = gpd.read_file(sphn_path)
    except Exception as exc:
        info["messages"].append(f"Failed to load SPHN layer: {exc}")

    return info


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
        ["File_creat", "Compliance", "Remain_in_", "Flight_dat", "Woodland_o"],
    )
    if year_series is None:
        messages.append("SPHN layer has no recognised year or date field.")
        return None, messages

    df = df.loc[year_series == int(year)].copy()
    if df.empty:
        return None, messages
    return df, messages


def build_real_host_landscape(
    host_gdf: Any | None,
    survey_layers: dict[str, Any] | None = None,
    host_column_name: str | None = None,
    host_values: list[str] | tuple[str, ...] | set[str] | None = None,
    include_unmatched_survey_polygons: bool = True,
):
    # Build the real host landscape from polygon data.
    # Host polygons are the main denominator. Survey-only polygons can also be
    # added if the host coverage file is incomplete.
    if not _gis_available():
        return None

    host_proj = _to_bng(host_gdf)
    host_frames: list[Any] = []
    if host_proj is not None and len(host_proj) > 0:
        # Start with the host coverage polygons, such as the Larch coverage file.
        host_work = host_proj.copy()
        host_col = str(host_column_name or "").strip()
        if host_col and host_col != "__all__" and host_col in host_work.columns:
            # Optional filter: use only host polygons whose selected column value
            # matches the user selected values.
            selected_values = {_normalize_status_text(x) for x in (host_values or []) if _normalize_status_text(x)}
            if selected_values:
                keep_mask = host_work[host_col].fillna("").astype(str).map(_normalize_status_text).isin(selected_values)
                host_work = host_work.loc[keep_mask].copy()
        host_work = host_work.loc[host_work.geometry.notna()].copy()
        host_work = host_work.loc[~host_work.geometry.is_empty].copy()
        host_work = host_work.reset_index(drop=True)
        host_work = host_work[["geometry"]].copy()
        host_work["cluster_id"] = [f"host_{i}" for i in range(len(host_work))]
        host_work["cluster_source"] = "host_polygon"
        host_frames.append(host_work)

    survey_frames: list[Any] = []
    for gdf in (survey_layers or {}).values():
        # Collect all survey polygons so we can identify survey areas that are
        # outside the host coverage file.
        if gdf is None or len(gdf) == 0:
            continue
        survey_proj = _to_bng(gdf)
        if survey_proj is not None and len(survey_proj) > 0:
            work = survey_proj[["geometry"]].copy()
            work = work.loc[work.geometry.notna()].copy()
            work = work.loc[~work.geometry.is_empty].copy()
            if len(work) > 0:
                survey_frames.append(work)

    uncovered = None
    if include_unmatched_survey_polygons and survey_frames:
        # Some DEFRA survey polygons are not inside the host coverage polygons.
        # To avoid throwing those surveys away, we can treat those survey only
        # polygons as extra host clusters.
        survey_all = gpd.GeoDataFrame(pd.concat(survey_frames, ignore_index=True), geometry="geometry", crs=survey_frames[0].crs)
        if host_frames:
            try:
                host_union = host_frames[0].geometry.union_all()
            except Exception:
                host_union = host_frames[0].geometry.unary_union
            uncovered = survey_all.loc[~survey_all.geometry.intersects(host_union)].copy()
        else:
            uncovered = survey_all.copy()
        if uncovered is not None and len(uncovered) > 0:
            uncovered = uncovered.reset_index(drop=True)
            uncovered["geom_key"] = uncovered.geometry.apply(lambda geom: geom.wkb_hex if geom is not None else None)
            uncovered = uncovered.drop_duplicates(subset=["geom_key"]).drop(columns=["geom_key"])
            uncovered["cluster_id"] = [f"survey_only_{i}" for i in range(len(uncovered))]
            uncovered["cluster_source"] = "survey_only_polygon"
            host_frames.append(uncovered[["geometry", "cluster_id", "cluster_source"]].copy())

    if not host_frames:
        return None

    combined = gpd.GeoDataFrame(pd.concat(host_frames, ignore_index=True), geometry="geometry", crs=host_frames[0].crs)
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
    # Turn messy survey polygons into one summary row per host polygon.
    # We assume a fixed number of hosts per km2, so surveyed area becomes an
    # estimated number of hosts sampled.
    cols = [
        "cluster_id",
        "cluster_source",
        "cluster_area_km2",
        "cluster_host_count",
        "survey_polygons",
        "surveyed_area_km2",
        "infected_polygons",
        "infected_area_km2",
        "surveyed_hosts_est",
        "infected_hosts_est",
        "estimated_prevalence",
    ]
    if not _gis_available() or host_landscape is None or len(host_landscape) == 0 or survey_gdf is None or len(survey_gdf) == 0:
        return pd.DataFrame(columns=cols)

    clusters = _to_bng(host_landscape).copy()
    surveys = _to_bng(survey_gdf).copy()
    if clusters is None or surveys is None or len(clusters) == 0 or len(surveys) == 0:
        return pd.DataFrame(columns=cols)

    density = max(float(host_density_km2), 0.0)
    clusters = clusters.reset_index(drop=True)
    # Estimate how many hosts each host polygon contains from its area.
    clusters["cluster_area_km2"] = pd.to_numeric(clusters.geometry.area, errors="coerce").fillna(0.0) / 1_000_000.0
    clusters["cluster_host_count"] = np.maximum(1.0, clusters["cluster_area_km2"] * density)
    clusters["_cluster_index"] = np.arange(len(clusters), dtype=int)

    surveys = surveys.reset_index(drop=True)
    surveys["_survey_index"] = np.arange(len(surveys), dtype=int)
    status_col = _status_column(surveys, preferred=status_column_name)
    if status_col is not None:
        # Keep only survey outcomes that the user has said mean positive or negative.
        # Anything else is ignored because it does not clearly say disease present/absent.
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

    cluster_join = gpd.sjoin(
        # Spatial join, which host cluster does each survey polygon touch.
        surveys[["_survey_index", "positive_flag", "geometry"]].copy(),
        clusters[["_cluster_index", "cluster_id", "cluster_source", "cluster_area_km2", "cluster_host_count", "geometry"]].copy(),
        how="left",
        predicate="intersects",
    )
    if cluster_join.empty:
        return pd.DataFrame(columns=cols)

    # If one survey polygon touches more than one host polygon, assign it to the
    # host polygon it overlaps most. This avoids double counting the same survey.
    cluster_join = cluster_join.merge(
        clusters[["_cluster_index", "geometry"]].rename(columns={"geometry": "cluster_geometry"}),
        on="_cluster_index",
        how="left",
    )
    cluster_join["intersect_area_km2"] = cluster_join.apply(
        lambda row: float(row.geometry.intersection(row.cluster_geometry).area) / 1_000_000.0
        if row.geometry is not None and row.cluster_geometry is not None
        else 0.0,
        axis=1,
    )
    cluster_join = cluster_join.sort_values(["_survey_index", "intersect_area_km2"], ascending=[True, False])
    cluster_join = cluster_join.drop_duplicates(subset=["_survey_index"], keep="first").copy()
    # Positive survey area is converted into an estimated number of infected hosts.
    cluster_join["surveyed_area_km2"] = pd.to_numeric(cluster_join.geometry.area, errors="coerce").fillna(0.0) / 1_000_000.0
    cluster_join["infected_area_km2"] = np.where(cluster_join["positive_flag"], cluster_join["surveyed_area_km2"], 0.0)
    cluster_join["surveyed_hosts_raw"] = cluster_join["surveyed_area_km2"] * density
    cluster_join["infected_hosts_raw"] = cluster_join["infected_area_km2"] * density

    out = (
        # After every survey polygon has been assigned to one host cluster, sum
        # the results by host cluster.
        cluster_join.groupby("cluster_id", as_index=False)
        .agg(
            cluster_source=("cluster_source", "first"),
            cluster_area_km2=("cluster_area_km2", "first"),
            cluster_host_count=("cluster_host_count", "first"),
            survey_polygons=("_survey_index", "nunique"),
            surveyed_area_km2=("surveyed_area_km2", "sum"),
            infected_polygons=("positive_flag", "sum"),
            infected_area_km2=("infected_area_km2", "sum"),
            surveyed_hosts_raw=("surveyed_hosts_raw", "sum"),
            infected_hosts_raw=("infected_hosts_raw", "sum"),
        )
        .copy()
    )
    # Do not allow the estimated surveyed/infected hosts to exceed the host count
    # in that cluster. This keeps the conversion physically sensible.
    out["surveyed_hosts_est"] = np.minimum(out["surveyed_hosts_raw"], out["cluster_host_count"])
    out["infected_hosts_est"] = np.minimum(out["infected_hosts_raw"], out["surveyed_hosts_est"])
    out["estimated_prevalence"] = np.where(out["surveyed_hosts_est"] > 0, out["infected_hosts_est"] / out["surveyed_hosts_est"], np.nan)
    return out[cols]


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
    # Find the survey result/status column.  The preferred column is used first,
    # otherwise the app tries common names.
    if gdf is None:
        return None
    if preferred is not None:
        pref = str(preferred).strip()
        if pref and pref != "__auto__" and pref in gdf.columns:
            return pref
    for name in ("Site status", "Site status ", "Site_statu"):
        if name in gdf.columns:
            return name
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
    # Default guess for survey outcomes that mean disease is present.
    # Users can change this in the Results settings.
    vals = series.fillna("").astype(str).str.strip().str.lower()
    positive_tokens = [
        "clear evidence",
        "confirmed infected",
        "assumed infected",
    ]
    return vals.apply(lambda x: any(tok in x for tok in positive_tokens))


def _default_negative_status_mask(series: pd.Series) -> pd.Series:
    # Default guess for survey outcomes that mean disease is absent.
    # Ambiguous statuses, such as "awaiting site visit", are not counted as negative.
    vals = series.fillna("").astype(str).str.strip().str.lower()
    negative_tokens = [
        "no evidence",
        "no_evidence",
    ]
    return vals.apply(lambda x: any(tok in x for tok in negative_tokens))


def _retrospective_benchmark_assumptions(mode: str) -> dict[str, Any]:
    mode = str(mode)
    mapping = {
        "optimistic": {
            "label": "Optimistic benchmark",
            "fixed_phi": 1.0,
            "sampling_mode": "srs",
            "within_cluster_cap": 5,
        },
        "moderate": {
            "label": "Moderate clustering benchmark",
            "fixed_phi": 1.5,
            "sampling_mode": "multistage",
            "within_cluster_cap": 5,
        },
        "cautious": {
            "label": "Cautious clustering benchmark",
            "fixed_phi": 2.5,
            "sampling_mode": "multistage",
            "within_cluster_cap": 5,
        },
    }
    return mapping.get(mode, mapping["moderate"]).copy()


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
        positive_flag = _default_positive_status_mask(series)
        include_flag = pd.Series(True, index=series.index)
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


def compute_n_appendix_c_cells(
    p0: float,
    delta: float = DEFAULT_APPENDIX_C_DELTA,
    corr: float = DEFAULT_APPENDIX_C_CORR,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    population_size: int | float | None = None,
    design_effect: float = 1.0,
) -> int:
    # Change Method sample-size formula.
    # It estimates how many hosts are needed in the first and final surveys.
    # alpha controls how strict the evidence threshold is, and power controls
    # how likely the method is to detect the chosen change if it is truly there.
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_beta = NormalDist().inv_cdf(power)
    p0 = _clamp_open01(float(p0))
    n = (2.0 * p0 * (1.0 - p0) * (1.0 - float(corr)) * (z_alpha + z_beta) ** 2) / (float(delta) ** 2)
    n = float(design_effect) * n
    return _apply_fpc(n, population_size)


def compute_n_initial_prev_ci_cells(
    p0_upper: float = DEFAULT_APPENDIX_E_INITIAL_UPPER,
    conf_level: float = DEFAULT_APPENDIX_E_INITIAL_CONF,
    width: float = DEFAULT_APPENDIX_E_INITIAL_WIDTH,
    population_size: int | float | None = None,
) -> int:
    # Pilot sample size used before the Regression Method calculation.
    # It gives a first estimate of prevalence.
    # The pilot is needed because the Regression Method formula needs a design
    # prevalence before the main survey sizes can be calculated.
    prop = _clamp_open01(float(p0_upper) / 2.0)
    alpha = 1.0 - float(conf_level)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    half_width = float(width) / 2.0
    if half_width <= 0:
        return 1
    n = (z**2 * prop * (1.0 - prop)) / (half_width**2)
    return _apply_fpc(n, population_size)


def compute_n_total_appendix_e_cells(
    pi0: float,
    pi_target: float,
    alpha: float,
    power: float,
    t_vec: np.ndarray,
    population_size: int | float | None = None,
    phi: float = 1.0,
) -> int:
    # Regression Method sample-size formula.
    # It estimates the total sample size needed to detect a prevalence trend.
    # It uses the starting prevalence, target prevalence, survey times, alpha,
    # power, and any clustering inflation factor.
    pi0 = _clamp_open01(float(pi0))
    pi_target = _clamp_open01(float(pi_target))
    t_vec = np.asarray(t_vec, dtype=float)
    if t_vec.size < 2:
        t_vec = np.array([0.0, 1.0], dtype=float)

    t_end = float(np.max(t_vec))
    beta_star = (_logit(pi_target) - _logit(pi0)) / max(t_end, 1e-8)
    # beta_star is the slope the study is designed to detect.
    eta = _logit(pi0) + beta_star * t_vec
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    w = p * (1.0 - p)
    X = np.column_stack([np.ones_like(t_vec), t_vec])
    # The information matrix tells us how much information the survey schedule
    # gives about the slope.
    info = X.T @ np.diag(w) @ X + np.eye(2) * 1e-8
    try:
        v_per_obs = float(np.linalg.inv(info)[1, 1])
    except np.linalg.LinAlgError:
        return _apply_fpc(float(population_size) if population_size else 1.0, population_size)

    z_alpha = NormalDist().inv_cdf(1.0 - alpha)
    z_beta = NormalDist().inv_cdf(power)
    beta_abs = abs(beta_star)
    if not np.isfinite(beta_abs) or beta_abs < 1e-10 or not np.isfinite(v_per_obs) or v_per_obs <= 0:
        raw = float(population_size) if population_size and np.isfinite(population_size) else 1.0
        return _apply_fpc(raw, population_size)

    n_raw = ((z_alpha + z_beta) / beta_abs) ** 2 * v_per_obs
    return _apply_fpc(float(phi) * float(np.ceil(n_raw)), population_size)


# ============================================================================
# Results Tables, Trend Fitting, And Spatial Diagnostics
# ============================================================================


def run_appendix_e_real(prevalence_ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_appendix_e_real_with_model(prevalence_ts, model_form="logistic")


def run_appendix_e_real_with_model(prevalence_ts: pd.DataFrame, model_form: str = "logistic") -> tuple[pd.DataFrame, pd.DataFrame]:
    # Fit the Regression Method to the yearly prevalence estimates.
    # The input table has one row per year: number sampled, number positive,
    # and estimated prevalence.
    if prevalence_ts.empty or len(prevalence_ts) < 2:
        return (
            pd.DataFrame([{"metric": "Appendix E note", "value": "At least two yearly observations are required"}]),
            prevalence_ts.copy(),
        )

    fit = prevalence_ts.copy()
    # Use years since the first survey year as the time variable.
    t = fit["year"].to_numpy(dtype=float) - float(fit["year"].min())
    model_form = str(model_form)
    if model_form == "logistic":
        model_x = t
    elif model_form == "fp2":
        model_x = (t + 1.0) ** 2
    else:
        model_x = (t + 1.0) ** 3
    x = fit["x"].to_numpy(dtype=float)
    n = fit["n"].to_numpy(dtype=float)
    beta, cov = _logistic_irls(model_x, x, n)
    # Convert the fitted model back into prevalence values between 0 and 1.
    eta = beta[0] + beta[1] * model_x
    fit["model_pi_hat"] = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    z = 1.96
    vec = np.column_stack([np.ones_like(model_x), model_x])
    var_eta = np.einsum("ij,jk,ik->i", vec, cov, vec)
    se_eta = np.sqrt(np.clip(var_eta, 0.0, None))
    fit["model_ci_low"] = 1.0 / (1.0 + np.exp(-np.clip(eta - z * se_eta, -30.0, 30.0)))
    fit["model_ci_high"] = 1.0 / (1.0 + np.exp(-np.clip(eta + z * se_eta, -30.0, 30.0)))

    slope = float(beta[1])
    slope_se = float(np.sqrt(max(cov[1, 1], 1e-12)))
    slope_low = slope - z * slope_se
    slope_high = slope + z * slope_se
    declining = bool(slope < 0)
    delta = float(fit["model_pi_hat"].iloc[-1] - fit["model_pi_hat"].iloc[0])

    summary = pd.DataFrame(
        [
            {"metric": "Model form", "value": model_form},
            {"metric": "Appendix E yearly points", "value": int(len(fit))},
            {"metric": "Modelled prevalence start", "value": round(float(fit["model_pi_hat"].iloc[0]), 4)},
            {"metric": "Modelled prevalence end", "value": round(float(fit["model_pi_hat"].iloc[-1]), 4)},
            {"metric": "Modelled change over timeframe", "value": round(delta, 4)},
            {"metric": "Trend slope (logit scale)", "value": round(slope, 4)},
            {"metric": "Trend slope 95% CI low", "value": round(float(slope_low), 4)},
            {"metric": "Trend slope 95% CI high", "value": round(float(slope_high), 4)},
            {"metric": "Declining trend", "value": declining},
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
    for year in range(year_min, year_max + 1):
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
                "survey_polygons": int(len(gdf_year)),
                "surveyed_area_km2": surveyed_area_km2,
                "infected_polygons": int(cluster_alloc["infected_polygons"].sum()) if not cluster_alloc.empty else 0,
                "infected_area_km2": infected_area_km2,
                "surveyed_hosts_est": surveyed_hosts_est,
                "infected_hosts_est": infected_hosts_est,
                "estimated_prevalence": prevalence,
            }
        )
    return pd.DataFrame(rows)


def plot_quick_results_trend(prevalence_ts: pd.DataFrame, appendix_e_fit: pd.DataFrame, year_before: int, year_after: int):
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    if prevalence_ts.empty:
        ax.set_title("Estimated prevalence over time")
        ax.text(0.5, 0.5, "Generate results to view prevalence estimates.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    ax.plot(
        prevalence_ts["year"],
        prevalence_ts["pi_hat"],
        color="#d95f02",
        marker="o",
        linewidth=0,
        label="Yearly survey estimates",
    )
    if "model_pi_hat" in appendix_e_fit.columns:
        ax.plot(appendix_e_fit["year"], appendix_e_fit["model_pi_hat"], color="#2b8cbe", linewidth=2, label="Regression Method")

    before_row = prevalence_ts.loc[prevalence_ts["year"] == year_before]
    after_row = prevalence_ts.loc[prevalence_ts["year"] == year_after]
    if not before_row.empty and not after_row.empty:
        ax.plot(
            [year_before, year_after],
            [float(before_row["pi_hat"].iloc[0]), float(after_row["pi_hat"].iloc[0])],
            color="#7b3294",
            linewidth=2,
            linestyle="--",
            marker="s",
            label="Change Method",
        )

    ymax_source = prevalence_ts["ci_high"] if "ci_high" in prevalence_ts.columns else pd.Series(dtype=float)
    ymax = float(np.nanmax(ymax_source)) if len(ymax_source) > 0 and np.isfinite(ymax_source).any() else float(np.nanmax(prevalence_ts["pi_hat"]))
    ax.set_title("Estimated prevalence over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Prevalence")
    ax.set_ylim(0, min(1.0, max(0.05, ymax * 1.15)))
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
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
# This section turns the numerical outputs into a PDF.  The first page gives the main conclusion, headline figures, and
# caveats.  Later pages show supporting plots and tables.


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


def _overall_trend_label(simple_change: Any, regression_change: Any) -> str:
    simple_dir = _direction_label(simple_change)
    regression_dir = _direction_label(regression_change)
    if simple_dir == "increase" and regression_dir == "increase":
        return "Increasing"
    if simple_dir == "decrease" and regression_dir == "decrease":
        return "Decreasing"
    if simple_dir == "no_clear_change" and regression_dir == "no_clear_change":
        return "Stable"
    if simple_dir is None and regression_dir is None:
        return "Unclear"
    return "Mixed"


def _method_agreement_label(simple_change: Any, regression_change: Any) -> str:
    simple_dir = _direction_label(simple_change)
    regression_dir = _direction_label(regression_change)
    if simple_dir is None or regression_dir is None:
        return "insufficient evidence"
    if simple_dir == regression_dir:
        return "method agreement"
    return "method disagreement"


def _latest_reported_prevalence(prevalence_ts_df: pd.DataFrame, regression_fit_df: pd.DataFrame) -> tuple[int | None, float | None]:
    if regression_fit_df is not None and not regression_fit_df.empty and "model_pi_hat" in regression_fit_df.columns:
        row = regression_fit_df.iloc[-1]
        return int(row["year"]), float(row["model_pi_hat"])
    if prevalence_ts_df is not None and not prevalence_ts_df.empty:
        row = prevalence_ts_df.iloc[-1]
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
    # Convert the fitted yearly change into a plain-English forecast sentence.
    # This is a simple straight-line projection, not a new epidemic model.
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
        return "At the current trend, prevalence appears broadly stable, so there is no clear projected milestone year."
    if change_per_year < 0:
        if prev_now <= target:
            return f"On the current trend, prevalence is already below the design prevalence of {_format_percent(target)}."
        years_to_target = (prev_now - target) / abs(change_per_year)
        if not np.isfinite(years_to_target) or years_to_target < 0:
            return None
        projected_year = int(np.ceil(year_now + years_to_target))
        return f"On the current trend, prevalence is expected to fall below the design prevalence of {_format_percent(target)} around {projected_year}."
    if prev_now >= 1.0:
        return "On the current trend, prevalence is already at or above 100%."
    years_to_full = (1.0 - prev_now) / change_per_year
    if not np.isfinite(years_to_full) or years_to_full < 0:
        return None
    projected_year = int(np.ceil(year_now + years_to_full))
    return f"On the current trend, prevalence would reach 100% around {projected_year} if that trajectory continued."


def build_results_report_caveats(
    effort_df: pd.DataFrame,
    spatial_diagnostics_df: pd.DataFrame,
    regression_change: Any,
    simple_change: Any,
    host_density_km2: float,
) -> list[str]:
    # Build the "points to keep in mind" section for the PDF.
    # These are practical warnings about sampling effort, survey bias, and assumptions.
    caveats: list[str] = []
    if effort_df is not None and not effort_df.empty:
        # Check whether observed survey effort was below the indicative benchmark.
        effort = effort_df.copy()
        for col in ("observed_hosts", "simple_required_hosts", "regression_required_hosts"):
            effort[col] = pd.to_numeric(effort[col], errors="coerce")
        simple_shortfall = effort.loc[
            effort["simple_required_hosts"].notna() & effort["observed_hosts"].notna() & (effort["observed_hosts"] < effort["simple_required_hosts"]),
            "year",
        ].astype(int).tolist()
        regression_shortfall = effort.loc[
            effort["regression_required_hosts"].notna() & effort["observed_hosts"].notna() & (effort["observed_hosts"] < effort["regression_required_hosts"]),
            "year",
        ].astype(int).tolist()
        if simple_shortfall:
            caveats.append(
                f"Observed survey effort was below the indicative Change Method benchmark in {', '.join(str(y) for y in simple_shortfall)}."
            )
        if regression_shortfall:
            caveats.append(
                f"Observed survey effort was below the indicative Regression Method benchmark in {', '.join(str(y) for y in regression_shortfall)}."
            )

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

    if _method_agreement_label(simple_change, regression_change) == "method disagreement":
        caveats.append(
            "The two methods do not tell exactly the same story, so the overall trend should be interpreted cautiously."
        )

    caveats.append(
        f"Estimated host numbers were derived from mapped area using an assumed host density of {_format_report_value(host_density_km2, digits=1)} hosts per km^2."
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
    simple_change = _method_metric_value(method_df, "Simple Change Method", "Estimated change")
    regression_change = _method_metric_value(method_df, "Regression Method", "Estimated change")
    # Compare the direction of change estimated by the two methods.
    simple_dir = _direction_label(simple_change)
    regression_dir = _direction_label(regression_change)

    if simple_dir == "decrease" and regression_dir == "decrease":
        lines.append(f"Disease prevalence has decreased between {before_year} and {after_year}, and both methods indicate a decline.")
    elif simple_dir == "increase" and regression_dir == "increase":
        lines.append(f"Disease prevalence has increased between {before_year} and {after_year}, and both methods indicate an increase.")
    elif simple_dir == "no_clear_change" and regression_dir == "no_clear_change":
        lines.append(f"The results do not show a clear overall change in disease prevalence between {before_year} and {after_year}.")
    elif simple_dir is None and regression_dir is None:
        lines.append("The current results are not sufficient to state an overall trend.")
    elif simple_dir == regression_dir and simple_dir is not None:
        lines.append(f"Both methods point to the same overall direction of change between {before_year} and {after_year}.")
    else:
        lines.append(f"The two methods do not fully agree on the direction of change between {before_year} and {after_year}.")

    if simple_dir is not None:
        lines.append(f"Estimated change from the Change Method: {_format_percentage_points(simple_change)}.")
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
    target_prevalence: float = DEFAULT_APPENDIX_E_TARGET,
):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    simple_change = _method_metric_value(method_df, "Simple Change Method", "Estimated change")
    simple_final = _method_metric_value(method_df, "Simple Change Method", "Final prevalence")
    regression_change = _method_metric_value(method_df, "Regression Method", "Estimated change")
    regression_final = _method_metric_value(method_df, "Regression Method", "Final prevalence")
    trend_label = _overall_trend_label(simple_change, regression_change)
    agreement_label = _method_agreement_label(simple_change, regression_change)
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
        simple_change=simple_change,
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
        _format_percentage_points(simple_change),
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
        _format_percent(simple_final),
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
    headers = ["Year", "Survey polygons", "Positive polygons", "Estimated hosts surveyed", "Estimated prevalence"]
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
    headers = ["Year", "Observed hosts", "Observed prevalence", "Change benchmark hosts", "Regression benchmark hosts"]
    if effort_df is None or effort_df.empty:
        return _html_report_table(headers, [], "No sampling-effort summary is available.")
    rows: list[list[Any]] = []
    for row in effort_df.itertuples(index=False):
        values = row._asdict()

        def host_count(name: str) -> str:
            value = values.get(name)
            return "" if value is None or pd.isna(value) else f"{int(round(float(value))):,}"

        rows.append(
            [
                int(values.get("year")) if pd.notna(values.get("year")) else "",
                host_count("observed_hosts"),
                _format_percent(values.get("observed_prevalence")),
                host_count("simple_required_hosts"),
                host_count("regression_required_hosts"),
            ]
        )
    return _html_report_table(
        headers,
        rows,
        "No sampling-effort summary is available.",
        widths=[9, 18, 19, 25, 29],
        extra_class="effort-table",
    )


def _build_report_overview_lines(
    method_df: pd.DataFrame,
    prevalence_ts_df: pd.DataFrame,
    regression_fit_df: pd.DataFrame,
    before_year: int,
    after_year: int,
    survey_years: int,
    survey_polygons: int,
    surveyed_hosts: float,
    target_prevalence: float,
) -> tuple[list[str], str]:
    # Create the short overview bullets on the first page of the PDF.
    # These lines are designed to be understood before reading any tables.
    simple_change = _method_metric_value(method_df, "Simple Change Method", "Estimated change")
    regression_change = _method_metric_value(method_df, "Regression Method", "Estimated change")
    trend_label = _overall_trend_label(simple_change, regression_change)
    latest_year, latest_prev = _latest_reported_prevalence(prevalence_ts_df, regression_fit_df)
    annual_change = _trend_change_per_year(regression_fit_df)
    trend_projection = _trend_projection_sentence(
        latest_year=latest_year,
        latest_prev=latest_prev,
        annual_change=annual_change,
        target_prevalence=target_prevalence,
    )
    overview_lines: list[str] = [
        f"Prevalence was {trend_label.lower()} between {before_year} and {after_year}.",
        "Both estimation methods point in the same overall direction." if _method_agreement_label(simple_change, regression_change) == "method agreement"
        else "The two estimation methods do not fully agree, so the overall trend should be treated with caution.",
    ]
    if latest_year is not None and latest_prev is not None:
        overview_lines.append(f"Estimated prevalence reached {_format_percent(latest_prev)} in {latest_year}.")
    if trend_projection:
        overview_lines.append(trend_projection)
    metadata = ""
    if survey_years > 0:
        metadata = (
            f"Dataset used: {survey_years} survey years, {survey_polygons:,} survey polygons, "
            f"and an estimated {_format_report_value(surveyed_hosts, digits=1)} surveyed hosts."
        )
    return overview_lines, metadata


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
) -> str:
    # Build the final HTML document that is converted into a PDF.
    # HTML/CSS is used because it gives more reliable layout than drawing every
    # item manually with matplotlib.
    simple_change = _method_metric_value(method_df, "Simple Change Method", "Estimated change")
    simple_final = _method_metric_value(method_df, "Simple Change Method", "Final prevalence")
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
        before_year=before_year,
        after_year=after_year,
        survey_years=survey_years,
        survey_polygons=survey_polygons,
        surveyed_hosts=surveyed_hosts,
        target_prevalence=target_prevalence,
    )
    caveats = build_results_report_caveats(
        effort_df=effort_df,
        spatial_diagnostics_df=spatial_diagnostics_df,
        regression_change=regression_change,
        simple_change=simple_change,
        host_density_km2=host_density_km2,
    )
    coverage_prop = _spatial_metric_value(spatial_diagnostics_df, "Host polygon coverage proportion")
    footer_items = []
    try:
        if coverage_prop is not None and np.isfinite(float(coverage_prop)):
            footer_items.append(f"Survey coverage: {_format_percent(coverage_prop)} of host polygons")
    except Exception:
        pass
    footer_items.append(f"Host density assumption: {_format_report_value(host_density_km2, digits=1)} hosts per km^2")
    text_load = sum(len(str(item)) for item in overview_lines + caveats) + len(overview_meta)
    # Use a compact layout if there is a lot of text to fit on the first page.
    layout_class = "compact" if len(caveats) > 5 or text_load > 1350 else "standard"
    yearly_table_html = _yearly_report_table_html(year_summary_df)
    effort_table_html = _effort_report_table_html(effort_df)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    @page {{
      size: A4;
      margin: 13mm 15mm 12mm 15mm;
    }}
    body {{
      font-family: Arial, Helvetica, sans-serif;
      color: #22332d;
      margin: 0;
      font-size: 10.8pt;
      line-height: 1.35;
    }}
    .page {{
      width: 100%;
      box-sizing: border-box;
    }}
    .hero {{
      background: #eef5f2;
      padding: 15px 18px 13px 18px;
      margin-bottom: 15px;
    }}
    .title {{
      font-size: 27pt;
      line-height: 1.05;
      font-weight: 800;
      color: #17352d;
      margin: 0 0 7px 0;
    }}
    .meta {{
      font-size: 10pt;
      color: #567066;
      margin: 0;
    }}
    .section-title {{
      font-size: 16pt;
      font-weight: 800;
      color: #17352d;
      margin: 6px 0 8px 0;
    }}
    .overview {{
      margin-bottom: 15px;
    }}
    .overview-panel {{
      background: #f8fbf9;
      padding: 11px 14px 10px 14px;
    }}
    .bullets {{
      margin: 0;
      padding-left: 0;
    }}
    .bullet-row {{
      margin: 0 0 5px 0;
    }}
    .bullet-mark {{
      color: #2d6a56;
      font-weight: 700;
    }}
    .overview-meta {{
      margin-top: 8px;
      font-size: 9.5pt;
      color: #5c6e68;
    }}
    .kpi-section {{
      margin-bottom: 17px;
    }}
    .kpi-grid {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 8px 0;
      table-layout: fixed;
    }}
    .kpi-card {{
      width: 33.333%;
      background: #eef5f2;
      border: 1px solid #d6e5dd;
      padding: 14px 15px;
      vertical-align: top;
    }}
    .kpi-title {{
      font-size: 10.5pt;
      color: #436257;
      font-weight: 700;
      margin: 0 0 10px 0;
      line-height: 1.25;
    }}
    .kpi-label {{
      font-size: 9pt;
      color: #51665f;
      margin: 0 0 2px 0;
    }}
    .kpi-value {{
      font-size: 21pt;
      line-height: 1.05;
      font-weight: 800;
      color: #17352d;
      margin: 0 0 8px 0;
    }}
    .kpi-unit {{
      font-size: 9pt;
      font-weight: 700;
      color: #436257;
      margin: -3px 0 8px 0;
    }}
    .kpi-divider {{
      height: 1px;
      background: #d7e7de;
      margin: 7px 0 7px 0;
    }}
    .notes {{
      margin-bottom: 14px;
    }}
    .notes-list {{
      margin: 0;
      padding-left: 0;
    }}
    .footer {{
      border-top: 1px solid #d7e7de;
      padding-top: 7px;
      font-size: 9pt;
      color: #6c7b75;
    }}
    body.compact {{
      font-size: 9.7pt;
      line-height: 1.27;
    }}
    body.compact .hero {{ padding: 11px 15px 10px 15px; margin-bottom: 10px; }}
    body.compact .title {{ font-size: 24pt; }}
    body.compact .section-title {{ font-size: 14.5pt; margin-bottom: 5px; }}
    body.compact .overview {{ margin-bottom: 10px; }}
    body.compact .overview-panel {{ padding: 8px 11px 7px 11px; }}
    body.compact .bullet-row {{ margin-bottom: 3px; }}
    body.compact .kpi-section {{ margin-bottom: 11px; }}
    body.compact .kpi-card {{ padding: 10px 11px; }}
    body.compact .kpi-title {{ font-size: 9.5pt; margin-bottom: 6px; }}
    body.compact .kpi-label {{ font-size: 8pt; }}
    body.compact .kpi-value {{ font-size: 18pt; margin-bottom: 5px; }}
    body.compact .kpi-unit {{ font-size: 8pt; margin-bottom: 5px; }}
    body.compact .notes {{ margin-bottom: 8px; }}
    body.compact .footer {{ font-size: 8pt; padding-top: 5px; }}
    .page-break {{
      break-before: page;
      page-break-before: always;
    }}
    .figure-title {{
      font-size: 17pt;
      font-weight: 800;
      color: #17352d;
      margin: 0 0 12px 0;
    }}
    .figure-panel {{
      margin-bottom: 18px;
    }}
    .figure-panel img {{
      width: 160mm;
      height: auto;
      border: 1px solid #dfe8e4;
      border-radius: 10px;
      display: block;
      margin: 0 auto 10px auto;
    }}
    .table-title {{
      font-size: 13pt;
      font-weight: 800;
      color: #17352d;
      margin: 10px 0 7px 0;
    }}
    .report-table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 8.5pt;
    }}
    .report-table th {{
      background: #dcebe4;
      color: #17352d;
      border: 1px solid #c9ddd3;
      padding: 5px 4px;
      font-weight: 700;
      text-align: left;
      word-wrap: break-word;
      -pdf-keep-in-frame-mode: shrink;
    }}
    .report-table td {{
      border: 1px solid #dbe5e0;
      padding: 4px;
      color: #2c3d37;
      word-wrap: break-word;
      -pdf-keep-in-frame-mode: shrink;
    }}
    .report-table tr:nth-child(even) td {{
      background: #f6f9f7;
    }}
    .effort-table {{
      font-size: 8pt;
    }}
    .table-empty {{
      color: #6c7b75;
      font-size: 9.5pt;
      padding: 8px 0;
    }}
  </style>
</head>
<body class="{layout_class}">
  <div class="page">
    <div class="hero">
      <div class="title">Plant Pest Prevalence Report</div>
      <p class="meta">Report period: {escape(str(before_year))} to {escape(str(after_year))}</p>
    </div>

    <section class="overview">
      <div class="section-title">Overview</div>
      <div class="overview-panel">
        {_html_bullet_list(overview_lines)}
        {"<div class='overview-meta'>" + escape(overview_meta) + "</div>" if overview_meta else ""}
      </div>
    </section>

    <section class="kpi-section">
      <div class="section-title">Key figures</div>
      <table class="kpi-grid" role="presentation"><tr>
        <td class="kpi-card">
          <div class="kpi-title">Estimated change in prevalence</div>
          <div class="kpi-label">Change Method</div>
          <div class="kpi-value">{escape(_format_signed_percent(simple_change))}</div>
          <div class="kpi-divider"></div>
          <div class="kpi-label">Regression Method</div>
          <div class="kpi-value">{escape(_format_signed_percent(regression_change))}</div>
        </td>
        <td class="kpi-card">
          <div class="kpi-title">Prevalence trend</div>
          <div class="kpi-value">{escape(_format_signed_percent(annual_change))}</div>
          <div class="kpi-unit">Change / year</div>
          <div class="kpi-label">Average yearly change from the fitted regression trend</div>
        </td>
        <td class="kpi-card">
          <div class="kpi-title">Estimated final prevalence</div>
          <div class="kpi-label">Change Method</div>
          <div class="kpi-value">{escape(_format_percent(simple_final))}</div>
          <div class="kpi-divider"></div>
          <div class="kpi-label">Regression Method</div>
          <div class="kpi-value">{escape(_format_percent(regression_final))}</div>
        </td>
      </tr></table>
    </section>

    <section class="notes">
      <div class="section-title">Points to keep in mind</div>
      {_html_bullet_list(caveats, "notes-list")}
    </section>

    <div class="footer">
      {escape("  |  ".join(footer_items))}
    </div>
  </div>

  <div class="page-break"></div>
  <div class="page">
    <div class="figure-panel">
      <div class="figure-title">Estimated prevalence over time</div>
      <img src="{trend_plot_uri}" alt="Estimated prevalence over time plot" />
      <div class="table-title">Yearly survey summary</div>
      {yearly_table_html}
    </div>
  </div>
  <div class="page-break"></div>
  <div class="page">
    <div class="figure-panel">
      <div class="figure-title">Observed effort vs indicative required effort</div>
      <img src="{effort_plot_uri}" alt="Observed effort vs indicative required effort plot" />
      <div class="table-title">Indicative effort summary</div>
      {effort_table_html}
    </div>
  </div>
</body>
</html>
"""


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
    simple_change_settings: dict[str, Any],
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
            target_prevalence=float(regression_settings.get("target", DEFAULT_APPENDIX_E_TARGET)),
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
    simple_change_settings: dict[str, Any],
    regression_settings: dict[str, Any],
    benchmark_label: str | None = None,
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
            simple_change_settings=simple_change_settings,
            regression_settings=regression_settings,
            benchmark_label=benchmark_label,
        )
        return

    trend_fig = plot_quick_results_trend(prevalence_ts_df, regression_fit_df, before_year, after_year)
    effort_fig = _plot_sampling_effort_combined_figure(effort_df, benchmark_label=benchmark_label)
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
            target_prevalence=float(regression_settings.get("target", DEFAULT_APPENDIX_E_TARGET)),
            trend_plot_uri=_figure_to_data_uri(trend_fig),
            effort_plot_uri=_figure_to_data_uri(effort_fig),
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
        plt.close(effort_fig)


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
    if not regression_fit_df.empty and "model_pi_hat" in regression_fit_df.columns:
        slope_val = _series_lookup_value(regression_summary_df, "metric", "Trend slope (logit scale)")
        rows.extend(
            [
                {"method": "Regression Method", "metric": "Slope", "value": slope_val},
                {"method": "Regression Method", "metric": "Final prevalence", "value": round(float(regression_fit_df["model_pi_hat"].iloc[-1]), 4)},
                {"method": "Regression Method", "metric": "Estimated change", "value": round(float(regression_fit_df["model_pi_hat"].iloc[-1] - regression_fit_df["model_pi_hat"].iloc[0]), 4)},
            ]
        )
    else:
        rows.append({"method": "Regression Method", "metric": "Note", "value": "At least two yearly observations are required"})
    if before_row.empty or after_row.empty:
        rows.append({"method": "Simple Change Method", "metric": "Note", "value": "Selected years need observed survey data"})
    else:
        rows.extend(
            [
                {"method": "Simple Change Method", "metric": "Final prevalence", "value": round(float(after_row["pi_hat"].iloc[0]), 4)},
                {"method": "Simple Change Method", "metric": "Estimated change", "value": round(float(after_row["pi_hat"].iloc[0] - before_row["pi_hat"].iloc[0]), 4)},
            ]
        )
    return pd.DataFrame(rows)


def _plot_sampling_effort_combined_figure(effort_df: pd.DataFrame, benchmark_label: str | None = None):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    if effort_df.empty:
        ax.set_title("Observed effort vs indicative required effort")
        ax.text(0.5, 0.5, "No sampling-effort data available.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    years = effort_df["year"].astype(int).tolist()
    x = np.arange(len(years), dtype=float)
    width = 0.23
    observed = pd.to_numeric(effort_df["observed_prop_population"], errors="coerce").to_numpy(dtype=float)
    simple_req = pd.to_numeric(effort_df["simple_required_prop_population"], errors="coerce").to_numpy(dtype=float)
    reg_req = pd.to_numeric(effort_df["regression_required_prop_population"], errors="coerce").to_numpy(dtype=float)
    ax.bar(x - width, observed, width=width, color="#7f7f7f", label="Observed")
    ax.bar(x, simple_req, width=width, color="#7b3294", label="Change Method benchmark")
    ax.bar(x + width, reg_req, width=width, color="#2b8cbe", label="Regression Method benchmark")
    ax.set_title("Observed effort vs indicative required effort")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of total host population")
    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in years])
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


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
    appendix_c_delta: float,
    appendix_c_corr: float,
    appendix_alpha: float,
    appendix_power: float,
    appendix_e_target: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_cols = [
        "year",
        "observed_hosts",
        "observed_infected_hosts",
        "observed_prevalence",
        "appendixE_required_hosts",
        "appendixE_observed_ratio",
        "appendixC_required_hosts",
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
    )
    effort["appendixE_required_hosts"] = np.nan

    before_row = effort.loc[effort["year"] == year_before]
    after_row = effort.loc[effort["year"] == year_after]

    if not before_row.empty:
        pi0 = float(before_row["observed_prevalence"].iloc[0])
        n_initial = compute_n_initial_prev_ci_cells(
            p0_upper=DEFAULT_APPENDIX_E_INITIAL_UPPER,
            conf_level=DEFAULT_APPENDIX_E_INITIAL_CONF,
            width=DEFAULT_APPENDIX_E_INITIAL_WIDTH,
            population_size=None,
        )
        effort.loc[effort["year"] == year_before, "appendixE_required_hosts"] = n_initial

        n_followups = max(1, int(year_after - year_before))
        t_vec = np.arange(1, n_followups + 1, dtype=float)
        n_total = compute_n_total_appendix_e_cells(
            pi0=pi0,
            pi_target=appendix_e_target,
            alpha=appendix_alpha,
            power=appendix_power,
            t_vec=t_vec,
            population_size=None,
            phi=1.0,
        )
        n_per_round = int(np.ceil(n_total / len(t_vec)))
        followup_mask = (effort["year"] > year_before) & (effort["year"] <= year_after)
        effort.loc[followup_mask, "appendixE_required_hosts"] = n_per_round

    effort["appendixE_observed_ratio"] = effort["observed_hosts"] / effort["appendixE_required_hosts"].replace({0: np.nan})

    appendix_c_required = np.nan
    if not before_row.empty and not after_row.empty:
        p0 = float(before_row["observed_prevalence"].iloc[0])
        appendix_c_required = compute_n_appendix_c_cells(
            p0=p0,
            delta=appendix_c_delta,
            corr=appendix_c_corr,
            alpha=appendix_alpha,
            power=appendix_power,
            population_size=None,
        )
    effort["appendixC_required_hosts"] = np.nan
    if np.isfinite(appendix_c_required):
        effort.loc[effort["year"].isin([year_before, year_after]), "appendixC_required_hosts"] = appendix_c_required

    chart_df = pd.concat(
        [
            effort[["year", "observed_hosts"]].rename(columns={"observed_hosts": "n_hosts"}).assign(series="Observed"),
            effort[["year", "appendixE_required_hosts"]].rename(columns={"appendixE_required_hosts": "n_hosts"}).assign(series="Appendix E recommended"),
            effort.loc[effort["year"].isin([year_before, year_after]), ["year", "appendixC_required_hosts"]]
            .rename(columns={"appendixC_required_hosts": "n_hosts"})
            .assign(series="Appendix C paired requirement"),
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
    appendix_e = chart_df[chart_df["series"] == "Appendix E recommended"].set_index("year")["n_cells"]
    appendix_c = chart_df[chart_df["series"] == "Appendix C paired requirement"].set_index("year")["n_cells"]

    ax.bar(
        x - width,
        [float(observed.get(year, np.nan)) for year in years],
        width=width,
        color="#7f7f7f",
        label="Observed",
    )
    ax.bar(
        x,
        [float(appendix_e.get(year, np.nan)) for year in years],
        width=width,
        color="#2b8cbe",
        label="Appendix E recommended",
    )

    c_years = [year_before, year_after]
    c_x = [year_to_x[year] + width for year in c_years if year in year_to_x and year in appendix_c.index]
    c_vals = [float(appendix_c.loc[year]) for year in c_years if year in year_to_x and year in appendix_c.index]
    if c_x:
        ax.bar(
            c_x,
            c_vals,
            width=width,
            color="#de2d26",
            alpha=0.85,
            label="Appendix C paired requirement",
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
        "⚙",
        class_="btn-outline-secondary btn-sm",
        style="min-width:2.4rem;padding-left:0.55rem;padding-right:0.55rem;",
    )


app_ui = ui.page_navbar(
    ui.nav_panel(
        "Home",
        ui.layout_column_wrap(
            ui.card(
                ui.card_header("About This Application"),
                ui.h3("DEFRA Plant Pest Dashboard"),
                ui.p(
                    "This dashboard brings together survey maps, host coverage maps, and control-zone maps so you can "
                    "see what was surveyed and what the survey record suggests about disease prevalence over time."
                ),
                ui.p(
                    "It is designed for operational use. Load the data here, inspect it in Data Viewer, then generate "
                    "results and a PDF report in Results."
                ),
                ui.h5("What each window is for"),
                ui.p("Home: load the map data and check that the file paths are correct."),
                ui.p("Data Viewer: inspect the loaded survey, host, and SPHN polygons year by year."),
                ui.p("Results: estimate prevalence over time, compare observed effort with benchmark effort, and create a PDF report."),
                ui.h5("What to do next"),
                ui.p("Start by loading your data on this page. Then open Data Viewer to check the map, or go straight to Results if the data has already loaded correctly."),
            ),
            ui.card(
                ui.card_header("Load Map Data"),
                ui.div(
                    ui.h4("Load your map data", class_="landing-load-title"),
                    ui.p(
                        "The dashboard will try to load the default files automatically when it opens. "
                        "If your files are stored somewhere else, update the paths below and press 'Load Data'.",
                        class_="landing-load-intro",
                    ),
                    class_="landing-load-hero",
                ),
                ui.div(
                    ui.div(
                        ui.tags.div("Survey polygons", class_="landing-path-label"),
                        ui.p("The shapefile or folder containing the survey polygons.", class_="landing-path-help"),
                        ui.input_text(
                            "home_target_sites_dir",
                            "",
                            value=str(TARGET_SITES_DIR),
                        ),
                        class_="landing-path-block",
                    ),
                    ui.div(
                        ui.tags.div("Host coverage polygons", class_="landing-path-label"),
                        ui.p("The shapefile or folder containing the mapped host distribution.", class_="landing-path-help"),
                        ui.input_text(
                            "home_host_layer_path",
                            "",
                            value=str(LARCH_HOST_DIR),
                        ),
                        class_="landing-path-block",
                    ),
                    ui.div(
                        ui.tags.div("SPHN polygons", class_="landing-path-label"),
                        ui.p("The shapefile or folder containing the SPHN polygons shown in the map view.", class_="landing-path-help"),
                        ui.input_text(
                            "home_sphn_layer_path",
                            "",
                            value=str(SPHN_DIR),
                        ),
                        class_="landing-path-block",
                    ),
                    class_="landing-path-grid",
                ),
                ui.div(
                    ui.input_action_button("btn_home_load_data", "Load Data", class_="btn-primary"),
                    ui.tags.span("You can reload at any time after changing the paths.", class_="landing-load-note"),
                    class_="landing-load-actions",
                ),
                ui.output_ui("home_data_status_panel"),
            ),
            width=1 / 2,
        ),
    ),
    ui.nav_panel(
        "Data Viewer",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h3("Data Viewer"),
                ui.p(
                    "This window shows the loaded map data. Use it to view survey sites, host distribution, "
                    "and SPHN sites for a selected year.",
                    width="100%",
                ),
                ui.p(
                    "Use the controls below to change the year, map view, and visible layers. When you are happy with the loaded data, go to Results to generate the prevalence summary.",
                    class_="text-muted",
                    width="100%",
                ),
                ui.input_slider(
                    "year_selected_interactive",
                    "Year",
                    min=YEAR_MIN,
                    max=YEAR_MAX,
                    value=2024,
                    step=1,
                    sep="",
                ),
                ui.input_select(
                    "view_mode_interactive",
                    "View",
                    choices={
                        "uk": "Whole UK",
                        "all_zoom": "Zoom to loaded polygons",
                        "survey": "Survey sites",
                        "larch": "Host distribution",
                        "sphn": "SPHN sites",
                    },
                    selected="all_zoom",
                ),
                ui.input_checkbox_group(
                    "visible_layers_interactive",
                    "Visible layers",
                    choices={
                        "survey": "Survey sites",
                        "larch": "Host distribution",
                        "sphn": "SPHN sites",
                    },
                    selected=[],
                ),
            ),
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("UK Map"),
                    ui.output_ui("survey_map_interactive_trial"),
                ),
                width=1,
            ),
        ),
    ),
    ui.nav_panel(
        "Results",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h3("Results"),
                ui.p(
                    "Use this window to turn the loaded survey data into prevalence estimates, trend summaries, and a printable PDF report.",
                    width="100%",
                ),
                ui.p(
                    "Press 'Generate Results' first. Open Settings only if you need to change how survey outcomes or host polygons are interpreted.",
                    class_="text-muted",
                    width="100%",
                ),
                ui.div(
                    ui.input_action_button("btn_quick_results", "Generate Results", class_="btn-primary"),
                    ui.popover(
                        _settings_cog_button("btn_results_settings"),
                        ui.div(
                            ui.h5("Survey Data Settings"),
                            _settings_field(
                                "Survey status column",
                                "Choose the column that records the survey outcome, such as infected, no evidence, or suspicious.",
                                ui.input_select(
                                    "quick_status_column",
                                    "",
                                    choices={"__auto__": "Auto-detect"},
                                    selected="__auto__",
                                ),
                            ),
                            _settings_field(
                                "Outcomes counted as positive",
                                "Tick the survey outcomes that should count as disease present.",
                                ui.input_checkbox_group(
                                    "quick_positive_statuses",
                                    "",
                                    choices={},
                                    selected=[],
                                ),
                            ),
                            _settings_field(
                                "Outcomes counted as negative",
                                "Tick the survey outcomes that should count as disease absent. Outcomes left unticked are ignored.",
                                ui.input_checkbox_group(
                                    "quick_negative_statuses",
                                    "",
                                    choices={},
                                    selected=[],
                                ),
                            ),
                            ui.hr(),
                            ui.h5("Host Landscape Settings"),
                            _settings_field(
                                "Host category column",
                                "Usually leave this unchanged. Only use it if your host map contains several host categories and you need to keep only part of it, such as one species.",
                                ui.input_select(
                                    "quick_host_column",
                                    "",
                                    choices={"__all__": "Use all host polygons"},
                                    selected="__all__",
                                ),
                            ),
                            _settings_field(
                                "Host categories to include",
                                "If you selected a host category column above, tick the values that should count as part of the host landscape.",
                                ui.input_checkbox_group(
                                    "quick_host_values",
                                    "",
                                    choices={},
                                    selected=[],
                                ),
                            ),
                            _settings_field(
                                "Survey polygons outside the host map",
                                "Turn this on if surveyed areas outside the host map should still be treated as part of the wider host landscape.",
                                ui.input_checkbox(
                                    "quick_include_unmatched_surveys",
                                    "Include surveyed areas outside the host map",
                                    value=True,
                                ),
                            ),
                            _settings_field(
                                "Host density (hosts per km^2)",
                                "This is the assumed number of host plants per square kilometre. The app uses it to convert mapped area into an estimated number of hosts.",
                                ui.input_numeric(
                                    "quick_host_density_km2",
                                    "",
                                    value=2500.0,
                                    min=0.0,
                                    step=100.0,
                                ),
                            ),
                            ui.hr(),
                            ui.h5("Effort Benchmark Settings"),
                            _settings_field(
                                "Benchmark assumption",
                                "Choose how cautious the dashboard should be when comparing observed survey effort with an indicative benchmark. This is a reference scenario, not a reconstruction of the true historical survey design.",
                                ui.input_select(
                                    "shared_effort_benchmark",
                                    "",
                                    choices=RETRO_BENCHMARK_CHOICES,
                                    selected="moderate",
                                ),
                            ),
                            _settings_field(
                                "Significance level (alpha)",
                                "This controls how strict the benchmark is about false alarms. Smaller values usually increase the indicative benchmark effort.",
                                ui.input_numeric(
                                    "shared_alpha",
                                    "",
                                    value=DEFAULT_ALPHA,
                                    min=0.001,
                                    max=0.2,
                                    step=0.005,
                                ),
                            ),
                            _settings_field(
                                "Power",
                                "This controls how likely the benchmark is to detect a real change or trend if one exists. Higher values usually increase the indicative benchmark effort.",
                                ui.input_numeric(
                                    "shared_power",
                                    "",
                                    value=DEFAULT_POWER,
                                    min=0.5,
                                    max=0.999,
                                    step=0.01,
                                ),
                            ),
                            ui.hr(),
                            ui.h5("Change Method"),
                            _settings_field(
                                "First survey year",
                                "The Change Method compares estimated prevalence in this starting year against the selected final year.",
                                ui.input_select(
                                    "simple_change_before_year",
                                    "",
                                    choices={str(y): str(y) for y in range(YEAR_MIN, YEAR_MAX)},
                                    selected="2017",
                                ),
                            ),
                            _settings_field(
                                "Last survey year",
                                "This is the end year used when calculating overall change between the two selected years.",
                                ui.input_select(
                                    "simple_change_after_year",
                                    "",
                                    choices={str(y): str(y) for y in range(YEAR_MIN + 1, YEAR_MAX + 1)},
                                    selected="2024",
                                ),
                            ),
                            _settings_field(
                                "Minimum change to detect",
                                "This is the smallest change in prevalence that you want the method to be able to detect reliably.",
                                ui.input_numeric(
                                    "simple_change_delta",
                                    "",
                                    value=DEFAULT_APPENDIX_C_DELTA,
                                    min=0.001,
                                    max=0.5,
                                    step=0.005,
                                ),
                            ),
                            ui.hr(),
                            ui.h5("Regression Method"),
                            _settings_field(
                                "Initial survey year",
                                "This is the starting year for the trend fitted through the yearly prevalence estimates.",
                                ui.input_select(
                                    "regression_before_year",
                                    "",
                                    choices={str(y): str(y) for y in range(YEAR_MIN, YEAR_MAX)},
                                    selected="2017",
                                ),
                            ),
                            _settings_field(
                                "Final survey year",
                                "This is the last year included when fitting the trend line.",
                                ui.input_select(
                                    "regression_after_year",
                                    "",
                                    choices={str(y): str(y) for y in range(YEAR_MIN + 1, YEAR_MAX + 1)},
                                    selected="2024",
                                ),
                            ),
                            _settings_field(
                                "Regression model",
                                "This chooses the shape of the smooth trend line fitted through the yearly prevalence estimates. Most users should leave it as Fractional polynomial 2.",
                                ui.input_select(
                                    "quick_regression_model",
                                    "",
                                    choices={
                                        "logistic": "Logistic",
                                        "fp2": "Fractional polynomial 2",
                                        "fp3": "Fractional polynomial 3",
                                    },
                                    selected="fp2",
                                ),
                            ),
                            _settings_field(
                                "Target prevalence",
                                "This is the low prevalence level used by the trend-based effort benchmark, for example a control or eradication target.",
                                ui.input_numeric(
                                    "regression_target",
                                    "",
                                    value=DEFAULT_APPENDIX_E_TARGET,
                                    min=0.0001,
                                    max=0.5,
                                    step=0.001,
                                ),
                            ),
                            style="min-width:46rem;max-width:72rem;",
                        ),
                        title="Results settings",
                        placement="right",
                        options={"customClass": "results-settings-popover"},
                    ),
                    ui.download_button("download_results_pdf", "Results to PDF", class_="btn-outline-secondary"),
                    style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;",
                ),
            ),
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Estimated Prevalence Over Time"),
                    ui.output_plot("appendix_e_plot", height="360px"),
                ),
                ui.card(
                    ui.card_header("Yearly Survey Summary"),
                    ui.output_data_frame("quick_results_summary_table"),
                    ui.hr(),
                    ui.h5("Method Summary"),
                    ui.output_data_frame("quick_results_method_table"),
                ),
                ui.card(
                    ui.card_header("Observed Effort vs Indicative Required Effort"),
                    ui.output_plot("sampling_effort_plot_combined", height="380px"),
                ),
                ui.card(
                    ui.card_header("Indicative Effort Summary"),
                    ui.output_data_frame("sampling_effort_table_combined"),
                ),
                width=1 / 2,
            ),
        ),
    ),
    title="DEFRA Plant Pest Dashboard",
    id="top_nav",
    header=ui.tags.head(
        ui.tags.style(
            """
            .landing-load-hero {
              background: linear-gradient(135deg, #eef6f1 0%, #f8fbf9 100%);
              border: 1px solid #d7e7de;
              border-radius: 16px;
              padding: 1rem 1.1rem 0.85rem 1.1rem;
              margin-bottom: 1rem;
            }
            .landing-load-title {
              margin: 0 0 0.35rem 0;
              color: #17352d;
              font-weight: 700;
            }
            .landing-load-intro {
              margin: 0;
              color: #51665f;
              line-height: 1.45;
            }
            .landing-path-grid {
              display: grid;
              grid-template-columns: 1fr;
              gap: 0.9rem;
            }
            .landing-path-block {
              background: #fbfcfb;
              border: 1px solid #e3ece7;
              border-radius: 14px;
              padding: 0.95rem 1rem 0.8rem 1rem;
            }
            .landing-path-label {
              font-weight: 700;
              color: #17352d;
              margin-bottom: 0.2rem;
            }
            .landing-path-help {
              font-size: 0.92rem;
              color: #5d716a;
              margin-bottom: 0.55rem;
            }
            .landing-load-actions {
              display: flex;
              align-items: center;
              gap: 0.8rem;
              margin-top: 1rem;
              margin-bottom: 1rem;
              flex-wrap: wrap;
            }
            .landing-load-note {
              color: #5d716a;
              font-size: 0.92rem;
            }
            .landing-status-grid {
              display: grid;
              grid-template-columns: repeat(3, minmax(0, 1fr));
              gap: 0.75rem;
              margin-top: 0.35rem;
              margin-bottom: 0.85rem;
            }
            .landing-status-card {
              border: 1px solid #dfe9e4;
              border-radius: 14px;
              padding: 0.85rem 0.95rem;
              background: #fafcfb;
            }
            .landing-status-label {
              font-size: 0.88rem;
              color: #5d716a;
              margin-bottom: 0.2rem;
            }
            .landing-status-value {
              font-size: 1.02rem;
              font-weight: 700;
              color: #17352d;
            }
            .landing-status-ok {
              color: #1c7c54;
            }
            .landing-status-missing {
              color: #a35a17;
            }
            .landing-status-notes {
              background: #faf7f1;
              border: 1px solid #eadcc5;
              border-radius: 14px;
              padding: 0.9rem 1rem;
              color: #4a3c28;
            }
            .landing-status-notes-title {
              font-weight: 700;
              margin-bottom: 0.35rem;
            }
            @media (max-width: 900px) {
              .landing-status-grid {
                grid-template-columns: 1fr;
              }
            }
            .results-settings-popover {
              max-width: min(88vw, 1180px) !important;
              width: max-content !important;
            }
            .data-viewer-settings-popover {
              max-width: min(72vw, 900px) !important;
              width: max-content !important;
            }
            .popover .popover-body {
              max-height: 75vh;
              overflow-y: auto;
            }
            """
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

    def _empty_layer_info(message: str | None = None) -> dict[str, Any]:
        # Standard empty result for a folder of survey layers.
        out = {"available_files": {}, "layers": {}, "messages": []}
        if message:
            out["messages"].append(message)
        return out

    def _empty_single_layer_info(message: str | None = None) -> dict[str, Any]:
        # Standard empty result for one optional polygon layer.
        out = {"available_file": False, "layer": None, "messages": []}
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
            "simple_effort_df": pd.DataFrame(),
            "regression_effort_df": pd.DataFrame(),
            "spatial_diagnostics_df": pd.DataFrame(),
            "total_host_population": np.nan,
            "benchmark_label": "",
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

    def _load_shared_data_sources(
        target_dir_text: str,
        host_path_text: str,
        sphn_path_text: str,
    ) -> list[str]:
        # Load all data paths from the Home page once, then share the same data
        # with the Data Viewer and Results tabs.
        messages: list[str] = []

        if target_dir_text:
            layer_info = load_target_sites(Path(target_dir_text).expanduser())
        else:
            layer_info = _empty_layer_info("Survey layer path was left blank, so survey polygons were not loaded.")
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
            host_info = load_host_layer(Path(host_path_text).expanduser())
        else:
            host_info = _empty_single_layer_info("Host layer path was left blank, so the host layer was not loaded.")
        rv_host_info_interactive.set(host_info)
        rv_quick_host_info.set(host_info)
        messages.extend(host_info.get("messages", []))

        if sphn_path_text:
            sphn_info = load_sphn_layer(Path(sphn_path_text).expanduser())
        else:
            sphn_info = _empty_single_layer_info("SPHN layer path was left blank, so the SPHN layer was not loaded.")
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
        return _retrospective_benchmark_assumptions(str(input.shared_effort_benchmark()))

    @reactive.calc
    def simple_change_effort_selected():
        return results_bundle_selected()["simple_effort_df"]

    @reactive.calc
    def regression_effort_selected():
        return results_bundle_selected()["regression_effort_df"]

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
        finally:
            ui.notification_remove(notif_id, session=session)

    @reactive.effect
    @reactive.event(input.btn_quick_results)
    def _load_quick_results_inputs():
        # Main Results button. This converts polygons into yearly prevalence,
        # fits the two methods, calculates effort benchmarks, and stores outputs.
        notif_id = _show_wait_notification()
        try:
            rv_quick_layer_info.set(rv_layer_info_interactive.get())
            rv_quick_host_info.set(rv_host_info_interactive.get())
            _sync_quick_results_choices(rv_layer_info_interactive.get(), rv_host_info_interactive.get())
            layers = rv_layer_info_interactive.get().get("layers", {})
            host_gdf = rv_host_info_interactive.get().get("layer")
            host_density = float(input.quick_host_density_km2())
            positive_statuses = list(input.quick_positive_statuses())
            negative_statuses = list(input.quick_negative_statuses())
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
            )
            # Convert yearly sampled/infected host counts into a prevalence time series.
            prevalence_ts_df = summarise_prevalence_timeseries_from_hosts(year_summary_df)
            # Fit the regression method to all yearly prevalence estimates.
            regression_summary_df, regression_fit_df = run_appendix_e_real_with_model(
                prevalence_ts_df,
                model_form=str(input.quick_regression_model()),
            )
            # Compare the first and final years using the Change Method.
            method_df = build_quick_results_method_summary(
                prevalence_ts_df,
                regression_summary_df,
                regression_fit_df,
                int(input.simple_change_before_year()),
                int(input.simple_change_after_year()),
            )
            # Estimate the denominator: how many hosts exist in the mapped host landscape.
            total_host_population = estimate_total_host_population(
                host_gdf,
                host_density_km2=host_density,
                layers=layers,
                host_column_name=host_column_name,
                host_values=host_values,
                include_unmatched_survey_polygons=include_unmatched,
            )
            bench = benchmark_assumptions_selected()
            # These effort tables are indicative benchmarks, not proof that the
            # historical surveys followed a clean statistical design.
            simple_effort_df = build_simple_change_effort_from_hosts(
                year_summary_df,
                year_before=int(input.simple_change_before_year()),
                year_after=int(input.simple_change_after_year()),
                pilot_year=int(input.simple_change_before_year()),
                delta=float(input.simple_change_delta()),
                corr=0.0,
                alpha=float(input.shared_alpha()),
                power=float(input.shared_power()),
                total_population_size=total_host_population,
                layers=layers,
                host_gdf=host_gdf,
                host_density_km2=host_density,
                sampling_mode=str(bench["sampling_mode"]),
                cluster_inflation_mode="fixed",
                within_cluster_cap=int(bench["within_cluster_cap"]),
                fixed_phi=float(bench["fixed_phi"]),
                positive_status_values=positive_statuses,
                negative_status_values=negative_statuses,
                status_column_name=status_column_name,
                host_column_name=host_column_name,
                host_values=host_values,
                include_unmatched_survey_polygons=include_unmatched,
            )
            regression_effort_df = build_regression_effort_from_hosts(
                year_summary_df,
                year_before=int(input.regression_before_year()),
                year_after=int(input.regression_after_year()),
                alpha=float(input.shared_alpha()),
                power=float(input.shared_power()),
                appendix_e_target=float(input.regression_target()),
                total_population_size=total_host_population,
                layers=layers,
                host_gdf=host_gdf,
                host_density_km2=host_density,
                pilot_year=int(input.regression_before_year()),
                sampling_mode=str(bench["sampling_mode"]),
                cluster_inflation_mode="fixed",
                within_cluster_cap=int(bench["within_cluster_cap"]),
                fixed_phi=float(bench["fixed_phi"]),
                positive_status_values=positive_statuses,
                negative_status_values=negative_statuses,
                status_column_name=status_column_name,
                host_column_name=host_column_name,
                host_values=host_values,
                include_unmatched_survey_polygons=include_unmatched,
            )
            spatial_diagnostics_df = build_spatial_survey_diagnostics(
                layers,
                host_gdf,
                host_column_name=host_column_name,
                host_values=host_values,
                include_unmatched_survey_polygons=include_unmatched,
            )
            # Store everything together so plots, tables, and the PDF all use the
            # exact same calculated results.
            rv_results_bundle.set(
                {
                    "year_summary_df": year_summary_df,
                    "prevalence_ts_df": prevalence_ts_df,
                    "regression_summary_df": regression_summary_df,
                    "regression_fit_df": regression_fit_df,
                    "method_df": method_df,
                    "simple_effort_df": simple_effort_df,
                    "regression_effort_df": regression_effort_df,
                    "spatial_diagnostics_df": spatial_diagnostics_df,
                    "total_host_population": total_host_population,
                    "benchmark_label": str(bench["label"]),
                }
            )
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
        # Small Home-page status display showing whether each data source loaded.
        layer_info = rv_layer_info_interactive.get()
        host_info = rv_host_info_interactive.get()
        sphn_info = rv_sphn_info_interactive.get()

        def status_card(label: str, loaded: bool):
            value = "Loaded" if loaded else "Not loaded"
            klass = "landing-status-value landing-status-ok" if loaded else "landing-status-value landing-status-missing"
            return ui.div(
                ui.div(label, class_="landing-status-label"),
                ui.div(value, class_=klass),
                class_="landing-status-card",
            )

        messages: list[str] = []
        messages.extend(layer_info.get("messages", []))
        messages.extend(host_info.get("messages", []))
        messages.extend(sphn_info.get("messages", []))

        notes_block = None
        if messages:
            notes_block = ui.div(
                ui.div("Loading notes", class_="landing-status-notes-title"),
                ui.tags.ul(*[ui.tags.li(msg) for msg in messages[:6]], style="margin-bottom:0;"),
                class_="landing-status-notes",
            )

        return ui.div(
            ui.div(
                status_card("Survey polygons", bool(layer_info.get("layers"))),
                status_card("Host coverage", host_info.get("layer") is not None),
                status_card("SPHN polygons", sphn_info.get("layer") is not None),
                class_="landing-status-grid",
            ),
            notes_block if notes_block is not None else ui.div(),
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
                "surveyed_hosts_est": "Estimated hosts surveyed",
                "infected_hosts_est": "Estimated infected hosts",
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
                "Simple Change Method": "Change Method",
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
    def simple_change_sampling_effort_table():
        out = simple_change_effort_selected().copy()
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
        # Join the Change Method and Regression Method effort benchmarks into
        # one table so users can compare them against the observed survey effort.
        simple_df = simple_change_effort_selected().copy()
        reg_df = regression_effort_selected().copy()
        bundle = results_bundle_selected()
        if simple_df.empty and reg_df.empty:
            return pd.DataFrame(
                columns=[
                    "year",
                    "observed_hosts",
                    "observed_prop_population",
                    "observed_prevalence",
                    "simple_required_hosts",
                    "simple_required_prop_population",
                    "regression_required_hosts",
                    "regression_required_prop_population",
                    "benchmark_assumption",
                ]
            )
        if reg_df.empty:
            merged = simple_df.loc[:, ["year", "observed_hosts", "observed_prop_population", "observed_prevalence"]].copy()
            merged["regression_required_hosts"] = np.nan
            merged["regression_required_prop_population"] = np.nan
        else:
            merged = reg_df.rename(
                columns={
                    "required_hosts": "regression_required_hosts",
                    "required_prop_population": "regression_required_prop_population",
                }
            )
            merged = merged[["year", "observed_hosts", "observed_prop_population", "observed_prevalence", "regression_required_hosts", "regression_required_prop_population"]]
        if not simple_df.empty:
            simple_keep = simple_df.rename(
                columns={
                    "required_hosts": "simple_required_hosts",
                    "required_prop_population": "simple_required_prop_population",
                }
            )[["year", "simple_required_hosts", "simple_required_prop_population"]]
            merged = merged.merge(simple_keep, on="year", how="outer")
        else:
            merged["simple_required_hosts"] = np.nan
            merged["simple_required_prop_population"] = np.nan
        merged = merged.sort_values("year").reset_index(drop=True)
        merged["benchmark_assumption"] = str(bundle.get("benchmark_label", ""))
        return merged

    @output
    @render.data_frame
    def sampling_effort_table_combined():
        # Display the combined effort table with friendly names and percentages.
        out = _build_combined_sampling_effort_table().copy()
        if out.empty:
            return render_grid(out)
        for col in ("observed_prop_population", "simple_required_prop_population", "regression_required_prop_population", "observed_prevalence"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        for col in ("observed_hosts", "simple_required_hosts", "regression_required_hosts"):
            out[col] = out[col].apply(lambda x: "" if pd.isna(x) else int(round(float(x))))
        out["observed_prop_population"] = out["observed_prop_population"].apply(_format_percent)
        out["simple_required_prop_population"] = out["simple_required_prop_population"].apply(_format_percent)
        out["regression_required_prop_population"] = out["regression_required_prop_population"].apply(_format_percent)
        out["observed_prevalence"] = out["observed_prevalence"].apply(_format_percent)
        out = out.rename(
            columns={
                "year": "Year",
                "observed_hosts": "Observed hosts",
                "observed_prop_population": "Observed share of host population",
                "observed_prevalence": "Observed prevalence",
                "simple_required_hosts": "Change Method benchmark hosts",
                "simple_required_prop_population": "Change Method benchmark share",
                "regression_required_hosts": "Regression Method benchmark hosts",
                "regression_required_prop_population": "Regression Method benchmark share",
                "benchmark_assumption": "Benchmark assumption",
            }
        )
        return render_grid(out)

    @output
    @render.plot
    def appendix_e_plot():
        # Main results plot: yearly prevalence estimates plus the fitted trend.
        return plot_quick_results_trend(
            prevalence_ts_selected(),
            regression_fit_selected(),
            int(input.simple_change_before_year()),
            int(input.simple_change_after_year()),
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
        ax.bar(x - width / 2, effort_df["observed_prop_population"].astype(float), width=width, color="#7f7f7f", label="Observed")
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
    def simple_change_sampling_effort_plot():
        return _plot_effort_props(simple_change_effort_selected(), "Observed effort vs benchmark effort: Change Method")

    @output
    @render.plot
    def regression_sampling_effort_plot():
        return _plot_effort_props(regression_effort_selected(), "Observed effort vs benchmark effort: Regression Method")

    @output
    @render.plot
    def sampling_effort_plot_combined():
        return _plot_sampling_effort_combined_figure(
            _build_combined_sampling_effort_table(),
            benchmark_label=str(results_bundle_selected().get("benchmark_label", "")),
        )

    @output
    @render.download(filename="defra_results_report.pdf", media_type="application/pdf")
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
            before_year = int(input.simple_change_before_year())
            after_year = int(input.simple_change_after_year())
            regression_model = str(input.quick_regression_model())

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
                    simple_change_settings={
                        "sampling_mode": str(benchmark_assumptions_selected()["sampling_mode"]),
                        "overlap_type": "cross_sectional_benchmark",
                        "alpha": float(input.shared_alpha()),
                        "power": float(input.shared_power()),
                        "delta": float(input.simple_change_delta()),
                    },
                    regression_settings={
                        "sampling_mode": str(benchmark_assumptions_selected()["sampling_mode"]),
                        "alpha": float(input.shared_alpha()),
                        "power": float(input.shared_power()),
                        "target": float(input.regression_target()),
                    },
                    benchmark_label=str(bundle.get("benchmark_label", "")),
                )
                with tmp_path.open("rb") as f:
                    yield f.read()
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        finally:
            ui.notification_remove(notif_id, session=session)


def compute_n_total_appendix_e_hosts(
    pi0: float,
    piD: float,
    alpha: float,
    power: float,
    t_vec: list[int] | np.ndarray,
    population_size: int | float | None = None,
    phi: float = 1.0,
) -> dict[str, float]:
    pi0 = _clamp_open01(float(pi0))
    piD = _clamp_open01(float(piD))
    t_arr = np.asarray(t_vec, dtype=float)
    if t_arr.size == 0:
        t_arr = np.array([1.0], dtype=float)
    t_end = float(np.max(t_arr))
    beta_star = (_logit(piD) - _logit(pi0)) / max(t_end, 1e-8)
    eta = _logit(pi0) + beta_star * t_arr
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    w = p * (1.0 - p)
    x = np.column_stack([np.ones_like(t_arr), t_arr])
    info = x.T @ np.diag(w) @ x + np.eye(2) * 1e-8
    try:
        v_per_obs = float(np.linalg.inv(info)[1, 1])
    except np.linalg.LinAlgError:
        fallback = float(population_size) if population_size is not None and np.isfinite(population_size) else 1.0
        return {"n_total": float(_apply_fpc(fallback, population_size)), "beta_star": float(beta_star), "V": float("nan")}
    z_alpha = NormalDist().inv_cdf(1.0 - float(alpha))
    z_beta = NormalDist().inv_cdf(float(power))
    beta_abs = abs(beta_star)
    if not np.isfinite(beta_abs) or beta_abs < 1e-10 or not np.isfinite(v_per_obs) or v_per_obs <= 0:
        fallback = float(population_size) if population_size is not None and np.isfinite(population_size) else 1.0
        return {"n_total": float(_apply_fpc(fallback, population_size)), "beta_star": float(beta_star), "V": float(v_per_obs)}
    n_raw = ((z_alpha + z_beta) / beta_abs) ** 2 * v_per_obs
    n_total = _apply_fpc(float(phi) * float(np.ceil(n_raw)), population_size)
    return {"n_total": float(n_total), "beta_star": float(beta_star), "V": float(v_per_obs)}


def _estimate_design_effect_from_binary_cluster_sample(sampled_df: pd.DataFrame, planned_m: int | None = None) -> dict[str, float]:
    if sampled_df is None or sampled_df.empty or "cell_id" not in sampled_df.columns or "detected" not in sampled_df.columns:
        m = float(planned_m) if planned_m is not None and planned_m > 0 else 1.0
        return {"phi": 1.0, "rho": 0.0, "m": m, "m_observed": 1.0, "estimator": "degenerate"}
    work = sampled_df.loc[:, ["cell_id", "detected"]].copy()
    work["cell_id"] = work["cell_id"].astype(str)
    work["detected"] = (pd.to_numeric(work["detected"], errors="coerce").fillna(0).astype(int) > 0).astype(int)
    grouped = work.groupby("cell_id")["detected"].agg(["size", "mean"]).reset_index()
    if grouped.empty:
        m = float(planned_m) if planned_m is not None and planned_m > 0 else 1.0
        return {"phi": 1.0, "rho": 0.0, "m": m, "m_observed": 1.0, "estimator": "degenerate"}
    sizes = grouped["size"].to_numpy(dtype=float)
    means = grouped["mean"].to_numpy(dtype=float)
    m_observed = float(np.mean(sizes))
    m = float(planned_m) if planned_m is not None and planned_m > 0 else m_observed
    if len(grouped) < 2 or np.all(work["detected"].to_numpy() == work["detected"].iloc[0]):
        return {"phi": 1.0, "rho": 0.0, "m": m, "m_observed": m_observed, "estimator": "degenerate"}
    overall = float(np.average(means, weights=sizes))
    ssb = float(np.sum(sizes * (means - overall) ** 2))
    dfb = max(len(grouped) - 1, 1)
    msb = ssb / dfb
    within_num = float(np.sum(np.maximum(sizes - 1.0, 0.0) * means * (1.0 - means)))
    within_den = float(np.sum(np.maximum(sizes - 1.0, 0.0)))
    msw = within_num / within_den if within_den > 0 else 0.0
    n_bar_num = float(np.sum(sizes) - (np.sum(sizes**2) / max(np.sum(sizes), 1.0)))
    n_bar_den = max(len(grouped) - 1, 1)
    n_bar = n_bar_num / n_bar_den if n_bar_den > 0 else m_observed
    denom = msb + max(n_bar - 1.0, 0.0) * msw
    rho = 0.0 if denom <= 0 else max(0.0, min(1.0, (msb - msw) / denom))
    phi = max(1.0, 1.0 + (m - 1.0) * rho)
    return {"phi": float(phi), "rho": float(rho), "m": float(m), "m_observed": float(m_observed), "estimator": "anova"}


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


def build_simple_change_effort_from_hosts(
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
    cols = ["year", "observed_hosts", "observed_prop_population", "required_hosts", "required_prop_population", "observed_prevalence"]
    if year_summary_df is None or year_summary_df.empty:
        return pd.DataFrame(columns=cols)
    work = year_summary_df.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    effort = work.loc[work["year"].isin([int(year_before), int(year_after)]), ["year", "surveyed_hosts_est", "estimated_prevalence"]].copy()
    if effort.empty:
        return pd.DataFrame(columns=cols)
    effort = effort.rename(columns={"surveyed_hosts_est": "observed_hosts", "estimated_prevalence": "observed_prevalence"})
    total_pop = float(total_population_size) if total_population_size is not None and np.isfinite(total_population_size) and total_population_size > 0 else float("nan")
    phi_info = _resolve_quick_results_phi(
        layers=layers or {},
        host_gdf=host_gdf,
        pilot_year=int(pilot_year),
        host_density_km2=float(host_density_km2),
        sampling_mode=str(sampling_mode),
        cluster_inflation_mode=str(cluster_inflation_mode),
        within_cluster_cap=int(within_cluster_cap),
        fixed_phi=float(fixed_phi),
        positive_status_values=positive_status_values,
        negative_status_values=negative_status_values,
        status_column_name=status_column_name,
        host_column_name=host_column_name,
        host_values=host_values,
        include_unmatched_survey_polygons=include_unmatched_survey_polygons,
    )
    before_row = effort.loc[effort["year"] == int(year_before)]
    appendix_c_required = np.nan
    if not before_row.empty and np.isfinite(before_row["observed_prevalence"].iloc[0]):
        appendix_c_required = compute_n_appendix_c_cells(
            p0=float(before_row["observed_prevalence"].iloc[0]),
            delta=float(delta),
            corr=float(corr),
            alpha=float(alpha),
            power=float(power),
            population_size=total_pop if np.isfinite(total_pop) else None,
            design_effect=float(phi_info["phi"]),
        )
    effort["required_hosts"] = appendix_c_required if np.isfinite(appendix_c_required) else np.nan
    effort["observed_prop_population"] = effort["observed_hosts"] / total_pop if np.isfinite(total_pop) and total_pop > 0 else np.nan
    effort["required_prop_population"] = effort["required_hosts"] / total_pop if np.isfinite(total_pop) and total_pop > 0 else np.nan
    return effort[cols]


def build_regression_effort_from_hosts(
    year_summary_df: pd.DataFrame,
    year_before: int,
    year_after: int,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    appendix_e_target: float = DEFAULT_APPENDIX_E_TARGET,
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
    cols = ["year", "observed_hosts", "observed_prop_population", "required_hosts", "required_prop_population", "observed_prevalence"]
    if year_summary_df is None or year_summary_df.empty:
        return pd.DataFrame(columns=cols)
    work = year_summary_df.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.loc[work["year"].between(float(year_before), float(year_after), inclusive="both"), ["year", "surveyed_hosts_est", "estimated_prevalence"]].copy()
    work = work.rename(columns={"surveyed_hosts_est": "observed_hosts", "estimated_prevalence": "observed_prevalence"}).sort_values("year")
    if work.empty:
        return pd.DataFrame(columns=cols)
    total_pop = float(total_population_size) if total_population_size is not None and np.isfinite(total_population_size) and total_population_size > 0 else float("nan")
    baseline_year = int(year_before)
    design_pilot_year = int(pilot_year if pilot_year is not None else baseline_year)
    p0_row_source = year_summary_df.copy()
    p0_row_source["year"] = pd.to_numeric(p0_row_source["year"], errors="coerce")
    p0_row = p0_row_source.loc[p0_row_source["year"] == float(baseline_year)]
    if p0_row.empty or not np.isfinite(pd.to_numeric(p0_row["estimated_prevalence"], errors="coerce").iloc[0]):
        return pd.DataFrame(columns=cols)
    p0_hat = float(pd.to_numeric(p0_row["estimated_prevalence"], errors="coerce").iloc[0])
    phi_info = _resolve_quick_results_phi(
        layers=layers or {},
        host_gdf=host_gdf,
        pilot_year=int(design_pilot_year),
        host_density_km2=float(host_density_km2),
        sampling_mode=str(sampling_mode),
        cluster_inflation_mode=str(cluster_inflation_mode),
        within_cluster_cap=int(within_cluster_cap),
        fixed_phi=float(fixed_phi),
        positive_status_values=positive_status_values,
        negative_status_values=negative_status_values,
        status_column_name=status_column_name,
        host_column_name=host_column_name,
        host_values=host_values,
        include_unmatched_survey_polygons=include_unmatched_survey_polygons,
    )
    out = work.copy()
    out["required_hosts"] = np.nan
    out["observed_prop_population"] = out["observed_hosts"] / total_pop if np.isfinite(total_pop) and total_pop > 0 else np.nan
    n_initial = compute_n_initial_prev_ci_cells(
        p0_upper=DEFAULT_APPENDIX_E_INITIAL_UPPER,
        conf_level=DEFAULT_APPENDIX_E_INITIAL_CONF,
        width=DEFAULT_APPENDIX_E_INITIAL_WIDTH,
        population_size=total_pop if np.isfinite(total_pop) else None,
    )
    out.loc[out["year"] == float(baseline_year), "required_hosts"] = float(n_initial)
    followup_years = [int(y) for y in out.loc[(out["year"] > float(baseline_year)) & (out["observed_hosts"] > 0), "year"].astype(int).tolist()]
    if followup_years:
        t_vec = np.arange(1, len(followup_years) + 1, dtype=float)
        e_design = compute_n_total_appendix_e_hosts(
            pi0=float(p0_hat),
            piD=float(appendix_e_target),
            alpha=float(alpha),
            power=float(power),
            t_vec=t_vec,
            population_size=total_pop if np.isfinite(total_pop) else None,
            phi=float(phi_info["phi"]),
        )
        per_round = int(np.ceil(float(e_design["n_total"]) / max(len(followup_years), 1)))
        per_round = max(1, per_round)
        out.loc[out["year"].isin(followup_years), "required_hosts"] = float(per_round)
    out["required_prop_population"] = out["required_hosts"] / total_pop if np.isfinite(total_pop) and total_pop > 0 else np.nan
    return out[cols]


app = App(app_ui, server)
