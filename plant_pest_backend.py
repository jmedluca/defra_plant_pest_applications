"""Shared simulation code for the plant pest apps.

This file is the "engine room" used by the simulator app.  It does not build
the Shiny screens itself.  Instead, it builds fake host landscapes, creates a
fake epidemic, simulates surveys, and returns tables/plots for the app to show.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statistics import NormalDist
from shiny import render

APP_DIR = Path(__file__).resolve().parent
DEFRA_DATA_DIR = APP_DIR / "DEFRA_data"
TARGET_SITES_DIR = DEFRA_DATA_DIR / "Survey_data"
LARCH_HOST_DIR = DEFRA_DATA_DIR / "Host_coverage_data"
LARCH_HOST_PATH = LARCH_HOST_DIR / "larch_coverage.shp"

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_APPENDIX_C_DELTA = 0.03
DEFAULT_APPENDIX_C_CORR = 0.0
DEFAULT_APPENDIX_E_TARGET = 0.005
DEFAULT_APPENDIX_E_INITIAL_CONF = 0.95
DEFAULT_APPENDIX_E_INITIAL_WIDTH = 0.025
DEFAULT_APPENDIX_E_INITIAL_UPPER = 0.1

SIMX_DEFAULT_HOST_DENSITY_KM2 = 2500.0
SIMX_DEFAULT_SURVEY_HOSTS_PER_ROUND = 600
SIMX_DEFAULT_PSU_BLOCK_SIDE = 4
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

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    from scipy.spatial import cKDTree
except Exception: 
    cKDTree = None

# ============================================================================
# Shared setup, data loading, and formulas
# ============================================================================
#
# These helpers are used by the simulator app. They live here so the backend
# does not need to import from either Shiny app.


def render_grid(df: pd.DataFrame):
    return render.DataGrid(df, width="100%")


def _resolve_shapefile_path(path: Path, preferred_stem: str | None = None) -> Path:
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


def load_host_layer(host_path: Path) -> dict[str, Any]:
    host_path = _resolve_shapefile_path(host_path, preferred_stem=LARCH_HOST_PATH.stem)
    info: dict[str, Any] = {"available_file": host_path.exists(), "layer": None, "messages": []}
    if gpd is None:
        info["messages"].append("Geospatial packages are not installed, so the host layer could not be loaded.")
        return info
    if not host_path.exists():
        info["messages"].append(f"Missing host layer: {host_path}")
        return info
    try:
        info["layer"] = gpd.read_file(host_path)
    except Exception as exc:
        info["messages"].append(f"Failed to load host layer: {exc}")
    return info


def _clamp_open01(x: float, eps: float = 1e-8) -> float:
    return float(min(1.0 - eps, max(eps, x)))


def _logit(x: float) -> float:
    x = _clamp_open01(float(x))
    return float(np.log(x / (1.0 - x)))


def _inv_logit(eta: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0))))


def _apply_fpc(n: float, population_size: int | float | None) -> int:
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
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_beta = NormalDist().inv_cdf(power)
    p0 = _clamp_open01(float(p0))
    n = (2.0 * p0 * (1.0 - p0) * (1.0 - float(corr)) * (z_alpha + z_beta) ** 2) / (float(delta) ** 2)
    return _apply_fpc(float(design_effect) * n, population_size)


def compute_n_initial_prev_ci_cells(
    p0_upper: float = DEFAULT_APPENDIX_E_INITIAL_UPPER,
    conf_level: float = DEFAULT_APPENDIX_E_INITIAL_CONF,
    width: float = DEFAULT_APPENDIX_E_INITIAL_WIDTH,
    population_size: int | float | None = None,
) -> int:
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
    pi0 = _clamp_open01(float(pi0))
    pi_target = _clamp_open01(float(pi_target))
    t_vec = np.asarray(t_vec, dtype=float)
    if t_vec.size < 2:
        t_vec = np.array([0.0, 1.0], dtype=float)
    t_end = float(np.max(t_vec))
    beta_star = (_logit(pi_target) - _logit(pi0)) / max(t_end, 1e-8)
    eta = _logit(pi0) + beta_star * t_vec
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    w = p * (1.0 - p)
    X = np.column_stack([np.ones_like(t_vec), t_vec])
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


def _sim_clamp01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def _sim_logit_linear_curve(t: np.ndarray, p0: float, p_end: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(_logit(p0) + (_logit(p_end) - _logit(p0)) * t)))


def _sim_raw_linear_curve(t: np.ndarray, p0: float, p_end: float) -> np.ndarray:
    return _sim_clamp01(p0 + (p_end - p0) * t)


def _sim_logistic_transition_curve(t: np.ndarray, p0: float, p_end: float, k: float = 10.0, tm: float = 0.5) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-k * (t - tm)))
    s0 = 1.0 / (1.0 + np.exp(-k * (0.0 - tm)))
    s1 = 1.0 / (1.0 + np.exp(-k * (1.0 - tm)))
    w = np.zeros_like(t) if abs(s1 - s0) < 1e-10 else (s - s0) / (s1 - s0)
    return _sim_clamp01(p0 + (p_end - p0) * w)


def _sim_make_prevalence_schedule(
    curve_type: str,
    p0: float,
    p_end: float,
    prevalence_shift: float,
    n_rounds: int = 6,
) -> pd.DataFrame:
    t_scaled = np.linspace(0.0, 1.0, n_rounds)
    if curve_type == "logit_linear":
        prev = _sim_logit_linear_curve(t_scaled, p0, p_end)
    elif curve_type == "linear":
        prev = _sim_raw_linear_curve(t_scaled, p0, p_end)
    else:
        prev = _sim_logistic_transition_curve(t_scaled, p0, p_end)
    prev = _sim_clamp01(prev + float(prevalence_shift))
    return pd.DataFrame(
        {
            "round_id": np.arange(n_rounds, dtype=int),
            "year": t_scaled * 5.0,
            "t": t_scaled * 5.0,
            "t_scaled": t_scaled,
            "true_prev": prev,
        }
    )


def _sim_overlap_config(overlap_type: str, overlap_prop_rot: float = 0.7) -> dict[str, float]:
    if overlap_type == "cross_sectional":
        return {"overlap_prop": 0.0, "corr_c": 0.0}
    if overlap_type == "rotating_panel":
        return {"overlap_prop": float(overlap_prop_rot), "corr_c": 0.95 * float(overlap_prop_rot)}
    if overlap_type == "longitudinal":
        return {"overlap_prop": 1.0, "corr_c": 0.95}
    raise ValueError(f"Unsupported overlap_type: {overlap_type}")


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
    cov = np.linalg.pinv(XtW @ X)
    return beta, cov


def _sim_fit_logistic_counts(t: np.ndarray, y: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, cov = _logistic_irls(np.asarray(t, dtype=float), np.asarray(y, dtype=float), np.asarray(n, dtype=float))
    return np.asarray(beta, dtype=float), np.asarray(cov, dtype=float)


def _sim_predict_logistic(beta: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    eta = beta[0] + beta[1] * t_arr
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


def _sim_true_beta_from_schedule(schedule: pd.DataFrame) -> float:
    t = schedule["t"].to_numpy(dtype=float)
    y = np.round(schedule["true_prev"].to_numpy(dtype=float) * 100000.0)
    n = np.full_like(y, 100000.0, dtype=float)
    beta, _ = _sim_fit_logistic_counts(t, y, n)
    return float(beta[1])


def _simx_estimate_design_effect(sampled_hosts_df: pd.DataFrame, planned_m: int | None = None) -> dict[str, float]:
    if sampled_hosts_df is None or sampled_hosts_df.empty:
        m_observed = 1.0
        m = float(planned_m) if planned_m is not None and planned_m > 0 else m_observed
        return {"phi": 1.0, "rho": 0.0, "m": m, "m_observed": m_observed, "estimator": "degenerate"}
    cluster_col = "cluster_id" if "cluster_id" in sampled_hosts_df.columns else "cell_id"
    work = sampled_hosts_df.loc[:, [cluster_col, "detected"]].copy()
    work["survey_cell_id"] = work[cluster_col].astype(str)
    tab = work["survey_cell_id"].value_counts()
    m_observed = float(tab.mean()) if len(tab) else 1.0
    m = float(planned_m) if planned_m is not None and planned_m > 0 else m_observed
    if len(tab) < 2 or int(work["detected"].sum()) in (0, len(work)):
        return {"phi": 1.0, "rho": 0.0, "m": m, "m_observed": m_observed, "estimator": "degenerate"}
    cluster_stats = work.groupby("survey_cell_id")["detected"].agg(["mean", "size"])
    k = int(len(cluster_stats))
    n = int(len(work))
    if k < 2 or n <= k:
        return {"phi": 1.0, "rho": 0.0, "m": m, "m_observed": m_observed, "estimator": "degenerate"}
    grand_mean = float(work["detected"].mean())
    ssb = float(np.sum(cluster_stats["size"].to_numpy(dtype=float) * (cluster_stats["mean"].to_numpy(dtype=float) - grand_mean) ** 2))
    msb = ssb / max(k - 1, 1)
    work2 = work.merge(cluster_stats[["mean"]], left_on="survey_cell_id", right_index=True, how="left")
    ssw = float(np.sum((work2["detected"].to_numpy(dtype=float) - work2["mean"].to_numpy(dtype=float)) ** 2))
    msw = ssw / max(n - k, 1)
    rho = 0.0
    denom = msb + (m_observed - 1.0) * msw
    if np.isfinite(denom) and denom > 0:
        rho = max(0.0, min(1.0, (msb - msw) / denom))
    phi = max(1.0, 1.0 + (m - 1.0) * rho)
    return {"phi": float(phi), "rho": float(rho), "m": float(m), "m_observed": float(m_observed), "estimator": "anova_icc"}


# ============================================================================
# Simulator display helpers
# ============================================================================


def _simx_infection_round_summary_table(epidemic: dict[str, Any] | None, round_index: int) -> pd.DataFrame:
    # Summarise one year/round of the epidemic.
    # This gives users a quick check that the infection landscape looks sensible.
    if epidemic is None or not epidemic.get("rounds"):
        return pd.DataFrame(columns=["metric", "value"])
    # Keep the requested round inside the valid range, so the app does not crash
    # if a slider still points to an old round number.
    round_index = max(0, min(int(round_index), len(epidemic["rounds"]) - 1))
    round_df = epidemic["rounds"][round_index]
    # Work out which clusters are most infected in this round.
    cluster_prev = (
        round_df.groupby("cluster_id", as_index=False)
        .agg(cluster_prev=("cluster_prev", "first"), cluster_size=("cluster_size", "first"))
        .sort_values("cluster_prev", ascending=False)
    )
    return pd.DataFrame(
        [
            {"metric": "Round", "value": int(round_df["round_id"].iloc[0]) + 1},
            {"metric": "Year", "value": round(float(round_df["year"].iloc[0]), 3)},
            {"metric": "True prevalence", "value": round(float(round_df["true_prev"].iloc[0]), 6)},
            {"metric": "Initial hotspot clusters", "value": int(len(epidemic.get("hotspots", [])))},
            {"metric": "Neighbouring clusters used for pressure", "value": 5},
            {"metric": "Infected hosts", "value": int(round_df["infected_count"].sum())},
            {"metric": "Active clusters", "value": int(round_df["cluster_id"].nunique())},
            {"metric": "Infected clusters", "value": int((cluster_prev["cluster_prev"] > 0).sum())},
            {"metric": "Max cluster prevalence", "value": round(float(cluster_prev["cluster_prev"].max()) if not cluster_prev.empty else 0.0, 6)},
            {"metric": "Mean infected pressure", "value": round(float(pd.to_numeric(round_df["neighbor_infected"], errors="coerce").fillna(0.0).mean()), 6)},
        ]
    )


def _simx_infection_hotspot_table(epidemic: dict[str, Any] | None, round_index: int) -> pd.DataFrame:
    # Show the most infected clusters for the selected round.
    # These are labelled as hotspots because they are the places with the
    # highest simulated infection pressure or prevalence.
    if epidemic is None or not epidemic.get("rounds"):
        return pd.DataFrame(columns=["cluster_id", "cluster_prev", "infected_hosts", "cluster_size", "cluster_x", "cluster_y"])
    round_index = max(0, min(int(round_index), len(epidemic["rounds"]) - 1))
    round_df = epidemic["rounds"][round_index]
    hotspot_df = (
        round_df.groupby("cluster_id", as_index=False)
        .agg(
            cluster_prev=("cluster_prev", "first"),
            infected_hosts=("infected_count", "sum"),
            cluster_size=("cluster_size", "first"),
            cluster_x=("cluster_x", "first"),
            cluster_y=("cluster_y", "first"),
        )
        .sort_values(["cluster_prev", "infected_hosts"], ascending=[False, False])
        .head(10)
        .reset_index(drop=True)
    )
    if hotspot_df.empty:
        return hotspot_df
    hotspot_df["cluster_prev"] = hotspot_df["cluster_prev"].astype(float).round(6)
    hotspot_df["cluster_x"] = hotspot_df["cluster_x"].astype(float).round(1)
    hotspot_df["cluster_y"] = hotspot_df["cluster_y"].astype(float).round(1)
    return hotspot_df


def _simx_plot_cluster_size_distribution(host_landscape: dict[str, Any] | None):
    # Plot how many clusters are small, medium, or large.
    # This helps users see whether the landscape is dominated by many small
    # sites or a few very large sites.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    freq = _simx_cluster_distribution_table(host_landscape)
    if freq.empty:
        ax.text(0.5, 0.5, "Generate the host landscape to view the cluster size distribution.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    sizes = freq["cells_per_cluster"].to_numpy(dtype=float)
    if len(sizes) == 0:
        ax.text(0.5, 0.5, "No cluster sizes are available.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    bins = min(20, max(5, len(np.unique(sizes))))
    ax.hist(np.repeat(freq["cells_per_cluster"].to_numpy(dtype=float), freq["n_clusters"].to_numpy(dtype=int)), bins=bins, color="#2ca25f", edgecolor="white")
    ax.set_title("Cluster size distribution")
    ax.set_xlabel("Hosts per cluster")
    ax.set_ylabel("Number of clusters")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig


def _simx_plot_macro_host_landscape(host_landscape: dict[str, Any] | None):
    # Plot the host landscape as a bubble chart.
    # Each bubble is one host cluster, and larger bubbles contain more hosts.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    if host_landscape is None or host_landscape.get("clusters") is None:
        ax.text(0.5, 0.5, "Generate the host landscape to view the host clusters.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    clusters = host_landscape["clusters"].copy()
    if clusters.empty:
        ax.text(0.5, 0.5, "No host clusters are available.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    host_counts = np.maximum(pd.to_numeric(clusters["host_count"], errors="coerce").fillna(1.0).to_numpy(dtype=float), 1.0)
    bubble_area = 8.0 + 52.0 * np.sqrt(host_counts / max(float(host_counts.max()), 1.0))
    ax.scatter(
        clusters["x"].to_numpy(dtype=float),
        clusters["y"].to_numpy(dtype=float),
        s=bubble_area,
        color="#2b8cbe",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.72,
    )
    ax.set_title("Macro host landscape")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def _simx_plot_macro_survey_landscape(round_df: pd.DataFrame | None, title: str):
    # Plot where simulated survey effort went in one round.
    # Darker blue means more host samples were taken from that cluster.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    if round_df is None or round_df.empty:
        ax.text(0.5, 0.5, title, ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    cluster_df = (
        round_df.groupby("cluster_id", as_index=False)
        .agg(
            cluster_x=("cluster_x", "first"),
            cluster_y=("cluster_y", "first"),
            cluster_size=("cluster_size", "first"),
            sampled_n=("sampled_n", "sum"),
        )
        .sort_values("sampled_n", ascending=False)
    )
    sc = ax.scatter(
        cluster_df["cluster_x"].to_numpy(dtype=float),
        cluster_df["cluster_y"].to_numpy(dtype=float),
        c=cluster_df["sampled_n"].to_numpy(dtype=float),
        cmap="Blues",
        s=np.clip(np.sqrt(np.maximum(cluster_df["cluster_size"].to_numpy(dtype=float), 1.0)) * 10.0, 10.0, 180.0),
        linewidths=0.2,
        edgecolors="white",
        alpha=0.95,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.ax.set_ylabel("Samples taken in cluster", rotation=90)
    ax.set_title(title)
    ax.set_xlabel("Projected x")
    ax.set_ylabel("Projected y")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def _simx_fit_methods(epidemic: dict[str, Any], surveys: dict[str, Any], model_form: str = "logistic") -> dict[str, Any]:
    # Fit the two estimation methods to the simulated survey results.
    del model_form
    schedule = epidemic["schedule"].copy()
    survey_df_e = surveys.get("appendix_e", surveys.get("static", {"survey_df": pd.DataFrame()}))["survey_df"].copy()
    survey_df_c = surveys.get("appendix_c", surveys.get("static", {"survey_df": pd.DataFrame()}))["survey_df"].copy()
    if survey_df_e.empty:
        return {
            "survey_df": survey_df_e,
            "appendix_c_survey_df": survey_df_c,
            "appendix_e_curve": pd.DataFrame(),
            "appendix_c_curve": pd.DataFrame(),
        }
    beta, cov = _sim_fit_logistic_counts(
        survey_df_e["year"].to_numpy(dtype=float),
        survey_df_e["y"].to_numpy(dtype=float),
        survey_df_e["n"].to_numpy(dtype=float),
    )
    del cov
    curve_year = np.linspace(float(schedule["year"].min()), float(schedule["year"].max()), 201)
    curve_prev = _sim_predict_logistic(beta, curve_year)
    appendix_e_curve = pd.DataFrame({"year": curve_year, "appendix_e_prev": curve_prev})
    c_source = survey_df_c.loc[survey_df_c["n"] > 0].copy() if not survey_df_c.empty else pd.DataFrame()
    if len(c_source) >= 2:
        c_points = c_source.iloc[[0, -1]].copy()
    else:
        c_points = pd.DataFrame(columns=["year", "prev_hat"])
    appendix_c_curve = pd.DataFrame(
        {
            "year": c_points["year"].to_numpy(dtype=float) if not c_points.empty else np.array([], dtype=float),
            "appendix_c_prev": c_points["prev_hat"].to_numpy(dtype=float) if not c_points.empty else np.array([], dtype=float),
        }
    )
    return {
        "survey_df": survey_df_e,
        "appendix_c_survey_df": survey_df_c,
        "appendix_e_curve": appendix_e_curve,
        "appendix_c_curve": appendix_c_curve,
    }


def _simx_plot_method_comparison(epidemic: dict[str, Any] | None, fit_result: dict[str, Any] | None, regression_model: str | None = None):
    # Plot true prevalence against the estimates from the two methods.
    # This is the main visual check of whether the methods recovered the truth.
    del regression_model
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    if epidemic is None or fit_result is None:
        ax.text(0.5, 0.5, "Fit the methods after simulating surveys.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    schedule = epidemic["schedule"]
    survey_df = fit_result.get("survey_df", pd.DataFrame())
    survey_df_c = fit_result.get("appendix_c_survey_df", pd.DataFrame())
    ax.plot(schedule["year"], schedule["true_prev"], color="#1b7837", linewidth=2.2, label="True prevalence")
    if not survey_df.empty:
        ax.scatter(survey_df["year"], survey_df["prev_hat"], color="#d95f02", s=34, label="Regression Method estimates")
    if not survey_df_c.empty:
        c_points = survey_df_c.loc[survey_df_c["n"] > 0].copy()
        if not c_points.empty:
            ax.scatter(c_points["year"], c_points["prev_hat"], color="#7b3294", s=34, marker="s", label="Change Method survey points")
    appendix_e_curve = fit_result.get("appendix_e_curve", pd.DataFrame())
    if not appendix_e_curve.empty:
        ax.plot(appendix_e_curve["year"], appendix_e_curve["appendix_e_prev"], color="#2b8cbe", linewidth=2.0, label="Regression Method")
    appendix_c_curve = fit_result.get("appendix_c_curve", pd.DataFrame())
    if not appendix_c_curve.empty:
        ax.plot(appendix_c_curve["year"], appendix_c_curve["appendix_c_prev"], color="#7b3294", linewidth=2.0, linestyle="--", marker="o", label="Change Method")
    ax.set_title("True prevalence and fitted method results")
    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Prevalence")
    ax.set_ylim(0, min(1.0, max(0.05, float(schedule["true_prev"].max()) * 1.2)))
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Simulator
#
#   - A cluster is a field, woodland, crop block, or similar host area.
#   - A host ID is one individual plant/tree inside a cluster.
#   - The epidemic stores whether each host ID is infected in each round.
#   - Surveys choose host IDs and then record how many sampled hosts are positive.


def _simx_scale01_local(values: np.ndarray) -> np.ndarray:
    # Put any numeric signal onto a 0 to 1 scale.
    # This is useful when combining different risk signals, such as distance
    # pressure and hotspot pressure, which may originally be on different scales.
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _simx_sample_ids(rng: np.random.Generator, ids: np.ndarray, n: int, weights: np.ndarray | None = None) -> np.ndarray:
    # Pick host IDs or cluster IDs without replacement.
    # If weights are supplied, larger weights mean that item is more likely to be chosen.
    ids = np.asarray(ids, dtype=int)
    n = min(max(0, int(n)), len(ids))
    if n <= 0:
        return np.asarray([], dtype=int)
    if weights is None:
        return rng.choice(ids, size=n, replace=False)
    w = np.asarray(weights, dtype=float)
    if len(w) != len(ids) or not np.isfinite(w).all() or float(w.sum()) <= 0:
        return rng.choice(ids, size=n, replace=False)
    return rng.choice(ids, size=n, replace=False, p=w / float(w.sum()))


def _simx_allocate_counts_capped_local(total: int, capacity: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Split a total number across clusters, but never give a cluster more than
    # its remaining capacity. For example, a cluster with 3 uninfected hosts
    # cannot receive 10 new infections.
    total = int(max(0, total))
    capacity = np.maximum(0, np.asarray(capacity, dtype=int))
    out = np.zeros(len(capacity), dtype=int)
    remaining = min(total, int(capacity.sum()))
    if remaining <= 0:
        return out
    weights = np.maximum(0.0, np.asarray(weights, dtype=float))
    while remaining > 0:
        # Only clusters with spare capacity can receive more items.
        available = capacity - out
        mask = available > 0
        if not np.any(mask):
            break
        # The weight decides which available clusters are favoured.
        # Multiplying by capacity means a large cluster can receive more items.
        w = weights[mask] * available[mask].astype(float)
        if float(w.sum()) <= 0 or not np.isfinite(w).all():
            w = available[mask].astype(float)
        draw = rng.multinomial(remaining, w / float(w.sum()))
        draw = np.minimum(draw, available[mask])
        idx = np.flatnonzero(mask)
        out[idx] += draw
        new_remaining = total - int(out.sum())
        if new_remaining == remaining:
            j = rng.choice(idx, p=available[mask] / float(available[mask].sum()))
            out[int(j)] += 1
            new_remaining -= 1
        remaining = max(0, new_remaining)
    return out


def _simx_host_counts_from_area(total_hosts: int, areas: np.ndarray) -> np.ndarray:
    # Convert cluster areas into host counts.
    # Bigger clusters get more hosts, and all counts are adjusted so the final
    # total equals the requested total host population.
    areas = np.maximum(0.0, np.asarray(areas, dtype=float))
    if float(areas.sum()) <= 0:
        weights = np.full(len(areas), 1.0 / max(len(areas), 1))
    else:
        weights = areas / float(areas.sum())
    raw = weights * int(total_hosts)
    # Start with rounded-down counts, then adjust the left-over hosts below.
    counts = np.floor(raw).astype(int)
    if len(counts) == 0:
        return counts
    counts[counts < 1] = 1
    diff = int(total_hosts) - int(counts.sum())
    frac_order = np.argsort(raw - np.floor(raw))[::-1]
    if diff > 0:
        for idx in np.resize(frac_order, diff):
            counts[int(idx)] += 1
    elif diff < 0:
        removable = np.where(counts > 1)[0]
        for idx in np.resize(removable, abs(diff)):
            if counts[int(idx)] > 1:
                counts[int(idx)] -= 1
    return counts


def _simx_make_host_id_landscape(clusters: pd.DataFrame, total_area_km2: float, host_density_km2: float, source: str) -> dict[str, Any]:
    # Build the main host landscape used by the simulator.
    # Each row in clusters is like a field/forest/site.
    # Each host gets a unique host_id and belongs to one cluster_id.
    clusters = clusters.copy().reset_index(drop=True)
    clusters["cluster_id"] = np.arange(len(clusters), dtype=int)
    total_hosts = max(1, int(round(float(total_area_km2) * float(host_density_km2))))
    clusters["host_count"] = _simx_host_counts_from_area(total_hosts, clusters["area_km2"].to_numpy(dtype=float))
    clusters["cluster_size"] = clusters["host_count"].astype(int)
    clusters["simulated_cells"] = 0
    clusters["active"] = True
    clusters["cluster_x"] = clusters["x"].astype(float)
    clusters["cluster_y"] = clusters["y"].astype(float)
    # This array says which cluster each host ID belongs to.
    # Example: [0, 0, 0, 1, 1] means hosts 0-2 are in cluster 0 and hosts 3-4 are in cluster 1.
    cluster_id_by_host = np.repeat(clusters["cluster_id"].to_numpy(dtype=int), clusters["host_count"].to_numpy(dtype=int))
    host_id = np.arange(len(cluster_id_by_host), dtype=int)
    hosts = pd.DataFrame({"host_id": host_id, "cluster_id": cluster_id_by_host})
    # Store the host IDs inside each cluster so sampling can be fast later.
    hosts_by_cluster = {
        int(cid): host_id[cluster_id_by_host == int(cid)]
        for cid in clusters["cluster_id"].to_numpy(dtype=int)
    }
    # Compatibility table for older plotting/table helpers. These rows are clusters, not model cells.
    cells = clusters.rename(columns={"area_km2": "cluster_area_km2"}).copy()
    cells["cell_id"] = cells["cluster_id"].astype(int)
    cells["host_count"] = cells["host_count"].astype(int)
    cells["active"] = True
    return {
        "model": "host_id_cluster",
        "clusters": clusters,
        "cells": cells,
        "hosts": hosts,
        "hosts_by_cluster": hosts_by_cluster,
        "cluster_id_by_host": cluster_id_by_host,
        "total_hosts": int(len(host_id)),
        "simulated_total_hosts": int(len(host_id)),
        "estimated_total_hosts": int(total_hosts),
        "total_area_km2": float(total_area_km2),
        "host_density_km2": float(host_density_km2),
        "landscape_source": source,
    }


def _simx_make_macro_host_landscape_synthetic(
    n_clusters: int,
    total_area_km2: float,
    host_density_km2: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    # Create a fake landscape from scratch.
    # We first make cluster centre points, then give each cluster a random area,
    # then assign host IDs to clusters in proportion to area.
    n_clusters = max(1, int(n_clusters))
    # Put the synthetic clusters inside a square with roughly the requested area.
    side_km = np.sqrt(max(float(total_area_km2), 1e-9))
    # The gamma distribution gives uneven cluster sizes, which is closer to real
    # landscapes than giving every cluster the same area.
    areas = rng.gamma(shape=0.7, scale=1.0, size=n_clusters)
    areas = areas / float(areas.sum()) * float(total_area_km2)
    clusters = pd.DataFrame(
        {
            "x": rng.uniform(0.0, side_km, size=n_clusters),
            "y": rng.uniform(0.0, side_km, size=n_clusters),
            "area_km2": areas,
        }
    )
    host = _simx_make_host_id_landscape(clusters, total_area_km2, host_density_km2, "Synthetic clusters")
    host["requested_n_clusters"] = int(n_clusters)
    return host


def _simx_make_macro_host_landscape_from_polygons(host_gdf: Any | None, host_density_km2: float) -> dict[str, Any]:
    # Convert real host polygons into the same structure as the synthetic landscape.
    # Each polygon becomes one cluster, and host count is estimated from area.
    if host_gdf is None or len(host_gdf) == 0:
        raise ValueError("The host polygon layer is empty.")
    work = host_gdf.to_crs(27700) if getattr(host_gdf, "crs", None) else host_gdf
    # Areas must be measured in metres, so polygons are converted to British
    # National Grid before converting area into km2.
    area_km2 = pd.to_numeric(work.geometry.area, errors="coerce").fillna(0.0).to_numpy(dtype=float) / 1_000_000.0
    centroids = work.geometry.centroid
    clusters = pd.DataFrame(
        {
            "x": centroids.x.to_numpy(dtype=float),
            "y": centroids.y.to_numpy(dtype=float),
            "area_km2": np.maximum(area_km2, 0.0),
        }
    )
    clusters = clusters.loc[clusters["area_km2"] > 0].reset_index(drop=True)
    if clusters.empty:
        raise ValueError("The host polygon layer has no positive-area polygons.")
    total_area_km2 = float(clusters["area_km2"].sum())
    return _simx_make_host_id_landscape(clusters, total_area_km2, host_density_km2, "Host polygons")


def _simx_cluster_knn_from_clusters(clusters: pd.DataFrame, k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Find the nearest neighbouring clusters.
    # This lets infection pressure spread more strongly to nearby clusters than far away ones.
    coords = clusters[["x", "y"]].to_numpy(dtype=float)
    if len(coords) <= 1:
        return coords, np.empty((len(coords), 0), dtype=int), np.empty((len(coords), 0), dtype=float)
    if cKDTree is None:
        raise RuntimeError("Spatial clustering requires scipy. Run Setup.bat to install the current requirements.")
    n_neighbors = min(max(1, int(k)), len(coords) - 1)
    distances, indices = cKDTree(coords).query(coords, k=n_neighbors + 1)
    return coords, np.asarray(indices)[:, 1:].astype(int), np.asarray(distances)[:, 1:].astype(float)


def _simx_draw_macro_hotspot_count(level: str, n_clusters: int, rng: np.random.Generator) -> int:
    # Draw a small random number of starting hotspots.
    # Low clustering has more hotspots; high clustering has fewer hotspots.
    mean_sd = {"low": (14.0, 3.0), "medium": (7.0, 2.0), "high": (3.0, 1.0)}
    mean, sd = mean_sd.get(level, (0.0, 0.0))
    if mean <= 0:
        return 0
    return max(1, min(int(n_clusters), int(round(rng.normal(mean, sd)))))


def _simx_make_round_clusters(
    clusters: pd.DataFrame,
    infected_counts: np.ndarray,
    neighbor_pressure: np.ndarray,
    round_id: int,
    year: float,
    true_prev: float,
) -> pd.DataFrame:
    # Make the table used for plotting one epidemic round.
    # Each row is a cluster with its current infected host count and prevalence.
    out = clusters.copy()
    out["round_id"] = int(round_id)
    out["year"] = float(year)
    out["true_prev"] = float(true_prev)
    out["infected_count"] = np.asarray(infected_counts, dtype=int)
    out["cluster_prev"] = out["infected_count"].astype(float) / np.maximum(out["host_count"].astype(float), 1.0)
    out["neighbor_infected"] = np.asarray(neighbor_pressure, dtype=float)
    out["sampled_n"] = 0
    return out


def _simx_make_macro_epidemic(
    host_landscape: dict[str, Any],
    prevalence_schedule: pd.DataFrame,
    infection_clustering_level: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    # Build the infection history across all survey rounds.
    # The true prevalence curve tells us how many hosts should be infected each round.
    # The clustering setting controls where those infected hosts are placed.
    clusters = host_landscape["clusters"].copy().reset_index(drop=True)
    host_counts = clusters["host_count"].to_numpy(dtype=int)
    total_hosts = int(host_counts.sum())
    # This tells us, for every host ID, which cluster it sits in.
    cluster_for_host = np.asarray(host_landscape["cluster_id_by_host"], dtype=int)
    hosts_by_cluster = host_landscape["hosts_by_cluster"]
    level = str(infection_clustering_level)
    coords, neighbor_idx, neighbor_dist = _simx_cluster_knn_from_clusters(clusters, k=5)
    positive_dist = neighbor_dist[np.isfinite(neighbor_dist) & (neighbor_dist > 0)]
    distance_scale = float(np.median(positive_dist)) if positive_dist.size else 1.0
    cfg = {
        # random_mix controls how much of the allocation remains broad/random.
        # Low clustering is intentionally close to random spread; high clustering is dominated by
        # existing infected clusters, nearby infected clusters, and initial hotspot pressure.
        "random": {"background": 1.0, "w_cluster": 0.0, "w_between": 0.0, "w_seed": 0.0, "selection_power": 1.0, "random_mix": 1.0},
        "low": {"background": 0.25, "w_cluster": 0.35, "w_between": 0.18, "w_seed": 0.15, "selection_power": 1.25, "random_mix": 0.55},
        "medium": {"background": 0.04, "w_cluster": 0.9, "w_between": 0.55, "w_seed": 0.4, "selection_power": 2.1, "random_mix": 0.25},
        "high": {"background": 0.003, "w_cluster": 1.8, "w_between": 1.1, "w_seed": 0.85, "selection_power": 3.8, "random_mix": 0.06},
    }[level]
    infected = np.zeros(total_hosts, dtype=bool)
    # infected is one True/False value per host ID.
    # True means the host is infected in the current round.
    seed_signal = np.zeros(len(clusters), dtype=float)
    hotspots = pd.DataFrame(columns=["x", "y"])
    round_tables: list[pd.DataFrame] = []
    infected_by_round: list[np.ndarray] = []
    achieved_prevalence: list[float] = []

    def infected_counts_by_cluster() -> np.ndarray:
        # Count how many infected host IDs are currently inside each cluster.
        return np.bincount(cluster_for_host, weights=infected.astype(float), minlength=len(clusters)).astype(int)

    def cluster_pressure() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Calculate how attractive each cluster is for new infections.
        # The score combines existing prevalence, nearby infected clusters,
        # hotspot pressure, and a random/background component.
        inf_counts = infected_counts_by_cluster()
        prev = inf_counts.astype(float) / np.maximum(host_counts.astype(float), 1.0)
        if neighbor_idx.shape[1] == 0:
            between = np.zeros(len(clusters), dtype=float)
        else:
            # Nearby infected clusters add more pressure than distant infected clusters.
            # Only the five nearest neighbours are used to keep the calculation fast.
            dist_weight = np.exp(-neighbor_dist / max(distance_scale, 1e-12))
            between = np.sum(inf_counts[neighbor_idx] * dist_weight, axis=1)
            between = _simx_scale01_local(between)
        pressure = (
            float(cfg["background"])
            + float(cfg["w_cluster"]) * prev
            + float(cfg["w_between"]) * between
            + float(cfg["w_seed"]) * seed_signal
        )
        pressure = np.maximum(1e-12, pressure) ** float(cfg["selection_power"])
        random_mix = float(cfg.get("random_mix", 0.0))
        if random_mix > 0:
            pressure = (1.0 - random_mix) * pressure + random_mix * np.ones_like(pressure, dtype=float)
        return inf_counts, between, pressure

    def add_infections(add_n: int) -> None:
        # Add infection to host IDs until we reach the target prevalence for this round.
        # The cluster pressure decides which clusters receive the new infections.
        nonlocal infected
        add_n = int(max(0, add_n))
        if add_n <= 0:
            return
        inf_counts, _, pressure = cluster_pressure()
        capacity = host_counts - inf_counts
        if level == "random":
            pressure = np.ones(len(clusters), dtype=float)
        add_counts = _simx_allocate_counts_capped_local(add_n, capacity, pressure, rng)
        for cid, count in enumerate(add_counts):
            if int(count) <= 0:
                continue
            ids = hosts_by_cluster[int(cid)]
            candidates = ids[~infected[ids]]
            chosen = _simx_sample_ids(rng, candidates, int(count))
            infected[chosen] = True

    def remove_infections(rem_n: int) -> None:
        # Remove infection status when the true prevalence curve is going down.
        # This is a simple abstraction for eradication or infection decline.
        nonlocal infected
        rem_n = int(max(0, rem_n))
        if rem_n <= 0:
            return
        inf_counts, _, pressure = cluster_pressure()
        if level == "random":
            remove_weights = np.ones(len(clusters), dtype=float)
        else:
            remove_weights = 1.0 / np.maximum(pressure, 1e-12)
        rem_counts = _simx_allocate_counts_capped_local(rem_n, inf_counts, remove_weights, rng)
        for cid, count in enumerate(rem_counts):
            if int(count) <= 0:
                continue
            ids = hosts_by_cluster[int(cid)]
            candidates = ids[infected[ids]]
            chosen = _simx_sample_ids(rng, candidates, int(count))
            infected[chosen] = False

    for i, row in enumerate(prevalence_schedule.itertuples(index=False)):
        # For every survey round, force the total infected host count to match
        # the requested true prevalence as closely as possible.
        target_prev = float(np.clip(row.true_prev, 0.0, 1.0))
        target_n = int(round(target_prev * total_hosts))
        if i == 0 and level != "random" and target_n > 0:
            # At the first round, clustered epidemics start from a small number
            # of hotspot clusters.  Later infections then tend to grow around
            # those hotspots.
            n_hotspots = _simx_draw_macro_hotspot_count(level, len(clusters), rng)
            hotspot_positions = rng.choice(len(clusters), size=n_hotspots, replace=False)
            hotspot_coords = coords[hotspot_positions]
            delta = coords[:, None, :] - hotspot_coords[None, :, :]
            min_dist = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
            spread = {"low": 3.0, "medium": 2.0, "high": 1.25}[level]
            seed_signal = np.exp(-min_dist / max(distance_scale * spread, 1e-12))
            seed_signal[hotspot_positions] = 1.0
            seed_signal = _simx_scale01_local(seed_signal)
            hotspots = pd.DataFrame({"x": hotspot_coords[:, 0], "y": hotspot_coords[:, 1]})
        current_n = int(infected.sum())
        # Add or remove infected host IDs so the total prevalence follows the
        # requested true prevalence curve.
        if target_n > current_n:
            add_infections(target_n - current_n)
        elif target_n < current_n:
            remove_infections(current_n - target_n)
        inf_counts, between_pressure, _ = cluster_pressure()
        actual_prev = float(infected.mean()) if total_hosts else 0.0
        achieved_prevalence.append(actual_prev)
        infected_by_round.append(infected.copy())
        round_tables.append(
            _simx_make_round_clusters(
                clusters=clusters,
                infected_counts=inf_counts,
                neighbor_pressure=between_pressure,
                round_id=int(row.round_id),
                year=float(row.year),
                true_prev=actual_prev,
            )
        )

    achieved_schedule = prevalence_schedule.copy()
    achieved_schedule["target_prev"] = achieved_schedule["true_prev"].astype(float)
    achieved_schedule["true_prev"] = np.asarray(achieved_prevalence, dtype=float)
    hosts = host_landscape["hosts"].copy()
    return {
        "model": "host_id_cluster",
        "rounds": round_tables,
        "schedule": achieved_schedule,
        "hosts": hosts,
        "hosts_by_cluster": hosts_by_cluster,
        "cluster_id_by_host": cluster_for_host,
        "infected_by_round": infected_by_round,
        "hotspots": hotspots,
        "infection_clustering": {"level": level, **cfg},
    }


def _simx_macro_simulate_surveys(
    epidemic: dict[str, Any],
    n_hosts_per_round: int | list[int] | np.ndarray,
    sampling_mode: str,
    overlap_type: str,
    targeted_mode: str,
    within_cluster_cap: int,
    overlap_prop: float,
    targeting_level: str,
    rng: np.random.Generator,
    method_sens: float,
) -> dict[str, Any]:
    # Simulate field surveys by sampling host IDs.
    # SRS samples host IDs from the whole population.
    # MSS samples clusters first, then host IDs inside those clusters.
    # Overlap reuses some previously sampled host IDs.
    # Targeting makes infected hosts more likely to be sampled.
    rounds = epidemic["rounds"]
    hosts = epidemic["hosts"]
    hosts_by_cluster = epidemic["hosts_by_cluster"]
    total_hosts = len(hosts)
    host_cluster = hosts["cluster_id"].to_numpy(dtype=int)
    n_rounds = len(rounds)
    if np.isscalar(n_hosts_per_round):
        sizes = [int(n_hosts_per_round)] * n_rounds
    else:
        sizes = [int(x) for x in list(n_hosts_per_round)]
        sizes = (sizes + [sizes[-1] if sizes else 0] * n_rounds)[:n_rounds]
    overlap_type = str(overlap_type)
    targeting_level = str(targeting_level)
    targeted = str(targeted_mode) != "none" and targeting_level != "random"
    target_mult = 1.0 if not targeted else {"low": 2.0, "medium": 5.0, "high": 10.0}.get(targeting_level, 5.0)
    overlap = 1.0 if overlap_type == "longitudinal" else float(overlap_prop if overlap_type == "rotating_panel" else 0.0)
    survey_rows: list[dict[str, Any]] = []
    round_tables: list[pd.DataFrame] = []
    sampled_hosts_by_round: list[pd.DataFrame] = []
    previous_sample: np.ndarray = np.asarray([], dtype=int)

    for ridx, round_df in enumerate(rounds):
        # Work through one survey round at a time and store both the summary
        # result and the sampled host IDs.
        n = min(max(0, int(sizes[ridx])), total_hosts)
        infected = np.asarray(epidemic["infected_by_round"][ridx], dtype=bool)
        reused = np.asarray([], dtype=int)
        if ridx > 0 and overlap > 0 and previous_sample.size > 0 and n > 0:
            # Overlap means re-sampling some of the same host IDs as last round.
            reuse_n = min(int(round(n * overlap)), len(previous_sample))
            reused = _simx_sample_ids(rng, previous_sample, reuse_n)
        new_n = max(0, n - len(reused))
        blocked = set(reused.tolist())

        if str(sampling_mode) == "multistage":
            # Multistage sampling: choose clusters, then take host IDs from those clusters.
            m = max(1, int(within_cluster_cap))
            n_clusters = max(1, int(np.ceil(new_n / m))) if new_n > 0 else 0
            cluster_df = round_df.copy().reset_index(drop=True)
            if targeted:
                cluster_weights = (1.0 + target_mult * cluster_df["cluster_prev"].to_numpy(dtype=float)) * np.maximum(cluster_df["host_count"].to_numpy(dtype=float), 1.0) ** 0.25
            else:
                cluster_weights = np.ones(len(cluster_df), dtype=float)
            chosen_clusters = _simx_sample_ids(rng, cluster_df["cluster_id"].to_numpy(dtype=int), min(n_clusters, len(cluster_df)), cluster_weights)
            fresh_parts = []
            for cid in chosen_clusters:
                ids = hosts_by_cluster[int(cid)]
                if blocked:
                    ids = np.asarray([x for x in ids if int(x) not in blocked], dtype=int)
                if len(ids) == 0:
                    continue
                if targeted:
                    weights = np.where(infected[ids], target_mult, 1.0)
                else:
                    weights = None
                take = min(m, new_n - sum(len(x) for x in fresh_parts), len(ids))
                if take <= 0:
                    break
                fresh_parts.append(_simx_sample_ids(rng, ids, take, weights))
            fresh = np.concatenate(fresh_parts).astype(int) if fresh_parts else np.asarray([], dtype=int)
        else:
            # Simple random sampling: choose directly from all host IDs.
            candidate_ids = np.arange(total_hosts, dtype=int)
            if blocked:
                candidate_ids = np.asarray([x for x in candidate_ids if int(x) not in blocked], dtype=int)
            if targeted:
                weights = np.where(infected[candidate_ids], target_mult, 1.0)
            else:
                weights = None
            fresh = _simx_sample_ids(rng, candidate_ids, new_n, weights)

        sampled = np.concatenate([reused, fresh]).astype(int) if len(reused) or len(fresh) else np.asarray([], dtype=int)
        previous_sample = sampled.copy()
        # Detection sensitivity lets the simulated survey miss some infected hosts.
        # With sensitivity 1.0 every infected sampled host is detected.
        detected_prob = np.where(infected[sampled], float(method_sens), 0.0) if len(sampled) else np.asarray([], dtype=float)
        detected = rng.binomial(1, np.clip(detected_prob, 0.0, 1.0)).astype(int) if len(sampled) else np.asarray([], dtype=int)
        y = int(detected.sum())
        prev_hat = float(y / len(sampled)) if len(sampled) else np.nan
        survey_rows.append(
            {
                "round_id": int(round_df["round_id"].iloc[0]),
                "year": float(round_df["year"].iloc[0]),
                "n": int(len(sampled)),
                "y": y,
                "prev_hat": prev_hat,
                "true_prev": float(round_df["true_prev"].iloc[0]),
            }
        )
        sampled_df = pd.DataFrame(
            {
                "host_id": sampled,
                "cluster_id": host_cluster[sampled] if len(sampled) else np.asarray([], dtype=int),
                "infected": infected[sampled].astype(int) if len(sampled) else np.asarray([], dtype=int),
                "detected": detected,
                "round_id": int(round_df["round_id"].iloc[0]),
            }
        )
        sampled_hosts_by_round.append(sampled_df)
        cluster_sample = sampled_df.groupby("cluster_id", as_index=False).size().rename(columns={"size": "sampled_n"}) if not sampled_df.empty else pd.DataFrame(columns=["cluster_id", "sampled_n"])
        plot_round = round_df.copy()
        plot_round = plot_round.drop(columns=["sampled_n"], errors="ignore").merge(cluster_sample, on="cluster_id", how="left")
        plot_round["sampled_n"] = pd.to_numeric(plot_round["sampled_n"], errors="coerce").fillna(0).astype(int)
        round_tables.append(plot_round)
    return {"rounds": round_tables, "survey_df": pd.DataFrame(survey_rows), "sampled_hosts_by_round": sampled_hosts_by_round}


def _simx_macro_plan_sampling_effort(
    epidemic: dict[str, Any],
    sampling_mode: str,
    overlap_type: str,
    targeted_mode: str,
    within_cluster_cap: int,
    overlap_prop: float,
    targeting_level: str,
    p0_mode: str,
    assumed_p0: float,
    appendix_alpha: float,
    appendix_power: float,
    appendix_c_delta: float,
    appendix_c_corr: float,
    appendix_e_target: float,
    cluster_inflation_mode: str,
    fixed_phi: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    # Estimate required sample sizes for the simulator.
    # This uses the same Appendix C / Appendix E formulas as the dashboard,
    # but the pilot survey is now drawn from host IDs rather than cells.
    del p0_mode, assumed_p0
    schedule = epidemic["schedule"].copy()
    total_hosts = int(len(epidemic["hosts"]))
    n_initial_raw = compute_n_initial_prev_ci_cells(
        p0_upper=DEFAULT_APPENDIX_E_INITIAL_UPPER,
        conf_level=DEFAULT_APPENDIX_E_INITIAL_CONF,
        width=DEFAULT_APPENDIX_E_INITIAL_WIDTH,
        population_size=total_hosts,
    )
    n_initial = int(n_initial_raw)
    # Take a simulated pilot survey in the first round.  Its prevalence estimate
    # is used as the design prevalence in the sample-size formulas.
    pilot = _simx_macro_simulate_surveys(
        epidemic={**epidemic, "rounds": [epidemic["rounds"][0]], "infected_by_round": [epidemic["infected_by_round"][0]]},
        sampling_mode=sampling_mode,
        n_hosts_per_round=[n_initial],
        within_cluster_cap=within_cluster_cap,
        overlap_type="cross_sectional",
        targeted_mode=targeted_mode,
        overlap_prop=0.0,
        targeting_level=targeting_level,
        method_sens=1.0,
        rng=rng,
    )
    p0_hat = float(pilot["survey_df"]["prev_hat"].iloc[0]) if not pilot["survey_df"].empty else float(schedule["true_prev"].iloc[0])
    p0_design = min(max(p0_hat, 1e-8), 1.0 - 1e-8)
    # phi is the inflation factor for clustered sampling.  If phi is 2, the
    # required sample size is doubled to account for correlation within clusters.
    if cluster_inflation_mode == "fixed":
        phi_info = {"phi": max(1.0, float(fixed_phi)), "rho": np.nan, "m": float(within_cluster_cap), "m_observed": float(within_cluster_cap), "estimator": "fixed"}
    elif cluster_inflation_mode == "pilot":
        phi_info = _simx_estimate_design_effect(pilot.get("sampled_hosts_by_round", [pd.DataFrame()])[0], planned_m=within_cluster_cap if sampling_mode == "multistage" else 1)
    else:
        phi_info = {"phi": 1.0, "rho": 0.0, "m": 1.0 if sampling_mode == "srs" else float(within_cluster_cap), "m_observed": 1.0 if sampling_mode == "srs" else float(within_cluster_cap), "estimator": "none"}
    t_vec = schedule["year"].to_numpy(dtype=float)[1:]
    # Regression Method: calculate the total sample size needed after the pilot,
    # then spread it across the remaining survey rounds.
    e_n_total = compute_n_total_appendix_e_cells(
        pi0=p0_design,
        pi_target=appendix_e_target,
        alpha=appendix_alpha,
        power=appendix_power,
        t_vec=t_vec,
        population_size=total_hosts,
        phi=float(phi_info["phi"]),
    )
    e_n_per_round = int(np.ceil(e_n_total / max(len(t_vec), 1)))
    c_n_per_round = compute_n_appendix_c_cells(
        p0=p0_design,
        delta=appendix_c_delta,
        corr=appendix_c_corr,
        alpha=appendix_alpha,
        power=appendix_power,
        population_size=total_hosts,
    )
    c_n_per_round = int(np.ceil(c_n_per_round * float(phi_info["phi"])))
    table = pd.DataFrame(
        [
            {"metric": "Pilot initial sample size", "value": int(n_initial)},
            {"metric": "Pilot prevalence estimate", "value": round(float(p0_hat), 4)},
            {"metric": "Design prevalence used in formulas", "value": round(float(p0_design), 4)},
            {"metric": "Clustering inflation phi", "value": round(float(phi_info["phi"]), 4)},
            {"metric": "Phi estimator", "value": phi_info["estimator"]},
            {"metric": "Alpha used", "value": round(float(appendix_alpha), 4)},
            {"metric": "Power used", "value": round(float(appendix_power), 4)},
            {"metric": "Change Method detectable change", "value": round(float(appendix_c_delta), 4)},
            {"metric": "Regression Method design prevalence", "value": round(float(appendix_e_target), 4)},
            {"metric": "Regression Method recommended per round", "value": int(max(1, e_n_per_round))},
            {"metric": "Change Method recommended per round", "value": int(max(1, c_n_per_round))},
        ]
    )
    return {
        "n_initial": int(n_initial),
        "appendix_e_n_per_round": int(max(1, e_n_per_round)),
        "appendix_c_n_per_round": int(max(1, c_n_per_round)),
        "appendix_e_n_total": int(max(1, e_n_total)),
        "p0_hat": float(p0_hat),
        "p0_design": float(p0_design),
        "phi": phi_info,
        "pilot_round": pilot["rounds"][0],
        "table": table,
    }


def _simx_summary_table(host_landscape: dict[str, Any] | None) -> pd.DataFrame:
    # Small table shown in the simulator to describe the generated landscape.
    if host_landscape is None:
        return pd.DataFrame(columns=["metric", "value"])
    clusters = host_landscape.get("clusters", pd.DataFrame())
    host_counts = pd.to_numeric(clusters.get("host_count", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return pd.DataFrame(
        [
            {"metric": "Landscape source", "value": host_landscape.get("landscape_source", host_landscape.get("model", ""))},
            {"metric": "Total landscape area (km^2)", "value": round(float(host_landscape.get("total_area_km2", np.nan)), 3)},
            {"metric": "Host density (hosts per km^2)", "value": round(float(host_landscape.get("host_density_km2", np.nan)), 3)},
            {"metric": "Total hosts", "value": int(host_landscape.get("total_hosts", 0))},
            {"metric": "Active clusters", "value": int(len(clusters))},
            {"metric": "Median hosts per cluster", "value": int(np.nanmedian(host_counts)) if len(host_counts) else np.nan},
            {"metric": "Largest cluster (hosts)", "value": int(np.nanmax(host_counts)) if len(host_counts) else np.nan},
        ]
    )


def _simx_effort_by_round_table(epidemic: dict[str, Any] | None, effort: dict[str, Any] | None) -> pd.DataFrame:
    # Show the sample size that each method would use in each survey round.
    if epidemic is None or effort is None:
        return pd.DataFrame(columns=["round_id", "year", "regression_n", "change_n"])
    schedule = epidemic["schedule"].copy()
    rows = []
    n_rounds = len(schedule)
    for i, row in enumerate(schedule.itertuples(index=False), start=1):
        regression_n = int(effort["n_initial"]) if i == 1 else int(effort["appendix_e_n_per_round"])
        change_n = int(effort["appendix_c_n_per_round"]) if i in {1, n_rounds} else 0
        rows.append(
            {
                "round_id": i,
                "year": float(row.year),
                "regression_n": regression_n,
                "change_n": change_n,
            }
        )
    return pd.DataFrame(rows)


def _simx_cluster_distribution_table(host_landscape: dict[str, Any] | None) -> pd.DataFrame:
    # Count how many clusters have each host-population size.
    # The old column name is kept so the existing plotting code still works.
    if host_landscape is None or host_landscape.get("clusters") is None:
        return pd.DataFrame(columns=["cells_per_cluster", "n_clusters"])
    sizes = pd.to_numeric(host_landscape["clusters"]["host_count"], errors="coerce").fillna(0).astype(int)
    out = sizes.value_counts().sort_index().reset_index()
    out.columns = ["cells_per_cluster", "n_clusters"]
    return out


def _simx_round_summary(epidemic: dict[str, Any], surveys: dict[str, Any] | None = None) -> pd.DataFrame:
    # Join the true epidemic state with any simulated survey estimates.
    # This is the table shown under the survey-design page.
    rows = []
    survey_df_e = pd.DataFrame()
    survey_df_c = pd.DataFrame()
    if surveys:
        if "appendix_e" in surveys:
            survey_df_e = surveys["appendix_e"]["survey_df"]
        elif "static" in surveys:
            survey_df_e = surveys["static"]["survey_df"]
        if "appendix_c" in surveys:
            survey_df_c = surveys["appendix_c"]["survey_df"]
        elif "static" in surveys:
            survey_df_c = surveys["static"]["survey_df"]
    for round_df in epidemic.get("rounds", []):
        rid = int(round_df["round_id"].iloc[0])
        row = {
            "round_id": rid,
            "year": float(round_df["year"].iloc[0]),
            "true_prev": float(round_df["true_prev"].iloc[0]),
            "infected_hosts": int(round_df["infected_count"].sum()),
        }
        if not survey_df_e.empty and rid in set(survey_df_e["round_id"].astype(int)):
            s = survey_df_e.loc[survey_df_e["round_id"].astype(int) == rid].iloc[0]
            row.update({"regression_n": int(s["n"]), "regression_prev_hat": float(s["prev_hat"])})
        if not survey_df_c.empty and rid in set(survey_df_c["round_id"].astype(int)):
            s = survey_df_c.loc[survey_df_c["round_id"].astype(int) == rid].iloc[0]
            row.update({"change_n": int(s["n"]), "change_prev_hat": float(s["prev_hat"])})
        rows.append(row)
    return pd.DataFrame(rows)


def _simx_targeting_comparison_table(surveys: dict[str, Any] | None) -> pd.DataFrame:
    # Compare the prevalence estimates from untargeted and targeted surveys.
    if surveys is None or "untargeted_all_rounds" not in surveys or "targeted_all_rounds" not in surveys:
        return pd.DataFrame(columns=["round_id", "year", "true_prev", "untargeted_prev_hat", "targeted_prev_hat", "targeting_bias_shift"])
    untargeted = surveys["untargeted_all_rounds"]["survey_df"].copy()
    targeted = surveys["targeted_all_rounds"]["survey_df"].copy()
    merged = untargeted.merge(
        targeted,
        on=["round_id", "year", "true_prev"],
        how="outer",
        suffixes=("_untargeted", "_targeted"),
    ).sort_values("round_id")
    merged["targeting_bias_shift"] = merged["prev_hat_targeted"] - merged["prev_hat_untargeted"]
    return merged.rename(
        columns={
            "n_untargeted": "untargeted_n",
            "prev_hat_untargeted": "untargeted_prev_hat",
            "n_targeted": "targeted_n",
            "prev_hat_targeted": "targeted_prev_hat",
        }
    )[
        ["round_id", "year", "true_prev", "untargeted_n", "untargeted_prev_hat", "targeted_n", "targeted_prev_hat", "targeting_bias_shift"]
    ]


def _simx_plot_macro_infection_landscape(round_df: pd.DataFrame | None, hotspot_cluster_ids: Any | None = None):
    # Bubble plot of the infection landscape.
    # Bubble size is cluster population and colour is cluster prevalence.
    import matplotlib.pyplot as plt

    del hotspot_cluster_ids
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    if round_df is None or round_df.empty:
        ax.text(0.5, 0.5, "Generate the infection landscape to view infected clusters.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    sizes = np.sqrt(np.maximum(round_df["host_count"].to_numpy(dtype=float), 1.0))
    sizes = 12.0 + 160.0 * sizes / max(float(sizes.max()), 1.0)
    sc = ax.scatter(round_df["x"], round_df["y"], c=round_df["cluster_prev"], s=sizes, cmap="Reds", vmin=0, vmax=1, edgecolors="white", linewidths=0.25, alpha=0.9)
    fig.colorbar(sc, ax=ax, shrink=0.82).ax.set_ylabel("Infected proportion in cluster")
    ax.set_title(f"Epidemic by cluster, round {int(round_df['round_id'].iloc[0]) + 1}, prevalence={float(round_df['true_prev'].iloc[0]):.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def _simx_plot_effort_bars(epidemic: dict[str, Any] | None, effort: dict[str, Any] | None):
    # Bar chart of the sample sizes calculated for each method.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    effort_df = _simx_effort_by_round_table(epidemic, effort)
    if effort_df.empty:
        ax.text(0.5, 0.5, "Estimate sample sizes to view the per-round effort from both formulas.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    x = np.arange(len(effort_df), dtype=float)
    width = 0.38
    ax.bar(x - width / 2, effort_df["regression_n"].to_numpy(dtype=float), width=width, color="#2b8cbe", label="Regression Method")
    ax.bar(x + width / 2, effort_df["change_n"].to_numpy(dtype=float), width=width, color="#7b3294", label="Change Method")
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{rid}" for rid in effort_df["round_id"].astype(int)], rotation=0)
    ax.set_title("Required sample size by round")
    ax.set_xlabel("Survey round")
    ax.set_ylabel("Required sample size")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _simx_plot_prevalence_curve(schedule: pd.DataFrame):
    # Plot the true prevalence curve chosen by the user.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if schedule.empty:
        ax.text(0.5, 0.5, "Generate a prevalence schedule to view the curve.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    ax.plot(schedule["year"], schedule["true_prev"], color="#1b7837", linewidth=2.5)
    ax.scatter(schedule["year"], schedule["true_prev"], color="#1b7837", s=28)
    ax.set_title("True prevalence schedule")
    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Prevalence")
    ax.set_ylim(0, min(1.0, max(0.05, float(schedule["true_prev"].max()) * 1.2)))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _simx_plot_targeting_comparison(epidemic: dict[str, Any] | None, surveys: dict[str, Any] | None):
    # Show how targeted surveys can change the estimated prevalence.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    if epidemic is None or surveys is None or "untargeted_all_rounds" not in surveys or "targeted_all_rounds" not in surveys:
        ax.text(0.5, 0.5, "Simulate surveys to compare untargeted and targeted prevalence estimates.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    schedule = epidemic["schedule"].copy()
    untargeted = surveys["untargeted_all_rounds"]["survey_df"].copy()
    targeted = surveys["targeted_all_rounds"]["survey_df"].copy()
    ax.plot(schedule["year"], schedule["true_prev"], color="#1b7837", linewidth=2.2, label="True prevalence")
    if not untargeted.empty:
        ax.plot(untargeted["year"], untargeted["prev_hat"], color="#2b8cbe", marker="o", linewidth=1.8, label="Untargeted surveys")
    if not targeted.empty:
        ax.plot(targeted["year"], targeted["prev_hat"], color="#d95f02", marker="s", linewidth=1.8, label="Targeted surveys")
    ax.set_title("Targeted vs untargeted survey estimates")
    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Estimated prevalence")
    ax.set_ylim(0, min(1.0, max(0.05, float(schedule["true_prev"].max()) * 1.25)))
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# ============================================================================
# Public exports used by the apps
# ============================================================================


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_APPENDIX_C_DELTA",
    "DEFAULT_APPENDIX_E_TARGET",
    "DEFAULT_POWER",
    "LARCH_HOST_DIR",
    "SIM_CURVE_CHOICES",
    "SIM_HOST_CLUSTERING_CHOICES",
    "SIM_INFECTION_CLUSTERING_CHOICES",
    "SIM_OVERLAP_CHOICES",
    "SIM_SURVEY_TARGETING_CHOICES",
    "SIMX_DEFAULT_HOST_DENSITY_KM2",
    "SIMX_DEFAULT_PSU_BLOCK_SIDE",
    "SIMX_DEFAULT_SURVEY_HOSTS_PER_ROUND",
    "SIMX_DEFAULT_TOTAL_AREA_KM2",
    "_sim_fit_logistic_counts",
    "_sim_make_prevalence_schedule",
    "_sim_overlap_config",
    "_sim_predict_logistic",
    "_sim_true_beta_from_schedule",
    "_simx_effort_by_round_table",
    "_simx_fit_methods",
    "_simx_infection_hotspot_table",
    "_simx_infection_round_summary_table",
    "_simx_macro_plan_sampling_effort",
    "_simx_macro_simulate_surveys",
    "_simx_make_macro_epidemic",
    "_simx_make_macro_host_landscape_from_polygons",
    "_simx_make_macro_host_landscape_synthetic",
    "_simx_plot_cluster_size_distribution",
    "_simx_plot_effort_bars",
    "_simx_plot_macro_host_landscape",
    "_simx_plot_macro_infection_landscape",
    "_simx_plot_macro_survey_landscape",
    "_simx_plot_method_comparison",
    "_simx_plot_prevalence_curve",
    "_simx_plot_targeting_comparison",
    "_simx_round_summary",
    "_simx_summary_table",
    "_simx_targeting_comparison_table",
    "load_host_layer",
    "render_grid",
]
