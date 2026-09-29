"""Statistical backend for the Plant Pest Prevalence project.

PURPOSE
=======
This file is the project's single statistical implementation.  The dashboard
and simulator call the functions here instead of maintaining their own copies
of the Change Method, Regression Method, model-fitting or simulation logic.

AUDIT MAP
=========
The module is intentionally organised in the same order that information flows
through the project:

1. Configuration and revision labels
   Default planning parameters, simulator choices and implementation-version
   labels used to prevent results from different revisions being mixed.
2. Data-loading and mathematical utilities
   Shapefile resolution, host-layer loading, probability transforms and the
   finite-population correction.
3. Monitoring-method planning calculations
   ``compute_n_baseline_prevalence_cells`` provides the shared baseline
   precision reference; ``compute_n_change_method_cells`` provides the
   Change Method endpoint reference; and
   ``compute_n_total_regression_method_cells`` provides the Regression Method
   follow-up trend-detection requirement.
4. Reference validation
   ``validate_efsa_reference_examples`` reproduces the published benchmark
   arithmetic used to check that the planning calculations have not drifted.
5. Prevalence trajectories and regression fitting
   Helpers for true simulated curves, grouped-binomial fitting, model
   diagnostics, prediction and robust covariance estimation.
6. Survey-design and simulation engine
   Construction of host landscapes, spatial infection patterns, SRS/MSS
   surveys, resampling, targeting, Design Effect estimation and sample-size
   planning.
7. Display/report helpers
   Tables and plots consumed by the simulator application.
8. Public exports
   ``__all__`` documents the functions intentionally exposed to the apps.

IMPORTANT DESIGN DISTINCTIONS
=============================
* Survey overlap/retention is not the same quantity as the Change Method
  planning correlation.
* The shared baseline precision reference and the Change Method endpoint
  requirement are different planning calculations; they need not be equal.
* The Regression Method follow-up allocation is designed to detect a specified
  decline.  It is not a general guarantee of precise annual prevalence.
* Design Effect inflation changes nominal sample-size requirements but does not
  make a targeted or otherwise non-representative survey representative.
* Simulation functions know the full host-level truth.  The operational
  dashboard does not; it must translate polygon data into host counts using
  explicit assumptions outside this backend.

MAINTENANCE RULE
================
Changes to the three planning functions, regression fitting, survey sampling or
infection allocation should be followed by ``python -m py_compile`` and
``validate_efsa_reference_examples()``.  Simulation revision labels should be
updated when a scientific implementation change would make old and new Monte
Carlo outputs non-comparable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statistics import NormalDist
from scipy.stats import beta as beta_dist

try:
    from shiny import render
except Exception:
    render = None

APP_DIR = Path(__file__).resolve().parent
DEFRA_DATA_DIR = APP_DIR / "DEFRA_data"
TARGET_SITES_DIR = DEFRA_DATA_DIR / "Survey_data"
LARCH_HOST_DIR = DEFRA_DATA_DIR / "Host_coverage_data"
LARCH_HOST_PATH = LARCH_HOST_DIR / "larch_coverage.shp"

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_CHANGE_METHOD_DELTA = 0.03
DEFAULT_CHANGE_METHOD_CORR = 0.0
DEFAULT_REGRESSION_METHOD_TARGET = 0.005
DEFAULT_BASELINE_PREVALENCE_CONF = 0.95
DEFAULT_BASELINE_PREVALENCE_WIDTH = 0.025
DEFAULT_BASELINE_PREVALENCE_UPPER = 0.1

SIMX_DEFAULT_HOST_DENSITY_KM2 = 2500.0
SIMX_DEFAULT_SURVEY_HOSTS_PER_ROUND = 600
SIMX_DEFAULT_PSU_BLOCK_SIDE = 4
SIMX_DEFAULT_TOTAL_AREA_KM2 = 250.0

# Revision labels are written into every Monte Carlo result.  The plotting
# script checks these labels so that output from different implementations
# cannot be mixed accidentally.
SIMULATION_REVISION = "hpc_final_study_v8"
INFECTION_ALLOCATION_REVISION = "per_host_risk_once_v2"
MULTISTAGE_DESIGN_REVISION = "pps_with_replacement_visits_v3"
REGRESSION_FIT_REVISION = "irls_convergence_diagnostics_v1"
CHANGE_CORRELATION_REVISION = "independent_planning_primary_v2"

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

# ============================================================================
# 1. Shared infrastructure: rendering, file loading and numeric utilities
# ============================================================================
# These helpers are deliberately dependency-light.  They are used by the apps
# but do not import application code, keeping the dependency direction one-way:
# dashboard/simulator -> backend.


def render_grid(df: pd.DataFrame):
    """Return a Shiny DataGrid when Shiny is available, otherwise return the DataFrame unchanged."""
    if render is None:
        return df
    return render.DataGrid(df, width="100%")


def _resolve_shapefile_path(path: Path, preferred_stem: str | None = None) -> Path:
    """Resolve a file-or-folder input to one shapefile, preferring a named stem when available."""
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
    """Load the host polygon layer used by the simulator and return data plus user-facing diagnostics."""
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
    """Clamp a scalar probability away from exactly 0 and 1 for stable logit calculations."""
    return float(min(1.0 - eps, max(eps, x)))


def _logit(x: float) -> float:
    """Return the scalar logit after protecting against probabilities of exactly 0 or 1."""
    x = _clamp_open01(float(x))
    return float(np.log(x / (1.0 - x)))


def _inv_logit(eta: float) -> float:
    """Return the inverse-logit while clipping extreme linear predictors for numerical stability."""
    return float(1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0))))


def _apply_fpc(n: float, population_size: int | float | None) -> int:
    """Apply the project finite-population correction when the nominal sample exceeds 5% of the population."""
    if population_size is None or not np.isfinite(population_size) or population_size <= 0:
        return max(1, int(np.ceil(n)))
    n = max(1.0, float(n))
    population_size = float(population_size)
    if n <= 0.05 * population_size:
        return max(1, int(np.ceil(n)))
    n_adj = (n * population_size) / (population_size + n)
    return max(1, int(np.ceil(n_adj)))



# ============================================================================
# 2. Monitoring-method planning calculations
# ============================================================================
# These are the primary planning functions used by the operational dashboard.
# Keep their inputs explicit and their validation strict: silent coercion here
# would make sample-size results difficult to audit.

def compute_n_change_method_cells(
    p0: float,
    delta: float = DEFAULT_CHANGE_METHOD_DELTA,
    corr: float = DEFAULT_CHANGE_METHOD_CORR,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    population_size: int | float | None = None,
    design_effect: float = 1.0,
) -> int:
    """Change Method endpoint-change sample size.

    ``corr`` is the prespecified correlation between the two endpoint
    prevalence estimators.  It is a planning assumption, distinct from the
    realised paired-host correlation later estimated from simulated data.
    """
    delta = float(delta)
    corr = float(corr)
    alpha = float(alpha)
    power = float(power)
    design_effect = float(design_effect)
    if delta <= 0:
        raise ValueError("Change Method detectable change delta must be positive.")
    if not 0 < alpha < 1:
        raise ValueError("Change Method alpha must be between 0 and 1.")
    if not 0 < power < 1:
        raise ValueError("Change Method power must be between 0 and 1.")
    if not -1 < corr < 1:
        raise ValueError("Change Method planning correlation must be strictly between -1 and 1.")
    if not np.isfinite(design_effect) or design_effect < 1:
        raise ValueError("Change Method design effect must be finite and at least 1.")
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_beta = NormalDist().inv_cdf(power)
    p0 = _clamp_open01(float(p0))
    n = (2.0 * p0 * (1.0 - p0) * (1.0 - corr) * (z_alpha + z_beta) ** 2) / (delta ** 2)
    return _apply_fpc(design_effect * n, population_size)


def _clopper_pearson_width(n: int, prop: float, conf_level: float) -> float:
    """Expected two-sided Clopper--Pearson interval width at ``prop``.

    EFSA's worked initial-prevalence example is based on an exact binomial
    interval rather than the simpler Wald approximation.  The expected number
    of positives is rounded to the nearest integer for planning.
    """
    n = max(1, int(n))
    prop = _clamp_open01(float(prop))
    alpha = 1.0 - float(conf_level)
    x = int(round(n * prop))
    lower = 0.0 if x <= 0 else float(beta_dist.ppf(alpha / 2.0, x, n - x + 1))
    upper = 1.0 if x >= n else float(beta_dist.ppf(1.0 - alpha / 2.0, x + 1, n - x))
    return float(upper - lower)


@lru_cache(maxsize=64)
def _exact_initial_prev_n_cached(p0_upper: float, conf_level: float, width: float) -> int:
    """Return a conservative exact-binomial sample size for initial prevalence."""
    if width <= 0:
        return 1
    lo, hi = 1, 256
    while _clopper_pearson_width(hi, p0_upper, conf_level) > width:
        lo, hi = hi + 1, hi * 2
        if hi > 100_000_000:
            raise ValueError("Unable to find a finite initial-prevalence sample size.")
    while lo < hi:
        mid = (lo + hi) // 2
        if _clopper_pearson_width(mid, p0_upper, conf_level) <= width:
            hi = mid
        else:
            lo = mid + 1
    # Check a short neighbourhood because rounding the expected positive count
    # makes the exact width only approximately monotone.  The one-observation
    # safety margin reproduces EFSA's published 2,292 example.
    candidates = [n for n in range(max(1, lo - 12), lo + 13) if _clopper_pearson_width(n, p0_upper, conf_level) <= width]
    return int(min(candidates) + 1 if candidates else lo + 1)


def compute_n_baseline_prevalence_cells(
    p0_upper: float = DEFAULT_BASELINE_PREVALENCE_UPPER,
    conf_level: float = DEFAULT_BASELINE_PREVALENCE_CONF,
    width: float = DEFAULT_BASELINE_PREVALENCE_WIDTH,
    population_size: int | float | None = None,
) -> int:
    """Initial-prevalence sample size using an exact binomial CI criterion."""
    raw_n = _exact_initial_prev_n_cached(
        round(float(p0_upper), 12),
        round(float(conf_level), 12),
        round(float(width), 12),
    )
    return _apply_fpc(float(raw_n), population_size)


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
    """Required total follow-up sample for the Regression Method trend test.

    The time-zero survey contributes information to the slope estimate.  The
    follow-up total is allocated equally over ``t_vec`` and the smallest total
    giving the requested one-sided power is found numerically.  Cluster
    inflation is represented by dividing actual sample sizes by ``phi`` when
    constructing the Fisher information.
    """
    pi0 = _clamp_open01(float(pi0))
    pi_target = _clamp_open01(float(pi_target))
    t_vec = np.asarray(t_vec, dtype=float)
    t_vec = t_vec[np.isfinite(t_vec)]
    if t_vec.size < 1:
        t_vec = np.array([1.0], dtype=float)
    t_end = float(np.max(t_vec))
    if t_end <= 0:
        raise ValueError("Regression Method follow-up times must be greater than zero.")
    # Regression Method is a one-sided design for detecting a *declining* trend from
    # an initial prevalence above the design prevalence.  If the baseline is
    # already at or below the target, there is no negative target slope to
    # power and the planning problem is not applicable.
    if pi0 <= pi_target:
        raise ValueError(
            "Regression Method trend planning requires initial prevalence to be greater "
            "than the design prevalence."
        )
    beta_star = (_logit(pi_target) - _logit(pi0)) / t_end
    beta_abs = -float(beta_star)
    if not np.isfinite(beta_abs) or beta_abs <= 0:
        raise ValueError("Regression Method could not define a finite negative target slope.")

    phi = max(1.0, float(phi))
    all_t = np.concatenate([[0.0], t_vec])
    eta = _logit(pi0) + beta_star * all_t
    probs = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    bernoulli_info = probs * (1.0 - probs)
    X = np.column_stack([np.ones_like(all_t), all_t])
    critical = NormalDist().inv_cdf(1.0 - float(alpha)) + NormalDist().inv_cdf(float(power))
    initial_actual = max(0.0, float(initial_n))
    n_followup_rounds = int(t_vec.size)

    def achieved_standardised_effect(total_followup_actual: int) -> float:
        per_round_actual = float(total_followup_actual) / n_followup_rounds
        effective_sizes = np.concatenate([[initial_actual / phi], np.full(n_followup_rounds, per_round_actual / phi)])
        info = X.T @ np.diag(effective_sizes * bernoulli_info) @ X
        try:
            slope_var = float(np.linalg.inv(info)[1, 1])
        except np.linalg.LinAlgError:
            return 0.0
        if not np.isfinite(slope_var) or slope_var <= 0:
            return 0.0
        return beta_abs / np.sqrt(slope_var)

    lo, hi = 1, max(32, n_followup_rounds)
    while achieved_standardised_effect(hi) < critical:
        lo, hi = hi + 1, hi * 2
        if population_size is not None and np.isfinite(population_size) and hi > 100 * float(population_size):
            return _apply_fpc(float(hi), population_size)
        if hi > 1_000_000_000:
            raise ValueError("Unable to find a finite Regression Method follow-up sample size.")
    while lo < hi:
        mid = (lo + hi) // 2
        if achieved_standardised_effect(mid) >= critical:
            hi = mid
        else:
            lo = mid + 1
    return _apply_fpc(float(lo), population_size)



# ============================================================================
# 3. Reference-value validation
# ============================================================================
# These checks are executable documentation for the planning arithmetic.

def validate_efsa_reference_examples(raise_on_failure: bool = True) -> pd.DataFrame:
    """Check the implemented formulae against EFSA's published examples."""
    checks = []

    initial_n = compute_n_baseline_prevalence_cells(0.1, 0.95, 0.025, None)
    checks.append({"check": "Shared baseline prevalence", "expected": 2292, "observed": initial_n, "tolerance": 1})

    e_total = compute_n_total_regression_method_cells(
        pi0=0.03098,
        pi_target=0.005,
        alpha=0.05,
        power=0.99,
        t_vec=np.arange(1.0, 6.0),
        population_size=None,
        phi=1.0,
        initial_n=2292,
    )
    checks.append({"check": "Regression Method five-year follow-up total", "expected": 1680, "observed": e_total, "tolerance": 5})

    change_method_examples = [(0.95, 78), (0.50, 784), (0.01, 1552)]
    for corr, expected in change_method_examples:
        observed = compute_n_change_method_cells(0.5, 0.05, corr, 0.05, 0.80, None, 1.0)
        checks.append({
            "check": f"Change Method correlation {corr:g}",
            "expected": expected,
            "observed": observed,
            "tolerance": max(2, int(np.ceil(expected * 0.003))),
        })

    overlap_mapping = {
        "cross_sectional": 0.01,
        "rotating_panel": 0.50,
        "longitudinal": 0.95,
    }
    for overlap_type, expected_corr in overlap_mapping.items():
        observed_corr = float(_sim_overlap_config(overlap_type, overlap_prop_rot=0.7)["corr_c"])
        checks.append({
            "check": f"Change Method overlap mapping {overlap_type}",
            "expected": expected_corr,
            "observed": observed_corr,
            "tolerance": 1e-12,
        })

    result = pd.DataFrame(checks)
    result["difference"] = result["observed"] - result["expected"]
    result["passed"] = result["difference"].abs() <= result["tolerance"]
    if raise_on_failure and not bool(result["passed"].all()):
        failed = result.loc[~result["passed"], ["check", "expected", "observed", "tolerance"]]
        raise AssertionError(f"EFSA reference validation failed:\n{failed.to_string(index=False)}")
    return result


# ============================================================================
# 4. Simulated prevalence trajectories and temporal design helpers
# ============================================================================

def _sim_clamp01(x: np.ndarray | float) -> np.ndarray | float:
    """Clamp simulated probabilities to the closed unit interval."""
    return np.clip(x, 0.0, 1.0)


def _sim_logit_linear_curve(t: np.ndarray, p0: float, p_end: float) -> np.ndarray:
    """Generate a prevalence trajectory that is linear on the logit scale between two endpoints."""
    return 1.0 / (1.0 + np.exp(-(_logit(p0) + (_logit(p_end) - _logit(p0)) * t)))


def _sim_raw_linear_curve(t: np.ndarray, p0: float, p_end: float) -> np.ndarray:
    """Generate a prevalence trajectory that changes linearly on the probability scale."""
    return _sim_clamp01(p0 + (p_end - p0) * t)


def _sim_logistic_transition_curve(t: np.ndarray, p0: float, p_end: float, k: float = 10.0, tm: float = 0.5) -> np.ndarray:
    """Generate a smooth sigmoid transition between the requested initial and final prevalences."""
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
    """Create the known prevalence history for one simulation scenario.

    The prevalence-level sensitivity study shifts the *two endpoints* before
    constructing the requested curve.  This is an important fairness detail.
    Adding a constant to an already generated logit-linear curve would make it
    non-logit-linear, so the study would change both prevalence level and model
    specification at once.  Shifting the endpoints first keeps a logit-linear
    scenario logit-linear while preserving the same absolute endpoint change.
    """
    t_scaled = np.linspace(0.0, 1.0, n_rounds)
    p0_shifted = float(_sim_clamp01(float(p0) + float(prevalence_shift)))
    p_end_shifted = float(_sim_clamp01(float(p_end) + float(prevalence_shift)))

    if curve_type == "logit_linear":
        prev = _sim_logit_linear_curve(t_scaled, p0_shifted, p_end_shifted)
    elif curve_type == "linear":
        prev = _sim_raw_linear_curve(t_scaled, p0_shifted, p_end_shifted)
    elif curve_type == "fp3":
        # Single-term fractional-polynomial truth used in EFSA Regression Method
        # replication scenarios: logit(p(t)) = a + b (t + 1)^3.
        years = t_scaled * 5.0
        x = np.power(years + 1.0, 3.0)
        b = (_logit(p_end_shifted) - _logit(p0_shifted)) / max(float(x[-1] - x[0]), 1e-12)
        a = _logit(p0_shifted) - b * float(x[0])
        prev = 1.0 / (1.0 + np.exp(-np.clip(a + b * x, -30.0, 30.0)))
    else:
        prev = _sim_logistic_transition_curve(t_scaled, p0_shifted, p_end_shifted)
    prev = _sim_clamp01(prev)
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
    """Return host overlap and the Change Method planning correlation.

    The survey simulator uses ``overlap_prop`` to decide how many host IDs are
    revisited.  The Change Method sample-size formula separately requires a
    prespecified correlation between the two endpoint prevalence estimates.
    For the report comparison we use the three worked EFSA planning values:

    - repeated cross-sectional: correlation 0.01;
    - rotating panel: correlation 0.50;
    - longitudinal panel: correlation 0.95.

    The realised paired-outcome correlation used for inference is still
    calculated from the simulated observations; these values affect planning
    only.  ``overlap_prop_rot`` remains the actual rotating-panel host overlap.
    """
    if overlap_type == "cross_sectional":
        return {"overlap_prop": 0.0, "corr_c": 0.01}
    if overlap_type == "rotating_panel":
        return {"overlap_prop": float(overlap_prop_rot), "corr_c": 0.50}
    if overlap_type == "longitudinal":
        return {"overlap_prop": 1.0, "corr_c": 0.95}
    raise ValueError(f"Unsupported overlap_type: {overlap_type}")


def _maximum_positive_bernoulli_correlation(p0: float, p1: float) -> float:
    """Return the largest feasible positive correlation for two binary states.

    A planning correlation cannot be chosen independently of the two endpoint
    prevalences.  When one prevalence is much smaller than the other, even a
    perfectly nested infection pattern cannot produce a correlation near one.
    The bound below follows directly from the Fréchet upper bound
    ``P(X=1,Y=1) <= min(p0,p1)``.
    """
    p0 = _clamp_open01(float(p0))
    p1 = _clamp_open01(float(p1))
    denom = np.sqrt(p0 * (1.0 - p0) * p1 * (1.0 - p1))
    if not np.isfinite(denom) or denom <= 0:
        return 0.0
    upper_joint = min(p0, p1)
    return float(np.clip((upper_joint - p0 * p1) / denom, 0.0, 1.0))


def _effective_change_method_correlation(
    requested_corr: float,
    overlap_prop: float,
    p0_design: float,
    detectable_change: float,
) -> dict[str, float | bool]:
    """Convert a nominal planning correlation into a feasible value.

    The historical study mapped cross-sectional, rotating and longitudinal
    designs directly to correlations 0.01, 0.50 and 0.95.  That is too
    aggressive when prevalence is expected to fall from about 0.033 to about
    0.003--0.005: a binary host cannot then have endpoint correlation 0.95.

    We therefore cap the requested value using two transparent limits:

    * only the deliberately repeated proportion of hosts can contribute
      endpoint covariance; and
    * the repeated hosts themselves cannot exceed the Bernoulli correlation
      allowed by the planned endpoint prevalences.

    This remains a planning approximation, but it prevents mathematically
    impossible correlation assumptions from creating unrealistically tiny
    longitudinal endpoint samples.
    """
    requested = float(np.clip(requested_corr, 0.0, 0.999999))
    overlap = float(np.clip(overlap_prop, 0.0, 1.0))
    p0 = _clamp_open01(float(p0_design))
    p1 = _clamp_open01(max(float(p0_design) - abs(float(detectable_change)), 1e-8))
    host_corr_max = _maximum_positive_bernoulli_correlation(p0, p1)
    estimator_corr_bound = float(np.clip(overlap * host_corr_max, 0.0, 0.999999))
    used = min(requested, estimator_corr_bound)
    return {
        "requested": requested,
        "used": float(used),
        "host_corr_max": float(host_corr_max),
        "estimator_corr_bound": estimator_corr_bound,
        "capped": bool(used < requested - 1e-12),
        "p1_design": float(p1),
    }



# ============================================================================
# 5. Regression model fitting, prediction and covariance
# ============================================================================
# Grouped-binomial point estimation is kept separate from dependence-aware
# covariance estimation so the source of each quantity remains visible.

def _sim_regression_feature(t: np.ndarray, model_form: str = "logistic") -> np.ndarray:
    """Build the regression design feature used for the requested temporal model form."""
    t_arr = np.asarray(t, dtype=float)
    model_form = str(model_form).lower()
    if model_form == "logistic":
        return t_arr
    if model_form == "fp2":
        return np.power(t_arr + 1.0, 2.0)
    if model_form == "fp3":
        return np.power(t_arr + 1.0, 3.0)
    if model_form.startswith("fp:"):
        power = float(model_form.split(":", 1)[1])
        if abs(power) < 1e-12:
            return np.log(t_arr + 1.0)
        return np.power(t_arr + 1.0, power)
    raise ValueError(f"Unsupported regression model form: {model_form}")


def _logistic_irls_feature(
    x_feature: np.ndarray,
    x: np.ndarray,
    n: np.ndarray,
    max_iter: int = 100,
    *,
    return_diagnostics: bool = False,
):
    """Fit a two-parameter grouped-binomial logistic model by IRLS.

    Earlier versions treated every finite coefficient vector as a valid fit.
    Under rare outcomes or highly clustered surveys, IRLS can instead run
    towards complete or quasi-complete separation and return enormous but
    finite slopes.  Those values dominated error variances while boxplots hid
    them as fliers.

    This implementation uses a smoothed starting value, limits individual
    Newton steps, records convergence and flags separation-like solutions.
    The ordinary unpenalised likelihood is still fitted; suspicious fits are
    reported as invalid rather than silently replaced by a different model.
    """
    x_feature = np.asarray(x_feature, dtype=float)
    y = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    X = np.column_stack([np.ones_like(x_feature, dtype=float), x_feature])

    valid_data = (
        len(y) >= 2
        and X.shape[0] == len(y)
        and np.isfinite(X).all()
        and np.isfinite(y).all()
        and np.isfinite(n).all()
        and np.all(n > 0)
        and np.all(y >= 0)
        and np.all(y <= n)
    )
    diagnostics = {
        "converged": False,
        "iterations": 0,
        "separation_suspected": False,
        "condition_number": np.nan,
        "max_abs_coefficient": np.nan,
        "fit_valid": False,
        "reason": "invalid_input",
    }
    if not valid_data:
        beta = np.full(2, np.nan)
        cov = np.full((2, 2), np.nan)
        return (beta, cov, diagnostics) if return_diagnostics else (beta, cov)

    # Haldane--Anscombe smoothing is used only for the starting values.  It
    # prevents a year with zero positives from starting at an infinite logit;
    # the subsequent score equations still use the original counts.
    p_start = np.clip((y + 0.5) / (n + 1.0), 1e-8, 1.0 - 1e-8)
    eta_start = np.log(p_start / (1.0 - p_start))
    beta = np.linalg.lstsq(X, eta_start, rcond=None)[0]

    converged = False
    reason = "maximum_iterations"
    for iteration in range(1, int(max_iter) + 1):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        w = n * p * (1.0 - p)
        score = X.T @ (y - n * p)
        info = X.T @ (X * w[:, None])
        try:
            step = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            reason = "singular_information"
            break
        if not np.isfinite(step).all():
            reason = "nonfinite_step"
            break

        # Separation can produce explosive Newton steps.  Step limiting keeps
        # the numerical path stable while allowing the convergence diagnostic
        # to reveal that no finite maximum has been found.
        max_step = float(np.max(np.abs(step)))
        if max_step > 5.0:
            step = step * (5.0 / max_step)
        beta_new = beta + step
        diagnostics["iterations"] = iteration
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            converged = True
            reason = "converged"
            break
        beta = beta_new

    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    w = n * p * (1.0 - p)
    info = X.T @ (X * w[:, None])
    cov = np.linalg.pinv(info)
    condition_number = float(np.linalg.cond(info)) if np.isfinite(info).all() else np.inf
    max_abs_beta = float(np.max(np.abs(beta))) if np.isfinite(beta).all() else np.inf

    # A finite numeric result is not automatically a finite MLE.  These broad
    # thresholds are intended to catch only unmistakable divergence, not to
    # reject legitimately steep but estimable trends.
    separation = bool(
        (not converged)
        or max_abs_beta > 50.0
        or not np.isfinite(condition_number)
        or condition_number > 1e12
    )
    fit_valid = bool(
        converged
        and not separation
        and np.isfinite(beta).all()
        and np.isfinite(cov).all()
        and np.all(np.diag(cov) >= 0)
    )
    if separation and reason == "converged":
        reason = "separation_or_ill_conditioning"

    diagnostics.update(
        {
            "converged": bool(converged),
            "separation_suspected": separation,
            "condition_number": condition_number,
            "max_abs_coefficient": max_abs_beta,
            "fit_valid": fit_valid,
            "reason": reason,
        }
    )
    return (beta, cov, diagnostics) if return_diagnostics else (beta, cov)


def _sim_fit_logistic_counts(
    t: np.ndarray,
    y: np.ndarray,
    n: np.ndarray,
    model_form: str = "logistic",
    *,
    return_diagnostics: bool = False,
):
    """Fit grouped-binomial prevalence counts and return coefficients plus model-based covariance."""
    x_feature = _sim_regression_feature(np.asarray(t, dtype=float), model_form=model_form)
    result = _logistic_irls_feature(
        x_feature,
        np.asarray(y, dtype=float),
        np.asarray(n, dtype=float),
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        beta, cov, diagnostics = result
        return np.asarray(beta, dtype=float), np.asarray(cov, dtype=float), diagnostics
    beta, cov = result
    return np.asarray(beta, dtype=float), np.asarray(cov, dtype=float)


def _sim_binomial_loglik(t: np.ndarray, y: np.ndarray, n: np.ndarray, beta: np.ndarray, model_form: str) -> float:
    """Evaluate the grouped-binomial log-likelihood for model comparison."""
    p = np.clip(_sim_predict_logistic(beta, np.asarray(t, dtype=float), model_form=model_form), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    return float(np.sum(y * np.log(p) + (n - y) * np.log(1.0 - p)))


def _sim_select_fractional_polynomial_counts(
    t: np.ndarray, y: np.ndarray, n: np.ndarray,
    powers: tuple[float, ...] = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
) -> tuple[str, np.ndarray, np.ndarray, pd.DataFrame]:
    """Select a one-term FP power by binomial AIC for EFSA replication."""
    rows = []
    best = None
    for power in powers:
        model = "logistic" if abs(power - 1.0) < 1e-12 else f"fp:{power:g}"
        beta, cov = _sim_fit_logistic_counts(t, y, n, model_form=model)
        ll = _sim_binomial_loglik(t, y, n, beta, model)
        aic = -2.0 * ll + 2.0 * 2
        rows.append({"model_form": model, "fp_power": power, "loglik": ll, "aic": aic})
        if best is None or aic < best[0]:
            best = (aic, model, beta, cov)
    assert best is not None
    return best[1], np.asarray(best[2]), np.asarray(best[3]), pd.DataFrame(rows)


def _sim_cluster_robust_covariance(
    beta: np.ndarray, model_form: str, sampled_hosts_by_round: list[pd.DataFrame], years: np.ndarray,
    cluster_on_field: bool = True,
) -> tuple[np.ndarray, int]:
    """Sandwich covariance using realised host-level survey observations.

    Field clusters are used for multistage surveys, naturally accounting for
    repeated hosts within the same field as well as within-field similarity.
    For SRS, host IDs are the independent sampling units.
    """
    parts = []
    for ridx, sampled in enumerate(sampled_hosts_by_round or []):
        if sampled is None or sampled.empty or ridx >= len(years):
            continue
        one = sampled.copy()
        one["year"] = float(years[ridx])
        parts.append(one[["host_id", "cluster_id", "detected", "year"]])
    if not parts:
        return np.asarray([[]], dtype=float), 0
    long = pd.concat(parts, ignore_index=True)
    xfeat = _sim_regression_feature(long["year"].to_numpy(dtype=float), model_form=model_form)
    X = np.column_stack([np.ones(len(long), dtype=float), xfeat])
    p = np.clip(1.0 / (1.0 + np.exp(-np.clip(X @ np.asarray(beta)[:2], -30.0, 30.0))), 1e-8, 1 - 1e-8)
    resid = long["detected"].to_numpy(dtype=float) - p
    bread = np.linalg.pinv(X.T @ (X * (p * (1.0 - p))[:, None]))
    key = long["cluster_id"].astype(str) if cluster_on_field else long["host_id"].astype(str)
    scores = pd.DataFrame({"key": key, "s0": X[:, 0] * resid, "s1": X[:, 1] * resid}).groupby("key")[["s0", "s1"]].sum()
    G = int(len(scores))
    if G < 2:
        return np.asarray([[]], dtype=float), G
    S = scores.to_numpy(dtype=float)
    meat = S.T @ S
    cov = bread @ meat @ bread
    N, k = len(long), 2
    if G > 1 and N > k:
        cov *= (G / (G - 1.0)) * ((N - 1.0) / (N - k))
    return np.asarray(cov, dtype=float), G


def _sim_predict_logistic(beta: np.ndarray, t: np.ndarray | float, model_form: str = "logistic") -> np.ndarray:
    """Predict prevalence from fitted regression coefficients at one or more time points."""
    t_arr = np.asarray(t, dtype=float)
    x_feature = _sim_regression_feature(t_arr, model_form=model_form)
    eta = beta[0] + beta[1] * x_feature
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


def _sim_true_beta_from_schedule(schedule: pd.DataFrame) -> float:
    """Fit the reference logit-linear slope to a known simulated prevalence schedule."""
    t = schedule["t"].to_numpy(dtype=float)
    y = np.round(schedule["true_prev"].to_numpy(dtype=float) * 100000.0)
    n = np.full_like(y, 100000.0, dtype=float)
    beta, _ = _sim_fit_logistic_counts(t, y, n)
    return float(beta[1])



# ============================================================================
# 6. Cluster dependence and Design Effect estimation
# ============================================================================

def _simx_estimate_design_effect(sampled_hosts_df: pd.DataFrame, planned_m: int | None = None) -> dict[str, float]:
    """Estimate the baseline ICC and corresponding Design Effect from sampled host outcomes."""
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
# 7. Simulator display helpers
# ============================================================================# ============================================================================
# Simulator display helpers
# ============================================================================


def _simx_infection_round_summary_table(epidemic: dict[str, Any] | None, round_index: int) -> pd.DataFrame:
    # Summarise one year/round of the epidemic.
    # This gives users a quick check that the infection landscape looks sensible.
    """Summarise the selected simulated epidemic round for display in the simulator."""
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
    """Return the most infected clusters for the selected simulated epidemic round."""
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
    """Plot the distribution of host counts across simulated physical clusters."""
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
    """Plot the simulated or polygon-derived host landscape."""
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
    """Plot realised survey allocation across physical host clusters."""
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



def _simx_fit_methods(
    epidemic: dict[str, Any],
    surveys: dict[str, Any],
    model_form: str = "logistic",
) -> dict[str, Any]:
    """Fit both monitoring methods and retain dependence-aware covariance.

    The regression coefficients are estimated from annual positive and sampled
    counts.  Standard errors are then calculated from the realised host-level
    observations:

    * MSS results are grouped by physical host cluster;
    * SRS results are grouped by host ID.

    Grouping SRS observations by host ID is essential when the same hosts are
    revisited, because annual counts are then correlated even though the design
    is not spatially clustered.
    """
    schedule = epidemic["schedule"].copy()
    survey_obj_e = surveys.get(
        "regression_method",
        surveys.get("static", {"survey_df": pd.DataFrame()}),
    )
    survey_obj_c = surveys.get(
        "change_method",
        surveys.get("static", {"survey_df": pd.DataFrame()}),
    )
    survey_df_e = survey_obj_e["survey_df"].copy()
    survey_df_c = survey_obj_c["survey_df"].copy()

    if survey_df_e.empty:
        return {
            "survey_df": survey_df_e,
            "change_method_survey_df": survey_df_c,
            "regression_method_curve": pd.DataFrame(),
            "change_method_curve": pd.DataFrame(),
            "regression_model": str(model_form),
            "regression_beta": np.asarray([], dtype=float),
            "regression_cov": np.asarray([[]], dtype=float),
            "regression_cov_model": np.asarray([[]], dtype=float),
            "regression_cov_type": "unavailable",
            "regression_robust_unit": "unavailable",
            "regression_cluster_count": 0,
        }

    requested_model = str(model_form).lower()
    fp_table = pd.DataFrame()
    fit_diagnostics = {
        "converged": True, "iterations": 0, "separation_suspected": False,
        "condition_number": np.nan, "max_abs_coefficient": np.nan,
        "fit_valid": True, "reason": "fp_selection",
    }
    if requested_model == "fp_auto":
        fitted_model, beta, cov_model, fp_table = (
            _sim_select_fractional_polynomial_counts(
                survey_df_e["year"].to_numpy(dtype=float),
                survey_df_e["y"].to_numpy(dtype=float),
                survey_df_e["n"].to_numpy(dtype=float),
            )
        )
    else:
        fitted_model = requested_model
        beta, cov_model, fit_diagnostics = _sim_fit_logistic_counts(
            survey_df_e["year"].to_numpy(dtype=float),
            survey_df_e["y"].to_numpy(dtype=float),
            survey_df_e["n"].to_numpy(dtype=float),
            model_form=fitted_model,
            return_diagnostics=True,
        )

    sampling_mode = str(survey_obj_e.get("sampling_mode", "srs"))
    robust_unit = "cluster_id" if sampling_mode == "multistage" else "host_id"
    robust_cov, cluster_count = _sim_cluster_robust_covariance(
        beta,
        fitted_model,
        survey_obj_e.get("sampled_hosts_by_round", []),
        survey_df_e["year"].to_numpy(dtype=float),
        cluster_on_field=(sampling_mode == "multistage"),
    )

    use_robust = (
        bool(fit_diagnostics.get("fit_valid", False))
        and robust_cov.shape == (2, 2)
        and np.isfinite(robust_cov).all()
    )
    cov = robust_cov if use_robust else cov_model

    curve_year = np.linspace(
        float(schedule["year"].min()),
        float(schedule["year"].max()),
        201,
    )
    curve_prev = _sim_predict_logistic(
        beta,
        curve_year,
        model_form=fitted_model,
    )
    regression_method_curve = pd.DataFrame(
        {"year": curve_year, "regression_method_prev": curve_prev}
    )

    c_source = (
        survey_df_c.loc[survey_df_c["n"] > 0].copy()
        if not survey_df_c.empty
        else pd.DataFrame()
    )
    c_points = (
        c_source.iloc[[0, -1]].copy()
        if len(c_source) >= 2
        else pd.DataFrame(columns=["year", "prev_hat"])
    )
    change_method_curve = pd.DataFrame(
        {
            "year": (
                c_points["year"].to_numpy(dtype=float)
                if not c_points.empty
                else np.array([], dtype=float)
            ),
            "change_method_prev": (
                c_points["prev_hat"].to_numpy(dtype=float)
                if not c_points.empty
                else np.array([], dtype=float)
            ),
        }
    )

    return {
        "survey_df": survey_df_e,
        "change_method_survey_df": survey_df_c,
        "regression_method_curve": regression_method_curve,
        "change_method_curve": change_method_curve,
        "regression_model": fitted_model,
        "regression_model_requested": requested_model,
        "regression_beta": np.asarray(beta, dtype=float),
        "regression_cov": np.asarray(cov, dtype=float),
        "regression_cov_model": np.asarray(cov_model, dtype=float),
        # Keep the established label for compatibility; robust_unit records
        # whether the grouping variable was a host or a physical field cluster.
        "regression_cov_type": "cluster_robust" if use_robust else "model_based",
        "regression_robust_unit": robust_unit if use_robust else "none",
        "regression_cluster_count": int(cluster_count),
        "regression_fit_valid": bool(fit_diagnostics.get("fit_valid", False)),
        "regression_converged": bool(fit_diagnostics.get("converged", False)),
        "regression_fit_iterations": int(fit_diagnostics.get("iterations", 0)),
        "regression_separation_suspected": bool(fit_diagnostics.get("separation_suspected", False)),
        "regression_information_condition_number": float(fit_diagnostics.get("condition_number", np.nan)),
        "regression_max_abs_coefficient": float(fit_diagnostics.get("max_abs_coefficient", np.nan)),
        "regression_fit_reason": str(fit_diagnostics.get("reason", "unknown")),
        "fp_selection_table": fp_table,
    }

def _simx_plot_method_comparison(epidemic: dict[str, Any] | None, fit_result: dict[str, Any] | None, regression_model: str | None = None):
    # Plot true prevalence against the estimates from the two methods.
    # This is the main visual check of whether the methods recovered the truth.
    """Plot true prevalence and fitted outputs from the Change and Regression Methods."""
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
    survey_df_c = fit_result.get("change_method_survey_df", pd.DataFrame())
    ax.plot(schedule["year"], schedule["true_prev"], color="#1b7837", linewidth=2.2, label="True prevalence")
    if not survey_df.empty:
        ax.scatter(survey_df["year"], survey_df["prev_hat"], color="#d95f02", s=34, label="Regression Method estimates")
    if not survey_df_c.empty:
        c_points = survey_df_c.loc[survey_df_c["n"] > 0].copy()
        if not c_points.empty:
            ax.scatter(c_points["year"], c_points["prev_hat"], color="#7b3294", s=34, marker="s", label="Change Method survey points")
    regression_method_curve = fit_result.get("regression_method_curve", pd.DataFrame())
    if not regression_method_curve.empty:
        ax.plot(regression_method_curve["year"], regression_method_curve["regression_method_prev"], color="#2b8cbe", linewidth=2.0, label="Regression Method")
    change_method_curve = fit_result.get("change_method_curve", pd.DataFrame())
    if not change_method_curve.empty:
        ax.plot(change_method_curve["year"], change_method_curve["change_method_prev"], color="#7b3294", linewidth=2.0, linestyle="--", marker="o", label="Change Method")
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



# ============================================================================
# 8. Host-landscape construction and sampling primitives
# ============================================================================
# Functions in this section create finite host populations and perform the
# low-level host/cluster sampling used by the survey simulator.

def _simx_scale01_local(values: np.ndarray) -> np.ndarray:
    # Put any numeric signal onto a 0 to 1 scale.
    # This is useful when combining different risk signals, such as distance
    # pressure and hotspot pressure, which may originally be on different scales.
    """Scale finite numeric values to the unit interval while handling constant inputs safely."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _simx_reflect_to_square_local(values: np.ndarray, side: float) -> np.ndarray:
    # Keep coordinates inside [0, side] without piling points on the boundary.
    # Reflection preserves clustered structure better than hard clipping.
    """Reflect simulated coordinates back into the square study region."""
    arr = np.asarray(values, dtype=float)
    side = max(float(side), 1e-12)
    period = 2.0 * side
    folded = np.mod(arr, period)
    return np.where(folded <= side, folded, period - folded)


def _simx_sample_ids(rng: np.random.Generator, ids: np.ndarray, n: int, weights: np.ndarray | None = None) -> np.ndarray:
    # Pick host IDs or cluster IDs without replacement.
    # If weights are supplied, larger weights mean that item is more likely to be chosen.
    """Sample host identifiers without replacement, optionally using selection weights."""
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


def _simx_allocate_counts_capped_local(
    total: int,
    capacity: np.ndarray,
    relative_risk: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate an exact count across clusters without exceeding capacity.

    Parameters
    ----------
    total:
        Total number of items to allocate, for example infected hosts.
    capacity:
        Maximum number that each cluster can contain.  For infection this is
        the number of hosts in the cluster.
    relative_risk:
        A *per-host* relative-risk score for each cluster.  Cluster size must
        not be included here because this function already multiplies risk by
        the number of remaining hosts in the cluster.
    rng:
        Random-number generator used for the multinomial allocation.

    Notes
    -----
    At each step the chance that the next item goes to a cluster is
    proportional to ``relative_risk * remaining_capacity``.  Consequently,
    equal relative risks allocate hosts approximately in proportion to cluster
    population, while a risk score above one raises the per-host chance of
    infection in that cluster.
    """
    total = int(max(0, total))
    capacity = np.maximum(0, np.asarray(capacity, dtype=int))
    out = np.zeros(len(capacity), dtype=int)
    remaining = min(total, int(capacity.sum()))
    if remaining <= 0:
        return out

    relative_risk = np.maximum(0.0, np.asarray(relative_risk, dtype=float))
    while remaining > 0:
        available = capacity - out
        mask = available > 0
        if not np.any(mask):
            break

        allocation_weight = relative_risk[mask] * available[mask].astype(float)
        if float(allocation_weight.sum()) <= 0 or not np.isfinite(allocation_weight).all():
            allocation_weight = available[mask].astype(float)

        draw = rng.multinomial(remaining, allocation_weight / float(allocation_weight.sum()))
        draw = np.minimum(draw, available[mask])
        cluster_index = np.flatnonzero(mask)
        out[cluster_index] += draw

        new_remaining = total - int(out.sum())
        if new_remaining == remaining:
            # Rounding/capping can occasionally leave the multinomial step
            # unable to place an item.  Place one item explicitly so the loop
            # always makes progress.
            chosen = rng.choice(
                cluster_index,
                p=available[mask] / float(available[mask].sum()),
            )
            out[int(chosen)] += 1
            new_remaining -= 1
        remaining = max(0, new_remaining)
    return out


def _simx_host_counts_from_area(total_hosts: int, areas: np.ndarray) -> np.ndarray:
    # Convert cluster areas into host counts.
    # Bigger clusters get more hosts, and all counts are adjusted so the final
    # total equals the requested total host population.
    """Allocate an exact total host count across polygons in proportion to their areas."""
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
    """Expand cluster-level host counts into the host-ID representation used by the simulator."""
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
    host_clustering_level: str | None = None,
) -> dict[str, Any]:
    # Create a fake landscape from scratch.
    # A custom landscape can still be requested with only n_clusters, but the
    # Monte Carlo studies pass an explicit host_clustering_level so the host
    # pattern classes are spatially distinct rather than being driven only by
    # the number of synthetic clusters.
    """Generate a finite synthetic host landscape with configurable spatial clustering."""
    n_clusters = max(1, int(n_clusters))
    # Put the synthetic clusters inside a square with roughly the requested area.
    side_km = np.sqrt(max(float(total_area_km2), 1e-9))
    level = None if host_clustering_level is None else str(host_clustering_level).lower()
    if level not in {None, "diffuse", "low", "medium", "high"}:
        raise ValueError(f"Unsupported host clustering level: {host_clustering_level}")

    if level == "diffuse":
        cfg = {
            "n_parents": n_clusters,
            "bg_share": 1.0,
            "parent_shape": 1.0,
            "parent_contrast": 1.0,
            "parent_sd": 0.0,
            "local_area_shape": 0.8,
        }
    elif level == "low":
        cfg = {
            "n_parents": max(24, int(round(n_clusters / 14))),
            "bg_share": 0.16,
            "parent_shape": 0.9,
            "parent_contrast": 1.25,
            "parent_sd": 0.085 * side_km,
            "local_area_shape": 0.75,
        }
    elif level == "medium":
        cfg = {
            "n_parents": max(12, int(round(n_clusters / 22))),
            "bg_share": 0.08,
            "parent_shape": 0.55,
            "parent_contrast": 1.9,
            "parent_sd": 0.055 * side_km,
            "local_area_shape": 0.7,
        }
    elif level == "high":
        cfg = {
            "n_parents": max(6, int(round(n_clusters / 32))),
            "bg_share": 0.03,
            "parent_shape": 0.28,
            "parent_contrast": 2.8,
            "parent_sd": 0.028 * side_km,
            "local_area_shape": 0.65,
        }
    else:
        # Custom user-specified landscapes fall back to a generic diffuse pattern.
        cfg = {
            "n_parents": n_clusters,
            "bg_share": 1.0,
            "parent_shape": 1.0,
            "parent_contrast": 1.0,
            "parent_sd": 0.0,
            "local_area_shape": 0.8,
        }

    n_parents = min(n_clusters, int(cfg["n_parents"]))
    n_background = int(round(float(cfg["bg_share"]) * n_clusters))
    n_background = min(n_clusters, max(0, n_background))
    n_clustered = max(0, n_clusters - n_background)
    parent_x = rng.uniform(0.0, side_km, size=n_parents)
    parent_y = rng.uniform(0.0, side_km, size=n_parents)
    parent_coords = np.column_stack([parent_x, parent_y])
    if n_clustered > 0:
        parent_weight = rng.gamma(shape=float(cfg["parent_shape"]), scale=1.0, size=n_parents)
        parent_weight = np.maximum(parent_weight, 1e-12)
        parent_weight = np.power(parent_weight / float(parent_weight.sum()), float(cfg["parent_contrast"]))
        parent_weight = parent_weight / float(parent_weight.sum())
        assigned_parent = rng.choice(np.arange(n_parents, dtype=int), size=n_clustered, replace=True, p=parent_weight)
        clustered_xy = parent_coords[assigned_parent] + rng.normal(0.0, float(cfg["parent_sd"]), size=(n_clustered, 2))
        clustered_xy[:, 0] = _simx_reflect_to_square_local(clustered_xy[:, 0], side_km)
        clustered_xy[:, 1] = _simx_reflect_to_square_local(clustered_xy[:, 1], side_km)
    else:
        assigned_parent = np.asarray([], dtype=int)
        clustered_xy = np.empty((0, 2), dtype=float)
    if n_background > 0:
        background_xy = np.column_stack(
            [
                rng.uniform(0.0, side_km, size=n_background),
                rng.uniform(0.0, side_km, size=n_background),
            ]
        )
    else:
        background_xy = np.empty((0, 2), dtype=float)
    coords = np.vstack([clustered_xy, background_xy])
    if len(coords) != n_clusters:
        raise RuntimeError("Synthetic host landscape generation did not create the requested number of clusters.")

    # Areas inherit both parent-level variation and local variation so dense host
    # regions contain several moderately sized neighbouring clusters rather than
    # identical patches.
    local_area = rng.gamma(shape=float(cfg["local_area_shape"]), scale=1.0, size=n_clusters)
    local_area = np.maximum(local_area, 1e-12)
    if n_clustered > 0:
        parent_area_weight = rng.gamma(shape=max(0.25, float(cfg["parent_shape"])), scale=1.0, size=n_parents)
        parent_area_weight = np.maximum(parent_area_weight, 1e-12)
        parent_area_weight = parent_area_weight / float(parent_area_weight.sum())
        area_parent = np.concatenate([parent_area_weight[assigned_parent], np.full(n_background, np.mean(parent_area_weight))])
    else:
        area_parent = np.ones(n_clusters, dtype=float)
    areas = np.maximum(local_area * area_parent, 1e-12)
    areas = areas / float(areas.sum()) * float(total_area_km2)
    clusters = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "area_km2": areas,
        }
    )
    source = "Synthetic clusters" if level is None else f"Synthetic clusters ({level})"
    host = _simx_make_host_id_landscape(clusters, total_area_km2, host_density_km2, source)
    host["requested_n_clusters"] = int(n_clusters)
    host["host_clustering_level"] = "custom" if level is None else level
    host["host_parent_count"] = int(n_parents)
    return host


def _simx_make_macro_host_landscape_from_polygons(host_gdf: Any | None, host_density_km2: float) -> dict[str, Any]:
    # Convert real host polygons into the same structure as the synthetic landscape.
    # Each polygon becomes one cluster, and host count is estimated from area.
    """Convert real host polygons into the simulator host-ID landscape using polygon area and host density."""
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
    """Aggregate host-level infection status to physical clusters for one epidemic round."""
    out = clusters.copy()
    out["round_id"] = int(round_id)
    out["year"] = float(year)
    out["true_prev"] = float(true_prev)
    out["infected_count"] = np.asarray(infected_counts, dtype=int)
    out["cluster_prev"] = out["infected_count"].astype(float) / np.maximum(out["host_count"].astype(float), 1.0)
    out["neighbor_infected"] = np.asarray(neighbor_pressure, dtype=float)
    out["sampled_n"] = 0
    return out



def _simx_epidemic_round_diagnostics(round_df: pd.DataFrame) -> dict[str, Any]:
    """Describe the realised spatial infection pattern in one epidemic year.

    These are *realised* diagnostics: they describe what the generator actually
    produced, rather than relying only on labels such as ``low`` or ``high``.
    They are useful for checking that a scenario behaves as intended and for
    explaining why a survey did or did not perform well.

    ``infection_icc_anova`` is an approximate host-level intracluster
    correlation calculated from cluster infection counts.  It is constrained
    to the interval 0--1 because the report uses it as a descriptive clustering
    measure rather than as an unrestricted variance-component estimate.
    """
    empty = {
        "host_weighted_prevalence": np.nan,
        "unweighted_cluster_prevalence": np.nan,
        "infected_cluster_count": 0,
        "infected_cluster_fraction": np.nan,
        "infection_icc_anova": np.nan,
        "cluster_size_prevalence_correlation": np.nan,
        "top_10pct_cluster_infection_share": np.nan,
    }
    if round_df is None or round_df.empty:
        return empty

    host_count = (
        pd.to_numeric(round_df["host_count"], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    infected_count = (
        pd.to_numeric(round_df["infected_count"], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    cluster_prev = infected_count / np.maximum(host_count, 1.0)

    total_hosts = float(host_count.sum())
    total_infected = float(infected_count.sum())
    weighted_prev = total_infected / total_hosts if total_hosts > 0 else np.nan
    cluster_mean = float(np.mean(cluster_prev)) if len(cluster_prev) else np.nan
    infected_mask = infected_count > 0

    if len(host_count) > 1 and np.std(host_count) > 0 and np.std(cluster_prev) > 0:
        size_prev_corr = float(np.corrcoef(host_count, cluster_prev)[0, 1])
    else:
        size_prev_corr = np.nan

    # One-way random-effects ANOVA ICC for unequal cluster sizes.  The within
    # sum of squares can be calculated directly from binary cluster counts:
    # n_i * p_i * (1-p_i).
    k = int(np.sum(host_count > 0))
    N = float(host_count.sum())
    infection_icc = np.nan
    if k > 1 and N > k and np.isfinite(weighted_prev):
        valid = host_count > 0
        n_i = host_count[valid]
        p_i = cluster_prev[valid]
        ss_between = float(np.sum(n_i * np.square(p_i - weighted_prev)))
        ms_between = ss_between / max(k - 1, 1)
        ss_within = float(np.sum(n_i * p_i * (1.0 - p_i)))
        ms_within = ss_within / max(N - k, 1.0)
        # Effective mean cluster size for an unequal-size one-way design.
        m0 = float((N - np.sum(np.square(n_i)) / max(N, 1.0)) / max(k - 1, 1))
        denominator = ms_between + (m0 - 1.0) * ms_within
        if np.isfinite(denominator) and denominator > 0:
            infection_icc = float(
                np.clip((ms_between - ms_within) / denominator, 0.0, 1.0)
            )

    top_n = max(1, int(np.ceil(0.10 * len(infected_count)))) if len(infected_count) else 0
    top_share = (
        float(np.sort(infected_count)[-top_n:].sum() / total_infected)
        if top_n > 0 and total_infected > 0
        else np.nan
    )

    return {
        "host_weighted_prevalence": weighted_prev,
        "unweighted_cluster_prevalence": cluster_mean,
        "infected_cluster_count": int(infected_mask.sum()),
        "infected_cluster_fraction": float(infected_mask.mean()) if len(infected_mask) else np.nan,
        "infection_icc_anova": infection_icc,
        "cluster_size_prevalence_correlation": size_prev_corr,
        "top_10pct_cluster_infection_share": top_share,
    }


def _simx_sample_multistage_hosts(
    *,
    hosts_by_cluster: list[np.ndarray] | dict[int, np.ndarray],
    cluster_ids: np.ndarray,
    requested_n: int,
    blocked_host_ids: set[int],
    hosts_per_cluster_visit: int,
    rng: np.random.Generator,
    cluster_risk_multiplier: dict[int, float] | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Draw a probability-proportional-to-size multistage host sample.

    Field interpretation
    --------------------
    One draw represents one visit to a host cluster such as a field or forest.

    1. Select a cluster with probability proportional to the number of eligible
       hosts remaining in that cluster.
    2. Inspect up to ``hosts_per_cluster_visit`` hosts during that visit.
    3. Return the cluster to the selection pool while it still has unsampled
       eligible hosts, so a large cluster can be visited more than once.
    4. Continue until the requested host sample has been collected.

    This is preferable to selecting every cluster at most once.  With a
    once-only rule, a five-host cap eventually gives small and very large
    clusters almost equal influence as the sample grows.  Allowing PPS
    reselection preserves approximately equal host-level inclusion
    probabilities while retaining genuine within-visit clustering.

    Targeted scenarios multiply the PPS weights by a cluster risk score.  Those
    scenarios are deliberately preferential and therefore are not expected to
    estimate population prevalence without additional weighting.
    """
    requested_n = max(0, int(requested_n))
    visit_size = max(1, int(hosts_per_cluster_visit))
    risk_lookup = cluster_risk_multiplier or {}

    blocked_array = (
        np.fromiter(blocked_host_ids, dtype=int)
        if blocked_host_ids
        else np.asarray([], dtype=int)
    )

    eligible_cluster_ids: list[int] = []
    remaining_ids: list[np.ndarray] = []
    risk_values: list[float] = []

    for cid_raw in np.asarray(cluster_ids, dtype=int):
        cid = int(cid_raw)
        ids = np.asarray(hosts_by_cluster[cid], dtype=int)
        if len(blocked_array):
            ids = ids[~np.isin(ids, blocked_array)]
        if len(ids) == 0:
            continue
        eligible_cluster_ids.append(cid)
        remaining_ids.append(ids.copy())
        risk_values.append(max(0.0, float(risk_lookup.get(cid, 1.0))))

    if requested_n <= 0 or not eligible_cluster_ids:
        return np.asarray([], dtype=int), {}

    total_available = int(sum(len(ids) for ids in remaining_ids))
    remaining_to_sample = min(requested_n, total_available)
    risk_array = np.asarray(risk_values, dtype=float)

    sampled_parts: list[np.ndarray] = []
    visit_counts: dict[int, int] = {}

    while remaining_to_sample > 0:
        capacities = np.asarray([len(ids) for ids in remaining_ids], dtype=float)
        active = capacities > 0
        if not np.any(active):
            break

        # PPS uses the number of currently eligible hosts.  This means that a
        # host in a large cluster is not penalised simply because their cluster
        # contains many hosts.
        weights = capacities[active] * risk_array[active]
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            weights = capacities[active]

        active_positions = np.flatnonzero(active)
        chosen_position = int(
            rng.choice(active_positions, p=weights / float(weights.sum()))
        )
        cid = int(eligible_cluster_ids[chosen_position])
        ids = remaining_ids[chosen_position]

        take = min(visit_size, remaining_to_sample, len(ids))
        chosen_local = np.asarray(
            rng.choice(len(ids), size=take, replace=False),
            dtype=int,
        )
        chosen_ids = ids[chosen_local]
        sampled_parts.append(chosen_ids.astype(int))

        # Remove inspected hosts so the same host cannot be selected twice in a
        # single survey round.  The physical cluster remains eligible for a
        # later visit if it still contains unsampled hosts.
        keep = np.ones(len(ids), dtype=bool)
        keep[chosen_local] = False
        remaining_ids[chosen_position] = ids[keep]

        visit_counts[cid] = visit_counts.get(cid, 0) + 1
        remaining_to_sample -= int(take)

    sampled = (
        np.concatenate(sampled_parts).astype(int)
        if sampled_parts
        else np.asarray([], dtype=int)
    )
    return sampled, visit_counts


# ============================================================================
# 9. Survey execution engine
# ============================================================================

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
    initial_sampled_hosts_df: pd.DataFrame | None = None,
    exclude_nonretained_panel_hosts: bool = False,
) -> dict[str, Any]:
    """Simulate all survey rounds for one monitoring method.

    The function supports three pieces of survey design:

    * ``sampling_mode`` chooses individual-host SRS or PPS cluster-visit MSS;
    * ``overlap_type`` controls deliberate reinspection of earlier hosts; and
    * ``targeting_level`` optionally favours clusters with higher infection.

    The returned ``survey_df`` contains both requested and achieved sample
    sizes, cluster coverage and shortfall indicators.  These diagnostics make
    it visible if a design cannot deliver its nominal sample size.
    """
    rounds = epidemic["rounds"]
    hosts = epidemic["hosts"]
    hosts_by_cluster = epidemic["hosts_by_cluster"]
    total_hosts = len(hosts)
    host_cluster = hosts["cluster_id"].to_numpy(dtype=int)
    n_rounds = len(rounds)

    if np.isscalar(n_hosts_per_round):
        requested_sizes = [int(n_hosts_per_round)] * n_rounds
    else:
        requested_sizes = [int(x) for x in list(n_hosts_per_round)]
        requested_sizes = (
            requested_sizes
            + [requested_sizes[-1] if requested_sizes else 0] * n_rounds
        )[:n_rounds]

    overlap_type = str(overlap_type)
    targeting_level = str(targeting_level)
    targeted = str(targeted_mode) != "none" and targeting_level != "random"
    target_multiplier = (
        1.0
        if not targeted
        else {"low": 2.0, "medium": 5.0, "high": 10.0}.get(targeting_level, 5.0)
    )
    overlap = (
        1.0
        if overlap_type == "longitudinal"
        else float(overlap_prop if overlap_type == "rotating_panel" else 0.0)
    )

    survey_rows: list[dict[str, Any]] = []
    round_tables: list[pd.DataFrame] = []
    sampled_hosts_by_round: list[pd.DataFrame] = []
    previous_nonempty_sample = np.asarray([], dtype=int)

    for round_index, round_df in enumerate(rounds):
        n_requested = max(0, int(requested_sizes[round_index]))
        n_target = min(n_requested, total_hosts)
        infected = np.asarray(epidemic["infected_by_round"][round_index], dtype=bool)
        use_shared_initial = (
            round_index == 0
            and initial_sampled_hosts_df is not None
            and not initial_sampled_hosts_df.empty
        )

        visit_counts: dict[int, int] = {}
        if use_shared_initial:
            supplied = initial_sampled_hosts_df.copy()
            supplied_ids = (
                pd.to_numeric(supplied.get("host_id", pd.Series(dtype=float)), errors="coerce")
                .dropna()
                .astype(int)
            )
            sampled = supplied_ids[
                (supplied_ids >= 0) & (supplied_ids < total_hosts)
            ].to_numpy(dtype=int)
            sample_source = np.repeat("shared_initial", len(sampled))

            if "detected" in supplied.columns and len(supplied) == len(sampled):
                detected = (
                    pd.to_numeric(supplied["detected"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .to_numpy(dtype=int)
                )
            else:
                detected_probability = np.where(
                    infected[sampled], float(method_sens), 0.0
                ) if len(sampled) else np.asarray([], dtype=float)
                detected = rng.binomial(
                    1,
                    np.clip(detected_probability, 0.0, 1.0),
                ).astype(int) if len(sampled) else np.asarray([], dtype=int)
        else:
            reused = np.asarray([], dtype=int)
            if (
                round_index > 0
                and overlap > 0
                and previous_nonempty_sample.size > 0
                and n_target > 0
            ):
                reuse_n = min(
                    int(round(n_target * overlap)),
                    len(previous_nonempty_sample),
                )
                reused = _simx_sample_ids(
                    rng,
                    previous_nonempty_sample,
                    reuse_n,
                )

            new_n = max(0, n_target - len(reused))
            # In a controlled overlap experiment, replacement hosts are drawn
            # from outside the previous panel. This makes the requested overlap
            # proportion the actual design intervention rather than allowing
            # accidental reselection among the nominally "new" observations.
            blocked_ids = (
                previous_nonempty_sample
                if exclude_nonretained_panel_hosts and previous_nonempty_sample.size > 0
                else reused
            )
            blocked = set(np.asarray(blocked_ids, dtype=int).tolist())

            if str(sampling_mode) == "multistage":
                cluster_df = round_df.copy().reset_index(drop=True)
                cluster_prev = cluster_df.set_index("cluster_id")["cluster_prev"].to_dict()
                risk_multiplier = (
                    {
                        int(cid): 1.0 + target_multiplier * float(cluster_prev.get(int(cid), 0.0))
                        for cid in cluster_df["cluster_id"].to_numpy(dtype=int)
                    }
                    if targeted
                    else None
                )
                fresh, visit_counts = _simx_sample_multistage_hosts(
                    hosts_by_cluster=hosts_by_cluster,
                    cluster_ids=cluster_df["cluster_id"].to_numpy(dtype=int),
                    requested_n=new_n,
                    blocked_host_ids=blocked,
                    hosts_per_cluster_visit=within_cluster_cap,
                    rng=rng,
                    cluster_risk_multiplier=risk_multiplier,
                )
            else:
                candidate_ids = np.arange(total_hosts, dtype=int)
                if blocked:
                    candidate_ids = candidate_ids[
                        ~np.isin(candidate_ids, np.fromiter(blocked, dtype=int))
                    ]
                weights = (
                    np.where(infected[candidate_ids], target_multiplier, 1.0)
                    if targeted
                    else None
                )
                fresh = _simx_sample_ids(rng, candidate_ids, new_n, weights)

            sampled = (
                np.concatenate([reused, fresh]).astype(int)
                if len(reused) or len(fresh)
                else np.asarray([], dtype=int)
            )
            sample_source = np.concatenate(
                [
                    np.repeat("panel_reinspection", len(reused)),
                    np.repeat("new_sample", len(fresh)),
                ]
            ) if len(sampled) else np.asarray([], dtype=str)

            detected_probability = np.where(
                infected[sampled], float(method_sens), 0.0
            ) if len(sampled) else np.asarray([], dtype=float)
            detected = rng.binomial(
                1,
                np.clip(detected_probability, 0.0, 1.0),
            ).astype(int) if len(sampled) else np.asarray([], dtype=int)

        # Preserve the most recent observed panel across years in which a method
        # does not survey.  This is essential for the Change Method, which has
        # no observations in years 1--4 but may deliberately revisit year-0
        # hosts at year 5.
        if sampled.size > 0:
            previous_nonempty_sample = sampled.copy()

        y = int(detected.sum())
        achieved_n = int(len(sampled))
        prevalence_hat = float(y / achieved_n) if achieved_n else np.nan
        sampled_cluster_ids = host_cluster[sampled] if achieved_n else np.asarray([], dtype=int)
        unique_clusters, per_cluster_n = (
            np.unique(sampled_cluster_ids, return_counts=True)
            if achieved_n
            else (np.asarray([], dtype=int), np.asarray([], dtype=int))
        )

        survey_rows.append(
            {
                "round_id": int(round_df["round_id"].iloc[0]),
                "year": float(round_df["year"].iloc[0]),
                "n_requested": int(n_requested),
                "n_planned": int(n_target),
                "n": achieved_n,
                "n_shortfall": int(max(0, n_requested - achieved_n)),
                "sample_fraction": float(achieved_n / total_hosts) if total_hosts else np.nan,
                "clusters_sampled": int(len(unique_clusters)),
                "cluster_selection_draws": int(sum(visit_counts.values())),
                "mean_hosts_per_sampled_cluster": float(np.mean(per_cluster_n)) if len(per_cluster_n) else 0.0,
                "max_hosts_in_sampled_cluster": int(np.max(per_cluster_n)) if len(per_cluster_n) else 0,
                "sampling_complete": bool(achieved_n == n_requested),
                "y": y,
                "prev_hat": prevalence_hat,
                "true_prev": float(round_df["true_prev"].iloc[0]),
            }
        )

        sampled_df = pd.DataFrame(
            {
                "host_id": sampled,
                "cluster_id": sampled_cluster_ids,
                "infected": infected[sampled].astype(int) if achieved_n else np.asarray([], dtype=int),
                "detected": detected,
                "sample_source": sample_source,
                "round_id": int(round_df["round_id"].iloc[0]),
            }
        )
        sampled_hosts_by_round.append(sampled_df)

        cluster_sample = (
            sampled_df.groupby("cluster_id", as_index=False)
            .size()
            .rename(columns={"size": "sampled_n"})
            if not sampled_df.empty
            else pd.DataFrame(columns=["cluster_id", "sampled_n"])
        )
        plot_round = round_df.copy()
        plot_round = (
            plot_round.drop(columns=["sampled_n"], errors="ignore")
            .merge(cluster_sample, on="cluster_id", how="left")
        )
        plot_round["sampled_n"] = (
            pd.to_numeric(plot_round["sampled_n"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        round_tables.append(plot_round)

    survey_df = pd.DataFrame(survey_rows)
    return {
        "rounds": round_tables,
        "survey_df": survey_df,
        "sampled_hosts_by_round": sampled_hosts_by_round,
        "sampling_mode": str(sampling_mode),
        "overlap_type": str(overlap_type),
        "multistage_design": (
            MULTISTAGE_DESIGN_REVISION
            if str(sampling_mode) == "multistage"
            else "not_applicable"
        ),
        "all_planned_samples_achieved": bool(
            survey_df["sampling_complete"].all()
        ) if not survey_df.empty else True,
    }



# ============================================================================
# 10. Simulator planning workflow
# ============================================================================
# This function deliberately mirrors the research design: shared baseline,
# Regression Method annual follow-up allocation, Change Method endpoint survey.

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
    planning_alpha: float,
    planning_power: float,
    change_method_delta: float,
    change_method_corr: float,
    regression_method_target: float,
    cluster_inflation_mode: str,
    fixed_phi: float,
    rng: np.random.Generator,
    method_sens: float = 1.0,
) -> dict[str, Any]:
    # Estimate required sample sizes for the simulator.
    # This uses the same Change Method / Regression Method formulas as the dashboard,
    # but the pilot survey is now drawn from host IDs rather than cells.
    """Plan the shared baseline, Regression Method follow-up, and Change Method endpoint sample sizes."""
    del p0_mode, assumed_p0
    schedule = epidemic["schedule"].copy()
    total_hosts = int(len(epidemic["hosts"]))
    n_initial_raw = compute_n_baseline_prevalence_cells(
        p0_upper=DEFAULT_BASELINE_PREVALENCE_UPPER,
        conf_level=DEFAULT_BASELINE_PREVALENCE_CONF,
        width=DEFAULT_BASELINE_PREVALENCE_WIDTH,
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
        method_sens=float(method_sens),
        rng=rng,
    )
    p0_hat = float(pilot["survey_df"]["prev_hat"].iloc[0]) if not pilot["survey_df"].empty else float(schedule["true_prev"].iloc[0])
    p0_design = min(max(p0_hat, 1e-8), 1.0 - 1e-8)
    # phi is the inflation factor for clustered sampling.  If phi is 2, the
    # required sample size is doubled to account for correlation within clusters.
    if cluster_inflation_mode == "fixed":
        phi_info = {"phi": max(1.0, float(fixed_phi)), "rho": np.nan, "m": float(within_cluster_cap), "m_observed": float(within_cluster_cap), "estimator": "fixed"}
    elif cluster_inflation_mode == "pilot":
        phi_info = _simx_estimate_design_effect(
            pilot.get("sampled_hosts_by_round", [pd.DataFrame()])[0],
            # Use the realised mean number of inspected hosts per physical
            # cluster.  A cluster may be revisited, so the visit cap is not the
            # same quantity as the final cluster sample size in the ICC formula.
            planned_m=None if sampling_mode == "multistage" else 1,
        )
    else:
        phi_info = {"phi": 1.0, "rho": 0.0, "m": 1.0 if sampling_mode == "srs" else float(within_cluster_cap), "m_observed": 1.0 if sampling_mode == "srs" else float(within_cluster_cap), "estimator": "none"}
    t_vec = schedule["year"].to_numpy(dtype=float)[1:]
    # Regression Method: calculate the total sample size needed after the pilot,
    # then spread it across the remaining survey rounds.
    regression_total_followup_n = compute_n_total_regression_method_cells(
        pi0=p0_design,
        pi_target=regression_method_target,
        alpha=planning_alpha,
        power=planning_power,
        t_vec=t_vec,
        population_size=total_hosts,
        phi=float(phi_info["phi"]),
        initial_n=int(pilot["survey_df"]["n"].iloc[0]) if not pilot["survey_df"].empty else int(n_initial),
    )
    regression_followup_n_per_round = int(np.ceil(regression_total_followup_n / max(len(t_vec), 1)))
    corr_info = _effective_change_method_correlation(
        requested_corr=change_method_corr,
        overlap_prop=overlap_prop,
        p0_design=p0_design,
        detectable_change=change_method_delta,
    )
    change_method_n_per_endpoint = compute_n_change_method_cells(
        p0=p0_design,
        delta=change_method_delta,
        corr=float(corr_info["used"]),
        alpha=planning_alpha,
        power=planning_power,
        population_size=total_hosts,
        design_effect=float(phi_info["phi"]),
    )
    table = pd.DataFrame(
        [
            {"metric": "Pilot initial sample size", "value": int(n_initial)},
            {"metric": "Pilot prevalence estimate", "value": round(float(p0_hat), 4)},
            {"metric": "Design prevalence used in formulas", "value": round(float(p0_design), 4)},
            {"metric": "Clustering inflation phi", "value": round(float(phi_info["phi"]), 4)},
            {"metric": "Phi estimator", "value": phi_info["estimator"]},
            {"metric": "Alpha used", "value": round(float(planning_alpha), 4)},
            {"metric": "Power used", "value": round(float(planning_power), 4)},
            {"metric": "Change Method detectable change", "value": round(float(change_method_delta), 4)},
            {"metric": "Requested endpoint correlation", "value": round(float(corr_info["requested"]), 4)},
            {"metric": "Feasible endpoint correlation used", "value": round(float(corr_info["used"]), 4)},
            {"metric": "Correlation capped for feasibility", "value": bool(corr_info["capped"])},
            {"metric": "Regression Method design prevalence", "value": round(float(regression_method_target), 4)},
            {"metric": "Regression Method recommended per round", "value": int(max(1, regression_followup_n_per_round))},
            {"metric": "Change Method recommended per round", "value": int(max(1, change_method_n_per_endpoint))},
        ]
    )
    return {
        "n_initial": int(n_initial),
        "regression_method_n_per_round": int(max(1, regression_followup_n_per_round)),
        "change_method_n_per_round": int(max(1, change_method_n_per_endpoint)),
        "regression_method_n_total": int(max(1, regression_total_followup_n)),
        "p0_hat": float(p0_hat),
        "p0_design": float(p0_design),
        "change_method_corr_requested": float(corr_info["requested"]),
        "change_method_corr_used": float(corr_info["used"]),
        "change_method_corr_host_max": float(corr_info["host_corr_max"]),
        "change_method_corr_feasible_bound": float(corr_info["estimator_corr_bound"]),
        "change_method_corr_capped": bool(corr_info["capped"]),
        "change_method_p1_design": float(corr_info["p1_design"]),
        "phi": phi_info,
        "pilot_round": pilot["rounds"][0],
        "pilot_survey": pilot,
        "pilot_n_achieved": int(pilot["survey_df"]["n"].iloc[0]) if not pilot["survey_df"].empty else 0,
        "table": table,
    }



# ============================================================================
# 11. Simulator summary tables and plots
# ============================================================================

def _simx_summary_table(host_landscape: dict[str, Any] | None) -> pd.DataFrame:
    # Small table shown in the simulator to describe the generated landscape.
    """Return high-level host-landscape diagnostics for simulator display."""
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
    """Expand method-specific planning outputs into the sample size requested at every survey round."""
    if epidemic is None or effort is None:
        return pd.DataFrame(columns=["round_id", "year", "regression_n", "change_n"])
    schedule = epidemic["schedule"].copy()
    rows = []
    n_rounds = len(schedule)
    for i, row in enumerate(schedule.itertuples(index=False), start=1):
        regression_n = int(effort["n_initial"]) if i == 1 else int(effort["regression_method_n_per_round"])
        # Both methods use the same initial survey.  The Change Method then has
        # no observations until its separately planned final endpoint survey.
        if i == 1:
            change_n = int(effort["n_initial"])
        elif i == n_rounds:
            change_n = int(effort["change_method_n_per_round"])
        else:
            change_n = 0
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
    """Summarise the number and size of simulated host clusters."""
    if host_landscape is None or host_landscape.get("clusters") is None:
        return pd.DataFrame(columns=["cells_per_cluster", "n_clusters"])
    sizes = pd.to_numeric(host_landscape["clusters"]["host_count"], errors="coerce").fillna(0).astype(int)
    out = sizes.value_counts().sort_index().reset_index()
    out.columns = ["cells_per_cluster", "n_clusters"]
    return out


def _simx_round_summary(epidemic: dict[str, Any], surveys: dict[str, Any] | None = None) -> pd.DataFrame:
    # Join the true epidemic state with any simulated survey estimates.
    # This is the table shown under the survey-design page.
    """Combine true epidemic prevalence with realised survey summaries by round."""
    rows = []
    survey_df_e = pd.DataFrame()
    survey_df_c = pd.DataFrame()
    if surveys:
        if "regression_method" in surveys:
            survey_df_e = surveys["regression_method"]["survey_df"]
        elif "static" in surveys:
            survey_df_e = surveys["static"]["survey_df"]
        if "change_method" in surveys:
            survey_df_c = surveys["change_method"]["survey_df"]
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
    """Compare prevalence estimates from targeted and untargeted simulated surveys."""
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
    """Plot the host landscape coloured by infection status for one simulation round."""
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
    """Plot the planned Change Method and Regression Method sample sizes by survey round."""
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
    """Plot the known simulated prevalence trajectory."""
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
    """Plot targeted and untargeted survey estimates against the known prevalence trajectory."""
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


# Main epidemic builder used by the simulator and Monte Carlo studies.
# Clustered scenarios choose hotspot centres from the realised host clusters
# themselves, then weight infection strongly by distance to those hotspot
# clusters. Each round still matches the requested total prevalence exactly.

# ============================================================================
# 12. Spatial epidemic construction
# ============================================================================
# The simulator allocates exact prevalence totals while varying where infected
# hosts are concentrated.  This is not a transmission model.

def _simx_make_macro_epidemic(
    host_landscape: dict[str, Any],
    prevalence_schedule: pd.DataFrame,
    infection_clustering_level: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Allocate infection across a finite host landscape for every scheduled prevalence round."""
    clusters = host_landscape["clusters"].copy().reset_index(drop=True)
    host_counts = clusters["host_count"].to_numpy(dtype=int)
    total_hosts = int(host_counts.sum())
    cluster_for_host = np.asarray(host_landscape["cluster_id_by_host"], dtype=int)
    hosts_by_cluster = host_landscape["hosts_by_cluster"]
    level = str(infection_clustering_level).lower()
    coords = clusters[["x", "y"]].to_numpy(dtype=float)
    if level not in {"random", "low", "medium", "high"}:
        raise ValueError(f"Unsupported infection clustering level: {infection_clustering_level}")

    positive_dist = []
    if len(coords) > 1:
        deltas = coords[:, None, :] - coords[None, :, :]
        dmat = np.sqrt(np.sum(deltas * deltas, axis=2))
        positive_dist = dmat[np.isfinite(dmat) & (dmat > 0)]
    distance_scale = float(np.median(positive_dist)) if np.size(positive_dist) else 1.0

    pattern_cfg = {
        "random": {
            "pattern": "random",
            "hotspot_rate_per_100": 0.0,
            "hotspot_rate_sd_fraction": 0.0,
            "hotspot_min_n": 0,
            "hotspot_decay_mult": 0.0,
            "hotspot_strength_shape": 1.0,
            "concentration_base": 1.0,
            "concentration_lowprev_gain": 0.0,
            "field_floor": 1.0,
            "background_weight": 1.0,
            "growth_midpoint": 0.0,
            "growth_steepness": 1.0,
            "growth_power": 1.0,
            "growth_floor": 1.0,
        },
        "low": {
            "pattern": "clustered",
            "hotspot_rate_per_100": 8.0,
            "hotspot_rate_sd_fraction": 0.24,
            "hotspot_min_n": 6,
            "hotspot_decay_mult": 0.15,
            "hotspot_strength_shape": 1.00,
            "concentration_base": 1.45,
            "concentration_lowprev_gain": 0.55,
            "field_floor": 0.003,
            "background_weight": 0.05,
            "growth_midpoint": 0.21,
            "growth_steepness": 6.0,
            "growth_power": 1.10,
            "growth_floor": 0.05,
        },
        "medium": {
            "pattern": "clustered",
            "hotspot_rate_per_100": 5.0,
            "hotspot_rate_sd_fraction": 0.22,
            "hotspot_min_n": 4,
            "hotspot_decay_mult": 0.13,
            "hotspot_strength_shape": 1.05,
            "concentration_base": 1.65,
            "concentration_lowprev_gain": 0.75,
            "field_floor": 0.002,
            "background_weight": 0.04,
            "growth_midpoint": 0.24,
            "growth_steepness": 7.0,
            "growth_power": 1.2,
            "growth_floor": 0.03,
        },
        "high": {
            "pattern": "clustered",
            "hotspot_rate_per_100": 2.2,
            "hotspot_rate_sd_fraction": 0.20,
            "hotspot_min_n": 2,
            "hotspot_decay_mult": 0.095,
            "hotspot_strength_shape": 1.10,
            "concentration_base": 2.20,
            "concentration_lowprev_gain": 1.25,
            "field_floor": 0.0005,
            "background_weight": 0.02,
            "growth_midpoint": 0.28,
            "growth_steepness": 9.5,
            "growth_power": 1.45,
            "growth_floor": 0.012,
        },
    }[level]

    infected = np.zeros(total_hosts, dtype=bool)
    infected_by_round: list[np.ndarray] = []
    round_tables: list[pd.DataFrame] = []
    achieved_prevalence: list[float] = []
    hotspot_n = 0
    hotspot_field = np.ones(len(clusters), dtype=float)
    hotspots = pd.DataFrame(columns=["x", "y"])

    def infected_counts_by_cluster() -> np.ndarray:
        return np.bincount(cluster_for_host, weights=infected.astype(float), minlength=len(clusters)).astype(int)

    def cluster_prev_from_counts(inf_counts: np.ndarray) -> np.ndarray:
        return inf_counts.astype(float) / np.maximum(host_counts.astype(float), 1.0)

    def score_clusters_against_hotspots() -> np.ndarray:
        nonlocal hotspot_n, hotspot_field, hotspots
        if level == "random" or len(clusters) == 0:
            hotspot_field = np.ones(len(clusters), dtype=float)
            hotspot_n = 0
            hotspots = pd.DataFrame(columns=["x", "y"])
            return hotspot_field
        mean = float(pattern_cfg["hotspot_rate_per_100"]) * max(int(len(clusters)), 1) / 100.0
        sd = max(0.75, mean * float(pattern_cfg["hotspot_rate_sd_fraction"]))
        hotspot_n = int(round(rng.normal(mean, sd)))
        hotspot_n = max(int(pattern_cfg["hotspot_min_n"]), min(int(len(clusters)), hotspot_n))
        hotspot_n = max(1, min(int(len(clusters)), int(hotspot_n)))
        hotspot_idx = np.asarray(rng.choice(len(clusters), size=hotspot_n, replace=False), dtype=int)
        hotspot_coords = coords[hotspot_idx]
        hotspot_strength = rng.gamma(shape=float(pattern_cfg["hotspot_strength_shape"]), scale=1.0, size=hotspot_n)
        hotspot_strength = np.maximum(hotspot_strength, 1e-12)
        hotspot_scale = max(distance_scale * float(pattern_cfg["hotspot_decay_mult"]), 1e-12)
        d_hot = np.sqrt(np.sum((coords[:, None, :] - hotspot_coords[None, :, :]) ** 2, axis=2))
        kernel = np.exp(-(d_hot * d_hot) / (2.0 * hotspot_scale * hotspot_scale))
        score = np.max(kernel * hotspot_strength[None, :], axis=1)
        hotspot_field = _simx_scale01_local(score)
        if float(np.max(hotspot_field)) <= 0:
            hotspot_field = np.ones(len(clusters), dtype=float)
        hotspot_field = np.maximum(hotspot_field, float(pattern_cfg["field_floor"]))
        hotspots = pd.DataFrame({"x": hotspot_coords[:, 0], "y": hotspot_coords[:, 1]})
        return hotspot_field

    def target_cluster_counts_from_field(target_n: int) -> np.ndarray:
        target_n = int(max(0, min(target_n, total_hosts)))
        if target_n <= 0:
            return np.zeros(len(clusters), dtype=int)
        if pattern_cfg["pattern"] == "random":
            return _simx_allocate_counts_capped_local(target_n, host_counts, np.ones(len(clusters), dtype=float), rng)

        target_prev = target_n / max(total_hosts, 1)
        concentration = float(pattern_cfg["concentration_base"]) + float(pattern_cfg["concentration_lowprev_gain"]) * (1.0 - target_prev)
        concentration = max(0.55, concentration)
        field = np.maximum(hotspot_field, 0.0)
        if float(np.sum(field)) <= 0:
            field = np.ones(len(clusters), dtype=float)
        bg = float(pattern_cfg.get("background_weight", 0.0))
        field = bg + (1.0 - bg) * field
        current_counts = infected_counts_by_cluster()
        current_prev = current_counts.astype(float) / np.maximum(host_counts.astype(float), 1.0)
        if int(current_counts.sum()) > 0:
            growth_gate = 1.0 / (
                1.0
                + np.exp(
                    -float(pattern_cfg["growth_steepness"])
                    * (current_prev - float(pattern_cfg["growth_midpoint"]))
                )
            )
            growth_gate = float(pattern_cfg["growth_floor"]) + np.power(growth_gate, float(pattern_cfg["growth_power"]))
        else:
            growth_gate = np.ones(len(clusters), dtype=float)
        # These are per-host relative-risk scores.  Do not multiply by
        # host_counts here: _simx_allocate_counts_capped_local() already
        # accounts for the number of available hosts in every cluster.
        relative_risk = (
            np.power(np.clip(field, 1e-12, None), concentration)
            * np.maximum(growth_gate, 1e-12)
        )
        return _simx_allocate_counts_capped_local(
            target_n,
            host_counts,
            relative_risk,
            rng,
        )

    def set_cluster_counts(target_counts: np.ndarray) -> None:
        nonlocal infected
        current_counts = infected_counts_by_cluster()
        target_counts = np.minimum(np.maximum(np.asarray(target_counts, dtype=int), 0), host_counts)

        for cid in np.flatnonzero(target_counts > current_counts):
            need = int(target_counts[int(cid)] - current_counts[int(cid)])
            ids = hosts_by_cluster[int(cid)]
            candidates = ids[~infected[ids]]
            chosen = _simx_sample_ids(rng, candidates, need)
            infected[chosen] = True

        for cid in np.flatnonzero(target_counts < current_counts):
            drop_n = int(current_counts[int(cid)] - target_counts[int(cid)])
            ids = hosts_by_cluster[int(cid)]
            candidates = ids[infected[ids]]
            chosen = _simx_sample_ids(rng, candidates, drop_n)
            infected[chosen] = False

    def initialise_infections(target_n: int) -> None:
        nonlocal infected
        target_n = int(max(0, target_n))
        if target_n <= 0:
            return
        if pattern_cfg["pattern"] == "random":
            chosen = _simx_sample_ids(rng, np.arange(total_hosts, dtype=int), target_n)
            infected[chosen] = True
            return
        score_clusters_against_hotspots()
        target_counts = target_cluster_counts_from_field(target_n)
        set_cluster_counts(target_counts)

    def add_infections(add_n: int) -> None:
        nonlocal infected
        add_n = int(max(0, add_n))
        if add_n <= 0:
            return
        susceptible_ids = np.flatnonzero(~infected)
        if len(susceptible_ids) == 0:
            return
        if pattern_cfg["pattern"] == "random":
            chosen = _simx_sample_ids(rng, susceptible_ids, add_n)
            infected[chosen] = True
            return
        target_total = int(infected.sum()) + add_n
        target_counts = target_cluster_counts_from_field(target_total)
        set_cluster_counts(target_counts)

    def remove_infections(rem_n: int) -> None:
        nonlocal infected
        rem_n = int(max(0, rem_n))
        if rem_n <= 0:
            return
        infected_ids = np.flatnonzero(infected)
        if len(infected_ids) == 0:
            return
        if pattern_cfg["pattern"] == "random":
            chosen = _simx_sample_ids(rng, infected_ids, rem_n)
            infected[chosen] = False
            return
        target_total = int(infected.sum()) - rem_n
        target_counts = target_cluster_counts_from_field(target_total)
        set_cluster_counts(target_counts)

    schedule_rows = list(prevalence_schedule.itertuples(index=False))
    if schedule_rows:
        first_row = schedule_rows[0]
        first_target_n = int(round(float(np.clip(first_row.true_prev, 0.0, 1.0)) * total_hosts))
        initialise_infections(first_target_n)

    for row in schedule_rows:
        if int(row.round_id) != int(schedule_rows[0].round_id):
            target_prev = float(np.clip(row.true_prev, 0.0, 1.0))
            target_n = int(round(target_prev * total_hosts))
            current_n = int(infected.sum())
            if target_n > current_n:
                add_infections(target_n - current_n)
            elif target_n < current_n:
                remove_infections(current_n - target_n)
        inf_counts = infected_counts_by_cluster()
        actual_prev = float(infected.mean()) if total_hosts else 0.0
        achieved_prevalence.append(actual_prev)
        infected_by_round.append(infected.copy())
        round_tables.append(
            _simx_make_round_clusters(
                clusters=clusters,
                infected_counts=inf_counts,
                neighbor_pressure=cluster_prev_from_counts(inf_counts),
                round_id=int(row.round_id),
                year=float(row.year),
                true_prev=actual_prev,
            )
        )

    achieved_schedule = prevalence_schedule.copy()
    achieved_schedule["target_prev"] = achieved_schedule["true_prev"].astype(float)
    achieved_schedule["true_prev"] = np.asarray(achieved_prevalence, dtype=float)
    hosts = host_landscape["hosts"].copy()

    # Store transparent diagnostics for every epidemic year.  These make it
    # possible to detect unintended association between cluster population
    # size and infection prevalence before survey results are interpreted.
    round_diagnostics = pd.DataFrame(
        [
            {
                "round_id": int(round_df["round_id"].iloc[0]),
                "year": float(round_df["year"].iloc[0]),
                **_simx_epidemic_round_diagnostics(round_df),
            }
            for round_df in round_tables
        ]
    )
    return {
        "model": "host_id_cluster",
        "rounds": round_tables,
        "schedule": achieved_schedule,
        "hosts": hosts,
        "hosts_by_cluster": hosts_by_cluster,
        "cluster_id_by_host": cluster_for_host,
        "infected_by_round": infected_by_round,
        "round_diagnostics": round_diagnostics,
        "hotspots": hotspots,
        "infection_clustering": {
            "level": level,
            "definition": {
                **pattern_cfg,
            },
            "realised": {
                "hotspot_count": int(hotspot_n),
                "hotspot_decay_scale": float(distance_scale * float(pattern_cfg["hotspot_decay_mult"])) if level != "random" else 0.0,
            },
        },
    }


# ============================================================================
# 13. Public exports used by the dashboard and simulator
# ============================================================================# ============================================================================
# Public exports used by the apps
# ============================================================================


__all__ = [
    # Public planning constants and calculations
    "DEFAULT_ALPHA",
    "DEFAULT_POWER",
    "DEFAULT_CHANGE_METHOD_DELTA",
    "DEFAULT_CHANGE_METHOD_CORR",
    "DEFAULT_REGRESSION_METHOD_TARGET",
    "DEFAULT_BASELINE_PREVALENCE_CONF",
    "DEFAULT_BASELINE_PREVALENCE_WIDTH",
    "DEFAULT_BASELINE_PREVALENCE_UPPER",
    "compute_n_baseline_prevalence_cells",
    "compute_n_change_method_cells",
    "compute_n_total_regression_method_cells",
    "validate_efsa_reference_examples",
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
    "MULTISTAGE_DESIGN_REVISION",
    "INFECTION_ALLOCATION_REVISION",
    "SIMULATION_REVISION",
    "REGRESSION_FIT_REVISION",
    "CHANGE_CORRELATION_REVISION",
    "_sim_fit_logistic_counts",
    "_sim_make_prevalence_schedule",
    "_sim_overlap_config",
    "_sim_predict_logistic",
    "_sim_true_beta_from_schedule",
    "_simx_effort_by_round_table",
    "_simx_fit_methods",
    "_simx_epidemic_round_diagnostics",
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
