"""Interactive simulator app for testing plant pest survey designs.

This app is meant to be a decision-support and learning tool.  A user builds a
fake host landscape, creates a fake epidemic with a known true prevalence curve,
simulates surveys, and then checks how well the Change Method and Regression
Method recover the true answer.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from plant_pest_backend import (
    DEFAULT_ALPHA,
    DEFAULT_APPENDIX_C_DELTA,
    DEFAULT_APPENDIX_E_TARGET,
    DEFAULT_POWER,
    LARCH_HOST_DIR,
    SIM_CURVE_CHOICES,
    SIM_HOST_CLUSTERING_CHOICES,
    SIM_INFECTION_CLUSTERING_CHOICES,
    SIM_OVERLAP_CHOICES,
    SIM_SURVEY_TARGETING_CHOICES,
    SIMX_DEFAULT_HOST_DENSITY_KM2,
    SIMX_DEFAULT_PSU_BLOCK_SIDE,
    SIMX_DEFAULT_SURVEY_HOSTS_PER_ROUND,
    SIMX_DEFAULT_TOTAL_AREA_KM2,
    _sim_fit_logistic_counts,
    _sim_make_prevalence_schedule,
    _sim_overlap_config,
    _sim_predict_logistic,
    _sim_true_beta_from_schedule,
    _simx_fit_methods,
    _simx_infection_hotspot_table,
    _simx_infection_round_summary_table,
    _simx_macro_plan_sampling_effort,
    _simx_macro_simulate_surveys,
    _simx_make_macro_epidemic,
    _simx_make_macro_host_landscape_from_polygons,
    _simx_make_macro_host_landscape_synthetic,
    _simx_plot_cluster_size_distribution,
    _simx_plot_effort_bars,
    _simx_plot_macro_host_landscape,
    _simx_plot_macro_infection_landscape,
    _simx_plot_macro_survey_landscape,
    _simx_plot_method_comparison,
    _simx_plot_prevalence_curve,
    _simx_plot_targeting_comparison,
    _simx_round_summary,
    _simx_summary_table,
    _simx_targeting_comparison_table,
    load_host_layer,
    render_grid,
)


SIMX_DEFAULT_SYNTHETIC_CLUSTERS = 600
SIMX_DEFAULT_SEED = 123
SIMX_BASELINE_P0 = 0.05
SIMX_BASELINE_P_END = 0.005
SIMX_BASELINE_DELTA = abs(SIMX_BASELINE_P_END - SIMX_BASELINE_P0)
GRID_TWO_COLUMNS = "display:grid;grid-template-columns:repeat(2, minmax(0, 1fr));gap:1rem;align-items:start;"
GRID_TWO_COLUMNS_WITH_TOP_MARGIN = f"{GRID_TWO_COLUMNS}margin-top:1rem;"

# This file defines the Shiny simulator app.
# The app is arranged like the simulation workflow:
#   1. make a host landscape
#   2. define the true prevalence curve
#   3. create the infection landscape
#   4. calculate sampling effort
#   5. simulate surveys
#   6. fit the two methods and compare results


# ============================================================================
# Sidebar Controls
# ============================================================================
#
# Each sidebar is the control panel for one workflow stage.  The matching main
# panel shows the plot or table produced by that stage.


def _status_banner(output_id: str) -> Any:
    # Small text area used to tell the user what just happened or what failed.
    return ui.div(
        ui.output_text_verbatim(output_id),
        style="margin:0 0 0.75rem 0;font-size:0.95rem;color:#495057;white-space:pre-wrap;",
    )


def _host_landscape_sidebar() -> Any:
    # Left-hand controls for creating the host population.
    # Users can either load polygons or generate a fake synthetic landscape.
    return ui.sidebar(
        ui.h3("Host Landscape"),
        ui.p("Build the host landscape either from a host polygon layer or from a synthetic cluster generator."),
        ui.input_select(
            "simx_landscape_source",
            "Landscape source",
            choices={
                "polygons": "Load host polygons",
                "synthetic": "Generate synthetic clusters",
            },
            selected="polygons",
        ),
        ui.panel_conditional(
            "input.simx_landscape_source === 'polygons'",
            # Polygon mode uses real host coverage polygons.  Each polygon is
            # turned into one cluster, and the host count comes from its area.
            ui.input_text(
                "simx_host_layer_path",
                "Host shapefile or folder",
                value=str(LARCH_HOST_DIR),
            ),
            ui.input_numeric(
                "simx_host_density_km2",
                "Host density (hosts per km^2)",
                value=SIMX_DEFAULT_HOST_DENSITY_KM2,
                min=1,
                step=100,
            ),
        ),
        ui.panel_conditional(
            "input.simx_landscape_source === 'synthetic'",
            # Synthetic mode makes a new landscape from scratch.  This is useful
            # for testing general plant pest scenarios, not just Larch/Ramorum.
            ui.input_numeric(
                "simx_synth_n_clusters",
                "Number of clusters",
                value=SIMX_DEFAULT_SYNTHETIC_CLUSTERS,
                min=10,
                step=10,
            ),
            ui.input_numeric(
                "simx_synth_total_area_km2",
                "Total area (km^2)",
                value=SIMX_DEFAULT_TOTAL_AREA_KM2,
                min=1,
                step=1,
            ),
            ui.input_numeric(
                "simx_synth_host_density_km2",
                "Host density (hosts per km^2)",
                value=SIMX_DEFAULT_HOST_DENSITY_KM2,
                min=1,
                step=100,
            ),
        ),
        ui.input_numeric(
            "simx_host_seed",
            "Landscape seed",
            value=SIMX_DEFAULT_SEED,
            min=1,
            step=1,
        ),
        ui.input_action_button("btn_simx_host", "Generate host landscape", class_="btn-primary"),
    )


def _prevalence_sidebar() -> Any:
    # Controls for the true prevalence curve.
    # This is the hidden truth that the survey methods are trying to estimate.
    return ui.sidebar(
        ui.h3("True Prevalence"),
        ui.p("Define the true prevalence curve that the simulator will enforce across the survey rounds."),
        ui.input_numeric("simx_n_rounds", "Number of rounds", value=6, min=2, max=12, step=1),
        ui.input_select("simx_curve_type", "True prevalence curve", choices=SIM_CURVE_CHOICES, selected="logit_linear"),
        ui.input_numeric("simx_p0", "Initial prevalence", value=SIMX_BASELINE_P0, min=0.0001, max=0.999, step=0.001),
        ui.input_numeric("simx_p_end", "Final prevalence", value=SIMX_BASELINE_P_END, min=0.0001, max=0.999, step=0.001),
        ui.input_action_button("btn_simx_curve", "Generate prevalence curve", class_="btn-primary"),
    )


def _infection_sidebar() -> Any:
    # Controls for placing infections across the host landscape.
    # The clustering choice changes whether infection is random or hotspot-driven.
    return ui.sidebar(
        ui.h3("Infection Landscape"),
        ui.p("Apply the prevalence curve to the host landscape and inspect infection spread round by round."),
        ui.input_select("simx_infection_clustering", "Infection clustering", choices=SIM_INFECTION_CLUSTERING_CHOICES, selected="random"),
        ui.p(
            "Clustered scenarios draw a small, variable number of initial hotspot clusters. New infections favour "
            "clusters that are already infected and clusters exposed to infected hosts in their five nearest neighbours.",
            class_="text-muted small",
        ),
        ui.input_numeric("simx_epidemic_seed", "Infection landscape seed", value=SIMX_DEFAULT_SEED + 101, min=1, step=1),
        ui.input_slider("simx_display_round", "Displayed round", min=1, max=12, value=1, step=1),
        ui.input_action_button("btn_simx_epidemic", "Generate infection landscape", class_="btn-primary"),
    )


def _effort_sidebar() -> Any:
    # Controls for the sample-size formulas.
    # These settings decide how many samples each method says are needed.
    return ui.sidebar(
        ui.h3("Sampling Effort"),
        ui.p("Estimate the required sampling effort for the Change Method and Regression Method."),
        ui.input_select(
            "simx_cluster_inflation",
            "Clustering inflation",
            choices={"none": "No inflation", "pilot": "Pilot-estimated phi"},
            selected="none",
        ),
        ui.input_numeric("simx_appendix_alpha", "Significance level (alpha)", value=DEFAULT_ALPHA, min=0.001, max=0.2, step=0.005),
        ui.input_numeric("simx_appendix_power", "Power", value=DEFAULT_POWER, min=0.5, max=0.999, step=0.01),
        ui.input_numeric("simx_appendix_c_delta", "Appendix C detectable change", value=SIMX_BASELINE_DELTA, min=0.001, max=0.5, step=0.005),
        ui.input_numeric("simx_appendix_e_target", "Appendix E target prevalence", value=DEFAULT_APPENDIX_E_TARGET, min=0.0001, max=0.5, step=0.001),
        ui.input_numeric("simx_effort_seed", "Sampling effort seed", value=SIMX_DEFAULT_SEED + 151, min=1, step=1),
        ui.input_action_button("btn_simx_effort", "Estimate sample sizes", class_="btn-primary"),
    )


def _survey_sidebar() -> Any:
    # Controls for how the simulated survey is actually carried out.
    # This is where SRS, MSS, targeting, overlap, and sensitivity are chosen.
    return ui.sidebar(
        ui.h3("Survey Design"),
        ui.p("Choose the survey design and simulate survey rounds across the epidemic."),
        ui.input_select(
            "simx_sample_size_mode",
            "Sample size mode",
            choices={
                "static": "Static per-round size",
                "method_specific": "Method-specific Appendix C/E sizes",
            },
            selected="method_specific",
        ),
        ui.input_select("simx_sampling_mode", "Sampling mode", choices={"srs": "SRS", "multistage": "MSS"}, selected="srs"),
        # For MSS, the app first chooses clusters, then chooses host IDs inside
        # each chosen cluster.  This field controls the second step.
        ui.input_numeric("simx_psu_block_side", "MSS groups sampled per cluster", value=SIMX_DEFAULT_PSU_BLOCK_SIDE, min=1, step=1),
        ui.input_select("simx_survey_targeting", "Targeted surveying", choices=SIM_SURVEY_TARGETING_CHOICES, selected="random"),
        ui.input_select("simx_overlap_type", "Overlap type", choices=SIM_OVERLAP_CHOICES, selected="cross_sectional"),
        ui.input_numeric("simx_survey_hosts_per_round", "Static survey hosts per round", value=SIMX_DEFAULT_SURVEY_HOSTS_PER_ROUND, min=1, step=10),
        ui.input_numeric("simx_within_cluster_cap", "Hosts sampled per MSS cluster", value=1, min=1, step=1),
        ui.input_numeric("simx_method_sensitivity", "Method sensitivity", value=1.0, min=0.0, max=1.0, step=0.01),
        ui.input_numeric("simx_biased_multiplier", "Biased sampling multiplier", value=1.0, min=1.0, step=0.1),
        ui.input_numeric("simx_survey_seed", "Survey seed", value=SIMX_DEFAULT_SEED + 202, min=1, step=1),
        ui.input_action_button("btn_simx_survey", "Simulate surveys", class_="btn-primary"),
    )


def _results_sidebar() -> Any:
    # The normal Results tab fits the two methods to one simulated survey run.
    return ui.sidebar(
        ui.h3("Results"),
        ui.p("Fit the Change Method and Regression Method to the simulated survey outputs."),
        ui.input_action_button("btn_simx_fit", "Fit Appendix C and E", class_="btn-primary"),
    )


def _comparison_scenario_controls(prefix: str, label: str) -> Any:
    # Controls for one side of the A/B comparison.
    # Scenario A and B use the same set of settings so users can change only
    # the part they want to test.
    return ui.card(
        ui.card_header(label),
        ui.h5("True Prevalence"),
        ui.input_select(f"{prefix}_curve_type", "Curve shape", choices=SIM_CURVE_CHOICES, selected="logit_linear"),
        ui.input_numeric(f"{prefix}_p0", "Initial prevalence", value=SIMX_BASELINE_P0, min=0.0001, max=0.999, step=0.001),
        ui.input_numeric(f"{prefix}_p_end", "Final prevalence", value=SIMX_BASELINE_P_END, min=0.0001, max=0.999, step=0.001),
        ui.hr(),
        ui.h5("Infection Landscape"),
        ui.input_select(f"{prefix}_infection_clustering", "Infection clustering", choices=SIM_INFECTION_CLUSTERING_CHOICES, selected="random"),
        ui.hr(),
        ui.h5("Sampling Effort"),
        # These settings affect how many samples the formulas recommend.
        # They do not directly change the true epidemic.
        ui.input_select(
            f"{prefix}_cluster_inflation",
            "Clustering inflation",
            choices={"none": "No inflation", "pilot": "Pilot-estimated phi"},
            selected="none",
        ),
        ui.input_numeric(f"{prefix}_appendix_c_delta", "Change Method detectable change", value=SIMX_BASELINE_DELTA, min=0.001, max=0.5, step=0.005),
        ui.input_numeric(f"{prefix}_appendix_e_target", "Regression Method design prevalence", value=DEFAULT_APPENDIX_E_TARGET, min=0.0001, max=0.5, step=0.001),
        ui.hr(),
        ui.h5("Survey Design"),
        # These settings affect where the simulated survey samples are taken.
        # This is where users can compare random surveying with targeted surveying.
        ui.input_select(f"{prefix}_sampling_mode", "Sampling mode", choices={"srs": "SRS", "multistage": "MSS"}, selected="srs"),
        ui.input_select(f"{prefix}_survey_targeting", "Targeted surveying", choices=SIM_SURVEY_TARGETING_CHOICES, selected="random"),
        ui.input_select(f"{prefix}_overlap_type", "Overlap type", choices=SIM_OVERLAP_CHOICES, selected="cross_sectional"),
        ui.input_numeric(f"{prefix}_within_cluster_cap", "Hosts sampled per MSS cluster", value=1, min=1, step=1),
        ui.input_numeric(f"{prefix}_method_sensitivity", "Detection sensitivity", value=1.0, min=0.0, max=1.0, step=0.01),
        ui.hr(),
        ui.h5("Analysis"),
        ui.input_select(f"{prefix}_regression_model", "Regression model", choices={"logistic": "Logistic"}, selected="logistic"),
    )


def _comparison_sidebar() -> Any:
    # Controls for repeated A/B experiments.
    # One run gives side-by-side fitted curves; many runs give error boxplots.
    return ui.sidebar(
        ui.h3("Scenario Comparison"),
        ui.p("Run repeated simulations for two configurations and compare the error distributions."),
        ui.input_numeric("simx_compare_iters", "Runs per scenario", value=100, min=1, max=1000, step=1),
        ui.input_numeric("simx_compare_seed", "Comparison seed", value=SIMX_DEFAULT_SEED + 500, min=1, step=1),
        ui.input_numeric("simx_compare_alpha", "Significance level", value=DEFAULT_ALPHA, min=0.001, max=0.2, step=0.005),
        ui.input_numeric("simx_compare_power", "Power", value=DEFAULT_POWER, min=0.5, max=0.999, step=0.01),
        ui.input_checkbox("simx_compare_use_current_host", "Use the generated host landscape from the Host Landscape page", value=True),
        ui.p(
            "If no host landscape has been generated, the comparison uses the baseline synthetic landscape.",
            class_="text-muted small",
        ),
        ui.input_action_button("btn_simx_compare", "Run Scenario Comparison", class_="btn-primary"),
    )


# ============================================================================
# Plot And Table Helpers
# ============================================================================
#
# These helpers keep the server code shorter.  They do not run new simulations;
# they only format already-created results for display.


def _simx_empty_comparison_plot(title: str):
    # Placeholder plot shown before the comparison has been run.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.text(0.5, 0.5, "Run the scenario comparison to view this error distribution.", ha="center", va="center")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def _simx_plot_comparison_boxplot(df: pd.DataFrame | None, metric: str, title: str, ylabel: str):
    # Draw one error boxplot panel for the Scenario Comparison tab.
    # The value plotted is always "estimate minus truth".
    import matplotlib.pyplot as plt

    if df is None or df.empty or metric not in set(df.get("metric", pd.Series(dtype=str))):
        return _simx_empty_comparison_plot(title)
    plot_df = df.loc[df["metric"] == metric].copy()
    scenarios = ["Scenario A", "Scenario B"]
    methods = [m for m in ["Change Method", "Regression Method"] if m in set(plot_df["method"])]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    colors = {"Change Method": "#7b3294", "Regression Method": "#2b8cbe"}
    offsets = np.linspace(-0.18, 0.18, max(len(methods), 1))
    width = 0.28 if len(methods) > 1 else 0.42
    for method, offset in zip(methods, offsets):
        # Draw one box for each method in each scenario.
        # This lets users see both average error and variability.
        data = []
        positions = []
        for idx, scenario in enumerate(scenarios, start=1):
            vals = plot_df.loc[(plot_df["scenario"] == scenario) & (plot_df["method"] == method), "value"].dropna().to_numpy(dtype=float)
            data.append(vals if len(vals) else np.array([np.nan]))
            positions.append(idx + offset)
        bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors.get(method, "#777777"))
            patch.set_alpha(0.78)
            patch.set_edgecolor("#222222")
        for part in ["whiskers", "caps", "medians"]:
            for line in bp[part]:
                line.set_color("#222222")
                line.set_linewidth(1.0)
    ax.axhline(0, color="#4d4d4d", linestyle="--", linewidth=1.0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.18)
    legend_handles = [
        plt.Line2D([0], [0], color=colors[m], linewidth=8, alpha=0.78, label=m)
        for m in methods
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", frameon=True)
    fig.tight_layout()
    return fig


def _simx_key_results_rows(epidemic: dict[str, Any] | None, fit: dict[str, Any] | None) -> list[dict[str, Any]]:
    # Build the five key numbers shown in the Results tab.
    # These compare the true answer with what the two methods estimated.
    if epidemic is None or fit is None:
        return []
    schedule = epidemic.get("schedule", pd.DataFrame()).copy()
    if schedule.empty:
        return []
    true_initial = float(schedule["true_prev"].iloc[0])
    true_final = float(schedule["true_prev"].iloc[-1])
    true_change = true_final - true_initial
    true_slope = float(_sim_true_beta_from_schedule(schedule))

    e_df = fit.get("survey_df", pd.DataFrame()).copy()
    # Regression Method uses all survey rounds to estimate a smooth trend.
    if e_df.empty:
        reg_initial = reg_final = reg_change = reg_slope = np.nan
    else:
        beta_hat, _ = _sim_fit_logistic_counts(
            e_df["year"].to_numpy(dtype=float),
            e_df["y"].to_numpy(dtype=float),
            e_df["n"].to_numpy(dtype=float),
        )
        first_year = float(e_df["year"].min())
        last_year = float(e_df["year"].max())
        reg_initial = float(_sim_predict_logistic(beta_hat, first_year))
        reg_final = float(_sim_predict_logistic(beta_hat, last_year))
        reg_change = reg_final - reg_initial
        reg_slope = float(beta_hat[1])

    c_df = fit.get("appendix_c_survey_df", pd.DataFrame()).copy()
    if c_df.empty or "n" not in c_df.columns:
        c_df = pd.DataFrame()
    else:
        c_df = c_df.loc[pd.to_numeric(c_df["n"], errors="coerce").fillna(0) > 0].copy()
    if len(c_df) >= 2:
        # Change Method only compares the first and final survey estimates.
        change_initial = float(c_df["prev_hat"].iloc[0])
        change_final = float(c_df["prev_hat"].iloc[-1])
        change_change = change_final - change_initial
    else:
        change_initial = change_final = change_change = np.nan

    return [
        {
            "metric": "Change Method final prevalence",
            "true_value": true_final,
            "estimated_value": change_final,
        },
        {
            "metric": "Regression Method final prevalence",
            "true_value": true_final,
            "estimated_value": reg_final,
        },
        {
            "metric": "Change Method prevalence change",
            "true_value": true_change,
            "estimated_value": change_change,
        },
        {
            "metric": "Regression Method prevalence change",
            "true_value": true_change,
            "estimated_value": reg_change,
        },
        {
            "metric": "Regression Method slope",
            "true_value": true_slope,
            "estimated_value": reg_slope,
        },
    ]


def _simx_key_results_table(epidemic: dict[str, Any] | None, fit: dict[str, Any] | None) -> pd.DataFrame:
    # Convert the key results into a neat table for the Shiny output.
    rows = _simx_key_results_rows(epidemic, fit)
    if not rows:
        return pd.DataFrame([{"metric": "Key results", "true_value": "Not yet available", "estimated_value": ""}])
    out = pd.DataFrame(rows)
    for col in ["true_value", "estimated_value"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(6)
    return out


def _host_landscape_panel() -> Any:
    # Page 1: create and inspect the host landscape.
    return ui.nav_panel(
        "Host Landscape",
        ui.layout_sidebar(
            _host_landscape_sidebar(),
            ui.div(
                _status_banner("simx_status_text_host"),
                ui.card(
                    ui.card_header("Landscape Cluster Plot"),
                    ui.output_plot("simx_host_plot", height="430px"),
                ),
                ui.div(
                    ui.card(
                        ui.card_header("Landscape Summary"),
                        ui.output_data_frame("simx_summary_table"),
                    ),
                    ui.card(
                        ui.card_header("Cluster Size Distribution"),
                        ui.output_plot("simx_cluster_hist_plot", height="320px"),
                    ),
                    style=GRID_TWO_COLUMNS_WITH_TOP_MARGIN,
                ),
            ),
        ),
    )


def _true_prevalence_panel() -> Any:
    # Page 2: define the true disease prevalence through time.
    return ui.nav_panel(
        "True Prevalence",
        ui.layout_sidebar(
            _prevalence_sidebar(),
            ui.div(
                _status_banner("simx_status_text_curve"),
                ui.card(
                    ui.card_header("True Prevalence Curve"),
                    ui.output_plot("simx_curve_plot", height="420px"),
                ),
            ),
        ),
    )


def _infection_landscape_panel() -> Any:
    # Page 3: place infection onto the host landscape for each round.
    return ui.nav_panel(
        "Infection Landscape",
        ui.layout_sidebar(
            _infection_sidebar(),
            ui.div(
                _status_banner("simx_status_text_infection"),
                ui.card(
                    ui.card_header("Global Infection Landscape"),
                    ui.output_plot("simx_infection_plot", height="430px"),
                ),
                ui.div(
                    ui.card(
                        ui.card_header("Infection Round Summary"),
                        ui.output_data_frame("simx_infection_round_table"),
                    ),
                    ui.card(
                        ui.card_header("Largest Infection Hotspots"),
                        ui.output_data_frame("simx_infection_hotspot_table"),
                    ),
                    style=GRID_TWO_COLUMNS_WITH_TOP_MARGIN,
                ),
            ),
        ),
    )


def _sampling_effort_panel() -> Any:
    # Page 4: calculate how many samples the two methods require.
    return ui.nav_panel(
        "Sampling Effort",
        ui.layout_sidebar(
            _effort_sidebar(),
            ui.div(
                _status_banner("simx_status_text_effort"),
                ui.div(
                    ui.card(
                        ui.card_header("Required Sampling Effort"),
                        ui.output_data_frame("simx_effort_table"),
                    ),
                    ui.card(
                        ui.card_header("Required Sample Size by Round"),
                        ui.output_plot("simx_effort_plot", height="340px"),
                    ),
                    style=GRID_TWO_COLUMNS,
                ),
            ),
        ),
    )


def _survey_design_panel() -> Any:
    # Page 5: simulate where samples are taken and what the survey observes.
    return ui.nav_panel(
        "Survey Design",
        ui.layout_sidebar(
            _survey_sidebar(),
            ui.div(
                _status_banner("simx_status_text_survey"),
                ui.card(
                    ui.card_header("Survey Allocation by Cluster"),
                    ui.output_plot("simx_survey_plot", height="430px"),
                ),
                ui.div(
                    ui.card(
                        ui.card_header("Survey Round Summary"),
                        ui.output_data_frame("simx_round_summary_table"),
                    ),
                    ui.card(
                        ui.card_header("Targeted vs Untargeted Comparison"),
                        ui.output_data_frame("simx_targeting_table"),
                    ),
                    style=GRID_TWO_COLUMNS_WITH_TOP_MARGIN,
                ),
                ui.card(
                    ui.card_header("Targeted vs Untargeted Prevalence Estimates"),
                    ui.output_plot("simx_targeting_plot", height="340px"),
                    style="margin-top:1rem;",
                ),
            ),
        ),
    )


def _results_panel() -> Any:
    # Page 6: fit the two methods and compare estimates with the true values.
    return ui.nav_panel(
        "Results",
        ui.layout_sidebar(
            _results_sidebar(),
            ui.div(
                _status_banner("simx_status_text_results"),
                ui.card(
                    ui.card_header("Method Comparison"),
                    ui.output_plot("simx_methods_plot", height="450px"),
                ),
                ui.card(
                    ui.card_header("Key Results"),
                    ui.output_data_frame("simx_key_results_table"),
                    style="margin-top:1rem;",
                ),
            ),
        ),
    )


def _scenario_comparison_panel() -> Any:
    # Extra page: compare two scenarios side by side.
    # This is useful for questions like "what if DEFRA used targeted surveying?"
    return ui.nav_panel(
        "Scenario Comparison",
        ui.layout_sidebar(
            _comparison_sidebar(),
            ui.div(
                _status_banner("simx_status_text_compare"),
                ui.div(
                    _comparison_scenario_controls("simx_a", "Scenario A: Baseline"),
                    _comparison_scenario_controls("simx_b", "Scenario B: Counterfactual"),
                    style=f"{GRID_TWO_COLUMNS}margin-bottom:1rem;",
                ),
                ui.output_ui("simx_compare_plot_panel"),
                ui.card(
                    ui.card_header("Comparison Summary"),
                    ui.output_data_frame("simx_compare_summary_table"),
                    style="margin-top:1rem;",
                ),
            ),
        ),
    )


# ============================================================================
# Shiny UI
# ============================================================================


app_ui = ui.page_navbar(
    # Each tab matches one stage of the simulator workflow.
    _host_landscape_panel(),
    _true_prevalence_panel(),
    _infection_landscape_panel(),
    _sampling_effort_panel(),
    _survey_design_panel(),
    _results_panel(),
    _scenario_comparison_panel(),
    title="Plant Pest Simulator",
)


# ============================================================================
# Shiny Server Logic
# ============================================================================


def server(input, output, session):
    # Reactive values store the current state of the simulator.
    # They start empty, then each button press fills in the next layer.
    rv_simx_host = reactive.Value(None)
    rv_simx_schedule = reactive.Value(pd.DataFrame())
    rv_simx_epidemic = reactive.Value(None)
    rv_simx_effort = reactive.Value(None)
    rv_simx_surveys = reactive.Value(None)
    rv_simx_fit = reactive.Value(None)
    rv_simx_compare = reactive.Value(None)
    rv_simx_message = reactive.Value("Generate the host landscape to start the simulator.")

    def _targeted_mode(targeting_level: str) -> str:
        # The backend expects "none" for random surveys and a named mode for
        # targeted surveys.
        return "none" if str(targeting_level) == "random" else "defra_targeted"

    def _overlap_prop(overlap_type: str) -> float:
        # Convert a plain-language overlap option into the proportion of host IDs
        # that are sampled again in the next round.
        if overlap_type == "longitudinal":
            return 1.0
        if overlap_type == "rotating_panel":
            return 0.7
        return 0.0

    def _clear_after_host_or_curve() -> None:
        # If the landscape or true curve changes, all later results are no
        # longer valid, so they are cleared.
        rv_simx_epidemic.set(None)
        rv_simx_effort.set(None)
        rv_simx_surveys.set(None)
        rv_simx_fit.set(None)

    def _clear_after_epidemic() -> None:
        rv_simx_effort.set(None)
        rv_simx_surveys.set(None)
        rv_simx_fit.set(None)

    def _clear_after_effort() -> None:
        rv_simx_surveys.set(None)
        rv_simx_fit.set(None)

    # ------------------------------------------------------------------
    # Small Reactive Helpers
    # ------------------------------------------------------------------

    @reactive.calc
    def simx_overlap_corr():
        # Appendix C uses a correlation value when surveys overlap between rounds.
        return float(_sim_overlap_config(str(input.simx_overlap_type()), overlap_prop_rot=0.7)["corr_c"])

    @reactive.calc
    def simx_display_round_index():
        # The round slider is 1-based for users, but Python list indexes are 0-based.
        epidemic = rv_simx_epidemic.get()
        if epidemic is None or not epidemic.get("rounds"):
            return 0
        return max(0, min(int(input.simx_display_round()) - 1, len(epidemic["rounds"]) - 1))

    def _comparison_overlap_prop(overlap_type: str) -> float:
        # Translate the overlap label into the proportion of host IDs reused.
        return _overlap_prop(overlap_type)

    def _comparison_spec(prefix: str) -> dict[str, Any]:
        # Read all settings for Scenario A or Scenario B into one dictionary.
        # This makes the comparison runner easier to call.
        overlap_type = str(getattr(input, f"{prefix}_overlap_type")())
        targeting_level = str(getattr(input, f"{prefix}_survey_targeting")())
        return {
            "curve_type": str(getattr(input, f"{prefix}_curve_type")()),
            "p0": float(getattr(input, f"{prefix}_p0")()),
            "p_end": float(getattr(input, f"{prefix}_p_end")()),
            "infection_clustering_level": str(getattr(input, f"{prefix}_infection_clustering")()),
            "sampling_mode": str(getattr(input, f"{prefix}_sampling_mode")()),
            "targeting_level": targeting_level,
            "targeted_mode": _targeted_mode(targeting_level),
            "overlap_type": overlap_type,
            "overlap_prop": _comparison_overlap_prop(overlap_type),
            "within_cluster_cap": int(getattr(input, f"{prefix}_within_cluster_cap")()),
            "method_sens": float(getattr(input, f"{prefix}_method_sensitivity")()),
            "cluster_inflation_mode": str(getattr(input, f"{prefix}_cluster_inflation")()),
            "appendix_c_delta": float(getattr(input, f"{prefix}_appendix_c_delta")()),
            "appendix_c_corr": float(_sim_overlap_config(overlap_type, overlap_prop_rot=0.7)["corr_c"]),
            "appendix_e_target": float(getattr(input, f"{prefix}_appendix_e_target")()),
            "regression_model": str(getattr(input, f"{prefix}_regression_model")()),
        }

    # ------------------------------------------------------------------
    # Scenario Comparison Runner
    # ------------------------------------------------------------------

    def _comparison_host(iter_seed: int) -> dict[str, Any]:
        # Use the currently generated host landscape if the user selected that option.
        # Otherwise make a fresh baseline synthetic landscape.
        current_host = rv_simx_host.get()
        if bool(input.simx_compare_use_current_host()) and current_host is not None:
            return current_host
        return _simx_make_macro_host_landscape_synthetic(
            n_clusters=SIMX_DEFAULT_SYNTHETIC_CLUSTERS,
            total_area_km2=SIMX_DEFAULT_TOTAL_AREA_KM2,
            host_density_km2=SIMX_DEFAULT_HOST_DENSITY_KM2,
            rng=np.random.default_rng(int(iter_seed)),
        )

    def _run_comparison_iteration(
        scenario_name: str,
        spec: dict[str, Any],
        host: dict[str, Any],
        iter_idx: int,
        iter_seed: int,
    ) -> dict[str, Any]:
        # Run one complete simulation for either Scenario A or Scenario B.
        # This does every layer: prevalence curve, epidemic, sample sizes,
        # surveys, fitted methods, and error metrics.
        # The returned errors are always estimated value minus true value.
        schedule = _sim_make_prevalence_schedule(
            curve_type=str(spec["curve_type"]),
            p0=float(spec["p0"]),
            p_end=float(spec["p_end"]),
            prevalence_shift=0.0,
            n_rounds=int(input.simx_n_rounds()),
        )
        epidemic = _simx_make_macro_epidemic(
            host_landscape=host,
            prevalence_schedule=schedule,
            infection_clustering_level=str(spec["infection_clustering_level"]),
            rng=np.random.default_rng(int(iter_seed) + 101),
        )
        effort = _simx_macro_plan_sampling_effort(
            epidemic=epidemic,
            sampling_mode=str(spec["sampling_mode"]),
            overlap_type=str(spec["overlap_type"]),
            targeted_mode=str(spec["targeted_mode"]),
            within_cluster_cap=int(spec["within_cluster_cap"]),
            overlap_prop=float(spec["overlap_prop"]),
            targeting_level=str(spec["targeting_level"]),
            p0_mode="pilot",
            assumed_p0=0.5,
            appendix_alpha=float(input.simx_compare_alpha()),
            appendix_power=float(input.simx_compare_power()),
            appendix_c_delta=float(spec["appendix_c_delta"]),
            appendix_c_corr=float(spec["appendix_c_corr"]),
            appendix_e_target=float(spec["appendix_e_target"]),
            cluster_inflation_mode=str(spec["cluster_inflation_mode"]),
            fixed_phi=1.0,
            rng=np.random.default_rng(int(iter_seed) + 151),
        )
        n_rounds = len(epidemic["rounds"])
        # Regression Method samples every round after an initial pilot.
        e_sizes = [int(effort["n_initial"])] + [int(effort["appendix_e_n_per_round"])] * (n_rounds - 1)
        # Change Method samples only the first and final rounds.
        c_sizes = [int(effort["appendix_c_n_per_round"])] + [0] * max(0, n_rounds - 2) + [int(effort["appendix_c_n_per_round"])]
        surveys = {
            "appendix_e": _simx_macro_simulate_surveys(
                epidemic=epidemic,
                n_hosts_per_round=e_sizes,
                sampling_mode=str(spec["sampling_mode"]),
                overlap_type=str(spec["overlap_type"]),
                targeted_mode=str(spec["targeted_mode"]),
                within_cluster_cap=int(spec["within_cluster_cap"]),
                overlap_prop=float(spec["overlap_prop"]),
                targeting_level=str(spec["targeting_level"]),
                rng=np.random.default_rng(int(iter_seed) + 202),
                method_sens=float(spec["method_sens"]),
            ),
            "appendix_c": _simx_macro_simulate_surveys(
                epidemic=epidemic,
                n_hosts_per_round=c_sizes,
                sampling_mode=str(spec["sampling_mode"]),
                overlap_type=str(spec["overlap_type"]),
                targeted_mode=str(spec["targeted_mode"]),
                within_cluster_cap=int(spec["within_cluster_cap"]),
                overlap_prop=float(spec["overlap_prop"]),
                targeting_level=str(spec["targeting_level"]),
                rng=np.random.default_rng(int(iter_seed) + 203),
                method_sens=float(spec["method_sens"]),
            ),
        }
        fit = _simx_fit_methods(epidemic=epidemic, surveys=surveys, model_form=str(spec["regression_model"]))
        true_schedule = epidemic["schedule"].copy()
        true_initial = float(true_schedule["true_prev"].iloc[0])
        true_final = float(true_schedule["true_prev"].iloc[-1])
        true_change = true_final - true_initial
        true_slope = float(_sim_true_beta_from_schedule(true_schedule))

        e_df = fit.get("survey_df", pd.DataFrame()).copy()
        if e_df.empty:
            reg_final = np.nan
            reg_change = np.nan
            reg_slope = np.nan
        else:
            beta_hat, _ = _sim_fit_logistic_counts(
                e_df["year"].to_numpy(dtype=float),
                e_df["y"].to_numpy(dtype=float),
                e_df["n"].to_numpy(dtype=float),
            )
            first_year = float(e_df["year"].min())
            last_year = float(e_df["year"].max())
            reg_initial = float(_sim_predict_logistic(beta_hat, first_year))
            reg_final = float(_sim_predict_logistic(beta_hat, last_year))
            reg_change = reg_final - reg_initial
            reg_slope = float(beta_hat[1])

        c_df = fit.get("appendix_c_survey_df", pd.DataFrame()).copy()
        if c_df.empty or "n" not in c_df.columns:
            c_df = pd.DataFrame()
        else:
            c_df = c_df.loc[pd.to_numeric(c_df["n"], errors="coerce").fillna(0) > 0].copy()
        if len(c_df) >= 2:
            change_final = float(c_df["prev_hat"].iloc[-1])
            change_change = float(c_df["prev_hat"].iloc[-1] - c_df["prev_hat"].iloc[0])
        else:
            change_final = np.nan
            change_change = np.nan

        metric_table = _simx_key_results_table(epidemic, fit).copy()
        metric_table.insert(0, "scenario", scenario_name)
        # Store error rows in a long table because that is easy to summarise and plot.
        common = {"scenario": scenario_name, "iter": int(iter_idx)}
        error_rows = [
            {**common, "metric": "Final prevalence error", "method": "Change Method", "value": change_final - true_final},
            {**common, "metric": "Final prevalence error", "method": "Regression Method", "value": reg_final - true_final},
            {**common, "metric": "Change in prevalence error", "method": "Change Method", "value": change_change - true_change},
            {**common, "metric": "Change in prevalence error", "method": "Regression Method", "value": reg_change - true_change},
            {**common, "metric": "Regression slope error", "method": "Regression Method", "value": reg_slope - true_slope},
        ]
        return {
            "errors": error_rows,
            "epidemic": epidemic,
            "fit": fit,
            "metrics": metric_table,
        }

    # ------------------------------------------------------------------
    # Button Actions
    # ------------------------------------------------------------------

    @reactive.effect
    @reactive.event(input.btn_simx_host)
    def _generate_host_landscape():
        # Button action: create the host landscape and clear later results.
        # Later layers are cleared because they depend on the host landscape.
        rv_simx_message.set("Generating host landscape...")
        try:
            rng = np.random.default_rng(int(input.simx_host_seed()))
            if str(input.simx_landscape_source()) == "polygons":
                host_info = load_host_layer(Path(str(input.simx_host_layer_path())).expanduser())
                host_layer = host_info.get("layer")
                if host_layer is None or len(host_layer) == 0:
                    raise ValueError("; ".join(host_info.get("messages", [])) or "The host polygon layer could not be loaded.")
                host = _simx_make_macro_host_landscape_from_polygons(
                    host_gdf=host_layer,
                    host_density_km2=float(input.simx_host_density_km2()),
                )
            else:
                host = _simx_make_macro_host_landscape_synthetic(
                    n_clusters=int(input.simx_synth_n_clusters()),
                    total_area_km2=float(input.simx_synth_total_area_km2()),
                    host_density_km2=float(input.simx_synth_host_density_km2()),
                    rng=rng,
                )
            rv_simx_host.set(host)
            _clear_after_host_or_curve()
            clusters = host.get("clusters", pd.DataFrame())
            n_clusters = int(clusters["cluster_id"].nunique()) if not clusters.empty and "cluster_id" in clusters.columns else 0
            rv_simx_message.set(
                f"Host landscape generated: {int(host['total_hosts'])} hosts across {n_clusters} clusters."
            )
        except Exception as exc:
            rv_simx_message.set(f"Host landscape failed: {exc}")

    @reactive.effect
    @reactive.event(input.btn_simx_curve)
    def _generate_prevalence_curve():
        # Button action: make the true prevalence curve and clear later results.
        try:
            rv_simx_schedule.set(
                _sim_make_prevalence_schedule(
                    curve_type=str(input.simx_curve_type()),
                    p0=float(input.simx_p0()),
                    p_end=float(input.simx_p_end()),
                    prevalence_shift=0.0,
                    n_rounds=int(input.simx_n_rounds()),
                )
            )
            _clear_after_host_or_curve()
            rv_simx_message.set("Prevalence curve generated.")
        except Exception as exc:
            rv_simx_message.set(f"Prevalence curve failed: {exc}")

    @reactive.effect
    @reactive.event(input.btn_simx_epidemic)
    def _generate_infection_landscape():
        # Button action: place infections across clusters/hosts for every round.
        host = rv_simx_host.get()
        schedule = rv_simx_schedule.get()
        if host is None or schedule.empty:
            rv_simx_message.set("Generate the host landscape and prevalence curve first.")
            return
        try:
            epidemic = _simx_make_macro_epidemic(
                host_landscape=host,
                prevalence_schedule=schedule,
                infection_clustering_level=str(input.simx_infection_clustering()),
                rng=np.random.default_rng(int(input.simx_epidemic_seed())),
            )
            rv_simx_epidemic.set(epidemic)
            _clear_after_epidemic()
            rv_simx_message.set("Infection landscape generated.")
        except Exception as exc:
            rv_simx_message.set(f"Infection landscape failed: {exc}")

    @reactive.effect
    @reactive.event(input.btn_simx_effort)
    def _estimate_sampling_effort():
        # Button action: calculate the required sample sizes for both methods.
        epidemic = rv_simx_epidemic.get()
        if epidemic is None:
            rv_simx_message.set("Generate the infection landscape before estimating sample sizes.")
            return
        try:
            effort = _simx_macro_plan_sampling_effort(
                epidemic=epidemic,
                sampling_mode=str(input.simx_sampling_mode()),
                overlap_type=str(input.simx_overlap_type()),
                targeted_mode=_targeted_mode(str(input.simx_survey_targeting())),
                within_cluster_cap=int(input.simx_within_cluster_cap()),
                overlap_prop=_overlap_prop(str(input.simx_overlap_type())),
                targeting_level=str(input.simx_survey_targeting()),
                p0_mode="pilot",
                assumed_p0=0.5,
                appendix_alpha=float(input.simx_appendix_alpha()),
                appendix_power=float(input.simx_appendix_power()),
                appendix_c_delta=float(input.simx_appendix_c_delta()),
                appendix_c_corr=float(simx_overlap_corr()),
                appendix_e_target=float(input.simx_appendix_e_target()),
                cluster_inflation_mode=str(input.simx_cluster_inflation()),
                fixed_phi=1.0,
                rng=np.random.default_rng(int(input.simx_effort_seed())),
            )
            rv_simx_effort.set(effort)
            _clear_after_effort()
            rv_simx_message.set("Sample sizes estimated.")
        except Exception as exc:
            rv_simx_message.set(f"Sample-size estimation failed: {exc}")

    @reactive.effect
    @reactive.event(input.btn_simx_survey)
    def _simulate_surveys():
        # Button action: run the simulated surveys using the chosen design.
        epidemic = rv_simx_epidemic.get()
        if epidemic is None:
            rv_simx_message.set("Generate the infection landscape before simulating surveys.")
            return
        try:
            sample_mode = str(input.simx_sample_size_mode())
            if sample_mode == "method_specific":
                effort = rv_simx_effort.get()
                if effort is None:
                    rv_simx_message.set("Estimate sample sizes before using method-specific sample sizes.")
                    return
                n_rounds = len(epidemic["rounds"])
                e_sizes = [int(effort["n_initial"])] + [int(effort["appendix_e_n_per_round"])] * (n_rounds - 1)
                c_sizes = [int(effort["appendix_c_n_per_round"])] + [0] * max(0, n_rounds - 2) + [int(effort["appendix_c_n_per_round"])]
                surveys = {
                    "appendix_e": _simx_macro_simulate_surveys(
                        epidemic=epidemic,
                        n_hosts_per_round=e_sizes,
                        sampling_mode=str(input.simx_sampling_mode()),
                        overlap_type=str(input.simx_overlap_type()),
                        targeted_mode=_targeted_mode(str(input.simx_survey_targeting())),
                        within_cluster_cap=int(input.simx_within_cluster_cap()),
                        overlap_prop=_overlap_prop(str(input.simx_overlap_type())),
                        targeting_level=str(input.simx_survey_targeting()),
                        rng=np.random.default_rng(int(input.simx_survey_seed())),
                        method_sens=float(input.simx_method_sensitivity()),
                    ),
                    "appendix_c": _simx_macro_simulate_surveys(
                        epidemic=epidemic,
                        n_hosts_per_round=c_sizes,
                        sampling_mode=str(input.simx_sampling_mode()),
                        overlap_type=str(input.simx_overlap_type()),
                        targeted_mode=_targeted_mode(str(input.simx_survey_targeting())),
                        within_cluster_cap=int(input.simx_within_cluster_cap()),
                        overlap_prop=_overlap_prop(str(input.simx_overlap_type())),
                        targeting_level=str(input.simx_survey_targeting()),
                        rng=np.random.default_rng(int(input.simx_survey_seed()) + 1),
                        method_sens=float(input.simx_method_sensitivity()),
                    ),
                }
                comparison_sizes = e_sizes
            else:
                surveys = {
                    "static": _simx_macro_simulate_surveys(
                        epidemic=epidemic,
                        n_hosts_per_round=int(input.simx_survey_hosts_per_round()),
                        sampling_mode=str(input.simx_sampling_mode()),
                        overlap_type=str(input.simx_overlap_type()),
                        targeted_mode=_targeted_mode(str(input.simx_survey_targeting())),
                        within_cluster_cap=int(input.simx_within_cluster_cap()),
                        overlap_prop=_overlap_prop(str(input.simx_overlap_type())),
                        targeting_level=str(input.simx_survey_targeting()),
                        rng=np.random.default_rng(int(input.simx_survey_seed())),
                        method_sens=float(input.simx_method_sensitivity()),
                    )
                }
                comparison_sizes = int(input.simx_survey_hosts_per_round())

            surveys["untargeted_all_rounds"] = _simx_macro_simulate_surveys(
                epidemic=epidemic,
                n_hosts_per_round=comparison_sizes,
                sampling_mode=str(input.simx_sampling_mode()),
                overlap_type=str(input.simx_overlap_type()),
                targeted_mode="none",
                within_cluster_cap=int(input.simx_within_cluster_cap()),
                overlap_prop=_overlap_prop(str(input.simx_overlap_type())),
                targeting_level="random",
                rng=np.random.default_rng(int(input.simx_survey_seed()) + 20),
                method_sens=float(input.simx_method_sensitivity()),
            )
            surveys["targeted_all_rounds"] = _simx_macro_simulate_surveys(
                epidemic=epidemic,
                n_hosts_per_round=comparison_sizes,
                sampling_mode=str(input.simx_sampling_mode()),
                overlap_type=str(input.simx_overlap_type()),
                targeted_mode="defra_targeted",
                within_cluster_cap=int(input.simx_within_cluster_cap()),
                overlap_prop=_overlap_prop(str(input.simx_overlap_type())),
                targeting_level=str(input.simx_survey_targeting()),
                rng=np.random.default_rng(int(input.simx_survey_seed()) + 21),
                method_sens=float(input.simx_method_sensitivity()),
            )
            rv_simx_surveys.set(surveys)
            rv_simx_fit.set(None)
            rv_simx_message.set("Survey rounds simulated.")
        except Exception as exc:
            rv_simx_message.set(f"Survey simulation failed: {exc}")

    @reactive.effect
    @reactive.event(input.btn_simx_fit)
    def _fit_methods():
        # Button action: fit the Change Method and Regression Method to the
        # simulated survey data.
        epidemic = rv_simx_epidemic.get()
        surveys = rv_simx_surveys.get()
        if epidemic is None or surveys is None:
            rv_simx_message.set("Generate the infection landscape and surveys before fitting the methods.")
            return
        try:
            fit_key = "appendix_e" if "appendix_e" in surveys else "static"
            fit_surveys = {
                "appendix_e": surveys[fit_key],
                "appendix_c": surveys["appendix_c"] if "appendix_c" in surveys else surveys[fit_key],
            }
            rv_simx_fit.set(_simx_fit_methods(epidemic=epidemic, surveys=fit_surveys, model_form="logistic"))
            rv_simx_message.set("Appendix C and Appendix E fitted to the simulated survey outputs.")
        except Exception as exc:
            rv_simx_message.set(f"Method fitting failed: {exc}")

    @reactive.effect
    @reactive.event(input.btn_simx_compare)
    def _run_scenario_comparison():
        # Button action: run Scenario A and Scenario B.
        # One run gives a visual example; many runs give error distributions.
        notif_id = ui.notification_show(
            "Please wait... running scenario comparison.",
            duration=None,
            close_button=False,
            type="message",
            session=session,
        )
        try:
            n_iters = int(input.simx_compare_iters())
            base_seed = int(input.simx_compare_seed())
            spec_a = _comparison_spec("simx_a")
            spec_b = _comparison_spec("simx_b")
            rows: list[dict[str, Any]] = []
            single_run_payload: dict[str, Any] = {}
            rv_simx_message.set(f"Running scenario comparison: 0/{n_iters} runs complete...")
            for i in range(1, n_iters + 1):
                # Use related random seeds so Scenario A and B are compared
                # under the same broad random conditions.
                iter_seed = base_seed + i
                host_a = _comparison_host(iter_seed)
                host_b = host_a
                result_a = _run_comparison_iteration("Scenario A", spec_a, host_a, i, iter_seed)
                result_b = _run_comparison_iteration("Scenario B", spec_b, host_b, i, iter_seed)
                rows.extend(result_a["errors"])
                rows.extend(result_b["errors"])
                if n_iters == 1:
                    single_run_payload = {"Scenario A": result_a, "Scenario B": result_b}
                if i == 1 or i == n_iters or i % 25 == 0:
                    rv_simx_message.set(f"Running scenario comparison: {i}/{n_iters} runs complete...")
            rv_simx_compare.set(
                {
                    "mode": "single" if n_iters == 1 else "mc",
                    "errors": pd.DataFrame(rows),
                    "single": single_run_payload,
                }
            )
            rv_simx_message.set(f"Scenario comparison complete: {n_iters} runs per scenario.")
            ui.notification_remove(notif_id, session=session)
            ui.notification_show(
                "Scenario comparison complete.",
                duration=5,
                type="message",
                session=session,
            )
        except Exception as exc:
            rv_simx_compare.set(None)
            rv_simx_message.set(f"Scenario comparison failed: {exc}")
            ui.notification_remove(notif_id, session=session)
            ui.notification_show(
                f"Scenario comparison failed: {exc}",
                duration=10,
                type="error",
                session=session,
            )

    # ------------------------------------------------------------------
    # Outputs Shared Across Tabs
    # ------------------------------------------------------------------
    #
    # Output functions are Shiny renderers.  They watch the reactive values
    # above and redraw tables/plots whenever those values change.

    def _simx_status_value() -> str:
        # All tabs show the same current status message.
        return str(rv_simx_message.get() or "")

    @output
    @render.text
    def simx_status_text_host():
        return _simx_status_value()

    @output
    @render.text
    def simx_status_text_curve():
        return _simx_status_value()

    @output
    @render.text
    def simx_status_text_infection():
        return _simx_status_value()

    @output
    @render.text
    def simx_status_text_effort():
        return _simx_status_value()

    @output
    @render.text
    def simx_status_text_survey():
        return _simx_status_value()

    @output
    @render.text
    def simx_status_text_results():
        return _simx_status_value()

    @output
    @render.text
    def simx_status_text_compare():
        return _simx_status_value()

    @output
    @render.data_frame
    def simx_summary_table():
        # Show a friendly placeholder before a host landscape exists.
        host = rv_simx_host.get()
        if host is None:
            return render_grid(pd.DataFrame([{"metric": "Host landscape", "value": "Not yet generated"}]))
        return render_grid(_simx_summary_table(host))

    @output
    @render.plot
    def simx_host_plot():
        return _simx_plot_macro_host_landscape(rv_simx_host.get())

    @output
    @render.plot
    def simx_cluster_hist_plot():
        return _simx_plot_cluster_size_distribution(rv_simx_host.get())

    @output
    @render.plot
    def simx_curve_plot():
        return _simx_plot_prevalence_curve(rv_simx_schedule.get())

    @output
    @render.plot
    def simx_infection_plot():
        # The infection plot uses the selected display round from the slider.
        epidemic = rv_simx_epidemic.get()
        if epidemic is None or not epidemic.get("rounds"):
            return _simx_plot_macro_infection_landscape(None, hotspot_cluster_ids=None)
        return _simx_plot_macro_infection_landscape(
            epidemic["rounds"][simx_display_round_index()],
            hotspot_cluster_ids=epidemic.get("hotspot_cluster_ids"),
        )

    @output
    @render.data_frame
    def simx_infection_round_table():
        return render_grid(_simx_infection_round_summary_table(rv_simx_epidemic.get(), simx_display_round_index()))

    @output
    @render.data_frame
    def simx_infection_hotspot_table():
        return render_grid(_simx_infection_hotspot_table(rv_simx_epidemic.get(), simx_display_round_index()))

    @output
    @render.data_frame
    def simx_effort_table():
        effort = rv_simx_effort.get()
        if effort is None:
            return render_grid(pd.DataFrame([{"metric": "Sampling effort", "value": "Not yet estimated"}]))
        return render_grid(effort["table"])

    @output
    @render.plot
    def simx_effort_plot():
        return _simx_plot_effort_bars(rv_simx_epidemic.get(), rv_simx_effort.get())

    @output
    @render.plot
    def simx_survey_plot():
        # If both method-specific survey outputs exist, the plot shows the
        # Regression Method allocation because it has every survey round.
        surveys = rv_simx_surveys.get()
        if surveys is None:
            return _simx_plot_macro_survey_landscape(None, "Survey allocation")
        survey_key = "appendix_e" if "appendix_e" in surveys else "static"
        survey_rounds = surveys[survey_key]["rounds"]
        round_idx = max(0, min(simx_display_round_index(), len(survey_rounds) - 1))
        return _simx_plot_macro_survey_landscape(survey_rounds[round_idx], f"Survey allocation: round {round_idx + 1}")

    @output
    @render.data_frame
    def simx_round_summary_table():
        epidemic = rv_simx_epidemic.get()
        surveys = rv_simx_surveys.get()
        if epidemic is None:
            return render_grid(pd.DataFrame(columns=["round_id", "year", "true_prev", "infected_hosts"]))
        return render_grid(_simx_round_summary(epidemic, surveys=surveys))

    @output
    @render.data_frame
    def simx_targeting_table():
        return render_grid(_simx_targeting_comparison_table(rv_simx_surveys.get()))

    @output
    @render.plot
    def simx_targeting_plot():
        return _simx_plot_targeting_comparison(rv_simx_epidemic.get(), rv_simx_surveys.get())

    @output
    @render.plot
    def simx_methods_plot():
        return _simx_plot_method_comparison(rv_simx_epidemic.get(), rv_simx_fit.get(), None)

    @output
    @render.data_frame
    def simx_key_results_table():
        return render_grid(_simx_key_results_table(rv_simx_epidemic.get(), rv_simx_fit.get()))

    @output
    @render.ui
    def simx_compare_plot_panel():
        # Switch the comparison output depending on the number of runs.
        # One run shows fitted curves; repeated runs show boxplots.
        result = rv_simx_compare.get()
        if isinstance(result, dict) and result.get("mode") == "single":
            return ui.div(
                ui.card(
                    ui.card_header("Scenario A Fitted Results"),
                    ui.output_plot("simx_compare_scenario_a_plot", height="410px"),
                ),
                ui.card(
                    ui.card_header("Scenario B Fitted Results"),
                    ui.output_plot("simx_compare_scenario_b_plot", height="410px"),
                ),
                style="display:grid;grid-template-columns:repeat(2, minmax(0, 1fr));gap:1rem;align-items:start;",
            )
        return ui.div(
            ui.card(
                ui.card_header("Final Prevalence Error"),
                ui.output_plot("simx_compare_final_plot", height="360px"),
            ),
            ui.card(
                ui.card_header("Prevalence Change Error"),
                ui.output_plot("simx_compare_change_plot", height="360px"),
            ),
            ui.card(
                ui.card_header("Regression Slope Error"),
                ui.output_plot("simx_compare_slope_plot", height="360px"),
            ),
            style="display:grid;grid-template-columns:repeat(3, minmax(0, 1fr));gap:1rem;align-items:start;",
        )

    def _simx_compare_errors_df() -> pd.DataFrame:
        # Pull the error table out of the comparison result object.
        result = rv_simx_compare.get()
        if isinstance(result, dict):
            errors = result.get("errors", pd.DataFrame())
            return errors if isinstance(errors, pd.DataFrame) else pd.DataFrame()
        if isinstance(result, pd.DataFrame):
            return result
        return pd.DataFrame()

    @output
    @render.plot
    def simx_compare_scenario_a_plot():
        result = rv_simx_compare.get()
        single = result.get("single", {}) if isinstance(result, dict) else {}
        payload = single.get("Scenario A", {})
        return _simx_plot_method_comparison(payload.get("epidemic"), payload.get("fit"), None)

    @output
    @render.plot
    def simx_compare_scenario_b_plot():
        result = rv_simx_compare.get()
        single = result.get("single", {}) if isinstance(result, dict) else {}
        payload = single.get("Scenario B", {})
        return _simx_plot_method_comparison(payload.get("epidemic"), payload.get("fit"), None)

    @output
    @render.plot
    def simx_compare_final_plot():
        return _simx_plot_comparison_boxplot(
            _simx_compare_errors_df(),
            metric="Final prevalence error",
            title="Error in estimated final prevalence",
            ylabel="Estimated prevalence - true prevalence",
        )

    @output
    @render.plot
    def simx_compare_change_plot():
        return _simx_plot_comparison_boxplot(
            _simx_compare_errors_df(),
            metric="Change in prevalence error",
            title="Error in estimated prevalence change",
            ylabel="Estimated change - true change",
        )

    @output
    @render.plot
    def simx_compare_slope_plot():
        return _simx_plot_comparison_boxplot(
            _simx_compare_errors_df(),
            metric="Regression slope error",
            title="Error in estimated regression slope",
            ylabel="Estimated slope - true slope",
        )

    @output
    @render.data_frame
    def simx_compare_summary_table():
        # For one run, show the true and estimated values side by side.
        # For many runs, show average error, median error, and RMSE.
        result = rv_simx_compare.get()
        if isinstance(result, dict) and result.get("mode") == "single":
            single = result.get("single", {})
            pivot_parts = []
            for scenario in ["Scenario A", "Scenario B"]:
                metrics = single.get(scenario, {}).get("metrics", pd.DataFrame())
                if isinstance(metrics, pd.DataFrame) and not metrics.empty:
                    part = metrics[["metric", "true_value", "estimated_value"]].copy()
                    part = part.rename(
                        columns={
                            "true_value": f"{scenario} true",
                            "estimated_value": f"{scenario} estimated",
                        }
                    )
                    pivot_parts.append(part)
            if pivot_parts:
                out = pivot_parts[0]
                for part in pivot_parts[1:]:
                    out = out.merge(part, on="metric", how="outer")
                for col in out.columns:
                    if col != "metric":
                        out[col] = pd.to_numeric(out[col], errors="coerce").round(6)
                return render_grid(out)
            return render_grid(pd.DataFrame([{"metric": "Scenario comparison", "value": "Not yet run"}]))

        df = _simx_compare_errors_df()
        if df is None or df.empty:
            return render_grid(pd.DataFrame([{"metric": "Scenario comparison", "value": "Not yet run"}]))
        summary = (
            df.groupby(["scenario", "metric", "method"], as_index=False)
            .agg(
                mean_error=("value", "mean"),
                median_error=("value", "median"),
                rmse=("value", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
            )
            .sort_values(["metric", "scenario", "method"])
        )
        for col in ["mean_error", "median_error", "rmse"]:
            summary[col] = summary[col].astype(float).round(6)
        return render_grid(summary)


app = App(app_ui, server)
