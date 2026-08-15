import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def format_p_value(p_value):
    if pd.isna(p_value):
        return "p = NA"
    if p_value < 0.0001:
        return "p < 0.0001"
    return f"p = {p_value:.4f}"


def pass_fail_label_from_pvalue(p_value, alpha=0.05):
    if pd.isna(p_value):
        return "CHECK", "darkorange"
    if p_value > alpha:
        return "OK", "#228B22"
    return "NOT OK", "#cc0000"


def pass_fail_label_from_dw(dw_stat):
    if pd.isna(dw_stat):
        return "CHECK", "darkorange"
    if 1.5 <= dw_stat <= 2.5:
        return "OK", "#228B22"
    return "NOT OK", "#cc0000"


def add_status_box(ax, top_text, bottom_text, label, label_color):
    ax.text(
        0.03, 0.97, f"{top_text}\n{bottom_text}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.95)
    )
    ax.text(
        0.97, 0.06, label,
        transform=ax.transAxes, va="bottom", ha="right",
        fontsize=14, fontweight="bold", color=label_color
    )


def add_lowess_line(ax, x, y, frac=0.65):
    if len(x) < 3:
        return
    smooth = lowess(y, x, frac=frac, return_sorted=True)
    ax.plot(smooth[:, 0], smooth[:, 1], color="#d62728", linewidth=1.5)


def build_diagnostic_panel(
    fitted_vals,
    residuals,
    scenario_name,
    rainbow_pvalue=np.nan,
    shapiro_pvalue=np.nan,
    bp_pvalue=np.nan,
    dw_stat=np.nan,
    marker_size=18
):
    fitted_vals = np.asarray(fitted_vals, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    resid_std = np.std(residuals, ddof=1)
    std_resid = residuals / resid_std if resid_std > 0 else residuals.copy()
    sqrt_abs_resid = np.sqrt(np.abs(residuals))
    order_idx = np.arange(1, len(residuals) + 1)

    max_points = 1000
    rng = np.random.default_rng(42)
    idx = rng.choice(len(fitted_vals), size=max_points, replace=False) if len(fitted_vals) > max_points else np.arange(len(fitted_vals))

    fitted_plot = fitted_vals[idx]
    residuals_plot = residuals[idx]
    sqrt_abs_resid_plot = sqrt_abs_resid[idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    # 1) Residuals vs Fitted
    ax = axes[0]
    ax.scatter(fitted_plot, residuals_plot, color="black", alpha=0.5, s=marker_size)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    add_lowess_line(ax, fitted_plot, residuals_plot)
    label, color = pass_fail_label_from_pvalue(rainbow_pvalue)
    add_status_box(ax, "Rainbow test", format_p_value(rainbow_pvalue), label, color)
    ax.set_title("Residuals vs Fitted")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")

    # 2) QQ plot
    ax = axes[1]
    osm, osr = stats.probplot(std_resid, dist="norm", fit=False)
    ax.scatter(osm, osr, color="black", alpha=0.5, s=marker_size)
    qq_min = min(np.min(osm), np.min(osr))
    qq_max = max(np.max(osm), np.max(osr))
    ax.plot([qq_min, qq_max], [qq_min, qq_max], linestyle="--", color="gray")
    label, color = pass_fail_label_from_pvalue(shapiro_pvalue)
    add_status_box(ax, "Shapiro-Wilk", format_p_value(shapiro_pvalue), label, color)
    ax.set_title("Q-Q Plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Standardized Residuals")

    # 3) Scale-location
    ax = axes[2]
    ax.scatter(fitted_plot, sqrt_abs_resid_plot, color="black", alpha=0.5, s=marker_size)
    add_lowess_line(ax, fitted_plot, sqrt_abs_resid_plot)
    label, color = pass_fail_label_from_pvalue(bp_pvalue)
    add_status_box(ax, "Breusch-Pagan", format_p_value(bp_pvalue), label, color)
    ax.set_title("Scale-Location")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Sqrt(|Residuals|)")

    # 4) Residual order
    ax = axes[3]
    ax.plot(order_idx, residuals, color="black", linewidth=0.6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    label, color = pass_fail_label_from_dw(dw_stat)
    dw_text = f"DW = {dw_stat:.4f}" if not pd.isna(dw_stat) else "DW = NA"
    add_status_box(ax, "Durbin-Watson", dw_text, label, color)
    ax.set_title("Residuals in Observation Order")
    ax.set_xlabel("Observation order")
    ax.set_ylabel("Residuals")

    fig.suptitle(f"OLS Diagnostics - {scenario_name}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig