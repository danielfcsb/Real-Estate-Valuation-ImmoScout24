import numpy as np
import pandas as pd


def evaluate_assumptions(
    bp_pvalue,
    jb_pvalue,
    shapiro_p,
    dw_stat,
    reset_pvalue,
    rainbow_pvalue,
    max_vif,
    alpha=0.05,
    dw_min=1.5,
    dw_max=2.5,
    max_allowed_vif=10.0
):
    checks = {
        "homoscedasticity_bp": bool(pd.notna(bp_pvalue) and bp_pvalue > alpha),
        "normality_jb": bool(pd.notna(jb_pvalue) and jb_pvalue > alpha),
        "normality_shapiro": bool(pd.notna(shapiro_p) and shapiro_p > alpha),
        "independence_dw": bool(pd.notna(dw_stat) and dw_min <= dw_stat <= dw_max),
        "specification_reset": bool(pd.notna(reset_pvalue) and reset_pvalue > alpha),
        "linearity_rainbow": bool(pd.notna(rainbow_pvalue) and rainbow_pvalue > alpha),
        "multicollinearity_vif": bool(pd.notna(max_vif) and max_vif <= max_allowed_vif),
    }
    failed = [k for k, ok in checks.items() if not ok]
    return checks, failed


def should_trigger_remediation(failed_checks, trigger=2):
    return len(failed_checks) >= trigger


def winsorize_series(y: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    lo = y.quantile(lower_q)
    hi = y.quantile(upper_q)
    return y.clip(lower=lo, upper=hi)


def apply_basic_remediation(
    y,
    failed_checks,
    enable_target_log=True,
    enable_winsor=True,
    lower_q=0.01,
    upper_q=0.99
):
    y_new = y.copy()
    actions = []
    fail_set = set(failed_checks)

    if enable_target_log and (
        "normality_jb" in fail_set
        or "normality_shapiro" in fail_set
        or "homoscedasticity_bp" in fail_set
    ):
        if (pd.Series(y_new) > 0).all():
            y_new = np.log1p(y_new)
            actions.append("Applied log1p to target")
        else:
            actions.append("Skipped log1p (target has non-positive values)")

    if enable_winsor and (
        "normality_jb" in fail_set or "normality_shapiro" in fail_set
    ):
        y_new = winsorize_series(pd.Series(y_new), lower_q=lower_q, upper_q=upper_q)
        actions.append(f"Winsorized target at q=({lower_q:.2f}, {upper_q:.2f})")

    return y_new, actions