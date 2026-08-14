import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset, linear_rainbow
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf


def run_model_scenario(
    df_original,
    scenario_name,
    scenario_code,
    max_missing_pct,
    use_imputation,
    *,
    target,
    features,
    binary_features,
    test_size,
    random_state,
    ridge_alphas,
    stepwise_p_threshold,
    stepwise_max_iter,
    vif_sample_size,
    add_section,
    add_subsection,
    add_text,
    add_figure_to_doc,
    add_engineered_features,
    normalize_binary_series,
    select_features_by_missing_threshold,
    make_preprocessor,
    get_clean_feature_names,
    add_pearson_correlation_plot,
    run_backward_stepwise,
    build_diagnostic_panel,
    # NEW
    assumption_alpha=0.05,
    dw_ok_min=1.5,
    dw_ok_max=2.5,
    max_allowed_vif=10.0,
    failed_checks_trigger=2,
    max_remediation_rounds=2,
    enable_target_log=True,
    enable_winsorization=True,
    winsor_lower_q=0.01,
    winsor_upper_q=0.99
):
    add_section(f"SCENARIO: {scenario_name}")

    df_model = df_original.copy()
    df_model[target] = pd.to_numeric(df_model[target], errors="coerce")
    df_model = add_engineered_features(df_model)

    rows_initial = len(df_model)
    df_model = df_model.dropna(subset=[target])
    rows_after_target_drop = len(df_model)

    X = df_model[features].copy()
    y = df_model[target].copy()

    n_features_before_threshold = X.shape[1]
    selected_features, removed_features, missing_feature_summary_df = select_features_by_missing_threshold(X, max_missing_pct)
    X = X[selected_features].copy()
    n_features_after_threshold = X.shape[1]

    add_subsection("Missing-value feature threshold")
    add_text(f"Scenario code: {scenario_code}")
    add_text(f"Maximum allowed missing-value share: {max_missing_pct * 100:.1f}%")
    add_text(f"Features before threshold filtering: {n_features_before_threshold}")
    add_text(f"Features after threshold filtering: {n_features_after_threshold}")

    if n_features_after_threshold == 0:
        add_text("No features remained after threshold filtering. Scenario skipped.")
        return None

    for col in binary_features:
        if col in X.columns:
            X[col] = normalize_binary_series(X[col])

    if not use_imputation:
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid_mask].copy()
        y = y.loc[valid_mask].copy()

    rows_final = len(X)
    add_text(f"Initial rows: {rows_initial}")
    add_text(f"Rows after dropping missing target: {rows_after_target_drop}")
    add_text(f"Rows used in this scenario: {rows_final}")
    add_text(f"Use imputation: {use_imputation}")

    if rows_final == 0:
        add_text("No rows are available for this scenario. Scenario skipped.")
        return None

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    binary_detected, numeric_non_binary = [], []
    for col in numeric_features:
        unique_vals = set(X[col].dropna().unique())
        if unique_vals.issubset({0, 1}):
            binary_detected.append(col)
        else:
            numeric_non_binary.append(col)

    add_subsection("Variable Type Summary")
    add_text(f"Number of binary variables: {len(binary_detected)}")
    add_text(f"Number of numeric non-binary variables: {len(numeric_non_binary)}")
    add_text(f"Number of categorical variables: {len(categorical_features)}")

    pearson_df = add_pearson_correlation_plot(
        X_input=X,
        y_input=y,
        target=target,
        scenario_name=scenario_name,
        scenario_code=scenario_code,
        use_imputation=use_imputation
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = make_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        use_imputation=use_imputation
    )

    X_train_proc = np.array(preprocessor.fit_transform(X_train), dtype=float)
    X_test_proc = np.array(preprocessor.transform(X_test), dtype=float)

    feature_names_out = get_clean_feature_names(preprocessor, numeric_features, categorical_features)
    if len(feature_names_out) != X_train_proc.shape[1]:
        feature_names_out = [f"x_{i}" for i in range(X_train_proc.shape[1])]

    X_train_sm = sm.add_constant(X_train_proc, has_constant="add")
    X_test_sm = sm.add_constant(X_test_proc, has_constant="add")

    ols_model = sm.OLS(y_train, X_train_sm).fit()
    y_pred_ols = ols_model.predict(X_test_sm)

    ols_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_ols)))
    ols_mae = float(mean_absolute_error(y_test, y_pred_ols))
    ols_r2 = float(r2_score(y_test, y_pred_ols))

    coef_df = pd.DataFrame({
        "Feature": ["const"] + feature_names_out,
        "Coefficient": ols_model.params,
        "p-value": ols_model.pvalues
    })

    ridge_scaler = StandardScaler()
    X_train_ridge = ridge_scaler.fit_transform(X_train_proc)
    X_test_ridge = ridge_scaler.transform(X_test_proc)

    ridge_model = RidgeCV(alphas=ridge_alphas, cv=5)
    ridge_model.fit(X_train_ridge, y_train)
    y_pred_ridge = ridge_model.predict(X_test_ridge)

    ridge_best_alpha = float(ridge_model.alpha_)
    ridge_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
    ridge_mae = float(mean_absolute_error(y_test, y_pred_ridge))
    ridge_r2 = float(r2_score(y_test, y_pred_ridge))

    ridge_coef_df = pd.DataFrame({
        "Feature": feature_names_out,
        "Standardized coefficient": ridge_model.coef_
    })
    ridge_coef_df["Absolute coefficient"] = ridge_coef_df["Standardized coefficient"].abs()
    ridge_coef_df = ridge_coef_df.sort_values("Absolute coefficient", ascending=False)

    add_section(f"RIDGE RESULTS - {scenario_name}")
    add_text(f"Best alpha: {ridge_best_alpha:.6f}")

    top_ridge_plot = ridge_coef_df.head(20).sort_values("Absolute coefficient", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_ridge_plot["Feature"], top_ridge_plot["Absolute coefficient"], color="steelblue")
    ax.set_xlabel("|Standardized coefficient|")
    ax.set_title(f"Top Ridge Coefficients - {scenario_name}")
    plt.tight_layout()
    add_figure_to_doc(fig, width=6.0)
    plt.close(fig)

    try:
        reset_res = linear_reset(ols_model, power=2, use_f=True)
        reset_f = float(reset_res.fvalue)
        reset_p = float(reset_res.pvalue)
    except Exception:
        reset_f = np.nan
        reset_p = np.nan

    try:
        stepwise_output = run_backward_stepwise(
            X_train_proc=X_train_proc,
            X_test_proc=X_test_proc,
            y_train=y_train,
            y_test=y_test,
            feature_names_out=feature_names_out,
            scenario_name=scenario_name,
            p_threshold=stepwise_p_threshold,
            max_iter=stepwise_max_iter
        )
        stepwise_rmse = stepwise_output["stepwise_rmse"]
        stepwise_mae = stepwise_output["stepwise_mae"]
        stepwise_r2 = stepwise_output["stepwise_r2"]
        stepwise_selected_df = stepwise_output["stepwise_selected_df"]
        stepwise_removed_df = stepwise_output["stepwise_removed_df"]
        stepwise_coef_df = stepwise_output["stepwise_coef_df"]
    except Exception as exc:
        add_section(f"STEPWISE REGRESSION - {scenario_name}")
        add_text(f"Stepwise regression failed: {exc}")
        stepwise_rmse = np.nan
        stepwise_mae = np.nan
        stepwise_r2 = np.nan
        stepwise_selected_df = pd.DataFrame()
        stepwise_removed_df = pd.DataFrame()
        stepwise_coef_df = pd.DataFrame()

    add_section(f"VIF - {scenario_name}")
    X_vif = X_train_proc.copy()
    if vif_sample_size and X_vif.shape[0] > vif_sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_vif.shape[0], size=vif_sample_size, replace=False)
        X_vif = X_vif[idx, :]

    vif_values = []
    for i in range(X_vif.shape[1]):
        try:
            vif_val = variance_inflation_factor(X_vif, i)
        except Exception:
            vif_val = np.inf
        vif_values.append(vif_val)

    vif_df = pd.DataFrame({"Feature": feature_names_out, "VIF": vif_values}).sort_values("VIF", ascending=False)
    for _, row in vif_df.head(20).iterrows():
        add_text(f"{row['Feature']}: {row['VIF']:.4f}")

    top_vif_plot = vif_df.head(20).sort_values("VIF", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_vif_plot["Feature"], top_vif_plot["VIF"], color="slateblue")
    ax.axvline(5, color="orange", linestyle="--", linewidth=1, label="VIF = 5")
    ax.axvline(10, color="red", linestyle="--", linewidth=1, label="VIF = 10")
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title(f"Top VIF Values - {scenario_name}")
    ax.legend()
    plt.tight_layout()
    add_figure_to_doc(fig, width=6.2)
    plt.close(fig)

    add_section(f"OLS DIAGNOSTICS - {scenario_name}")
    residuals = y_test - y_pred_ols
    fitted_vals = y_pred_ols

    try:
        rainbow_stat, rainbow_pvalue = linear_rainbow(ols_model)
        rainbow_stat, rainbow_pvalue = float(rainbow_stat), float(rainbow_pvalue)
    except Exception:
        rainbow_stat, rainbow_pvalue = np.nan, np.nan

    try:
        bp_test = het_breuschpagan(residuals, X_test_sm)
        bp_lm_stat, bp_pvalue, bp_f_stat, bp_f_pvalue = map(float, bp_test)
    except Exception:
        bp_lm_stat = bp_pvalue = bp_f_stat = bp_f_pvalue = np.nan

    try:
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        jb_stat, jb_pvalue = float(jb_stat), float(jb_pvalue)
    except Exception:
        jb_stat, jb_pvalue = np.nan, np.nan

    try:
        sample_size = min(5000, len(residuals))
        resid_sample = pd.Series(residuals).sample(sample_size, random_state=random_state)
        shapiro_stat, shapiro_p = stats.shapiro(resid_sample)
        shapiro_stat, shapiro_p = float(shapiro_stat), float(shapiro_p)
    except Exception:
        sample_size, shapiro_stat, shapiro_p = np.nan, np.nan, np.nan

    try:
        dw_stat = float(durbin_watson(residuals))
    except Exception:
        dw_stat = np.nan

    add_text(f"Breusch-Pagan p-value: {bp_pvalue}")
    add_text(f"Jarque-Bera p-value: {jb_pvalue}")
    add_text(f"Shapiro-Wilk p-value: {shapiro_p}")
    add_text(f"Durbin-Watson: {dw_stat}")
    add_text(f"RESET p-value: {reset_p}")
    add_text(f"Rainbow p-value: {rainbow_pvalue}")

    fig = build_diagnostic_panel(
        fitted_vals=fitted_vals,
        residuals=residuals,
        scenario_name=scenario_name,
        rainbow_pvalue=rainbow_pvalue,
        shapiro_pvalue=shapiro_p,
        bp_pvalue=bp_pvalue,
        dw_stat=dw_stat
    )
    add_figure_to_doc(fig, width=7.0)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="steelblue", ax=ax)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution - {scenario_name}")
    plt.tight_layout()
    add_figure_to_doc(fig, width=6.2)
    plt.close(fig)

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_acf(pd.Series(residuals).dropna(), lags=40, ax=ax)
        ax.set_title(f"Residual ACF - {scenario_name}")
        plt.tight_layout()
        add_figure_to_doc(fig, width=6.2)
        plt.close(fig)
    except Exception:
        pass

    max_vif = np.nan
    if not vif_df.empty:
        max_vif = float(vif_df["VIF"].replace([np.inf, -np.inf], np.nan).max())

    table_missing_data_scenario = pd.DataFrame([{
        "Scenario": scenario_code,
        "Scenario label": scenario_name,
        "Missing threshold (%)": max_missing_pct * 100,
        "Imputation": "Yes" if use_imputation else "No",
        "Initial rows": rows_initial,
        "Rows after dropping missing target": rows_after_target_drop,
        "Rows used for modelling": rows_final,
        "Features before threshold filtering": n_features_before_threshold,
        "Features after threshold filtering": n_features_after_threshold
    }])

    table_linear_performance = pd.DataFrame([
        {"Scenario": scenario_code, "Scenario label": scenario_name, "Imputation": "Yes" if use_imputation else "No", "Model": "OLS", "RMSE": ols_rmse, "MAE": ols_mae, "R2": ols_r2},
        {"Scenario": scenario_code, "Scenario label": scenario_name, "Imputation": "Yes" if use_imputation else "No", "Model": "Ridge", "RMSE": ridge_rmse, "MAE": ridge_mae, "R2": ridge_r2},
        {"Scenario": scenario_code, "Scenario label": scenario_name, "Imputation": "Yes" if use_imputation else "No", "Model": "Stepwise OLS", "RMSE": stepwise_rmse, "MAE": stepwise_mae, "R2": stepwise_r2}
    ])

    table_top_pearson = pearson_df.copy()
    if not table_top_pearson.empty:
        table_top_pearson = table_top_pearson.sort_values("Absolute Pearson correlation", ascending=False).head(15)

    table_ols_diagnostics = pd.DataFrame([{
        "Scenario": scenario_code,
        "Scenario label": scenario_name,
        "Imputation": "Yes" if use_imputation else "No",
        "Breusch-Pagan p-value": bp_pvalue,
        "Jarque-Bera p-value": jb_pvalue,
        "Shapiro-Wilk p-value": shapiro_p,
        "Durbin-Watson": dw_stat,
        "Ramsey RESET p-value": reset_p,
        "Rainbow p-value": rainbow_pvalue,
        "Maximum VIF": max_vif
    }])

    table_stepwise_removed = stepwise_removed_df.copy()
    if not table_stepwise_removed.empty:
        table_stepwise_removed.insert(0, "Scenario", scenario_code)
        table_stepwise_removed.insert(1, "Scenario label", scenario_name)
        table_stepwise_removed.insert(2, "Imputation", "Yes" if use_imputation else "No")

    full_vif_df = vif_df.copy()
    full_vif_df.insert(0, "Scenario", scenario_code)
    full_vif_df.insert(1, "Scenario label", scenario_name)
    full_vif_df.insert(2, "Imputation", "Yes" if use_imputation else "No")

    missing_feature_summary_df.insert(0, "Scenario", scenario_code)
    missing_feature_summary_df.insert(1, "Scenario label", scenario_name)
    missing_feature_summary_df.insert(2, "Missing threshold (%)", max_missing_pct * 100)

    result_row = {
        "scenario": scenario_name,
        "scenario_code": scenario_code,
        "max_missing_pct": max_missing_pct,
        "n_obs": rows_final,
        "use_imputation": use_imputation,
        "OLS_RMSE": ols_rmse,
        "OLS_MAE": ols_mae,
        "OLS_R2": ols_r2,
        "Ridge_Best_Alpha": ridge_best_alpha,
        "Ridge_RMSE": ridge_rmse,
        "Ridge_MAE": ridge_mae,
        "Ridge_R2": ridge_r2,
        "Stepwise_RMSE": stepwise_rmse,
        "Stepwise_MAE": stepwise_mae,
        "Stepwise_R2": stepwise_r2,
        "Jarque_Bera_p": jb_pvalue,
        "Shapiro_p": shapiro_p,
        "Durbin_Watson": dw_stat,
        "BP_pvalue": bp_pvalue,
        "RESET_pvalue": reset_p,
        "Rainbow_pvalue": rainbow_pvalue,
        "Maximum_VIF": max_vif
    }

    metrics_df = pd.DataFrame({"metric": list(result_row.keys()), "value": list(result_row.values())})

    # <- este bloco deve estar neste nível (4 espaços)
    y_current = y.copy()
    remediation_rows = []
    last_actions = []
    final_bundle = None

    for round_idx in range(max_remediation_rounds + 1):
        round_name = f"{scenario_name} | round_{round_idx}"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_current, test_size=test_size, random_state=random_state
        )

        preprocessor = make_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            use_imputation=use_imputation
        )

        X_train_proc = np.array(preprocessor.fit_transform(X_train), dtype=float)
        X_test_proc = np.array(preprocessor.transform(X_test), dtype=float)

        feature_names_out = get_clean_feature_names(preprocessor, numeric_features, categorical_features)
        if len(feature_names_out) != X_train_proc.shape[1]:
            feature_names_out = [f"x_{i}" for i in range(X_train_proc.shape[1])]

        X_train_sm = sm.add_constant(X_train_proc, has_constant="add")
        X_test_sm = sm.add_constant(X_test_proc, has_constant="add")

        ols_model = sm.OLS(y_train, X_train_sm).fit()
        y_pred_ols = ols_model.predict(X_test_sm)

        ols_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_ols)))
        ols_mae = float(mean_absolute_error(y_test, y_pred_ols))
        ols_r2 = float(r2_score(y_test, y_pred_ols))

        coef_df = pd.DataFrame({
            "Feature": ["const"] + feature_names_out,
            "Coefficient": ols_model.params,
            "p-value": ols_model.pvalues
        })

        ridge_scaler = StandardScaler()
        X_train_ridge = ridge_scaler.fit_transform(X_train_proc)
        X_test_ridge = ridge_scaler.transform(X_test_proc)

        ridge_model = RidgeCV(alphas=ridge_alphas, cv=5)
        ridge_model.fit(X_train_ridge, y_train)
        y_pred_ridge = ridge_model.predict(X_test_ridge)

        ridge_best_alpha = float(ridge_model.alpha_)
        ridge_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
        ridge_mae = float(mean_absolute_error(y_test, y_pred_ridge))
        ridge_r2 = float(r2_score(y_test, y_pred_ridge))

        ridge_coef_df = pd.DataFrame({
            "Feature": feature_names_out,
            "Standardized coefficient": ridge_model.coef_
        })
        ridge_coef_df["Absolute coefficient"] = ridge_coef_df["Standardized coefficient"].abs()
        ridge_coef_df = ridge_coef_df.sort_values("Absolute coefficient", ascending=False)

        try:
            reset_res = linear_reset(ols_model, power=2, use_f=True)
            reset_p = float(reset_res.pvalue)
        except Exception:
            reset_p = np.nan

        try:
            stepwise_output = run_backward_stepwise(
                X_train_proc=X_train_proc,
                X_test_proc=X_test_proc,
                y_train=y_train,
                y_test=y_test,
                feature_names_out=feature_names_out,
                scenario_name=round_name,
                p_threshold=stepwise_p_threshold,
                max_iter=stepwise_max_iter
            )
            stepwise_rmse = stepwise_output["stepwise_rmse"]
            stepwise_mae = stepwise_output["stepwise_mae"]
            stepwise_r2 = stepwise_output["stepwise_r2"]
            stepwise_selected_df = stepwise_output["stepwise_selected_df"]
            stepwise_removed_df = stepwise_output["stepwise_removed_df"]
            stepwise_coef_df = stepwise_output["stepwise_coef_df"]
        except Exception:
            stepwise_rmse = np.nan
            stepwise_mae = np.nan
            stepwise_r2 = np.nan
            stepwise_selected_df = pd.DataFrame()
            stepwise_removed_df = pd.DataFrame()
            stepwise_coef_df = pd.DataFrame()

        X_vif = X_train_proc.copy()
        if vif_sample_size and X_vif.shape[0] > vif_sample_size:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(X_vif.shape[0], size=vif_sample_size, replace=False)
            X_vif = X_vif[idx, :]

        vif_values = []
        for i in range(X_vif.shape[1]):
            try:
                vif_val = variance_inflation_factor(X_vif, i)
            except Exception:
                vif_val = np.inf
            vif_values.append(vif_val)

        vif_df = pd.DataFrame({"Feature": feature_names_out, "VIF": vif_values}).sort_values("VIF", ascending=False)
        max_vif = float(vif_df["VIF"].replace([np.inf, -np.inf], np.nan).max()) if not vif_df.empty else np.nan

        residuals = y_test - y_pred_ols
        fitted_vals = y_pred_ols

        try:
            _, rainbow_pvalue = linear_rainbow(ols_model)
            rainbow_pvalue = float(rainbow_pvalue)
        except Exception:
            rainbow_pvalue = np.nan

        try:
            bp_test = het_breuschpagan(residuals, X_test_sm)
            bp_pvalue = float(bp_test[1])
        except Exception:
            bp_pvalue = np.nan

        try:
            _, jb_pvalue = stats.jarque_bera(residuals)
            jb_pvalue = float(jb_pvalue)
        except Exception:
            jb_pvalue = np.nan

        try:
            resid_sample = pd.Series(residuals).sample(min(5000, len(residuals)), random_state=random_state)
            _, shapiro_p = stats.shapiro(resid_sample)
            shapiro_p = float(shapiro_p)
        except Exception:
            shapiro_p = np.nan

        try:
            dw_stat = float(durbin_watson(residuals))
        except Exception:
            dw_stat = np.nan

        checks, failed_checks = evaluate_assumptions(
            bp_pvalue=bp_pvalue,
            jb_pvalue=jb_pvalue,
            shapiro_p=shapiro_p,
            dw_stat=dw_stat,
            reset_pvalue=reset_p,
            rainbow_pvalue=rainbow_pvalue,
            max_vif=max_vif,
            alpha=assumption_alpha,
            dw_min=dw_ok_min,
            dw_max=dw_ok_max,
            max_allowed_vif=max_allowed_vif
        )

        trigger = should_trigger_remediation(failed_checks, trigger=failed_checks_trigger)

        remediation_rows.append({
            "Scenario": scenario_code,
            "Scenario label": scenario_name,
            "Round": round_idx,
            "Failed checks count": len(failed_checks),
            "Failed checks": ", ".join(failed_checks) if failed_checks else "",
            "Trigger remediation": bool(trigger),
            "Actions": "; ".join(last_actions) if last_actions else ""
        })

        final_bundle = {
            "X_test": X_test,
            "y_test": y_test,
            "fitted_vals": fitted_vals,
            "residuals": residuals,
            "coef_df": coef_df,
            "ridge_coef_df": ridge_coef_df,
            "vif_df": vif_df,
            "stepwise_selected_df": stepwise_selected_df,
            "stepwise_removed_df": stepwise_removed_df,
            "stepwise_coef_df": stepwise_coef_df,
            "ols_rmse": ols_rmse, "ols_mae": ols_mae, "ols_r2": ols_r2,
            "ridge_best_alpha": ridge_best_alpha, "ridge_rmse": ridge_rmse, "ridge_mae": ridge_mae, "ridge_r2": ridge_r2,
            "stepwise_rmse": stepwise_rmse, "stepwise_mae": stepwise_mae, "stepwise_r2": stepwise_r2,
            "bp_pvalue": bp_pvalue, "jb_pvalue": jb_pvalue, "shapiro_p": shapiro_p, "dw_stat": dw_stat,
            "reset_p": reset_p, "rainbow_pvalue": rainbow_pvalue, "max_vif": max_vif,
            "failed_checks": failed_checks
        }

        if (not trigger) or (round_idx == max_remediation_rounds):
            break

        y_next, actions = apply_basic_remediation(
            y=y_current,
            failed_checks=failed_checks,
            enable_target_log=enable_target_log,
            enable_winsor=enable_winsorization,
            lower_q=winsor_lower_q,
            upper_q=winsor_upper_q
        )
        if not actions or y_next.equals(y_current):
            break

        add_subsection(f"Remediation round {round_idx + 1}")
        add_text(f"Failed checks: {failed_checks}")
        for a in actions:
            add_text(f"- {a}")

        y_current = y_next
        last_actions = actions

    remediation_summary_df = pd.DataFrame(remediation_rows)

    return {
        "metrics_df": metrics_df,
        "coef_df": coef_df,
        "ridge_coef_df": ridge_coef_df,
        "vif_df": full_vif_df,
        "result_row": result_row,
        "stepwise_selected_df": stepwise_selected_df,
        "stepwise_removed_df": table_stepwise_removed,
        "stepwise_coef_df": stepwise_coef_df,
        "missing_feature_summary_df": missing_feature_summary_df,
        "table_missing_data_scenario": table_missing_data_scenario,
        "table_linear_performance": table_linear_performance,
        "table_top_pearson": table_top_pearson,
        "table_ols_diagnostics": table_ols_diagnostics,
        "remediation_summary_df": remediation_summary_df
    }

from src.remediation import (
    evaluate_assumptions,
    should_trigger_remediation,
    apply_basic_remediation
)