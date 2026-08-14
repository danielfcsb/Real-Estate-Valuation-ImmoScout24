import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_backward_stepwise(
    X_train_proc,
    X_test_proc,
    y_train,
    y_test,
    feature_names_out,
    scenario_name,
    target,
    p_threshold=0.05,
    max_iter=100,
    add_section=None,
    add_subsection=None,
    add_text=None,
    add_figure_to_doc=None
):
    if add_section:
        add_section(f"STEPWISE REGRESSION - {scenario_name}")

    X_train_sw = pd.DataFrame(X_train_proc, columns=feature_names_out, index=y_train.index)
    X_test_sw = pd.DataFrame(X_test_proc, columns=feature_names_out, index=y_test.index)

    selected_features = list(X_train_sw.columns)
    removed_features = []

    for iteration in range(max_iter):
        if not selected_features:
            break

        X_train_sm = sm.add_constant(X_train_sw[selected_features], has_constant="add")
        model = sm.OLS(y_train, X_train_sm).fit()
        pvals = model.pvalues.drop("const", errors="ignore").replace([np.inf, -np.inf], np.nan).dropna()

        if pvals.empty:
            break

        worst_feature = pvals.idxmax()
        worst_p = float(pvals.max())

        if worst_p > p_threshold:
            selected_features.remove(worst_feature)
            removed_features.append({
                "Iteration": iteration + 1,
                "Removed variable": worst_feature,
                "p-value": worst_p
            })
        else:
            break

    X_train_final = sm.add_constant(X_train_sw[selected_features], has_constant="add")
    X_test_final = sm.add_constant(X_test_sw[selected_features], has_constant="add")
    final_model = sm.OLS(y_train, X_train_final).fit()
    y_pred = final_model.predict(X_test_final)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    removed_df = pd.DataFrame(removed_features)
    selected_df = pd.DataFrame({"Selected feature": selected_features})
    coef_df = pd.DataFrame({
        "Feature": final_model.params.index,
        "Coefficient": final_model.params.values,
        "p-value": final_model.pvalues.values
    })

    if add_subsection and add_text:
        add_subsection("Stepwise Metrics")
        add_text(f"RMSE: {rmse:.4f}")
        add_text(f"MAE: {mae:.4f}")
        add_text(f"R2: {r2:.4f}")

    if add_figure_to_doc is not None:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_test, y_pred, alpha=0.25, color="black")
        min_v = min(np.min(y_test), np.min(y_pred))
        max_v = max(np.max(y_test), np.max(y_pred))
        ax.plot([min_v, max_v], [min_v, max_v], color="red", linestyle="--")
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"Stepwise Actual vs Predicted - {scenario_name}")
        plt.tight_layout()
        add_figure_to_doc(fig, width=6.0)
        plt.close(fig)

    return {
        "stepwise_model": final_model,
        "stepwise_rmse": rmse,
        "stepwise_mae": mae,
        "stepwise_r2": r2,
        "stepwise_selected_df": selected_df,
        "stepwise_removed_df": removed_df,
        "stepwise_coef_df": coef_df
    }