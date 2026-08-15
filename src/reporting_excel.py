import pandas as pd

def safe_sheet_name(name: str) -> str:
    invalid_chars = ["\\", "/", "*", "?", ":", "[", "]"]
    for ch in invalid_chars:
        name = name.replace(ch, "_")
    return name[:31]

def concat_export_table(all_exports, key):
    frames = []
    for export_data in all_exports.values():
        df_tmp = export_data.get(key, pd.DataFrame())
        if isinstance(df_tmp, pd.DataFrame) and not df_tmp.empty:
            frames.append(df_tmp)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def export_excel_report(
    output_excel,
    comparison_df,
    all_exports,
    safe_sheet_name,
    table_6_1_missing_data,
    table_6_2_linear_performance,
    table_6_3_top_pearson,
    table_6_4_ols_diagnostics,
    table_stepwise_removed_all,
    table_vif_all,
    table_missing_feature_summary_all,
    table_remediation_all=None
):
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        # Thesis-ready
        table_6_1_missing_data.to_excel(writer, index=False, sheet_name="Table_6_1_Missing_Data")
        table_6_2_linear_performance.to_excel(writer, index=False, sheet_name="Table_6_2_Linear_Perf")
        table_6_3_top_pearson.to_excel(writer, index=False, sheet_name="Table_6_3_Pearson")
        table_6_4_ols_diagnostics.to_excel(writer, index=False, sheet_name="Table_6_4_OLS_Diag")
        table_stepwise_removed_all.to_excel(writer, index=False, sheet_name="Appendix_Stepwise_Removed")
        table_vif_all.to_excel(writer, index=False, sheet_name="Appendix_VIF")
        table_missing_feature_summary_all.to_excel(writer, index=False, sheet_name="Appendix_Missing_Features")
        if table_remediation_all is not None and not table_remediation_all.empty:
            table_remediation_all.to_excel(writer, index=False, sheet_name="Appendix_Remediation")

        if not comparison_df.empty:
            comparison_df.to_excel(writer, index=False, sheet_name="Full_Comparison")

        # Detailed scenarios
        for scenario_name, export_data in all_exports.items():
            clean_name = scenario_name.replace(" ", "_").replace("-", "_")

            export_data["metrics_df"].to_excel(
                writer, index=False, sheet_name=safe_sheet_name(f"{clean_name}_metrics")
            )
            export_data["coef_df"].to_excel(
                writer, index=False, sheet_name=safe_sheet_name(f"{clean_name}_ols_coef")
            )
            export_data["ridge_coef_df"].to_excel(
                writer, index=False, sheet_name=safe_sheet_name(f"{clean_name}_ridge_coef")
            )
            export_data["stepwise_coef_df"].to_excel(
                writer, index=False, sheet_name=safe_sheet_name(f"{clean_name}_step_coef")
            )
            export_data["stepwise_selected_df"].to_excel(
                writer, index=False, sheet_name=safe_sheet_name(f"{clean_name}_step_selected")
            )