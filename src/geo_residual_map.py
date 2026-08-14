import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path


def _draw_germany_outline_from_geojson(ax):
    """
    Draw Germany polygon from local GeoJSON file.
    Returns True when the outline was drawn.
    """
    geojson_path = Path(__file__).resolve().parent / "data" / "germany.geojson"
    if not geojson_path.exists():
        return False

    try:
        with geojson_path.open("r", encoding="utf-8") as f:
            geo = json.load(f)
    except Exception:
        return False

    features = geo.get("features", [])
    if not features:
        return False

    geometry = features[0].get("geometry", {})
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if not coords:
        return False

    polygons = []
    if gtype == "Polygon":
        polygons = [coords]
    elif gtype == "MultiPolygon":
        polygons = coords
    else:
        return False

    for poly in polygons:
        if not poly:
            continue
        outer_ring = poly[0]
        if len(outer_ring) < 3:
            continue
        xs = [pt[0] for pt in outer_ring]
        ys = [pt[1] for pt in outer_ring]
        ax.fill(xs, ys, color="#f5f5f5", zorder=1)
        ax.plot(xs, ys, color="#666666", linewidth=0.9, zorder=2)

    return True


def _draw_bundesland_boundaries(ax):
    """
    Draw Bundesland internal boundaries from local GeoJSON.
    """
    geojson_path = Path(__file__).resolve().parent / "data" / "bundeslaender.geojson"
    if not geojson_path.exists():
        return False

    try:
        with geojson_path.open("r", encoding="utf-8") as f:
            geo = json.load(f)
    except Exception:
        return False

    features = geo.get("features", [])
    if not features:
        return False

    for feat in features:
        geometry = feat.get("geometry", {})
        gtype = geometry.get("type")
        coords = geometry.get("coordinates", [])
        if not coords:
            continue

        polygons = []
        if gtype == "Polygon":
            polygons = [coords]
        elif gtype == "MultiPolygon":
            polygons = coords
        else:
            continue

        for poly in polygons:
            if not poly:
                continue
            outer_ring = poly[0]
            if len(outer_ring) < 3:
                continue
            xs = [pt[0] for pt in outer_ring]
            ys = [pt[1] for pt in outer_ring]
            ax.plot(xs, ys, color="#8a8a8a", linewidth=0.45, alpha=0.95, zorder=2)

    return True


def sanitize_german_zip_series(zip_series: pd.Series) -> pd.Series:
    """
    Convert zip-like values to canonical 5-digit German postal code strings.
    """
    cleaned = (
        zip_series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
    )
    cleaned = cleaned.where(cleaned.str.len().between(4, 5))
    cleaned = cleaned.str.zfill(5)
    return cleaned


def build_residual_zip_tables(df_source, index, y_true, y_pred, zip_col, scenario_name):
    """
    Build test-level residuals and ZIP-level aggregates for Germany maps.
    """
    if zip_col not in df_source.columns:
        return pd.DataFrame(), pd.DataFrame()

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    base = pd.DataFrame({
        "row_index": index,
        "scenario": scenario_name,
        "actual": y_true_arr,
        "predicted": y_pred_arr
    })
    base["residual"] = base["actual"] - base["predicted"]
    base["abs_residual"] = base["residual"].abs()
    eps = 1e-9
    denom_actual = np.where(np.abs(base["actual"].to_numpy(dtype=float)) > eps, np.abs(base["actual"].to_numpy(dtype=float)), np.nan)
    base["pct_error"] = (base["residual"].to_numpy(dtype=float) / denom_actual) * 100.0
    base["abs_pct_error"] = np.abs(base["pct_error"].to_numpy(dtype=float))
    smape_denom = np.abs(base["actual"].to_numpy(dtype=float)) + np.abs(base["predicted"].to_numpy(dtype=float)) + eps
    base["smape_pct"] = 200.0 * np.abs(base["residual"].to_numpy(dtype=float)) / smape_denom
    base["zip_code"] = sanitize_german_zip_series(
        df_source.loc[index, zip_col]
    ).to_numpy()
    base = base.dropna(subset=["zip_code"]).copy()

    if base.empty:
        return base, pd.DataFrame()

    zip_summary = (
        base.groupby("zip_code", as_index=False)
        .agg(
            n_listings=("residual", "size"),
            mean_residual=("residual", "mean"),
            median_residual=("residual", "median"),
            mean_abs_residual=("abs_residual", "mean"),
            mean_pct_error=("pct_error", "mean"),
            median_pct_error=("pct_error", "median"),
            mean_abs_pct_error=("abs_pct_error", "mean"),
            mape_pct=("abs_pct_error", "mean"),
            smape_pct=("smape_pct", "mean")
        )
        .sort_values("mean_abs_residual", ascending=False)
    )

    return base, zip_summary


def plot_germany_residual_map(
    zip_summary_df,
    scenario_name,
    output_prefix,
    min_listings_per_zip=3,
    max_points=1200,
    color_column="mean_residual",
    cmap="RdBu_r",
    color_label=None,
    symmetric_scale=True,
    clip_percentile_low=2.0,
    clip_percentile_high=98.0,
    vmin=None,
    vmax=None
):
    """
    Plot a Germany map with ZIP centroids colored by mean residual.
    Returns generated image path or None.
    """
    if zip_summary_df.empty:
        return None

    plot_df = zip_summary_df.copy()
    if min_listings_per_zip > 1:
        plot_df = plot_df.loc[plot_df["n_listings"] >= min_listings_per_zip].copy()

    if plot_df.empty:
        return None

    try:
        import pgeocode
    except Exception:
        return None

    nomi = pgeocode.Nominatim("de")
    zip_geo = nomi.query_postal_code(plot_df["zip_code"].tolist())

    geo_df = pd.DataFrame({
        "zip_code": plot_df["zip_code"].values,
        "latitude": zip_geo["latitude"].values,
        "longitude": zip_geo["longitude"].values
    })

    map_df = plot_df.merge(geo_df, on="zip_code", how="left")
    map_df = map_df.dropna(subset=["latitude", "longitude"]).copy()
    if map_df.empty:
        return None

    if len(map_df) > max_points:
        map_df = map_df.nlargest(max_points, "n_listings").copy()

    if color_column not in map_df.columns:
        return None

    color_values = map_df[color_column].to_numpy(dtype=float)
    valid_colors = color_values[np.isfinite(color_values)]
    if valid_colors.size == 0:
        return None

    if clip_percentile_low is not None and clip_percentile_high is not None:
        lo = np.nanpercentile(valid_colors, clip_percentile_low)
        hi = np.nanpercentile(valid_colors, clip_percentile_high)
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            color_values = np.clip(color_values, lo, hi)

    if vmin is None or vmax is None:
        cmin = float(np.nanmin(color_values))
        cmax = float(np.nanmax(color_values))
        if symmetric_scale:
            cabs = max(abs(cmin), abs(cmax))
            vmin_eff, vmax_eff = -cabs, cabs
        else:
            vmin_eff, vmax_eff = cmin, cmax
    else:
        vmin_eff, vmax_eff = float(vmin), float(vmax)

    fig, ax = plt.subplots(figsize=(7.5, 9.2))

    germany_outline_drawn = _draw_germany_outline_from_geojson(ax)
    _draw_bundesland_boundaries(ax)

    size_scale = np.clip(map_df["n_listings"].to_numpy(dtype=float), 1.0, None)
    marker_sizes = 20.0 + 4.0 * np.sqrt(size_scale)

    sc = ax.scatter(
        map_df["longitude"],
        map_df["latitude"],
        c=color_values,
        s=marker_sizes,
        cmap=cmap,
        vmin=vmin_eff,
        vmax=vmax_eff,
        alpha=0.88,
        edgecolors="black",
        linewidths=0.2,
        zorder=3
    )

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    if color_label is None:
        default_labels = {
            "mean_residual": "Mean residual",
            "mean_abs_residual": "Mean absolute residual",
            "mean_pct_error": "Mean percentage error (%)",
            "mean_abs_pct_error": "Mean absolute percentage error (%)",
            "mape_pct": "MAPE (%)",
            "smape_pct": "sMAPE (%)"
        }
        color_label = default_labels.get(color_column, color_column)
    cbar.set_label(color_label)

    ax.set_title(f"Germany residual map by ZIP - {scenario_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25, linewidth=0.5)

    if not germany_outline_drawn:
        ax.set_xlim(5.5, 15.6)
        ax.set_ylim(47.0, 55.2)

    plt.tight_layout()

    safe_name = scenario_name.replace(" ", "_").replace("-", "_")
    output_path = f"{output_prefix}_{safe_name}.png"
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path
