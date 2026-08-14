from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def normalize_binary_series(series: pd.Series) -> pd.Series:
    s = series.astype("object")
    s = s.where(pd.notna(s), np.nan)

    def _normalize(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isclose(value, 1):
                return 1.0
            if np.isclose(value, 0):
                return 0.0
            return np.nan
        value = str(value).strip().upper()
        if value in {"Y", "YES", "TRUE", "T", "1", "1.0"}:
            return 1.0
        if value in {"N", "NO", "FALSE", "F", "0", "0.0"}:
            return 0.0
        return np.nan

    return s.map(_normalize).astype("float")


def make_ohe():
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def select_features_by_missing_threshold(X, max_missing_pct):
    missing_share = X.isna().mean().sort_values(ascending=False)
    selected_features = missing_share[missing_share <= max_missing_pct].index.tolist()
    removed_features = missing_share[missing_share > max_missing_pct].index.tolist()

    missing_summary_df = pd.DataFrame({
        "feature": missing_share.index,
        "missing_share": missing_share.values,
        "missing_percent": missing_share.values * 100,
        "kept": missing_share.index.isin(selected_features),
    })
    return selected_features, removed_features, missing_summary_df


def add_engineered_features(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    current_year = datetime.now().year

    if "obj_energyType" in df.columns:
        df["obj_energyType_cat"] = df["obj_energyType"].astype("object").where(pd.notna(df["obj_energyType"]), "Unknown").astype(str)
    else:
        df["obj_energyType_cat"] = "Unknown"

    if "obj_thermalChar" in df.columns:
        thermal_num = pd.to_numeric(df["obj_thermalChar"], errors="coerce")
        df["obj_thermalChar_num"] = thermal_num.fillna(0)
    else:
        df["obj_thermalChar_num"] = 0.0

    if "obj_numberOfFloors" in df.columns:
        floors_num = pd.to_numeric(df["obj_numberOfFloors"], errors="coerce")
        valid_floors = floors_num.notna()
        df["obj_hasNumberOfFloorsInfo"] = valid_floors.astype(float)
        df["obj_numberOfFloors_num"] = floors_num.fillna(0)
    else:
        df["obj_hasNumberOfFloorsInfo"] = 0.0
        df["obj_numberOfFloors_num"] = 0.0

    if "obj_noParkSpaces" in df.columns:
        parking_num = pd.to_numeric(df["obj_noParkSpaces"], errors="coerce")
        valid_parking = parking_num.notna()
        df["obj_hasParkingInfo"] = valid_parking.astype(float)
        df["obj_noParkSpaces_num"] = parking_num.fillna(0)
    else:
        df["obj_hasParkingInfo"] = 0.0
        df["obj_noParkSpaces_num"] = 0.0

    if "obj_lastRefurbish" in df.columns:
        last_refurbish_year = pd.to_numeric(df["obj_lastRefurbish"], errors="coerce")
        valid_refurbish = last_refurbish_year.notna() & (last_refurbish_year > 1800) & (last_refurbish_year <= current_year)
        df["obj_hasLastRefurbishInfo"] = valid_refurbish.astype(float)
        df["obj_yearsSinceLastRefurbish"] = np.where(valid_refurbish, current_year - last_refurbish_year, 0)
    else:
        last_refurbish_year = pd.Series(np.nan, index=df.index)
        valid_refurbish = pd.Series(False, index=df.index)
        df["obj_hasLastRefurbishInfo"] = 0.0
        df["obj_yearsSinceLastRefurbish"] = 0.0

    if "obj_yearConstructed" in df.columns:
        constructed_year = pd.to_numeric(df["obj_yearConstructed"], errors="coerce")
        valid_constructed = constructed_year.notna() & (constructed_year > 1800) & (constructed_year <= current_year)
        df["obj_buildingAge"] = np.where(valid_constructed, current_year - constructed_year, 0)
    else:
        constructed_year = pd.Series(np.nan, index=df.index)
        valid_constructed = pd.Series(False, index=df.index)
        df["obj_buildingAge"] = 0.0

    valid_both = valid_refurbish & valid_constructed
    df["obj_yearsBetweenConstructionAndRefurbish"] = np.where(
        valid_both & (last_refurbish_year >= constructed_year),
        last_refurbish_year - constructed_year,
        0
    )

    return df


def make_preprocessor(numeric_features, categorical_features, use_imputation=False):
    if use_imputation:
        numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ])
    else:
        numeric_transformer = "passthrough"
        categorical_transformer = Pipeline([("onehot", make_ohe())])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0
    )


def get_clean_feature_names(preprocessor, numeric_features, categorical_features):
    try:
        out = list(preprocessor.get_feature_names_out())
        return [x.replace("num__", "").replace("cat__", "") for x in out]
    except Exception:
        out = []
        out.extend(numeric_features)
        if len(categorical_features) > 0:
            cat_pipeline = preprocessor.named_transformers_["cat"]
            ohe = cat_pipeline.named_steps["onehot"]
            out.extend(list(ohe.get_feature_names_out(categorical_features)))
        return out