# matching/features/feature_builder.py



import pandas as pd
from rapidfuzz.distance import JaroWinkler

from matching.config.match_config import FEATURE_COLS
from matching.utils.validators import require_columns
from matching.features.address_features import compute_address_similarity
from matching.features.date_features import (
    dob_year_match,
    dob_month_match,
    dob_day_match,
    dob_levenshtein_similarity,
    dob_month_day_swapped,
)

from matching.features.name_normalization import normalize_name_series



# Integration in pipeline:
# Extend compute_similarity() to call new feature computations from these new modules.
# Update add_features() to include these additional features columns.

def safe_equal(a, b):
    if pd.isna(a) or pd.isna(b):
        return False
    return a == b

def compute_similarity(row):
    fn_att = row.get("first_name_att") or ""
    fn_vf  = row.get("first_name_vf") or ""
    ln_att = row.get("last_name_att") or ""
    ln_vf  = row.get("last_name_vf") or ""

    row["fn_jw"] = JaroWinkler.normalized_similarity(fn_att, fn_vf)
    row["ln_jw"] = JaroWinkler.normalized_similarity(ln_att, ln_vf)

    row["dob_exact"] = int(row["dob_norm_att"] == row["dob_norm_vf"])
    row["zip_exact"] = int(safe_equal(row["zip_norm_att"], row["zip_norm_vf"]))

    dob1 = row["dob_norm_att"]
    dob2 = row["dob_norm_vf"]

    row["dob_year_match"] = dob_year_match(dob1, dob2)
    row["dob_month_match"] = dob_month_match(dob1, dob2)
    row["dob_day_match"] = dob_day_match(dob1, dob2)
    row["dob_levenshtein_sim"] = dob_levenshtein_similarity(dob1, dob2)
    row["dob_month_day_swapped"] = dob_month_day_swapped(dob1, dob2)
    row["addr_jw"] = compute_address_similarity(
        row.get("voting_street_address_one", ""),
        row.get("residence_address_1", "")
    )
    

    return row

def add_features(df: pd.DataFrame):
    print(f"add_features: processing chunk with {len(df)} rows")

    needed = [
        "first_name_att", "first_name_vf",
        "last_name_att", "last_name_vf",
        "dob_norm_att", "dob_norm_vf",
        "zip_norm_att", "zip_norm_vf"
        
    ]
    # Ensure required columns exist (update if you add address etc.)
    require_columns(df, needed, "train_df")

    # Normalize names before similarity computation
    df["first_name_att"] = normalize_name_series(df["first_name_att"])
    df["last_name_att"] = normalize_name_series(df["last_name_att"])
    df["first_name_vf"] = normalize_name_series(df["first_name_vf"])
    df["last_name_vf"] = normalize_name_series(df["last_name_vf"])
    
    # Compute similarity features including your new date features
    df = df.apply(compute_similarity, axis=1)
    
    feature_cols = [
        "fn_jw",
        "ln_jw",
        "dob_exact",
        "zip_exact",
        "dob_year_match",
        "dob_month_match",
        "dob_day_match",
        "dob_levenshtein_sim",
        "dob_month_day_swapped",
        "addr_jw"  # if you have address features included
    ]
    # Check only if 'is_match' in columns
    if "is_match" in df.columns:
        needed.append("is_match")
    
    require_columns(df, needed, "train_df")

    # example features computed by compute_similarity
    X = df[feature_cols]
    y = df["is_match"] if "is_match" in df.columns else None
    return X, y

def get_feature_matrix(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["is_match"]
    return X, y