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
    """
    Safely compare two values for equality, returning False if any value is NaN.

    Args:
        a: First value to compare.
        b: Second value to compare.

    Returns:
        bool: True if values are equal and neither is NaN, False otherwise.
    """
    if pd.isna(a) or pd.isna(b):
        return False
    return a == b


def compute_similarity(row):
    """
    Compute similarity features for a given row of a DataFrame.

    This adds multiple similarity metrics:
        - Jaro-Winkler similarity for first and last names
        - Exact matches for date of birth and zip code
        - Several detailed date of birth feature comparisons
        - Address similarity

    Args:
        row (pd.Series): A single row of the DataFrame with necessary columns.

    Returns:
        pd.Series: The original row with added feature columns.
    """
    fn_att = row.get("first_name_att") or ""
    fn_vf = row.get("first_name_vf") or ""
    ln_att = row.get("last_name_att") or ""
    ln_vf = row.get("last_name_vf") or ""

    # Calculate Jaro-Winkler similarity for first and last names
    row["fn_jw"] = JaroWinkler.normalized_similarity(fn_att, fn_vf)
    row["ln_jw"] = JaroWinkler.normalized_similarity(ln_att, ln_vf)

    # Exact match for normalized date of birth
    row["dob_exact"] = int(row["dob_norm_att"] == row["dob_norm_vf"])

    # Exact match for zip code, safely handling NaNs
    row["zip_exact"] = int(safe_equal(row["zip_norm_att"], row["zip_norm_vf"]))

    dob1 = row["dob_norm_att"]
    dob2 = row["dob_norm_vf"]

    # Calculate additional DOB similarity features
    row["dob_year_match"] = dob_year_match(dob1, dob2)
    row["dob_month_match"] = dob_month_match(dob1, dob2)
    row["dob_day_match"] = dob_day_match(dob1, dob2)
    row["dob_levenshtein_sim"] = dob_levenshtein_similarity(dob1, dob2)
    row["dob_month_day_swapped"] = dob_month_day_swapped(dob1, dob2)

    # Calculate similarity for addresses
    row["addr_jw"] = compute_address_similarity(
        row.get("voting_street_address_one", ""),
        row.get("residence_address_1", "")
    )

    return row


def add_features(df: pd.DataFrame):
    """
    Prepare the DataFrame and add similarity features.

    This function:
    - Checks for required columns
    - Normalizes names
    - Applies compute_similarity row-wise
    - Extracts feature matrix X and optional target y

    Args:
        df (pd.DataFrame): DataFrame containing raw columns to compute features from.

    Returns:
        tuple: (X, y) where X is a DataFrame of feature columns,
               and y is the target series if available, otherwise None.
    """
    print(f"add_features: processing chunk with {len(df)} rows")

    needed = [
        "first_name_att", "first_name_vf",
        "last_name_att", "last_name_vf",
        "dob_norm_att", "dob_norm_vf",
        "zip_norm_att", "zip_norm_vf"
    ]

    # Ensure required columns exist
    require_columns(df, needed, "train_df")

    # Normalize name columns before similarity computation
    df["first_name_att"] = normalize_name_series(df["first_name_att"])
    df["last_name_att"] = normalize_name_series(df["last_name_att"])
    df["first_name_vf"] = normalize_name_series(df["first_name_vf"])
    df["last_name_vf"] = normalize_name_series(df["last_name_vf"])

    # Apply compute_similarity function to all rows
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
        "addr_jw"
    ]

    # If target column exists, add it to needed columns for validation
    if "is_match" in df.columns:
        needed.append("is_match")
    
    require_columns(df, needed, "train_df")

    # Extract feature matrix
    X = df[feature_cols]

    # Extract target vector if available
    y = df["is_match"] if "is_match" in df.columns else None

    return X, y


def get_feature_matrix(df: pd.DataFrame):
    """
    Extract features matrix and target vector from DataFrame.

    This uses predefined feature columns from config and expects "is_match" as target.

    Args:
        df (pd.DataFrame): DataFrame containing features and target.

    Returns:
        tuple: (X, y) feature matrix and target vector.
    """
    X = df[FEATURE_COLS]
    y = df["is_match"]
    return X, y