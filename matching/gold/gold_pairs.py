# matching/gold/gold_pairs.py

import pandas as pd
from matching.utils.normalization import (
    normalize_text_series,
    normalize_zip_series,
    normalize_dob_series,
)
from matching.utils.validators import require_columns
from matching.config.match_config import TRUE_MATCH_TYPES

def add_normalized_keys(att_df: pd.DataFrame, vf_df: pd.DataFrame):
    # attempts
    att_df["fn_norm_att"]  = normalize_text_series(att_df["first_name_att"])
    att_df["ln_norm_att"]  = normalize_text_series(att_df["last_name_att"])
    att_df["zip_norm_att"] = normalize_zip_series(att_df["zip_raw_att"])
    att_df["dob_norm_att"] = normalize_dob_series(att_df["dob_raw_att"])
    att_df["dob_year_att"] = att_df["dob_norm_att"].dt.year.astype("Int64")

    # voterfile
    vf_df["fn_norm_vf"]  = normalize_text_series(vf_df["first_name_vf"])
    vf_df["ln_norm_vf"]  = normalize_text_series(vf_df["last_name_vf"])
    vf_df["zip_norm_vf"] = normalize_zip_series(vf_df["zip_raw_vf"])
    vf_df["dob_norm_vf"] = normalize_dob_series(vf_df["dob_raw_vf"])
    vf_df["dob_year_vf"] = vf_df["dob_norm_vf"].dt.year.astype("Int64")

    return att_df, vf_df

def build_gold_pairs(att_df: pd.DataFrame, vf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a set of high-confidence ("gold") positive pairs for supervised matching.

    This function merges a DataFrame of normalized attempt records (e.g., from registration forms)
    with a DataFrame of normalized voterfile records using strict deterministic keys:
    first name, last name, date of birth, and ZIP code (all normalized). Only those
    registration attempts that match exactly one voterfile record (i.e., not ambiguous)
    are retained to ensure high-likelihood matches.

    The resulting DataFrame is labeled with `is_match` based on predefined true match types,
    returning only a minimal subset of columns needed for training downstream models.

    Parameters
    ----------
    att_df : pd.DataFrame
        DataFrame of registration attempt records (with normalized fields).
        Must include columns: registration_form_id, fn_norm_att, ln_norm_att,
        dob_norm_att, zip_norm_att, type_code, confidence_score.

    vf_df : pd.DataFrame
        DataFrame of normalized voterfile records.
        Must include columns: voter_id, fn_norm_vf, ln_norm_vf,
        dob_norm_vf, zip_norm_vf.

    Returns
    -------
    pos_df : pd.DataFrame
        DataFrame containing strictly matched positive pairs with columns for both
        sides (attempt and voterfile) and an `is_match` label, suitable for training
        matching models.
    """

    needed_att = ["registration_form_id", "fn_norm_att", "ln_norm_att", "dob_norm_att", "zip_norm_att"]
    needed_vf  = ["voter_id", "fn_norm_vf", "ln_norm_vf", "dob_norm_vf", "zip_norm_vf"]

    require_columns(att_df, needed_att + ["type_code", "confidence_score"], "attempts")
    require_columns(vf_df, needed_vf, "voterfile")

    # Strict deterministic join
    clean_pairs = att_df.merge(
        vf_df,
        left_on=["fn_norm_att", "ln_norm_att", "dob_norm_att", "zip_norm_att"],
        right_on=["fn_norm_vf", "ln_norm_vf", "dob_norm_vf", "zip_norm_vf"],
        how="inner",
        suffixes=("_att", "_vf"),
    )

    # Count how many voter_ids per registration_form_id
    match_counts = (
        clean_pairs.groupby("registration_form_id")["voter_id"]
        .nunique()
        .reset_index(name="n_voters")
    )

    good_ids = match_counts[match_counts["n_voters"] == 1]["registration_form_id"]
    gold_pairs = clean_pairs[clean_pairs["registration_form_id"].isin(good_ids)].copy()

    # is_match label driven by match type
    gold_pairs["is_match"] = gold_pairs["type_code"].isin(TRUE_MATCH_TYPES).astype(int)

    # Minimal positive training frame
    pos_df = clean_pairs[[
        "registration_form_id",
        "first_name_att", "last_name_att", "dob_norm_att", "zip_norm_att",
        "first_name_vf", "last_name_vf", "dob_norm_vf", "zip_norm_vf"
    ]].copy()
    pos_df["is_match"] = 1

    attempt_match_counts = (
        clean_pairs.groupby("registration_form_id")["voter_id"]
        .nunique()
        .reset_index(name="n_voters")
    )
    good_attempts = attempt_match_counts[attempt_match_counts["n_voters"] == 1]["registration_form_id"]

    gold_pairs = clean_pairs[clean_pairs["registration_form_id"].isin(good_attempts)].copy()

    return pos_df, gold_pairs, att_df  # or return att_df if you want to pass something else