# matching/loaders/load_attempts.py

import pandas as pd
import numpy as np
from matching.config.column_map import ATTEMPT_COL_MAP
from matching.utils.validators import require_columns

def load_matches(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_attempts(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def attach_match_labels(attempts_df, matches_df):
    keep = ["registration_form_id", "type_code", "confidence_score"]
    require_columns(matches_df, keep, "matches")
    return attempts_df.merge(matches_df[keep], on="registration_form_id", how="inner")

def standardize_attempt_columns(att_labeled: pd.DataFrame) -> pd.DataFrame:
    # keep only the columns we care about and rename
    cols_present = [c for c in ATTEMPT_COL_MAP.keys() if c in att_labeled.columns]
    att_small = att_labeled[cols_present].copy()
    att_small.rename(columns=ATTEMPT_COL_MAP, inplace=True)
    return att_small




def filter_dataframe_by_columns(df: pd.DataFrame, columns: list, unique_col: str = None) -> pd.DataFrame:
    """
    Filters the dataframe removing rows where any of the specified columns contain missing or invalid data.

    Args:
        df (pd.DataFrame): Input dataframe to filter
        columns (list of str): List of column names to check for missing data
        unique_col (str, optional): Column name to keep unique values for. Defaults to None.

    Returns:
        pd.DataFrame: Filtered dataframe with rows having no missing/invalid data in given columns
    """

    def clean_column(series):
        cleaned = series.astype(str).str.strip()
        cleaned_lower = cleaned.str.lower()
        null_equivalents = ['', 'na', 'nan', 'null', 'none', 'undefined']
        cleaned[cleaned_lower.isin(null_equivalents)] = np.nan
        return cleaned

    df_cleaned = df.copy()

    for col in columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = clean_column(df_cleaned[col])
        else:
            raise ValueError(f"Column '{col}' not found in dataframe")

    filtered_df = df_cleaned.dropna(subset=columns)

    if unique_col:
        if unique_col in filtered_df.columns:
            filtered_df = filtered_df.drop_duplicates(subset=unique_col)
        else:
            raise ValueError(f"Unique column '{unique_col}' not found in dataframe")

    return filtered_df