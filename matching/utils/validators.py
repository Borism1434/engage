# matching/utils/validators.py

def require_columns(df, cols, df_name="DataFrame"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")