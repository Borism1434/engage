# db.py
from sqlalchemy import create_engine
import pandas as pd

# --- Basic connection config ---
DB_USER = "postgres"
DB_PASSWORD = "1434"          # <- change if needed
DB_HOST = "100.64.23.82"      # Tailscale IP
DB_PORT = 5432                # change if your Postgres runs on a different port
DB_NAME = "fl_election"

# Build SQLAlchemy engine
def get_engine():
    conn_str = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@"
        f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(conn_str)

# Simple helper to run a query and return a DataFrame
def run_query(sql: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)

# Optional: helper to load a voterfile table by name
def load_voterfile(table_name: str, limit: int | None = None) -> pd.DataFrame:
    """
    Example: load_voterfile('voterfile_2024') 
    or load_voterfile('voterfile_2022', limit=10000)
    """
    sql = f"SELECT * FROM {table_name}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return run_query(sql)


import os
import pandas as pd
from glob import glob

def merge_chunk_csvs(output_path="predicted_matches_full.csv", chunk_folder=".", chunk_pattern="predicted_matches_chunk_*.csv"):
    """
    Merge chunk CSV files into one CSV.

    Args:
        output_path (str): Path to save the merged CSV.
        chunk_folder (str): Folder where chunk CSVs reside.
        chunk_pattern (str): Glob pattern for chunk CSV filenames.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    chunk_files = sorted(glob(os.path.join(chunk_folder, chunk_pattern)))
    if not chunk_files:
        print("No chunk files found.")
        return None

    print(f"Found {len(chunk_files)} chunk files. Starting merge...")
    df_list = []
    for f in chunk_files:
        print(f"Reading {f}")
        df_chunk = pd.read_csv(f)
        df_list.append(df_chunk)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved to {output_path} with {len(merged_df)} rows.")

    return merged_df

    # Usage example in notebook or script:
    # merged_df = merge_chunk_csvs()