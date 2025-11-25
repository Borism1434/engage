# merge_chunks.py
import os
import pandas as pd
from glob import glob
import re
from script_helper import add_repo_root_to_syspath; add_repo_root_to_syspath()

def get_chunk_num(filename):
    match = re.search(r'predicted_matches_chunk_(\d+)\.csv', filename)
    return int(match.group(1)) if match else -1

def merge_chunk_csvs(
    output_path="predicted_matches_full.parquet",
    best_matches_path="predicted_best_matches.parquet",
    chunk_folder="/Users/borismartinez/Documents/GitHub/engage/chunk_folder",
    chunk_pattern="predicted_matches_chunk_*.csv"
):
    """
    Merge chunk CSV files into one efficient Parquet file and save best matches per registration_form_id.

    Args:
        output_path (str): Path to save the merged Parquet.
        best_matches_path (str): Path to save the filtered best matches Parquet.
        chunk_folder (str): Folder where chunk CSVs reside.
        chunk_pattern (str): Glob pattern for chunk CSV filenames.

    Returns:
        pd.DataFrame: The merged DataFrame with all candidates.
    """
    # Find all chunk files and sort numerically
    chunk_files = glob(os.path.join(chunk_folder, chunk_pattern))
    if not chunk_files:
        print("No chunk files found.")
        return None

    chunk_files = sorted(chunk_files, key=get_chunk_num)
    print(f"Found {len(chunk_files)} chunk files. Starting merge...")

    # Read chunks and append to list
    df_list = []
    for f in chunk_files:
        print(f"Reading {f}")
        df_chunk = pd.read_csv(f, low_memory=False)
        df_list.append(df_chunk)

    # Concatenate all DataFrames and reset index
    merged_df = pd.concat(df_list, ignore_index=True).reset_index(drop=True)

    # Fix object columns: fill NaNs and cast to string to avoid PyArrow errors
    for col in merged_df.select_dtypes(include=['object']).columns:
        merged_df[col] = merged_df[col].fillna('').astype(str)

    # Convert match_prob to numeric and drop rows with NaNs
    merged_df['match_prob'] = pd.to_numeric(merged_df['match_prob'], errors='coerce')
    merged_df = merged_df.dropna(subset=['match_prob'])

    print(f"Merged data saved to {output_path} with {len(merged_df)} rows.")
    print(f"Unique registration_form_id count before filtering: {merged_df['registration_form_id'].nunique()}")

    # Filter to keep only highest match_prob per registration_form_id
    idx = merged_df.groupby('registration_form_id')['match_prob'].idxmax()
    best_matches = merged_df.loc[idx]

    print(f"Best matches saved to {best_matches_path} with {len(best_matches)} rows.")
    print(f"Unique registration_form_id count after filtering: {best_matches['registration_form_id'].nunique()}")

    # Save full merged data as Parquet
    merged_df.to_parquet(output_path, index=False)

    # Save best matches as Parquet
    best_matches.to_parquet(best_matches_path, index=False)

    return merged_df

if __name__ == "__main__":
    merge_chunk_csvs()