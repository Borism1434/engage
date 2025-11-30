import os
import json
import joblib
import pandas as pd
from script_helper import add_repo_root_to_syspath; add_repo_root_to_syspath()
from matching.loaders.load_attempts import load_attempts, filter_dataframe_by_columns
from matching.loaders.load_voterfile import load_voterfile_chunk
from matching.candidates.block_candidates import block_candidate_pairs
from matching.features.feature_builder import add_features
from matching.modeling.predict import predict_chunk
from matching.gold.gold_pairs import add_normalized_keys


PROGRESS_FILE = "progress.json"


def load_progress():
    """
    Load the progress state from a JSON file if it exists.

    Returns:
        dict: Progress mapping chunk indices (as strings) to booleans indicating processing done.
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    else:
        return {}


def save_progress(progress):
    """
    Save the progress state dict to a JSON file.

    Args:
        progress (dict): Mapping of chunk indices (as strings) to processed booleans.
    """
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def save_chunk_results(candidates_df, chunk_idx, output_dir="/Users/borismartinez/Documents/GitHub/engage/chunk_folder"):
    """
    Save candidate match predictions for a specific chunk to a CSV file.

    Args:
        candidates_df (pd.DataFrame): DataFrame containing candidate pairs and prediction results.
        chunk_idx (int): Index of the current chunk.
        output_dir (str, optional): Directory to save chunk files. Defaults to a fixed path.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"predicted_matches_chunk_{chunk_idx}.csv")
    candidates_df.to_csv(filename, index=False)
    print(f"Chunk {chunk_idx}: Saved {len(candidates_df)} rows to {filename}")


def stream_inference(attempts_path, db_engine, sql, model_path, chunksize=5000):
    """
    Stream voterfile data chunks and run inference on generated candidate pairs.

    This function:
    - Loads attempts data and filters necessary columns
    - Iterates over voterfile chunks from a SQL query
    - Generates candidate pairs by blocking
    - Computes features and runs model predictions
    - Saves progress and results incrementally

    Args:
        attempts_path (str): Path to CSV file containing attempts data.
        db_engine: Database engine for querying voterfile chunks.
        sql (str): SQL query string to select voterfile chunk data.
        model_path (str): Path to the trained model pickle file.
        chunksize (int, optional): Number of rows to load per voterfile chunk. Defaults to 5000.
    """
    # Load attempts data
    attempts = load_attempts(attempts_path)
    # Rename columns to normalized names for matching
    attempts = attempts.rename(columns={
        "first_name": "first_name_att",
        "last_name": "last_name_att",
        "date_of_birth": "dob_raw_att",
        "voting_zipcode": "zip_raw_att",
    })

    # Filter attempts dataframe to required columns
    required_columns = ["first_name_att", "last_name_att"]
    attempts = filter_dataframe_by_columns(attempts, required_columns)
    print(f"Attempts after required columns filter: {len(attempts)}")

    # Filter attempts by upload_time containing 2025
    attempts = attempts[attempts["upload_time"].astype(str).str.contains("2025", na=False)]
    print(f"Attempts after 'upload_time' filter: {len(attempts)}")

    # Print sample registration ids for verification
    print(f"Sample registration_form_id: {attempts['registration_form_id'].unique()}")

    # Load prediction model
    model = joblib.load(model_path)
    # Load processing progress from previous runs
    progress = load_progress()

    chunks_processed = 0

    # Iterate over SQL chunk generator
    for idx, vf_chunk in enumerate(load_voterfile_chunk(db_engine, sql, chunksize=chunksize)):
        # Skip chunk if already processed (according to progress file)
        if str(idx) in progress:
            print(f"Chunk {idx}: already processed, skipping")
            continue

        print(f"Chunk {idx}: processing ...")

        # Rename voterfile chunk columns to normalized names
        vf_chunk = vf_chunk.rename(columns={
            "first_name": "first_name_vf",
            "last_name": "last_name_vf",
            "residence_zipcode": "zip_raw_vf",
            "birth_date": "dob_raw_vf",
        })

        # Add normalized keys for attempts and voterfile chunks
        attempts_norm, vf_chunk_norm = add_normalized_keys(attempts, vf_chunk)

        # Generate candidate pairs via blocking
        candidates = block_candidate_pairs(attempts_norm, vf_chunk_norm)
        print(f"Chunk {idx}: Generated {len(candidates)} candidate pairs")
        print(f"Chunk {idx}: Unique registration_form_id in candidates: {candidates['registration_form_id'].nunique() if not candidates.empty else 0}")

        # If no candidates, mark chunk done and continue
        if candidates.empty:
            print(f"Chunk {idx}: No candidate pairs generated, skipping feature build and prediction")
            progress[str(idx)] = True
            save_progress(progress)
            continue

        # Compute features for candidates
        X, _ = add_features(candidates)
        print(f"Chunk {idx}: Features computed for {X.shape[0]} candidates")

        # Make predictions using loaded model
        probs = predict_chunk(model, X)
        candidates['match_prob'] = probs
        print(f"Chunk {idx}: Predictions made")

        # Save results for this chunk
        save_chunk_results(candidates, idx)

        # Update progress file
        progress[str(idx)] = True
        save_progress(progress)

        chunks_processed += 1

    print(f"Processing complete. Total chunks processed: {chunks_processed}")

    # Optional: remove progress file if all chunks processed
    # Use total_chunks to verify
    # if total_chunks and chunks_processed == total_chunks:
    #     os.remove(PROGRESS_FILE)
    #     print("All chunks processed. Progress file deleted.")


if __name__ == "__main__":
    import matching.utils.db as db

    # Prepare database engine for querying
    db_engine = db.get_engine()

    # Filepath to attempts CSV data
    attempts_path = "/Users/borismartinez/Documents/GitHub/engage/data/vr_blocks_export_no_na.csv"

    # SQL query to load voterfile data
    sql = "SELECT * FROM voterfile.election_detail_2024 WHERE county = 'DAD'"

    # Model file path
    model_path = "models/model.pkl"

    # Number of rows per chunk
    chunksize = 5000

    # Optionally clear chunk output directory before processing
    output_dir = "/Users/borismartinez/Documents/GitHub/engage/chunk_folder"
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.startswith("predicted_matches_chunk_") and f.endswith(".csv"):
                os.remove(os.path.join(output_dir, f))
    else:
        os.makedirs(output_dir)

    # Run streaming inference over chunks
    stream_inference(
        attempts_path=attempts_path,
        db_engine=db_engine,
        sql=sql,
        model_path=model_path,
        chunksize=chunksize
    )