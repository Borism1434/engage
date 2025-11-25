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
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

def save_chunk_results(candidates_df, chunk_idx, output_dir="/Users/borismartinez/Documents/GitHub/engage/chunk_folder"):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"predicted_matches_chunk_{chunk_idx}.csv")
    candidates_df.to_csv(filename, index=False)
    print(f"Chunk {chunk_idx}: Saved {len(candidates_df)} rows to {filename}")

def stream_inference(attempts_path, db_engine, sql, model_path, chunksize=5000):
    attempts = load_attempts(attempts_path)
    attempts = attempts.rename(columns={
        "first_name": "first_name_att",
        "last_name": "last_name_att",
        "date_of_birth": "dob_raw_att",
        "voting_zipcode": "zip_raw_att",
    })

    required_columns = ["first_name_att", "last_name_att"]
    attempts = filter_dataframe_by_columns(attempts, required_columns)
    print(f"Attempts after required columns filter: {len(attempts)}")

    attempts = attempts[attempts["upload_time"].astype(str).str.contains("2025", na=False)]
    print(f"Attempts after 'upload_time' filter: {len(attempts)}")

    # sample_n = min(1000, len(attempts))
    # attempts = attempts.sample(n=sample_n, random_state=42)
    # print(f"Attempts after sampling {sample_n} rows: {len(attempts)}")
    print(f"Sample registration_form_id: {attempts['registration_form_id'].unique()}")

    model = joblib.load(model_path)
    progress = load_progress()

    chunks_processed = 0
    total_chunks = None  # Optional, populate if you want

    for idx, vf_chunk in enumerate(load_voterfile_chunk(db_engine, sql, chunksize=chunksize)):
        if str(idx) in progress:
            print(f"Chunk {idx}: already processed, skipping")
            continue

        print(f"Chunk {idx}: processing ...")
        vf_chunk = vf_chunk.rename(columns={
            "first_name": "first_name_vf",
            "last_name": "last_name_vf",
            "residence_zipcode": "zip_raw_vf",
            "birth_date": "dob_raw_vf",
        })

        attempts_norm, vf_chunk_norm = add_normalized_keys(attempts, vf_chunk)

        candidates = block_candidate_pairs(attempts_norm, vf_chunk_norm)
        print(f"Chunk {idx}: Generated {len(candidates)} candidate pairs")
        print(f"Chunk {idx}: Unique registration_form_id in candidates: {candidates['registration_form_id'].nunique() if not candidates.empty else 0}")

        if candidates.empty:
            print(f"Chunk {idx}: No candidate pairs generated, skipping feature build and prediction")
            progress[str(idx)] = True
            save_progress(progress)
            continue

        X, _ = add_features(candidates)
        print(f"Chunk {idx}: Features computed for {X.shape[0]} candidates")

        probs = predict_chunk(model, X)
        candidates['match_prob'] = probs
        print(f"Chunk {idx}: Predictions made")

        save_chunk_results(candidates, idx)
        progress[str(idx)] = True
        save_progress(progress)
        chunks_processed += 1

    print(f"Processing complete. Total chunks processed: {chunks_processed}")

    # Optionally delete progress file if all chunks processed
    # To know if all done, you might want total_chunks count:
    # If total_chunks and chunks_processed == total_chunks:
    #     os.remove(PROGRESS_FILE)
    #     print("All chunks processed. Progress file deleted.")

if __name__ == "__main__":
    import matching.utils.db as db
    db_engine = db.get_engine()
    attempts_path = "/Users/borismartinez/Documents/GitHub/engage/data/vr_blocks_export.csv"
    sql = "SELECT * FROM voterfile.election_detail_2024 WHERE county = 'DAD'"
    model_path = "model.pkl"
    chunksize = 5000

    # Optional: clear chunk files before running
    output_dir = "/Users/borismartinez/Documents/GitHub/engage/chunk_folder"
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.startswith("predicted_matches_chunk_") and f.endswith(".csv"):
                os.remove(os.path.join(output_dir, f))
    else:
        os.makedirs(output_dir)

    stream_inference(
        attempts_path=attempts_path,
        db_engine=db_engine,
        sql=sql,
        model_path=model_path,
        chunksize=chunksize
    )