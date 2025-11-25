from script_helper import add_repo_root_to_syspath; add_repo_root_to_syspath()
add_repo_root_to_syspath()
import pandas as pd
import joblib
from matching.loaders.load_attempts import load_attempts, load_matches
from matching.gold.gold_pairs import add_normalized_keys  # for normalization
from matching.features.feature_builder import add_features
import matching.utils.db as db  # your DB module with run_query function


import sys
print("sys.path is now:")
for p in sys.path:
    print(p)

def generate_candidate_pairs(attempts_df, vf_df):
    # Same blocking logic
    attempts_df["ln0"] = attempts_df["last_name_att"].str[0].str.lower()
    attempts_df["dob_year"] = attempts_df["dob_norm_att"].dt.year

    
    vf_df["ln0"] = vf_df["last_name_vf"].str[0].str.lower()
    vf_df["dob_year"] = vf_df["dob_norm_vf"].dt.year

    candidates = attempts_df.merge(
        vf_df,
        how="inner",
        left_on=["ln0", "dob_year"],
        right_on=["ln0", "dob_year"],
        suffixes=('_att', '_vf'),
    )
    return candidates

def main():
    attempts = load_attempts("/Users/borismartinez/Documents/GitHub/engage/data/vr_blocks_export_no_na.csv")

    # Load voterfile from remote database
    query = "SELECT * FROM voterfile.election_detail_2024 WHERE county = 'DAD';"
    vf_extract = db.run_query(query)

    print([repr(c) for c in attempts.columns])

    attempts.columns = attempts.columns.str.strip()
    attempts.columns = attempts.columns.str.replace(r"['\"]", "", regex=True).str.strip()
    

    # Rename columns as per your training pipeline
    attempts = attempts.rename(columns={
    "first_name": "first_name_att",
    "last_name": "last_name_att",
    "date_of_birth": "dob_raw_att",
    "voting_zipcode": "zip_raw_att",
    })  

    vf_extract = vf_extract.rename(columns={
    "first_name": "first_name_vf",
    "last_name": "last_name_vf",
    "residence_zipcode": "zip_raw_vf",
    "birth_date": "dob_raw_vf",
    })
    
    print(attempts[["first_name_att", "last_name_att", "dob_raw_att", "zip_raw_att"]].head())
    print("Columns in attempts after rename:", attempts.columns.tolist())


    # Normalize names, DOBs, zips
    attempts, vf_extract = add_normalized_keys(attempts, vf_extract)

    # Generate candidate pairs blockingly
    candidates = generate_candidate_pairs(attempts, vf_extract)

    # Compute features for candidates
    X, _ = add_features(candidates)

    # Load trained model and predict
    model = joblib.load("logreg_model.pkl")
    candidates["match_prob"] = model.predict_proba(X)[:, 1]

    # Filter by threshold
    threshold = 0.8
    predicted_matches = candidates[candidates["match_prob"] >= threshold].copy()

    predicted_matches.to_csv("predicted_matches.csv", index=False)
    print(f"Predicted {len(predicted_matches)} matches above threshold {threshold}")

if __name__ == "__main__":
    main()