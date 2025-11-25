import pandas as pd
from script_helper import add_repo_root_to_syspath; add_repo_root_to_syspath()
from matching.loaders.load_attempts import load_attempts, load_matches # adapt as needed
from matching.gold.gold_pairs import build_gold_pairs, add_normalized_keys
from matching.negatives.hard_negatives import generate_hard_negatives
from matching.features.feature_builder import add_features
from matching.modeling.train_eval import train_and_evaluate
import matching.utils.db as db
import json
import joblib

def main():
    # 1. Load data (using your loader, DB module, or pandas if not yet modularized)
    matches = load_matches("/Users/borismartinez/Documents/GitHub/engage/data/vr_match_export.csv")
    attempts = load_attempts("/Users/borismartinez/Documents/GitHub/engage/data/vr_blocks_export.csv")
    
    pledges = pd.read_csv("/Users/borismartinez/Documents/GitHub/engage/data/pledge_data.csv")

    vf_2018 = db.run_query("SELECT * FROM voterfile.election_detail_2018 WHERE county = 'DAD';")

    att = attempts.merge(
    matches[["registration_form_id", "type_code", "confidence_score"]],
    on="registration_form_id",
    how="inner",
)

    # Rename columns in attempts and voterfile for normalization
    att = att.rename(columns={
        "first_name": "first_name_att",
        "last_name": "last_name_att",
        "date_of_birth": "dob_raw_att",
        "voting_zipcode": "zip_raw_att",
    })

    vf_2018 = vf_2018.rename(columns={
        "first_name": "first_name_vf",
        "last_name": "last_name_vf",
        "residence_zipcode": "zip_raw_vf",
        "birth_date": "dob_raw_vf",
    })

    # Normalize only attempts and vf_2018
    att, vf_2018 = add_normalized_keys(att, vf_2018)

    # Build gold pairs using normalized attempts and voterfile
    pos_df, vf_small, att_small = build_gold_pairs(att, vf_2018)

    # Continue with negatives, feature building, training etc.
    negatives = generate_hard_negatives(pos_df, vf_small)
    train_df = pd.concat([pos_df, negatives], ignore_index=True)
    X, y = add_features(train_df)
    model, metrics = train_and_evaluate(X, y)

    print("Model performance:")
    print(metrics)

    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")

    print("\n=== POSITIVE PAIRS DATA ===")
    print(f"Columns: {pos_df.columns.tolist()}")
    print(f"Shape: {pos_df.shape}")
    print(pos_df["is_match"].value_counts())

    print("\n=== TRAINING DATA ===")
    print(f"Shape: {train_df.shape}")
    print(train_df["is_match"].value_counts())

    print("\n=== FEATURES and TARGET ===")
    print(f"Features: {list(X.columns)}")
    print(f"Feature shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    pos_df.to_csv("pos_df_modular.csv", index=False)
    train_df.to_csv("train_df_modular.csv", index=False)

    
    with open("metrics_modular.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nFeature importance (coefficients):")
    print(pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False))

if __name__ == "__main__":
    main()