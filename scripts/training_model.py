import os
import pandas as pd
import json
import joblib
from tqdm import tqdm
from script_helper import add_repo_root_to_syspath; add_repo_root_to_syspath()
from matching.loaders.load_attempts import load_attempts, load_matches
from matching.gold.gold_pairs import build_gold_pairs, add_normalized_keys
from matching.negatives.hard_negatives import generate_hard_negatives
from matching.features.feature_builder import add_features
from matching.modeling.train_eval import train_and_evaluate
import matching.utils.db as db

def run_query_with_progress(sql):
    sql = sql.strip().rstrip(';')
    count_sql = f"SELECT COUNT(*) FROM ({sql}) AS subquery"
    total_rows = db.run_query(count_sql).iloc[0, 0]
    print(f"Total rows to read: {total_rows}")

    print("Running full query, please wait...")
    df = db.run_query(sql)
    print(f"Query finished. Read {len(df)} rows.")
    return df

def main(voterfile_sql="SELECT * FROM voterfile.election_detail_2018 WHERE county = 'DAD'", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    matches = load_matches("/Users/borismartinez/Documents/GitHub/engage/data/vr_match_export.csv")
    attempts = load_attempts("/Users/borismartinez/Documents/GitHub/engage/data/vr_blocks_export_no_na.csv")
    pledges = pd.read_csv("/Users/borismartinez/Documents/GitHub/engage/data/pledge_data.csv")
    
    # Load voterfile dataframe with progress bar
    vf_df = run_query_with_progress(voterfile_sql)

    att = attempts.merge(
        matches[["registration_form_id", "type_code", "confidence_score"]],
        on="registration_form_id",
        how="inner",
    )

    att = att.rename(columns={
        "first_name": "first_name_att",
        "last_name": "last_name_att",
        "date_of_birth": "dob_raw_att",
        "voting_zipcode": "zip_raw_att",
    })

    vf_df = vf_df.rename(columns={
        "first_name": "first_name_vf",
        "last_name": "last_name_vf",
        "residence_zipcode": "zip_raw_vf",
        "birth_date": "dob_raw_vf",
    })

    att, vf_df = add_normalized_keys(att, vf_df)

    pos_df, vf_small, att_small = build_gold_pairs(att, vf_df)

    negatives = generate_hard_negatives(pos_df, vf_small)
    train_df = pd.concat([pos_df, negatives], ignore_index=True)
    X, y = add_features(train_df)
    model, metrics = train_and_evaluate(X, y)

    print("Model performance:")
    print(metrics)

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved as {model_path}")

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

    pos_df.to_parquet("pos_df_modular.csv", index=False)
    train_df.to_parquet("train_df_modular.csv", index=False)

    with open("metrics_modular.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nFeature importance (coefficients):")
    print(pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False))

if __name__ == "__main__":
    main()