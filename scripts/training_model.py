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
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
import numpy as np

def train_logreg(X, y):
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    return model

def train_xgboost(X, y):
    model = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
    )
    model.fit(X, y)
    return model

def train_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    return model

def run_query_with_progress(sql):
    sql = sql.strip().rstrip(';')
    count_sql = f"SELECT COUNT(*) FROM ({sql}) AS subquery"
    total_rows = db.run_query(count_sql).iloc[0, 0]
    print(f"Total rows to read: {total_rows}")

    print("Running full query, please wait...")
    df = db.run_query(sql)
    print(f"Query finished. Read {len(df)} rows.")
    return df

def main(voterfile_sql="SELECT * FROM voterfile.election_detail_2024 WHERE county = 'DAD'",
         model_dir="models",
         model_choice="all",
         test_size=0.2,
         random_state=42):
    os.makedirs(model_dir, exist_ok=True)
    matches = load_matches("/Users/borismartinez/Documents/GitHub/engage/data/vr_match_export.csv")
    attempts = load_attempts("/Users/borismartinez/Documents/GitHub/engage/data/vr_blocks_export_no_na.csv")
    pledges = pd.read_csv("/Users/borismartinez/Documents/GitHub/engage/data/pledge_data.csv")

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

    # Train-test split for defensible evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models_to_run = []
    if model_choice == "all":
        models_to_run = ["logreg", "xgboost", "random_forest"]
    else:
        models_to_run = [model_choice]

    for m in models_to_run:
        print(f"\nTraining model: {m}")
        if m == "logreg":
            model = train_logreg(X_train, y_train)
        elif m == "xgboost":
            model = train_xgboost(X_train, y_train)
        elif m == "random_forest":
            model = train_random_forest(X_train, y_train)
        else:
            print(f"Unsupported model choice: {m}")
            continue

        # Predict on test set
        y_pred = model.predict(X_test)

        # y_proba (for AUC), check if supported
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            y_proba = scores
        else:
            y_proba = None

        class_report = classification_report(y_test, y_pred, output_dict=True)

        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = None

        metrics = {
            "classification_report": class_report,
            "auc": auc,
        }

        # Extract and add feature importance/coefs
        if hasattr(model, 'coef_'):
            coef_series = pd.Series(model.coef_[0], index=X.columns)
            metrics['feature_importance'] = coef_series.sort_values(ascending=False).to_dict()
        elif hasattr(model, 'feature_importances_'):
            fi_series = pd.Series(model.feature_importances_, index=X.columns)
            metrics['feature_importance'] = fi_series.sort_values(ascending=False).to_dict()
        else:
            metrics['feature_importance'] = {}

        # Add training and test data summary and other metadata
        metrics["training_data_summary"] = {
            "train_num_samples": len(y_train),
            "train_num_positive": int(np.sum(y_train == 1)),
            "train_num_negative": int(np.sum(y_train == 0)),
            "test_num_samples": len(y_test),
            "test_num_positive": int(np.sum(y_test == 1)),
            "test_num_negative": int(np.sum(y_test == 0)),
        }

        metrics["python_version"] = sys.version

        print("Model performance on TEST set:")
        print(classification_report(y_test, y_pred))
        if auc is not None:
            print(f"AUC: {auc:.6f}")

        print("Metrics dictionary:")
        print(metrics)

        model_path = os.path.join(model_dir, f"{m}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved as {model_path}")

        metrics_path = os.path.join(model_dir, f"{m}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved as {metrics_path}")

    pos_df.to_parquet("pos_df_modular.parquet", index=False)
    train_df.to_parquet("train_df_modular.parquet", index=False)
    print("Saved positive pairs and training datasets.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "xgboost", "random_forest", "all"], default="all",
                        help="Specify which model(s) to train and evaluate")
    args = parser.parse_args()

    main(model_choice=args.model)