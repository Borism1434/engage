# matching/pipeline.py

import pandas as pd

from matching.loaders.load_attempts import (
    load_matches,
    load_attempts,
    attach_match_labels,
    standardize_attempt_columns,
)
from matching.loaders.load_voterfile import (
    load_voterfile_2018_miami,
    standardize_voterfile_columns,
)
from matching.gold.gold_pairs import add_normalized_keys, build_gold_pairs
from matching.negatives.hard_negatives import generate_hard_negatives
from matching.features.feature_builder import add_features
from matching.modeling.train_eval import train_and_evaluate


MATCHES_PATH  = "/Users/borismartinez/Documents/GitHub/engage/data/vr_match_export.csv"
ATTEMPTS_PATH = "/Users/borismartinez/Documents/GitHub/engage/data/vr_blocks_export.csv"


def main():
    # Load CSVs
    matches = load_matches(MATCHES_PATH)
    attempts = load_attempts(ATTEMPTS_PATH)

    # Attach labels to attempts
    att_labeled = attach_match_labels(attempts, matches)

    # Standardize column names
    att_small = standardize_attempt_columns(att_labeled)
    vf_raw    = load_voterfile_2018_miami()
    vf_small  = standardize_voterfile_columns(vf_raw)

    # Add normalized keys (fn/ln/dob/zip)
    att_norm, vf_norm = add_normalized_keys(att_small, vf_small)

    # Build high-quality positive pairs
    pos_df = build_gold_pairs(att_norm, vf_norm)
    print("Positives:", len(pos_df))

    # Generate hard negatives
    neg_df = generate_hard_negatives(pos_df, vf_norm)
    print("Negatives:", len(neg_df))

    # Combine
    train_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # Add similarity features
    train_df = add_features(train_df)

    # Train + evaluate model
    model = train_and_evaluate(train_df)

    return model, train_df


if __name__ == "__main__":
    main()