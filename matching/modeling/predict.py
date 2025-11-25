import pandas as pd
from matching.loaders.load_voterfile import load_voterfile_chunk
from matching.gold.gold_pairs import add_normalized_keys  # full path to function
from matching.candidates.block_candidates import block_candidate_pairs
from matching.features.feature_builder import add_features
import joblib  # for model loading


def predict_chunk(model, features_df):
    """
    Predict match probabilities for the given features using the trained model.
    Parameters:
        model: Trained sklearn-like model.
        features_df: DataFrame of feature values to predict on.
    Returns:
        numpy array of predicted probabilities for the positive class.
    """

    print("Starting inference stream...")

def predict_chunk(model, features_df):
    return model.predict_proba(features_df)[:, 1]