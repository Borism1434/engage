# matching/negatives/hard_negatives.py

import random
from collections import defaultdict
import pandas as pd

from matching.utils.normalization import safe_first_letter
from matching.utils.validators import require_columns
from matching.negatives.address_negatives import generate_address_negatives
from matching.negatives.phonetic_negatives import generate_phonetic_negatives


# Same ZIP, different DOB
# Same DOB, different last name
# Similar first and last initials
# One pure random negative
# Ideas to strengthen negative sampling:
# Address-based negatives: pairs with similar addresses but different names/DOB etc.
# Partial DOB mismatches: e.g., same birth year but different day/month
# Phonetic or nickname-based negatives: to produce harder false negatives
# More systematic sampling: cap negatives per positive, stratify by certain fields

def build_vf_indexes(vf_df: pd.DataFrame):
    """
    Create dicts keyed by zip, dob, and (fn0, ln0).
    Assumes vf_df already has fn_norm_vf, ln_norm_vf, zip_norm_vf, dob_norm_vf.
    """
    zip_index  = defaultdict(list)
    dob_index  = defaultdict(list)
    name_index = defaultdict(list)

    for _, r in vf_df.iterrows():
        zip_index[r["zip_norm_vf"]].append(r)
        dob_index[r["dob_norm_vf"]].append(r)

        fn0 = safe_first_letter(r["fn_norm_vf"])
        ln0 = safe_first_letter(r["ln_norm_vf"])
        if fn0 and ln0:
            name_index[(fn0, ln0)].append(r)

    return zip_index, dob_index, name_index

def generate_hard_negatives(pos_df: pd.DataFrame, vf_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each positive pair, generate several hard negatives:
      - same ZIP, different DOB
      - same DOB, different last name
      - similar first/last initial
      - one pure random
    """
    require_columns(
        pos_df,
        ["registration_form_id", "first_name_att", "last_name_att",
         "dob_norm_att", "zip_norm_att"],
        "pos_df",
    )

    require_columns(
        vf_df,
        ["first_name_vf", "last_name_vf", "fn_norm_vf", "ln_norm_vf",
         "dob_norm_vf", "zip_norm_vf"],
        "vf_df",
    )

    zip_index, dob_index, name_index = build_vf_indexes(vf_df)
    
    negatives = []
    for _, row in pos_df.iterrows():
        # A) same ZIP, different DOB
        zip_list = zip_index.get(row["zip_norm_att"], [])
        zip_matches = [r for r in zip_list if r["dob_norm_vf"] != row["dob_norm_att"]]

        # B) same DOB, different last name
        dob_list = dob_index.get(row["dob_norm_att"], [])
        dob_matches = [r for r in dob_list if r["last_name_vf"] != row["last_name_att"]]

        # C) similar name initials
        fn0 = safe_first_letter(row["first_name_att"])
        ln0 = safe_first_letter(row["last_name_att"])
        name_matches = name_index.get((fn0, ln0), []) if fn0 and ln0 else []

        candidate_groups = [zip_matches, dob_matches, name_matches]

        for group in candidate_groups:
            if group:
                wrong = random.choice(group)
                negatives.append({
                    "registration_form_id": row["registration_form_id"],
                    "first_name_att": row["first_name_att"],
                    "last_name_att":  row["last_name_att"],
                    "dob_norm_att":   row["dob_norm_att"],
                    "zip_norm_att":   row["zip_norm_att"],

                    "first_name_vf": wrong["first_name_vf"],
                    "last_name_vf":  wrong["last_name_vf"],
                    "dob_norm_vf":   wrong["dob_norm_vf"],
                    "zip_norm_vf":   wrong["zip_norm_vf"],
                    "is_match": 0,
                })

        # one pure random negative
        wrong = vf_df.sample(1).iloc[0]
        negatives.append({
            "registration_form_id": row["registration_form_id"],
            "first_name_att": row["first_name_att"],
            "last_name_att":  row["last_name_att"],
            "dob_norm_att":   row["dob_norm_att"],
            "zip_norm_att":   row["zip_norm_att"],

            "first_name_vf": wrong["first_name_vf"],
            "last_name_vf":  wrong["last_name_vf"],
            "dob_norm_vf":   wrong["dob_norm_vf"],
            "zip_norm_vf":   wrong["zip_norm_vf"],
            "is_match": 0,
        })
    
    # Now generate address-based negatives
    address_negatives = generate_address_negatives(pos_df, vf_df, threshold=0.95)

    phonetic_negatives = generate_phonetic_negatives(pos_df, vf_df)

    # Combine all negatives into one DataFrame
    all_negatives = pd.concat(
        [pd.DataFrame(negatives), address_negatives, phonetic_negatives],
        ignore_index=True
    )
    return all_negatives
