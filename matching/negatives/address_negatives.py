from matching.features.address_features import compute_address_similarity
import pandas as pd


def generate_address_negatives(pos_df, vf_df, threshold=0.8):
    """
    Generate negatives where address similarity is above a threshold,
    but names or DOB differ enough to qualify as negatives.
    """
    negatives = []
    for _, row in pos_df.iterrows():
        vf_candidates = vf_df  # You could filter candidates by ZIP or region for efficiency

        for _, vf_row in vf_candidates.iterrows():
            addr_sim = compute_address_similarity(
                row.get("voting_street_address_one", ""),
                vf_row.get("residence_address_1", "")
            )
            # Basic filters - high address similarity but name or DOB mismatch
            if addr_sim >= threshold:
                # Check name or dob differences (you can expand this)
                if row["first_name_att"] != vf_row["first_name_vf"] or row["dob_norm_att"] != vf_row["dob_norm_vf"]:
                    negatives.append({
                        "registration_form_id": row["registration_form_id"],
                        "first_name_att": row["first_name_att"],
                        "last_name_att": row["last_name_att"],
                        "dob_norm_att": row["dob_norm_att"],
                        "zip_norm_att": row["zip_norm_att"],
                        "first_name_vf": vf_row["first_name_vf"],
                        "last_name_vf": vf_row["last_name_vf"],
                        "dob_norm_vf": vf_row["dob_norm_vf"],
                        "zip_norm_vf": vf_row["zip_norm_vf"],
                        "is_match": 0,
                    })
                    # Optional: limit negatives per positive here
                    if len(negatives) > 5:  # for example max 5 per positive
                        break
        if len(negatives) > 5000:  # global cap, adjust as needed
            break
    return pd.DataFrame(negatives)