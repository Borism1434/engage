import random
import fuzzy
from collections import defaultdict
import pandas as pd





soundex = fuzzy.Soundex(4)

def phonetic_code(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    return soundex(name.lower())



def build_phonetic_indexes(vf_df: pd.DataFrame) -> dict:
    last_name_index = defaultdict(list)
    first_name_index = defaultdict(list)
    for _, row in vf_df.iterrows():
        ln_code = phonetic_code(row["last_name_vf"])
        fn_code = phonetic_code(row["first_name_vf"])
        last_name_index[ln_code].append(row)
        first_name_index[fn_code].append(row)
    return first_name_index, last_name_index



def generate_phonetic_negatives(pos_df: pd.DataFrame, vf_df: pd.DataFrame) -> pd.DataFrame:
    first_name_index, last_name_index = build_phonetic_indexes(vf_df)
    negatives = []
    for _, row in pos_df.iterrows():
        fn_code = phonetic_code(row["first_name_att"])
        ln_code = phonetic_code(row["last_name_att"])

        # Phonetic candidates with same last name sound, different DOB
        candidates = [
            r for r in last_name_index.get(ln_code, [])
            if r["dob_norm_vf"] != row["dob_norm_att"] and r["registration_form_id"] != row["registration_form_id"]
        ]
        if candidates:
            wrong = random.choice(candidates)
            negatives.append({
                "registration_form_id": row["registration_form_id"],
                "first_name_att": row["first_name_att"],
                "last_name_att": row["last_name_att"],
                "dob_norm_att": row["dob_norm_att"],
                "zip_norm_att": row["zip_norm_att"],
                "first_name_vf": wrong["first_name_vf"],
                "last_name_vf": wrong["last_name_vf"],
                "dob_norm_vf": wrong["dob_norm_vf"],
                "zip_norm_vf": wrong["zip_norm_vf"],
                "is_match": 0,
            })
        # You can add more logic for first name phonetic matches similarly

    return pd.DataFrame(negatives)
