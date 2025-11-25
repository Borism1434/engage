# matching/utils/normalization.py

import re
import unicodedata
import pandas as pd

def normalize_text_series(s):
    return s.apply(
        lambda x: unicodedata.normalize("NFKD", x)
                  .encode("ascii", "ignore")
                  .decode("utf-8")
                  if pd.notna(x) else x
    )

def normalize_zip_series(z):
    z = z.astype("string")
    z = z.str.replace(r"\D", "", regex=True)
    return z.str.slice(0, 5)

def normalize_dob_series(d):
    return pd.to_datetime(d, errors="coerce")

def safe_first_letter(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    return x[0] if x else None