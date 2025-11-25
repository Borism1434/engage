# DOB partial matching
# Add boolean or similarity features comparing year, month, day separately


import pandas as pd
from rapidfuzz.distance import Levenshtein

def safe_str_date(d):
    """Convert date to a normalized string YYYYMMDD or empty string if NaT/NA."""
    if pd.isna(d):
        return ""
    return d.strftime("%Y%m%d")

def dob_exact_match(dob1, dob2):
    """Return 1 if dates exactly match, else 0."""
    if pd.isna(dob1) or pd.isna(dob2):
        return 0
    return int(dob1 == dob2)

def dob_year_match(dob1, dob2):
    if pd.isna(dob1) or pd.isna(dob2):
        return 0
    return int(dob1.year == dob2.year)

def dob_month_match(dob1, dob2):
    if pd.isna(dob1) or pd.isna(dob2):
        return 0
    return int(dob1.month == dob2.month)

def dob_day_match(dob1, dob2):
    if pd.isna(dob1) or pd.isna(dob2):
        return 0
    return int(dob1.day == dob2.day)

def dob_levenshtein_similarity(dob1, dob2):
    """Compute normalized Levenshtein similarity between DOB strings."""
    s1 = safe_str_date(dob1)
    s2 = safe_str_date(dob2)
    if not s1 or not s2:
        return 0.0
    # rapidfuzz distance returns similarity in [0,100]
    sim = Levenshtein.normalized_similarity(s1, s2) / 100.0
    return sim

def dob_month_day_swapped(dob1, dob2):
    """Check if month and day are swapped but year matches exactly."""
    if pd.isna(dob1) or pd.isna(dob2):
        return 0
    if dob1.year != dob2.year:
        return 0
    return int(dob1.month == dob2.day and dob1.day == dob2.month)