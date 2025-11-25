# Name normalization variants
# Implement logic to normalize Hispanic accented characters (José → Jose etc.)
# Use mapping dictionaries for common nicknames (William → Will, Bill)
# Consider maiden vs married surname variations, aliasesimport unicodedata
import unicodedata


def remove_accents(name: str) -> str:
    if not isinstance(name, str):
        return name
    nfkd_form = unicodedata.normalize('NFKD', name)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


NICKNAME_MAP = {
    "bill": "william",
    "will": "william",
    "billy": "william",
    "liz": "elizabeth",
    "beth": "elizabeth",
    "lizzy": "elizabeth",
    "joe": "joseph",
    "pepe": "joseph",
    # Add more as needed
}

def map_nickname(name: str) -> str:
    if not isinstance(name, str):
        return name
    name_lower = name.lower()
    return NICKNAME_MAP.get(name_lower, name_lower)



MAIDEN_MARRIED_MAP = {
    "smith-jones": "smith",
    "jones-smith": "smith",
    # You can extend or use fuzzy matching here
}

def normalize_surname(name: str) -> str:
    if not isinstance(name, str):
        return name
    name_lower = name.lower()
    return MAIDEN_MARRIED_MAP.get(name_lower, name_lower)



def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = remove_accents(name)
    name = map_nickname(name)
    name = normalize_surname(name)
    return name
    


import pandas as pd

def normalize_name_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().apply(normalize_name)