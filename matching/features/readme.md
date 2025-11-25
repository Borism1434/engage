# Features Module (`matching/features/`)

This module contains feature engineering utilities and tools essential for preparing data inputs for entity matching. The features capture various similarity metrics and normalized data representations used in model training and evaluation.

---

## Files and Functions

### `feature_builder.py`

- `add_features(df: pd.DataFrame) -> pd.DataFrame`  
  Computes similarity features like Jaro-Winkler scores on names and exact matches on DOB and ZIP.

- `compute_similarity(row: pd.Series) -> pd.Series`  
  Calculates individual similarity metrics for a single record pair.

### `address_features.py`

- `compute_address_similarity(addr1: str, addr2: str) -> float`  
  Measures similarity between two address strings using string matching algorithms.

### `date_features.py`

- `compute_partial_dob_match(dob1: datetime, dob2: datetime) -> Dict[str, bool]`  
  Creates features reflecting partial date of birth matches (year-only, month/day exact).

### `name_normalization.py`

- `normalize_name(name: str) -> str`  
  Normalizes accented characters and common variants in names.

---

_For further details on implementation and usage, refer to individual files._