# Gold Module (`matching/gold/`)

This module manages the creation and handling of high-confidence (“gold”) positive pairs used for supervised entity matching. It focuses on reliably identifying and preparing ground-truth matches between registration attempts and voterfile records.

---

## Files and Functions

### `gold_pairs.py`

- `add_normalized_keys(att_df: pd.DataFrame, vf_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]`  
  Adds normalized key columns (e.g., normalized first and last names, ZIP codes, and dates of birth) to both attempts and voterfile DataFrames to standardize values for matching.

- `build_gold_pairs(att_df: pd.DataFrame, vf_df: pd.DataFrame) -> pd.DataFrame`  
  Generates a set of high-confidence (“gold”) positive pairs by strictly joining normalized attempts and voterfile datasets on key identifiers. It filters to retain only uniquely matched pairs and labels them based on predefined true match types, returning a minimal subset for training matching models.

---

_For detailed implementations and usage examples, see the source files in this directory._