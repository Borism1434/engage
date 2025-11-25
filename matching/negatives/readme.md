# Negatives Module (`matching/negatives/`)

This module generates negative samples for supervised entity matching training. It creates hard negative examples that closely resemble positive matches but are incorrect, improving the model’s ability to discriminate true matches from near misses.

---

## Files and Functions

### `hard_negatives.py`

- `build_vf_indexes(vf_df: pd.DataFrame) -> Tuple`  
  Builds fast lookup indexes keyed by ZIP code, date of birth, and first/last name initials from the voterfile DataFrame to efficiently query candidate negatives.

- `generate_hard_negatives(pos_df: pd.DataFrame, vf_df: pd.DataFrame) -> pd.DataFrame`  
  For each positive pair, generates multiple types of hard negatives: same ZIP but different DOB, same DOB but different last name, similar first and last initials, plus one random negative for generalization.

---

_See source files in this directory for implementation details and usage patterns._