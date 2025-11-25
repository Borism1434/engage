# matching/loaders/load_voterfile.py

import pandas as pd
from matching.config.column_map import VOTERFILE_COL_MAP
from matching.config.match_config import MIAMI_DADE_CODE
from matching.utils.validators import require_columns
from sqlalchemy import create_engine
import matching.utils.db as db  # your existing helper

def load_voterfile_2018_miami() -> pd.DataFrame:
    # you already filtered county='DAD' in SQL; still useful in Python
    query = "SELECT * FROM voterfile.election_detail_2018 WHERE county = 'DAD';"
    vf = db.run_query(query)
    return vf

def standardize_voterfile_columns(vf_raw: pd.DataFrame) -> pd.DataFrame:
    cols_present = [c for c in VOTERFILE_COL_MAP.keys() if c in vf_raw.columns]
    vf_small = vf_raw[cols_present].copy()
    vf_small.rename(columns=VOTERFILE_COL_MAP, inplace=True)
    return vf_small

def load_voterfile_chunk(engine, sql_query, chunksize=100000):
    """Yield chunks of voterfile data from database."""
    # engine: SQLAlchemy engine
    yield from pd.read_sql_query(sql_query, engine, chunksize=chunksize)