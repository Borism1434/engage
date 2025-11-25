# matching/config/match_config.py

MIAMI_DADE_CODE = "DAD"   # how it appears in voterfile.election_detail_2018

TRUE_MATCH_TYPES = [
    "First time registrant",
    "Status change",
    "In-state move cross-county",
    "In-state move in-county",
    "Cross-state move",
]

# features we use in the toy model
FEATURE_COLS = [
    "fn_jw",
    "ln_jw",
    "dob_exact",
    "zip_exact",
]