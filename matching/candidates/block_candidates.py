def block_candidate_pairs(att_df, vf_chunk):
    """Block by last name initial and DOB year as example."""
    att_df["ln0"] = att_df["last_name_att"].str[0].str.lower()
    att_df["dob_year"] = att_df["dob_norm_att"].dt.year
    
    vf_chunk["ln0"] = vf_chunk["last_name_vf"].str[0].str.lower()
    vf_chunk["dob_year"] = vf_chunk["dob_norm_vf"].dt.year
    
    return att_df.merge(
        vf_chunk,
        how="inner",
        left_on=["ln0", "dob_year"],
        right_on=["ln0", "dob_year"],
        suffixes=('_att', '_vf'),
    )