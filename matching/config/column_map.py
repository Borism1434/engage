# matching/config/column_map.py

ATTEMPT_COL_MAP = {
    # labels from match result
    "type_code": "type_code",
    "confidence_score": "confidence_score",

    # identity
    "first_name": "first_name_att",
    "middle_name": "middle_name_att",
    "last_name": "last_name_att",
    "name_suffix": "suffix_att",
    "name_prefix": "prefix_att",
    "registration_form_id": "registration_form_id",

    # core voting address
    "voting_street_address_one": "voting_address_1_att",
    "voting_street_address_two": "voting_address_2_att",
    "voting_city": "voting_city_att",
    "voting_state": "voting_state_att",
    "voting_zipcode": "zip_raw_att",

    # mailing address
    "mailing_street_address_one": "mailing_address_1_att",
    "mailing_street_address_two": "mailing_address_2_att",
    "mailing_city": "mailing_city_att",
    "mailing_zipcode": "mailing_zip_att",

    # demographics
    "gender": "gender_att",
    "date_of_birth": "dob_raw_att",
    "party": "party_att",
    "ethnicity": "ethnicity_att",

    # contact
    "phone_number": "phone_att",
    "email_address": "email_att",

    # geospatial hints
    "data_entry_county": "data_entry_county",
    "program_state": "program_state",
    "collection_location_county": "collection_location_county",
    "collection_location_city": "collection_location_city",
    "collection_location_zip": "collection_location_zip",
    "voting_address_latitude": "voting_lat",
    "voting_address_longitude": "voting_lng",

    # quality flags
    "address_validated": "address_validated",
    "eligible_voting_age": "eligible_voting_age",
}

VOTERFILE_COL_MAP = {
    # identity
    "voter_id": "voter_id",
    "first_name": "first_name_vf",
    "middle_name": "middle_name_vf",
    "last_name": "last_name_vf",
    "suffix_name": "suffix_vf",

    # core residence address
    "residence_address_1": "res_address_1_vf",
    "residence_address_2": "res_address_2_vf",
    "residence_city_usps": "res_city_vf",
    "residence_state": "res_state_vf",
    "residence_zipcode": "zip_raw_vf",

    # mailing address
    "mailing_address_1": "mailing_address_1_vf",
    "mailing_address_2": "mailing_address_2_vf",
    "mailing_city": "mailing_city_vf",
    "mailing_state": "mailing_state_vf",
    "mailing_zipcode": "mailing_zip_vf",

    # demographics/registration
    "gender": "gender_vf",
    "race": "race_vf",
    "birth_date": "dob_raw_vf",
    "reg_date": "reg_date_vf",
    "party": "party_vf",

    # geography/districts
    "county": "county",
    "precinct": "precinct",
    "precinct_group": "precinct_group",
    "precinct_split": "precinct_split",
    "precinct_suffix": "precinct_suffix",
    "congressional_district": "cd",
    "house_district": "hd",
    "senate_district": "sd",
    "county_commission_district": "ccd",
    "school_board_district": "sbd",

    # contact
    "daytime_area_code": "daytime_area_code",
    "daytime_phone_num": "daytime_phone_num",
    "email": "email_vf",

    # status
    "voter_status": "voter_status",
}