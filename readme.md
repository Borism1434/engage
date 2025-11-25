# VR From Blocks Analysis

## Overview
This project aims to conduct a thorough audit and detailed analysis of voter registration (VR) data arising from blocks, using two main datasets: `vr_blocks_export` and `vr_match_export`.

- **vr_blocks_export** contains the recorded lifetime collection of voter registration blocks.
- **vr_match_export** consists of the subset of VR records successfully matched with the certified voter file by the State of Enrollment (SOE).

The match rate between these datasets is around 7%, and a core objective is to understand this rate, its drivers, and implications across various dimensions.

## Analysis Plan

1. **Overall Audit**
   - Assess row counts, uniqueness of key IDs (applicant, canvasser, turf), and basic schema validation.
   - Identify missing values and duplicate records in key fields.
   - Calculate and track the match rate between `vr_blocks_export` and `vr_match_export` over time and geography to identify trends and gaps.

2. **Time-Series Trends**
   - Visualize VR attempts and matches aggregated weekly and monthly.
   - Detect spikes around voter registration deadlines, special elections, and specific canvassing campaigns.

3. **Demographic Distributions**
   - Profile demographic characteristics (age bands, race/ethnicity, gender if available) for matched VRs.
   - Contrast with distributions in unmatched VRs to spot potential demographic biases or data quality concerns.
   - Note: Analytical caution is advised; no conclusions will be drawn at this stage.

4. **Geography and “Turf” Analysis**
   - Map registrations and matches by precinct, turf, or ZIP code.
   - Identify registration “hotspots” and “cold spots” relative to canvassing assignments.
   - Calculate match success rates per precinct/turf and across time intervals to evaluate canvassing effectiveness.

5. **Canvasser Performance**
   - Calculate total registrations, match rates, and average processing times per canvasser.
   - Perform comparative analyses across turfs and time periods to identify top performers and those needing support.

6. **Precinct-Level Voter Behavior**
   - Enrich matched VR data by linking turnout history (recent general, primary, special elections).
   - Analyze turnout by age, race, and precinct to explore how canvass-registered voters behave compared to the broader electorate.

7. **Application Pipeline Health**
   - Investigate reasons why some records in `vr_blocks_export` fail to appear in `vr_match_export` (e.g., missing signatures, address issues, duplicates).
   - Quantify failure reasons by canvasser and turf to target process improvements.

8. **Cross-Tabulations**
   - Explore compound relationships (e.g., age × precinct, race × canvasser, turf × turnout).
   - Use these insights to identify subgroups requiring targeted outreach or further investigation.

9. **Longitudinal Impact Analysis**
   - Track registration cohorts over time by campaign, canvasser, or precinct.
   - Assess persistence in registration and voting behavior across multiple election cycles.

## Contact Information
For questions related to these datasets and analysis, please contact the relevant teams:

- **vr_blocks_export** team — responsible for collect VRs lifetime dataset.
- **vr_match_export** team — responsible for matched VR rows aligned with the certified voter file by SOE.


## Next Steps and Improvements
To advance the matching model and analysis, the following practical development steps are planned:

1. **Refactor and add new feature modules:**
   - Develop `matching/features/address_features.py` containing functions like `compute_address_similarity(addr1, addr2)` to improve address matching.
   - Create `matching/features/date_features.py` to implement partial date-of-birth matching features (e.g., year-only, month/day separately).
   - Build `matching/features/name_normalization.py` to handle normalization of accented characters, Hispanic name variants, and common nicknames.

2. **Enhance the feature building pipeline:**
   - Update `matching/features/feature_builder.py` to import the new feature functions.
   - Extend the existing `compute_similarity()` to incorporate new features, and consider renaming it to `build_feature_vector()` for clarity and extensibility.

3. **Improve negative sample generation:**
   - Add additional negative generation functions focused on harder negative examples such as address similarities or partial DOB mismatches.
   - Modify `generate_hard_negatives()` (in `hard_negatives.py`) to integrate these new negative sampling strategies.

4. **Establish rigorous unit testing:**
   - Develop test cases in the `tests/` folder to validate correctness of new feature computations and negative generation methods.

5. **Iterate and evaluate model improvements:**
   - Perform training runs with the enriched feature set and augmented negatives.
   - Compare new model metrics against the version 1 baseline to quantify improvements and refine the model accordingly.

This structured approach will enable systematic enhancements to data quality assessment, feature richness, and matching accuracy, facilitating more reliable voter registration linkage and downstream analyses.
-- 