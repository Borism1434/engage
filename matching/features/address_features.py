# Features to add & where:
# You should expand and create new modular functions chiefly in matching/features.

# Address similarity
# Implement new similarity measure for address fields (e.g., Jaro-Winkler or token set ratios on address lines)


# matching/features/address_features.py
from rapidfuzz.distance import JaroWinkler

def compute_address_similarity(addr1: str, addr2: str) -> float:
    """
    Compute normalized similarity between two address strings using Jaro-Winkler.
    Handles None or empty addresses gracefully.
    """
    addr1 = addr1 or ""
    addr2 = addr2 or ""
    return JaroWinkler.normalized_similarity(addr1, addr2)