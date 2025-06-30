from collections import Counter
from typing import List

def get_mode(numbers: List[int]) -> int:
    """
    Returns the mode (most frequent number) from a list of numbers.
    If there are multiple modes, it returns the first one encountered.
    """
    if not numbers:
        raise ValueError("Input list cannot be empty")

    counts = Counter(numbers)
    # most_common(1) returns a list of tuples, where each tuple contains (element, count)
    # We only want the element, so we take the first element of the first tuple [0][0]
    mode = counts.most_common(1)[0][0]
    return mode