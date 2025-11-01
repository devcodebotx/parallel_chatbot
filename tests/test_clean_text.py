import pytest
from utils.clean_generated_text import clean_generated_text


def test_clean_generated_text():
    samples = {
        "Today's reflection: I felt exhausted after work.": "I felt exhausted after work.",
        "Today’s reflection I realized something new.": "I realized something new.",
        "Today: I found peace in silence.": "I found peace in silence.",
        "I feel the familiar sense of calm again today.": "Sense of calm again today.",
        "I felt the familiar ache of longing return.": "Ache of longing return.",
        "Nothing redundant here.": "Nothing redundant here.",
        "   Today’s reflection:   I learned patience.  ": "I learned patience."
    }

    for raw, expected in samples.items():
        cleaned = clean_generated_text(raw)
        assert cleaned == expected, f"Failed for input: {raw}"
        print(f"Input: {raw} | Cleaned: {cleaned}")
