import re


def clean_generated_text(text: str) -> str:
    """
    Cleans repetitive or redundant phrases from generated AI text.
    Example:
    - "Today's reflection: I felt exhausted..." -> "I felt exhausted..."
    - "I feel the familiar sense of calm again today." -> "Sense of calm again today."
    """
    if not text:
        return text

    # Remove redundant starts (handle both ' and ’)
    patterns = [
        r"^\s*(Today([’']s)?\s+reflection[:,]?\s*)",  # <-- fixed here
        r"^\s*(Today[:,]?\s*)",
        r"^\s*(I\s+feel\s+the\s+familiar\s+)",
        r"^\s*(I\s+felt\s+the\s+familiar\s+)",
    ]

    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.strip()

    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned
