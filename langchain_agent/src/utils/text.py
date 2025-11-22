def preview_text(text: str, max_len: int = 600) -> str:
    """Truncate text for logging, showing start and end for long content."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    half = (max_len - 5) // 2
    return f"{text[:half]} ... {text[-half:]}"
