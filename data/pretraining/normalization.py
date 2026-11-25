"""Normalization used for the original corpus deduplication and overlap guard."""

import unicodedata


def normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
