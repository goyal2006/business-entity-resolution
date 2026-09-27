from difflib import SequenceMatcher


def sequence_similarity(a, b):
    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def token_set_similarity(a, b):
    if not a or not b:
        return 0.0

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    return intersection / union


def token_overlap(a, b):
    if not a or not b:
        return 0.0

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)

    return intersection / min(len(tokens_a), len(tokens_b))


def character_similarity(a, b):
    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a.replace(" ", ""), b.replace(" ", "")).ratio()