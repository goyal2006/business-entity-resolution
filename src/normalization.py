import re
import unicodedata


LEGAL_SUFFIXES = {
    "incorporated": "inc",
    "corporation": "corp",
    "company": "co",
    "limited": "ltd",
    "private": "pvt",
    "llc": "llc",
    "l.l.c": "llc",
    "ltd.": "ltd",
    "inc.": "inc",
    "corp.": "corp",
    "co.": "co",
}


ADDRESS_REPLACEMENTS = {
    "street": "st",
    "st.": "st",
    "road": "rd",
    "rd.": "rd",
    "avenue": "ave",
    "ave.": "ave",
    "drive": "dr",
    "dr.": "dr",
    "boulevard": "blvd",
    "blvd.": "blvd",
    "highway": "hwy",
    "hwy.": "hwy",
    "lane": "ln",
    "ln.": "ln",
    "parkway": "pkwy",
    "place": "pl",
    "court": "ct",
}


def unicode_normalize(text):
    text = unicodedata.normalize("NFKC", text)
    return text


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    text = unicode_normalize(text)
    text = text.lower()

    text = text.replace("&", " and ")

    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_business_name(text):
    text = normalize_text(text)

    if not text:
        return ""

    tokens = text.split()

    normalized_tokens = []

    for token in tokens:
        normalized_tokens.append(
            LEGAL_SUFFIXES.get(token, token)
        )

    return " ".join(normalized_tokens)


def normalize_business_address(text):
    text = normalize_text(text)

    if not text:
        return ""

    tokens = text.split()

    normalized_tokens = []

    for token in tokens:
        normalized_tokens.append(
            ADDRESS_REPLACEMENTS.get(token, token)
        )

    return " ".join(normalized_tokens)