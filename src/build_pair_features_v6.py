from pathlib import Path
from collections import Counter
import re
import unicodedata

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

STUDENT_RESOURCE = ROOT.parent / "student_resource"

PAIR_FILE = ROOT.parent / "output" / "pair_training_data_v5.csv"

SOURCE1_FILE = (
    STUDENT_RESOURCE /
    "dataset" /
    "train" /
    "train_source1.tsv"
)

SOURCE2_FILE = (
    STUDENT_RESOURCE /
    "dataset" /
    "train" /
    "train_source2.tsv"
)

SOURCE3_FILE = (
    STUDENT_RESOURCE /
    "dataset" /
    "train" /
    "train_source3.tsv"
)

OUTPUT_FILE = (
    ROOT.parent /
    "output" /
    "pair_training_data_v6.csv"
)

CHUNK_SIZE = 250000


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
    "hwy": "hwy",
    "lane": "ln",
    "ln.": "ln",
    "parkway": "pkwy",
    "place": "pl",
    "court": "ct",
}


def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).strip()

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    text = text.replace(
        "&",
        " and "
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def normalize_business_name(text):

    text = normalize_text(text)

    if not text:
        return ""

    tokens = text.split()

    normalized_tokens = []

    for token in tokens:

        normalized_tokens.append(
            LEGAL_SUFFIXES.get(
                token,
                token
            )
        )

    return " ".join(
        normalized_tokens
    )


def normalize_business_address(text):

    text = normalize_text(text)

    if not text:
        return ""

    tokens = text.split()

    normalized_tokens = []

    for token in tokens:

        normalized_tokens.append(
            ADDRESS_REPLACEMENTS.get(
                token,
                token
            )
        )

    return " ".join(
        normalized_tokens
    )


def extract_numeric_tokens(address):

    if not address:
        return []

    tokens = re.findall(
        r"\d+",
        address
    )

    normalized = []

    for token in tokens:

        stripped = token.lstrip("0")

        if stripped == "":
            stripped = "0"

        normalized.append(
            stripped
        )

    return normalized


def extract_alpha_numeric_components(address):

    if not address:
        return []

    tokens = address.split()

    components = []

    for token in tokens:

        if any(
            char.isdigit()
            for char in token
        ):

            cleaned = re.sub(
                r"[^a-z0-9]",
                "",
                token
            )

            if cleaned:
                components.append(
                    cleaned
                )

    return components


def token_set(text):

    if not text:
        return set()

    return set(
        text.split()
    )


def safe_ratio(a, b):

    if b == 0:
        return 0.0

    return a / b


def dominant_script(text):

    if not text:
        return "none"

    counts = {
        "latin": 0,
        "devanagari": 0,
        "bengali": 0,
        "telugu": 0,
        "tamil": 0,
        "malayalam": 0,
        "other": 0,
    }

    for char in text:

        code = ord(char)

        if (
            0x0041 <= code <= 0x007A
            or
            0x00C0 <= code <= 0x024F
        ):

            counts["latin"] += 1

        elif 0x0900 <= code <= 0x097F:

            counts["devanagari"] += 1

        elif 0x0980 <= code <= 0x09FF:

            counts["bengali"] += 1

        elif 0x0C00 <= code <= 0x0C7F:

            counts["telugu"] += 1

        elif 0x0B80 <= code <= 0x0BFF:

            counts["tamil"] += 1

        elif 0x0D00 <= code <= 0x0D7F:

            counts["malayalam"] += 1

        elif char.isalpha():

            counts["other"] += 1

    return max(
        counts,
        key=counts.get
    )


print("Loading V5 pair data...")

pairs = pd.read_csv(
    PAIR_FILE
)

print(
    f"Pair rows: {len(pairs)}"
)

print(
    f"S1 entities: "
    f"{pairs['s1_entity_id'].nunique()}"
)


required_s1_ids = set(
    pairs["s1_entity_id"]
)

required_candidate_ids = set(
    pairs["candidate_entity_id"]
)


print()
print(
    "Required S1 records:",
    len(required_s1_ids)
)

print(
    "Required candidate records:",
    len(required_candidate_ids)
)


records = {}


def load_required_records(
    filename,
    source_name
):

    print()
    print(
        f"Loading {source_name}..."
    )

    found = 0

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = row.entity_id

            if (
                entity_id in required_s1_ids
                or
                entity_id in required_candidate_ids
            ):

                name = (
                    ""
                    if pd.isna(row.business_name)
                    else str(row.business_name)
                )

                address = (
                    ""
                    if pd.isna(row.business_address)
                    else str(row.business_address)
                )

                country = (
                    ""
                    if pd.isna(row.country)
                    else str(row.country)
                )

                records[entity_id] = {
                    "raw_name": name,
                    "raw_address": address,
                    "country": country,
                    "name": normalize_business_name(
                        name
                    ),
                    "address": normalize_business_address(
                        address
                    ),
                }

                found += 1

        print(
            f"  loaded relevant records: {found}"
        )

    print(
        f"{source_name} relevant records found: {found}"
    )


load_required_records(
    SOURCE1_FILE,
    "Source 1"
)

load_required_records(
    SOURCE2_FILE,
    "Source 2"
)

load_required_records(
    SOURCE3_FILE,
    "Source 3"
)


print()
print(
    f"Total records retained: "
    f"{len(records)}"
)


missing_records = []

for entity_id in (
    required_s1_ids |
    required_candidate_ids
):

    if entity_id not in records:

        missing_records.append(
            entity_id
        )


if missing_records:

    print()
    print(
        "WARNING: missing records:",
        len(missing_records)
    )

    print(
        missing_records[:20]
    )


print()
print(
    "Building global normalized-name frequencies..."
)


name_frequency = Counter()
address_frequency = Counter()


def build_global_frequency(
    filename,
    source_name
):

    print(
        f"Scanning {source_name}..."
    )

    rows = 0

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        dtype=str,
        usecols=[
            "business_name",
            "business_address"
        ],
        chunksize=CHUNK_SIZE
    ):

        names = (
            chunk["business_name"]
            .fillna("")
            .map(
                normalize_business_name
            )
        )

        addresses = (
            chunk["business_address"]
            .fillna("")
            .map(
                normalize_business_address
            )
        )

        for name in names:

            if name:

                name_frequency[name] += 1

        for address in addresses:

            if address:

                address_frequency[address] += 1

        rows += len(chunk)

        print(
            f"  scanned: {rows}"
        )


build_global_frequency(
    SOURCE2_FILE,
    "Source 2"
)

build_global_frequency(
    SOURCE3_FILE,
    "Source 3"
)


print()
print(
    f"Unique normalized names: "
    f"{len(name_frequency)}"
)

print(
    f"Unique normalized addresses: "
    f"{len(address_frequency)}"
)


print()
print(
    "Generating V6 features..."
)


new_rows = []


for position, row in enumerate(
    pairs.itertuples(
        index=False
    ),
    start=1
):

    s1_id = row.s1_entity_id

    candidate_id = row.candidate_entity_id


    s1 = records.get(
        s1_id,
        {
            "raw_name": "",
            "raw_address": "",
            "country": "",
            "name": "",
            "address": ""
        }
    )


    candidate = records.get(
        candidate_id,
        {
            "raw_name": "",
            "raw_address": "",
            "country": "",
            "name": "",
            "address": ""
        }
    )


    s1_name = s1["name"]

    candidate_name = candidate["name"]

    s1_address = s1["address"]

    candidate_address = candidate["address"]


    s1_name_tokens = token_set(
        s1_name
    )

    candidate_name_tokens = token_set(
        candidate_name
    )


    s1_address_tokens = token_set(
        s1_address
    )

    candidate_address_tokens = token_set(
        candidate_address
    )


    s1_numbers = extract_numeric_tokens(
        s1_address
    )

    candidate_numbers = extract_numeric_tokens(
        candidate_address
    )


    s1_number_set = set(
        s1_numbers
    )

    candidate_number_set = set(
        candidate_numbers
    )


    shared_numbers = (
        s1_number_set &
        candidate_number_set
    )


    unmatched_s1_numbers = (
        s1_number_set -
        candidate_number_set
    )


    unmatched_candidate_numbers = (
        candidate_number_set -
        s1_number_set
    )


    numeric_union = (
        s1_number_set |
        candidate_number_set
    )


    s1_components = (
        extract_alpha_numeric_components(
            s1_address
        )
    )


    candidate_components = (
        extract_alpha_numeric_components(
            candidate_address
        )
    )


    component_set_s1 = set(
        s1_components
    )


    component_set_candidate = set(
        candidate_components
    )


    shared_components = (
        component_set_s1 &
        component_set_candidate
    )


    s1_script = dominant_script(
        s1["raw_name"]
    )

    candidate_script = dominant_script(
        candidate["raw_name"]
    )


    same_script = int(
        s1_script == candidate_script
    )


    exact_name = int(
        bool(s1_name) and
        s1_name == candidate_name
    )


    exact_address = int(
        bool(s1_address) and
        s1_address == candidate_address
    )


    name_frequency_count = (
        name_frequency.get(
            s1_name,
            0
        )
        if s1_name
        else 0
    )


    candidate_name_frequency_count = (
        name_frequency.get(
            candidate_name,
            0
        )
        if candidate_name
        else 0
    )


    address_frequency_count = (
        address_frequency.get(
            s1_address,
            0
        )
        if s1_address
        else 0
    )


    candidate_address_frequency_count = (
        address_frequency.get(
            candidate_address,
            0
        )
        if candidate_address
        else 0
    )


    numeric_shared_count = len(
        shared_numbers
    )


    numeric_unmatched_s1_count = len(
        unmatched_s1_numbers
    )


    numeric_unmatched_candidate_count = len(
        unmatched_candidate_numbers
    )


    numeric_overlap_ratio = safe_ratio(
        numeric_shared_count,
        len(numeric_union)
    )


    numeric_s1_coverage = safe_ratio(
        numeric_shared_count,
        len(s1_number_set)
    )


    numeric_candidate_coverage = safe_ratio(
        numeric_shared_count,
        len(candidate_number_set)
    )


    numeric_set_equal = int(
        bool(s1_number_set) and
        bool(candidate_number_set) and
        s1_number_set ==
        candidate_number_set
    )


    numeric_has_shared = int(
        numeric_shared_count > 0
    )


    numeric_only_conflict = int(
        bool(s1_number_set) and
        bool(candidate_number_set) and
        numeric_shared_count == 0
    )


    numeric_both_have_unmatched = int(
        bool(unmatched_s1_numbers) and
        bool(unmatched_candidate_numbers)
    )


    component_overlap_ratio = safe_ratio(
        len(shared_components),
        len(
            component_set_s1 |
            component_set_candidate
        )
    )


    strong_name = int(
        (
            row.name_sequence_similarity >= 0.85
            and
            row.name_token_set_similarity >= 0.75
        )
        or
        row.name_exact == 1
    )


    strong_address = int(
        (
            row.address_sequence_similarity >= 0.85
            and
            row.address_token_set_similarity >= 0.70
        )
        or
        row.address_exact == 1
    )


    weak_name = int(
        row.name_sequence_similarity < 0.40
    )


    weak_address = int(
        row.address_sequence_similarity < 0.40
    )


    new_rows.append(
        {

            "name_frequency_count":
                name_frequency_count,

            "candidate_name_frequency_count":
                candidate_name_frequency_count,

            "name_frequency_inverse":
                (
                    1.0 /
                    np.log1p(
                        name_frequency_count
                    )
                    if name_frequency_count > 0
                    else 0.0
                ),

            "candidate_name_frequency_inverse":
                (
                    1.0 /
                    np.log1p(
                        candidate_name_frequency_count
                    )
                    if candidate_name_frequency_count > 0
                    else 0.0
                ),

            "address_frequency_count":
                address_frequency_count,

            "candidate_address_frequency_count":
                candidate_address_frequency_count,

            "numeric_shared_count":
                numeric_shared_count,

            "numeric_unmatched_s1_count":
                numeric_unmatched_s1_count,

            "numeric_unmatched_candidate_count":
                numeric_unmatched_candidate_count,

            "numeric_overlap_ratio":
                numeric_overlap_ratio,

            "numeric_s1_coverage":
                numeric_s1_coverage,

            "numeric_candidate_coverage":
                numeric_candidate_coverage,

            "numeric_set_equal":
                numeric_set_equal,

            "numeric_has_shared":
                numeric_has_shared,

            "numeric_only_conflict":
                numeric_only_conflict,

            "numeric_both_have_unmatched":
                numeric_both_have_unmatched,

            "address_component_overlap_ratio":
                component_overlap_ratio,

            "address_shared_component_count":
                len(shared_components),

            "same_name_script":
                same_script,

            "strong_name":
                strong_name,

            "strong_address":
                strong_address,

            "weak_name":
                weak_name,

            "weak_address":
                weak_address,

            "strong_name_strong_address":
                int(
                    strong_name and
                    strong_address
                ),

            "strong_name_numeric_consistent":
                int(
                    strong_name and
                    (
                        numeric_set_equal
                        or
                        numeric_overlap_ratio >= 0.5
                    )
                ),

            "strong_address_numeric_consistent":
                int(
                    strong_address and
                    (
                        numeric_set_equal
                        or
                        numeric_overlap_ratio >= 0.5
                    )
                ),

            "strong_address_numeric_conflict":
                int(
                    strong_address and
                    numeric_only_conflict
                ),

            "exact_name_rare":
                int(
                    exact_name and
                    name_frequency_count <= 2
                ),

            "exact_name_unique":
                int(
                    exact_name and
                    name_frequency_count == 1
                ),

            "exact_name_common":
                int(
                    exact_name and
                    name_frequency_count >= 10
                ),

            "exact_address_rare":
                int(
                    exact_address and
                    address_frequency_count <= 2
                ),

            "exact_address_common":
                int(
                    exact_address and
                    address_frequency_count >= 10
                ),

            "exact_name_missing_address":
                int(
                    exact_name and
                    (
                        not s1_address
                        or
                        not candidate_address
                    )
                ),

            "exact_name_candidate_missing_address":
                int(
                    exact_name and
                    not candidate_address
                ),

            "exact_name_both_addresses_missing":
                int(
                    exact_name and
                    not s1_address and
                    not candidate_address
                ),

            "exact_address_weak_name":
                int(
                    exact_address and
                    weak_name
                ),

            "strong_address_weak_name":
                int(
                    strong_address and
                    weak_name
                ),

            "strong_name_weak_address_v6":
                int(
                    strong_name and
                    weak_address
                ),

            "strong_name_strong_address_v6":
                int(
                    strong_name and
                    strong_address
                ),

            "strong_name_conflicting_numbers":
                int(
                    strong_name and
                    numeric_only_conflict
                ),

            "strong_address_conflicting_numbers":
                int(
                    strong_address and
                    numeric_only_conflict
                ),

            "name_and_address_script_same":
                int(
                    same_script and
                    bool(s1_name) and
                    bool(candidate_name)
                )
        }
    )


    if position % 1000 == 0:

        print(
            f"Processed: "
            f"{position} / {len(pairs)}"
        )


new_features = pd.DataFrame(
    new_rows,
    index=pairs.index
)


v6 = pd.concat(
    [
        pairs,
        new_features
    ],
    axis=1
)


print()
print(
    f"V5 columns: "
    f"{len(pairs.columns)}"
)

print(
    f"New V6 features: "
    f"{len(new_features.columns)}"
)

print(
    f"V6 columns: "
    f"{len(v6.columns)}"
)

print(
    f"Rows: "
    f"{len(v6)}"
)

print(
    f"Positive: "
    f"{int(v6['label'].sum())}"
)

print(
    f"Negative: "
    f"{int((v6['label'] == 0).sum())}"
)


v6.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print(
    f"Saved:\n{OUTPUT_FILE}"
)


print()
print("NEW FEATURES")
print("=" * 70)

for column in new_features.columns:

    print(column)


print()
print("=" * 70)
print("DONE")
print("=" * 70)