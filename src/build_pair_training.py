from pathlib import Path
import re
from difflib import SequenceMatcher

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT.parent / "student_resource" / "dataset"
TRAIN_ROOT = DATA_ROOT / "train"

SOURCE1_FILE = TRAIN_ROOT / "train_source1.tsv"
SOURCE2_FILE = TRAIN_ROOT / "train_source2.tsv"
SOURCE3_FILE = TRAIN_ROOT / "train_source3.tsv"
GT_FILE = TRAIN_ROOT / "train_ground_truth.tsv"

OUTPUT_FILE = ROOT.parent / "output" / "pair_training_data_v5.csv"

MAX_S1_ENTITIES = 1000
MAX_NEGATIVES_PER_ENTITY = 50
RARE_TOKEN_MAX_FREQ = 10

COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]


def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = text.replace("&", " and ")

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


def tokenize(text):

    if not text:
        return set()

    return set(text.split())


def extract_numbers(text):

    if not text:
        return set()

    return set(
        re.findall(
            r"\d+(?:\.\d+)?",
            text
        )
    )


def sequence_similarity(a, b):

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def character_similarity(a, b):

    if not a or not b:
        return 0.0

    a = a.replace(" ", "")
    b = b.replace(" ", "")

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def token_set_similarity(a, b):

    tokens_a = tokenize(a)
    tokens_b = tokenize(b)

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(
        tokens_a | tokens_b
    )


def token_overlap(a, b):

    tokens_a = tokenize(a)
    tokens_b = tokenize(b)

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / min(
        len(tokens_a),
        len(tokens_b)
    )


def length_ratio(a, b):

    if not a or not b:
        return 0.0

    return min(
        len(a),
        len(b)
    ) / max(
        len(a),
        len(b)
    )


def build_features(s1, candidate):

    name1 = s1["name"]
    name2 = candidate["name"]

    address1 = s1["address"]
    address2 = candidate["address"]

    name_tokens_1 = tokenize(name1)
    name_tokens_2 = tokenize(name2)

    address_tokens_1 = tokenize(address1)
    address_tokens_2 = tokenize(address2)

    shared_name = (
        name_tokens_1 &
        name_tokens_2
    )

    extra_name_candidate = (
        name_tokens_2 -
        name_tokens_1
    )

    extra_name_s1 = (
        name_tokens_1 -
        name_tokens_2
    )

    shared_address = (
        address_tokens_1 &
        address_tokens_2
    )

    extra_address_candidate = (
        address_tokens_2 -
        address_tokens_1
    )

    extra_address_s1 = (
        address_tokens_1 -
        address_tokens_2
    )

    numbers1 = extract_numbers(address1)
    numbers2 = extract_numbers(address2)

    shared_numbers = numbers1 & numbers2

    conflicting_numbers = numbers1 ^ numbers2

    name_seq = sequence_similarity(
        name1,
        name2
    )

    name_char = character_similarity(
        name1,
        name2
    )

    name_token_set = token_set_similarity(
        name1,
        name2
    )

    name_token_overlap = token_overlap(
        name1,
        name2
    )

    address_seq = sequence_similarity(
        address1,
        address2
    )

    address_char = character_similarity(
        address1,
        address2
    )

    address_token_set = token_set_similarity(
        address1,
        address2
    )

    address_token_overlap = token_overlap(
        address1,
        address2
    )

    name_exact = int(
        bool(name1) and
        bool(name2) and
        name1 == name2
    )

    address_exact = int(
        bool(address1) and
        bool(address2) and
        address1 == address2
    )

    address_missing = int(
        not address1 or not address2
    )

    both_fields_present = int(
        bool(name1) and
        bool(name2) and
        bool(address1) and
        bool(address2)
    )

    return {

        "name_sequence_similarity":
            name_seq,

        "name_character_similarity":
            name_char,

        "name_token_set_similarity":
            name_token_set,

        "name_token_overlap":
            name_token_overlap,

        "name_exact":
            name_exact,

        "name_length_ratio":
            length_ratio(
                name1,
                name2
            ),

        "name_token_count_difference":
            abs(
                len(name_tokens_1) -
                len(name_tokens_2)
            ),

        "address_sequence_similarity":
            address_seq,

        "address_character_similarity":
            address_char,

        "address_token_set_similarity":
            address_token_set,

        "address_token_overlap":
            address_token_overlap,

        "address_exact":
            address_exact,

        "address_length_ratio":
            length_ratio(
                address1,
                address2
            ),

        "address_token_count_difference":
            abs(
                len(address_tokens_1) -
                len(address_tokens_2)
            ),

        "country_equal":
            int(
                s1["country"].lower()
                ==
                candidate["country"].lower()
            ),

        "name_missing":
            int(
                not name1 or
                not name2
            ),

        "address_missing":
            address_missing,

        "both_fields_present":
            both_fields_present,

        "name_shared_token_count":
            len(shared_name),

        "name_extra_candidate_token_count":
            len(extra_name_candidate),

        "name_extra_s1_token_count":
            len(extra_name_s1),

        "name_token_count_s1":
            len(name_tokens_1),

        "name_token_count_candidate":
            len(name_tokens_2),

        "name_token_containment":
            (
                len(shared_name) /
                min(
                    len(name_tokens_1),
                    len(name_tokens_2)
                )
                if name_tokens_1 and name_tokens_2
                else 0.0
            ),

        "address_shared_token_count":
            len(shared_address),

        "address_extra_candidate_token_count":
            len(extra_address_candidate),

        "address_extra_s1_token_count":
            len(extra_address_s1),

        "address_token_count_s1":
            len(address_tokens_1),

        "address_token_count_candidate":
            len(address_tokens_2),

        "address_token_containment":
            (
                len(shared_address) /
                min(
                    len(address_tokens_1),
                    len(address_tokens_2)
                )
                if address_tokens_1 and address_tokens_2
                else 0.0
            ),

        "address_number_count_s1":
            len(numbers1),

        "address_number_count_candidate":
            len(numbers2),

        "address_shared_number_count":
            len(shared_numbers),

        "address_number_conflict_count":
            len(conflicting_numbers),

        "address_numbers_exact":
            int(
                bool(numbers1) and
                bool(numbers2) and
                numbers1 == numbers2
            ),

        "address_has_numbers_both":
            int(
                bool(numbers1) and
                bool(numbers2)
            ),

        "name_exact_address_missing":
            int(
                name_exact and
                address_missing
            ),

        "name_exact_address_present":
            int(
                name_exact and
                both_fields_present
            ),

        "exact_address_weak_name":
            int(
                address_exact and
                name_token_overlap < 0.5
            ),

        "strong_name_weak_address":
            int(
                name_token_overlap >= 0.8 and
                address_token_overlap < 0.5
            ),

        "strong_address_weak_name":
            int(
                address_token_overlap >= 0.8 and
                name_token_overlap < 0.5
            ),

        "name_candidate_has_extra_tokens":
            int(
                len(extra_name_candidate) > 0
            ),

        "address_candidate_has_extra_tokens":
            int(
                len(extra_address_candidate) > 0
            ),

        "numeric_address_agreement":
            int(
                len(shared_numbers) > 0
            ),

        "numeric_address_conflict":
            int(
                len(conflicting_numbers) > 0
            )
    }


def main():

    print("Loading Source 1...")

    s1 = pd.read_csv(
        SOURCE1_FILE,
        sep="\t",
        dtype=str,
        usecols=COLUMNS
    )

    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        dtype=str
    )

    s1 = s1.iloc[
        :MAX_S1_ENTITIES
    ].copy()

    gt = gt[
        gt["source1_entity_id"].isin(
            s1["entity_id"]
        )
    ]

    print(
        "Source 1 entities:",
        len(s1)
    )

    print("Loading Source 2...")

    s2 = pd.read_csv(
        SOURCE2_FILE,
        sep="\t",
        dtype=str,
        usecols=COLUMNS
    )

    print(
        "Source 2 rows:",
        len(s2)
    )

    print("Loading Source 3...")

    s3 = pd.read_csv(
        SOURCE3_FILE,
        sep="\t",
        dtype=str,
        usecols=COLUMNS
    )

    print(
        "Source 3 rows:",
        len(s3)
    )

    gt_map = {}

    for _, row in gt.iterrows():

        value = row["matched_entity_ids"]

        if pd.isna(value) or not str(value).strip():

            gt_map[
                row["source1_entity_id"]
            ] = set()

        else:

            gt_map[
                row["source1_entity_id"]
            ] = set(
                x.strip()
                for x in str(value).split(",")
                if x.strip()
            )

    print("Building compact indexes...")

    records = {}

    name_frequency = {}
    address_frequency = {}

    def process_source(df):

        for row in df.itertuples(
            index=False,
            name=None
        ):

            entity_id = row[0]
            raw_name = row[1]
            raw_address = row[2]
            country = row[3]

            name = normalize_text(
                raw_name
            )

            address = normalize_text(
                raw_address
            )

            name_tokens = tokenize(name)
            address_tokens = tokenize(address)

            records[entity_id] = (
                name,
                address,
                "" if pd.isna(country)
                else str(country).strip()
            )

            for token in name_tokens:
                name_frequency[token] = (
                    name_frequency.get(
                        token,
                        0
                    ) + 1
                )

            for token in address_tokens:
                address_frequency[token] = (
                    address_frequency.get(
                        token,
                        0
                    ) + 1
                )

    process_source(s2)

    del s2

    print("Source 2 indexed")

    process_source(s3)

    del s3

    print("Source 3 indexed")

    name_index = {}
    address_index = {}
    rare_name_index = {}
    rare_address_index = {}

    for entity_id, (
        name,
        address,
        country
    ) in records.items():

        if name:

            name_index.setdefault(
                name,
                set()
            ).add(entity_id)

        if address:

            address_index.setdefault(
                address,
                set()
            ).add(entity_id)

        for token in tokenize(name):

            if (
                name_frequency[token]
                <= RARE_TOKEN_MAX_FREQ
            ):

                rare_name_index.setdefault(
                    token,
                    set()
                ).add(entity_id)

        for token in tokenize(address):

            if (
                address_frequency[token]
                <= RARE_TOKEN_MAX_FREQ
            ):

                rare_address_index.setdefault(
                    token,
                    set()
                ).add(entity_id)

    print("Indexes ready")

    rows = []

    print("Generating pairs...")

    for counter, (_, row) in enumerate(
        s1.iterrows(),
        start=1
    ):

        s1_id = row["entity_id"]

        s1_record = {
            "name": normalize_text(
                row["business_name"]
            ),
            "address": normalize_text(
                row["business_address"]
            ),
            "country": (
                ""
                if pd.isna(row["country"])
                else str(row["country"]).strip()
            )
        }

        true_ids = gt_map.get(
            s1_id,
            set()
        )

        candidate_ids = set(
            true_ids
        )

        for token in tokenize(
            s1_record["name"]
        ):

            candidate_ids.update(
                rare_name_index.get(
                    token,
                    set()
                )
            )

        for token in tokenize(
            s1_record["address"]
        ):

            candidate_ids.update(
                rare_address_index.get(
                    token,
                    set()
                )
            )

        if s1_record["name"]:

            candidate_ids.update(
                name_index.get(
                    s1_record["name"],
                    set()
                )
            )

        if s1_record["address"]:

            candidate_ids.update(
                address_index.get(
                    s1_record["address"],
                    set()
                )
            )

        negative_ids = [
            x
            for x in candidate_ids
            if x not in true_ids
        ]

        negative_ids = negative_ids[
            :MAX_NEGATIVES_PER_ENTITY
        ]

        selected_ids = (
            list(true_ids)
            +
            negative_ids
        )

        for candidate_id in selected_ids:

            name2, address2, country2 = records[
                candidate_id
            ]

            candidate_record = {
                "name": name2,
                "address": address2,
                "country": country2
            }

            features = build_features(
                s1_record,
                candidate_record
            )

            features[
                "s1_entity_id"
            ] = s1_id

            features[
                "candidate_entity_id"
            ] = candidate_id

            features[
                "label"
            ] = int(
                candidate_id in true_ids
            )

            rows.append(
                features
            )

        if counter % 100 == 0:

            print(
                "Processed:",
                counter,
                "/",
                MAX_S1_ENTITIES
            )

    result = pd.DataFrame(
        rows
    )

    print()
    print("Rows:", len(result))

    print(
        "Positive:",
        int(result["label"].sum())
    )

    print(
        "Negative:",
        int(
            (result["label"] == 0).sum()
        )
    )

    print(
        "Columns:",
        len(result.columns)
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()