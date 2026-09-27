from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import re
import unicodedata
import time


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "student_resource" / "dataset"
TEST = DATASET / "test"

S1_PATH = TEST / "test_source1.tsv"
S2_PATH = TEST / "test_source2.tsv"
S3_PATH = TEST / "test_source3.tsv"

OUTPUT = ROOT / "output" / "candidate_pairs.tsv"

CHUNK_SIZE = 100_000
RARE_LIMIT = 10
MAX_POSTING = 500


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
    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(text):
    text = normalize_text(text)

    if not text:
        return ""

    return " ".join(
        LEGAL_SUFFIXES.get(token, token)
        for token in text.split()
    )


def normalize_address(text):
    text = normalize_text(text)

    if not text:
        return ""

    return " ".join(
        ADDRESS_REPLACEMENTS.get(token, token)
        for token in text.split()
    )


def tokens(text):
    if not text:
        return set()

    return {
        token
        for token in text.split()
        if len(token) >= 3
    }


def load_s1():

    print("Loading Source 1...")

    df = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    df["norm_name"] = df["business_name"].map(normalize_name)
    df["norm_address"] = df["business_address"].map(
        normalize_address
    )

    print(f"S1 rows: {len(df):,}")

    return df


def build_indexes():

    exact_name = defaultdict(list)
    exact_address = defaultdict(list)

    name_token_counts = Counter()
    address_token_counts = Counter()

    print("\nFirst pass: counting token frequencies...")

    total = 0

    for source_number, path in [(2, S2_PATH), (3, S3_PATH)]:

        print(f"Source {source_number}...")

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
            ],
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
        ):

            total += len(chunk)

            for row in chunk.itertuples(index=False):

                name = normalize_name(row.business_name)
                address = normalize_address(row.business_address)

                for token in tokens(name):
                    name_token_counts[token] += 1

                for token in tokens(address):
                    address_token_counts[token] += 1

        print(f"Rows processed: {total:,}")

    print("\nSecond pass: building indexes...")

    rare_name = defaultdict(list)
    rare_address = defaultdict(list)

    total = 0

    for source_number, path in [(2, S2_PATH), (3, S3_PATH)]:

        print(f"Source {source_number}...")

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
            ],
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
        ):

            total += len(chunk)

            for row in chunk.itertuples(index=False):

                entity_id = row.entity_id

                name = normalize_name(row.business_name)
                address = normalize_address(row.business_address)

                if name:
                    exact_name[name].append(entity_id)

                if address:
                    exact_address[address].append(entity_id)

                for token in tokens(name):

                    if name_token_counts[token] <= RARE_LIMIT:
                        rare_name[token].append(entity_id)

                for token in tokens(address):

                    if address_token_counts[token] <= RARE_LIMIT:
                        rare_address[token].append(entity_id)

        print(f"Rows indexed: {total:,}")

    return (
        exact_name,
        exact_address,
        rare_name,
        rare_address,
    )


def generate_candidates(
    row,
    exact_name,
    exact_address,
    rare_name,
    rare_address,
):

    candidates = set()

    name = row.norm_name
    address = row.norm_address

    if name:

        values = exact_name.get(name)

        if values:
            candidates.update(values)

    if address:

        values = exact_address.get(address)

        if values:
            candidates.update(values)

    for token in tokens(name):

        values = rare_name.get(token)

        if values:
            candidates.update(values)

    for token in tokens(address):

        values = rare_address.get(token)

        if values:
            candidates.update(values)

    return candidates


def write_header():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        f.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )


def append_rows(rows):

    with open(
        OUTPUT,
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        for s1_id, candidates in rows:

            f.write(
                s1_id
                + "\t"
                + ",".join(candidates)
                + "\n"
            )


def main():

    start = time.time()

    print("=" * 75)
    print("FAST TEST CANDIDATE GENERATION")
    print("=" * 75)

    s1 = load_s1()

    (
        exact_name,
        exact_address,
        rare_name,
        rare_address,
    ) = build_indexes()

    print("\nIndexes ready.")

    print(f"Exact names: {len(exact_name):,}")
    print(f"Exact addresses: {len(exact_address):,}")
    print(f"Rare name tokens: {len(rare_name):,}")
    print(f"Rare address tokens: {len(rare_address):,}")

    write_header()

    total_candidates = 0
    max_candidates = 0
    empty_entities = 0

    candidate_counts = []

    rows = []

    print("\nGenerating candidates...")

    for index, row in enumerate(
        s1.itertuples(index=False),
        start=1,
    ):

        candidates = generate_candidates(
            row,
            exact_name,
            exact_address,
            rare_name,
            rare_address,
        )

        candidates = sorted(candidates)

        count = len(candidates)

        total_candidates += count
        max_candidates = max(max_candidates, count)

        candidate_counts.append(count)

        if count == 0:
            empty_entities += 1

        rows.append(
            (
                row.entity_id,
                candidates,
            )
        )

        if len(rows) >= 10_000:

            append_rows(rows)
            rows.clear()

        if index % 25_000 == 0:

            elapsed = time.time() - start

            print(
                f"{index:,}/{len(s1):,} | "
                f"avg={total_candidates / index:,.1f} | "
                f"max={max_candidates:,} | "
                f"empty={empty_entities:,} | "
                f"time={elapsed / 60:.1f}m"
            )

    if rows:
        append_rows(rows)

    candidate_counts.sort()

    median = candidate_counts[
        len(candidate_counts) // 2
    ]

    p95 = candidate_counts[
        int(len(candidate_counts) * 0.95)
    ]

    p99 = candidate_counts[
        int(len(candidate_counts) * 0.99)
    ]

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    print(f"S1 entities:       {len(s1):,}")
    print(f"Total candidates:  {total_candidates:,}")
    print(
        f"Average candidates: "
        f"{total_candidates / len(s1):,.1f}"
    )
    print(f"Median candidates:  {median:,}")
    print(f"P95 candidates:     {p95:,}")
    print(f"P99 candidates:     {p99:,}")
    print(f"Maximum candidates: {max_candidates:,}")
    print(f"Empty entities:     {empty_entities:,}")

    print("\nSaved:")
    print(OUTPUT)

    print(
        f"\nTotal time: "
        f"{(time.time() - start) / 60:.2f} minutes"
    )

    print("\nDONE")


if __name__ == "__main__":
    main()