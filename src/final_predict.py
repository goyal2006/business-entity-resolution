from pathlib import Path
import re
import unicodedata
import pandas as pd
from rapidfuzz.fuzz import ratio


ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT.parent
DATASET = STUDENT / "student_resource" / "dataset"

CANDIDATES = ROOT / "output" / "candidate_pairs.tsv"
MATCHING_OUTPUT = ROOT / "output" / "matching_results.tsv"

S1_FILE = DATASET / "test" / "test_source1.tsv"
S2_FILE = DATASET / "test" / "test_source2.tsv"
S3_FILE = DATASET / "test" / "test_source3.tsv"

CHUNK_SIZE = 100_000


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


def normalize_text(x):
    if not x:
        return ""

    x = unicodedata.normalize("NFKC", str(x))
    x = x.lower()
    x = x.replace("&", " and ")
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def normalize_name(x):
    x = normalize_text(x)

    if not x:
        return ""

    return " ".join(
        LEGAL_SUFFIXES.get(t, t)
        for t in x.split()
    )


def normalize_address(x):
    x = normalize_text(x)

    if not x:
        return ""

    return " ".join(
        ADDRESS_REPLACEMENTS.get(t, t)
        for t in x.split()
    )


def nums(x):
    return set(re.findall(r"\d+", x))


def score_pair(a, b):
    n1 = a["name"]
    n2 = b["name"]

    ad1 = a["address"]
    ad2 = b["address"]

    if (
        a["country"]
        and
        b["country"]
        and
        a["country"].lower() != b["country"].lower()
    ):
        return -100.0

    nt1 = set(n1.split())
    nt2 = set(n2.split())

    at1 = set(ad1.split())
    at2 = set(ad2.split())

    shared_n = nt1 & nt2
    shared_a = at1 & at2

    name_exact = bool(n1 and n2 and n1 == n2)
    address_exact = bool(ad1 and ad2 and ad1 == ad2)

    name_overlap = (
        len(shared_n) / min(len(nt1), len(nt2))
        if nt1 and nt2
        else 0
    )

    address_overlap = (
        len(shared_a) / min(len(at1), len(at2))
        if at1 and at2
        else 0
    )

    name_ratio = (
        ratio(n1, n2) / 100
        if n1 and n2
        else 0
    )

    address_ratio = (
        ratio(ad1, ad2) / 100
        if ad1 and ad2
        else 0
    )

    num1 = nums(ad1)
    num2 = nums(ad2)

    shared_num = num1 & num2

    numeric_conflict = bool(num1 and num2 and not shared_num)

    score = 0.0

    if name_exact:
        score += 7.0

    if address_exact:
        score += 8.0

    score += 4.0 * name_overlap
    score += 5.0 * address_overlap

    score += 2.5 * name_ratio
    score += 3.0 * address_ratio

    if shared_num:
        score += 2.0

    if numeric_conflict:
        score -= 3.0

    if name_exact and not ad1:
        score += 1.5

    if name_exact and not ad2:
        score += 1.5

    if address_exact and name_ratio >= 0.45:
        score += 2.0

    if name_overlap >= 0.8 and address_overlap >= 0.5:
        score += 3.0

    if address_overlap >= 0.8 and name_overlap >= 0.5:
        score += 3.0

    return score


def load_records(path, ids):
    records = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=250_000,
        keep_default_na=False
    ):

        for row in chunk.itertuples(index=False):

            if row.entity_id not in ids:
                continue

            records[row.entity_id] = {
                "name": normalize_name(row.business_name),
                "address": normalize_address(row.business_address),
                "country": row.country
            }

    return records


print("=" * 70)
print("FAST FINAL TEST PREDICTION")
print("=" * 70)

print("\nChecking files...")

for path in [
    CANDIDATES,
    S1_FILE,
    S2_FILE,
    S3_FILE
]:
    print(path)
    if not path.exists():
        raise FileNotFoundError(path)

print("\nReading candidate IDs...")

s1_ids = set()
candidate_ids = set()

for chunk in pd.read_csv(
    CANDIDATES,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE,
    keep_default_na=False
):

    s1_ids.update(
        chunk["source1_entity_id"]
    )

    for value in chunk["candidate_entity_ids"]:

        if value:
            candidate_ids.update(
                value.split(",")
            )

print(
    f"S1 entities: {len(s1_ids):,}"
)

print(
    f"Candidate IDs: {len(candidate_ids):,}"
)

print("\nLoading S1...")

s1 = load_records(
    S1_FILE,
    s1_ids
)

print(
    f"S1 loaded: {len(s1):,}"
)

print("\nLoading S2...")

candidates = load_records(
    S2_FILE,
    candidate_ids
)

print(
    f"After S2: {len(candidates):,}"
)

print("\nLoading S3...")

candidates.update(
    load_records(
        S3_FILE,
        candidate_ids
    )
)

print(
    f"Candidate records: {len(candidates):,}"
)

print("\nGenerating final matches...")

MATCHING_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

processed = 0
matched_pairs = 0

with open(
    MATCHING_OUTPUT,
    "w",
    encoding="utf-8"
) as out:

    out.write(
        "source1_entity_id\tmatched_entity_ids\n"
    )

    for chunk in pd.read_csv(
        CANDIDATES,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        for row in chunk.itertuples(index=False):

            sid = row.source1_entity_id

            source = s1.get(sid)

            if source is None:
                out.write(f"{sid}\t\n")
                continue

            ids = (
                row.candidate_entity_ids.split(",")
                if row.candidate_entity_ids
                else []
            )

            matches = []

            for cid in ids:

                candidate = candidates.get(cid)

                if candidate is None:
                    continue

                score = score_pair(
                    source,
                    candidate
                )

                if score >= 8.0:
                    matches.append(
                        (score, cid)
                    )

            matches.sort(
                reverse=True
            )

            selected = [
                cid
                for score, cid in matches
            ]

            selected = list(
                dict.fromkeys(selected)
            )

            matched_pairs += len(selected)

            out.write(
                sid
                + "\t"
                + ",".join(selected)
                + "\n"
            )

            processed += 1

            if processed % 10_000 == 0:

                print(
                    f"Processed "
                    f"{processed:,} / "
                    f"{len(s1_ids):,} | "
                    f"Matches: "
                    f"{matched_pairs:,}"
                )

print("\n" + "=" * 70)
print("FINAL PREDICTION COMPLETE")
print("=" * 70)

print(
    f"S1 entities: {len(s1_ids):,}"
)

print(
    f"Predicted pairs: {matched_pairs:,}"
)

print(
    f"Output: {MATCHING_OUTPUT}"
)