from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

PAIR_FILE = (
    ROOT.parent /
    "output" /
    "pair_training_data_v6.csv"
)

THRESHOLD_OUTPUT = (
    ROOT.parent /
    "output" /
    "v6_rf_threshold_results.csv"
)

IMPORTANCE_OUTPUT = (
    ROOT.parent /
    "output" /
    "v6_rf_feature_importance.csv"
)


print("Loading V6 pair data...")

df = pd.read_csv(
    PAIR_FILE
)

print()
print("V6 RANDOM FOREST")
print("=" * 70)


feature_columns = [
    column
    for column in df.columns
    if column not in [
        "s1_entity_id",
        "candidate_entity_id",
        "label"
    ]
]


entities = sorted(
    df["s1_entity_id"].unique()
)


train_entities, validation_entities = train_test_split(
    entities,
    test_size=0.25,
    random_state=42
)


train_entities = set(
    train_entities
)

validation_entities = set(
    validation_entities
)


train_df = df[
    df["s1_entity_id"].isin(
        train_entities
    )
].copy()


validation_df = df[
    df["s1_entity_id"].isin(
        validation_entities
    )
].copy()


X_train = (
    train_df[
        feature_columns
    ]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


y_train = (
    train_df["label"]
    .astype(int)
)


X_validation = (
    validation_df[
        feature_columns
    ]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


y_validation = (
    validation_df["label"]
    .astype(int)
)


print(
    f"V6 entities: {len(entities)}"
)

print(
    f"Pair rows: {len(df)}"
)

print(
    f"Features: {len(feature_columns)}"
)

print(
    f"Training entities: "
    f"{len(train_entities)}"
)

print(
    f"Validation entities: "
    f"{len(validation_entities)}"
)

print(
    f"Training pairs: "
    f"{len(train_df)}"
)

print(
    f"Validation pairs: "
    f"{len(validation_df)}"
)

print(
    f"Training positives: "
    f"{int(y_train.sum())}"
)

print(
    f"Training negatives: "
    f"{int((y_train == 0).sum())}"
)

print(
    f"Validation positives: "
    f"{int(y_validation.sum())}"
)

print(
    f"Validation negatives: "
    f"{int((y_validation == 0).sum())}"
)


print()
print("Training Random Forest...")


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


probabilities = model.predict_proba(
    X_validation
)[:, 1]


def f05(
    precision,
    recall
):

    denominator = (
        0.25 * precision +
        recall
    )

    if denominator == 0:
        return 0.0

    return (
        1.25 *
        precision *
        recall /
        denominator
    )


def evaluate_threshold(
    threshold
):

    predictions = (
        probabilities >= threshold
    )


    prediction_series = pd.Series(
        predictions,
        index=validation_df.index
    )


    validation_labels = pd.Series(
        validation_df["label"].values,
        index=validation_df.index
    )


    entity_scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0


    for entity_id in validation_entities:

        entity_indices = validation_df[
            validation_df["s1_entity_id"]
            == entity_id
        ].index


        entity_true = (
            validation_labels
            .loc[entity_indices]
            .astype(bool)
        )


        entity_pred = (
            prediction_series
            .loc[entity_indices]
            .astype(bool)
        )


        tp = int(
            (
                entity_true &
                entity_pred
            ).sum()
        )


        fp = int(
            (
                ~entity_true &
                entity_pred
            ).sum()
        )


        fn = int(
            (
                entity_true &
                ~entity_pred
            ).sum()
        )


        total_tp += tp
        total_fp += fp
        total_fn += fn


        precision = (
            tp /
            (tp + fp)
            if tp + fp > 0
            else 1.0
        )


        recall = (
            tp /
            (tp + fn)
            if tp + fn > 0
            else 1.0
        )


        score = f05(
            precision,
            recall
        )


        entity_scores.append(
            score
        )


    macro_f05 = float(
        np.mean(entity_scores)
    )


    precision = (
        total_tp /
        (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )


    recall = (
        total_tp /
        (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )


    return (
        macro_f05,
        precision,
        recall,
        total_tp,
        total_fp,
        total_fn
    )


results = []


for threshold in np.round(
    np.arange(
        0.05,
        1.00,
        0.01
    ),
    2
):

    (
        macro_f05,
        precision,
        recall,
        tp,
        fp,
        fn
    ) = evaluate_threshold(
        threshold
    )


    results.append(
        {
            "threshold": threshold,
            "entity_macro_f05": macro_f05,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    )


results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "entity_macro_f05",
    ascending=False
)


best = results_df.iloc[0]


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Best threshold: "
    f"{best['threshold']:.2f}"
)

print(
    f"Entity macro F0.5: "
    f"{best['entity_macro_f05']:.6f}"
)

print(
    f"Pair precision: "
    f"{best['precision']:.6f}"
)

print(
    f"Pair recall: "
    f"{best['recall']:.6f}"
)

print(
    f"TP: "
    f"{int(best['tp'])}"
)

print(
    f"FP: "
    f"{int(best['fp'])}"
)

print(
    f"FN: "
    f"{int(best['fn'])}"
)


print()
print("TOP THRESHOLDS")
print("=" * 70)

print(
    results_df.head(10).to_string(
        index=False
    )
)


results_df.to_csv(
    THRESHOLD_OUTPUT,
    index=False
)


print()
print(
    f"Saved threshold results:"
)

print(
    THRESHOLD_OUTPUT
)


importance_df = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": model.feature_importances_
    }
)


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print()
print("TOP FEATURES")
print("=" * 70)

print(
    importance_df.head(25).to_string(
        index=False
    )
)


importance_df.to_csv(
    IMPORTANCE_OUTPUT,
    index=False
)


print()
print(
    f"Saved feature importance:"
)

print(
    IMPORTANCE_OUTPUT
)


print()
print("=" * 70)
print("COMPARISON WITH V5")
print("=" * 70)

v5_score = 0.977647

v6_score = float(
    best["entity_macro_f05"]
)

difference = (
    v6_score -
    v5_score
)

print(
    f"V5 Entity macro F0.5: "
    f"{v5_score:.6f}"
)

print(
    f"V6 Entity macro F0.5: "
    f"{v6_score:.6f}"
)

print(
    f"Difference: "
    f"{difference:+.6f}"
)


if difference > 0:

    print()
    print(
        "V6 IMPROVES OVER V5."
    )

elif difference < 0:

    print()
    print(
        "V6 DOES NOT IMPROVE OVER V5."
    )

else:

    print()
    print(
        "V6 MATCHES V5."
    )


print()
print("DONE")