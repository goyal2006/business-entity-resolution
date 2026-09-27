# Business Entity Resolution

## Team

- Team Name: Over The Counter
- Team Members: Paras Goyal, Naman Garg

## Overview

This solution addresses the Business Entity Resolution challenge by identifying matching business records across Source 1, Source 2, and Source 3. The approach combines text normalization, candidate generation (blocking), engineered name and address similarity features, and a Random Forest classifier.

## Methodology

### 1. Text Normalization

Business names and addresses are normalized before comparison.

The preprocessing includes:

- Unicode normalization
- Lowercasing
- Punctuation normalization
- Whitespace normalization
- Ampersand normalization
- Common legal suffix normalization for business names
- Common address abbreviation normalization

Examples include converting terms such as company/corporation/limited to consistent forms and standardizing address terms such as street/road/avenue.

### 2. Candidate Generation

Comparing every Source 1 entity against every Source 2 and Source 3 record would be computationally expensive.

Candidate generation therefore uses blocking based on normalized textual signals, including:

- Exact normalized business names
- Exact normalized addresses
- Rare normalized name tokens
- Rare normalized address tokens

The final test candidate set contains:

- Source 1 entities: 1,732,544
- Candidate pairs: 15,872,850
- Average candidates per Source 1 entity: 9.2
- Median candidates: 3
- 95th percentile: 35
- 99th percentile: 94
- Maximum candidates: 480

### 3. Matching Features

For each candidate pair, the model uses name, address, and structural comparison features.

Name features include:

- Sequence similarity
- Token-set similarity
- Token-overlap similarity
- Character similarity
- Exact-name equality
- Shared token counts
- Token containment
- Candidate and Source 1 extra-token counts
- Name token counts
- Name frequency features

Address features include:

- Sequence similarity
- Token-set similarity
- Token-overlap similarity
- Character similarity
- Shared token counts
- Token containment
- Address component overlap
- Address number overlap
- Numeric agreement
- Numeric conflicts
- Address frequency features

Additional structural features capture interactions between strong and weak name/address evidence, numeric conflicts, missing fields, and dominant text script.

### 4. Matching Model

The final matching model is a Random Forest classifier.

Configuration:

- Trees: 300
- Maximum depth: 12
- Minimum samples per leaf: 3
- Class weighting: balanced
- Random state: 42

The decision threshold was selected using entity-level validation with the challenge's macro F0.5 metric.

Final threshold:

0.79

### 5. Validation Results

Using an entity-level 75/25 validation split:

- Macro F0.5: 0.979150
- Pair precision: 0.991696
- Pair recall: 0.958716
- True positives: 836
- False positives: 7
- False negatives: 36

The F0.5 metric was used because the challenge places greater importance on precision and therefore penalizes incorrect entity merges more heavily than missed matches.

### 6. Error Analysis

False positives were primarily associated with records having strong address similarity or highly similar business information despite representing different entities.

False negatives occurred in cases including:

- Exact or strong business-name matches with weak or missing addresses
- Strong address matches with weak business names
- Numeric address conflicts
- Cross-script records
- Records where multiple signals were individually informative but the combined model score remained below the threshold

### 7. Final Prediction

The final Random Forest model was applied to the generated test candidate pairs.

Final outputs:

- output/matching_results.tsv
- output/candidate_pairs.tsv

The final outputs were validated using the official challenge validator, including ID-existence checking.

## Reproduction

The source code is located under:

code/business_entity_resolution/src/

The main stages are:

1. Normalize source records.
2. Build training candidate pairs.
3. Generate V6 matching features.
4. Train/evaluate the Random Forest matcher.
5. Generate test candidates.
6. Run final prediction.

The generated TSV files are provided under the output/ directory.

## Data Policy

No external business databases, APIs, registries, geocoding services, or external business identity information were used. The solution relies on the provided challenge data and algorithmic matching techniques.
