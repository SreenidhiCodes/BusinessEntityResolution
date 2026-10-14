# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Elixora]
**Team Members:** Santhi Ranimekala, Ragipindi Dipika Reddy, Rebbapragada V N S Sreenidhi (Andhra University College of Engineering (AUCE), Visakhapatnam)
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We treat entity resolution as **assignment**: each Source-2/3 record retrieves its most similar Source-1 records using sparse TF-IDF over transliterated, normalized name and address tokens. A classifier trained on ground-truth pairs scores each candidate. Each S2/S3 record is assigned to **at most one** S1 entity: its highest-scoring candidate, if it clears a country-specific threshold tuned for macro F0.5. On held-out training entities, the final pipeline achieved **macro F0.5 = 0.961** (US 0.974, India 0.942) and scored **0.92 on the public leaderboard**.

---

## 2. Methodology

### 2.1 Problem Analysis
Key findings from EDA on the training data:
- **One-owner structure.** Every one of the 7.64M matched S2/S3 IDs in the ground truth belongs to exactly one S1 entity (no ID is shared). About 74.6% of S2/S3 records match some S1 entity, and an S1 entity has on average 3.5 matches (0–11). Only 5.6% of S1 entities are singletons.
- **Reused generic names.** Business names repeat heavily across different entities (e.g. 18 distinct US entities called "Capital Medical Center"), so the address is at least as discriminative as the name.
- **Scripts.** 18.5% of Indian and 6.8% of US S3 names are non-Latin (Devanagari, Telugu, …), e.g. "जय सॉल्यूशंस प्राइवेट लिमिटेड" ↔ "Jai Solutions Private Limited".
- **Address noise.** US states are written out on one side and abbreviated on the other ("Texas"/"TX"); Indian states likewise ("MH"/"Maharashtra"). Other noise includes filler tokens ("null", "N/A"), component reordering, dropped units and typos in street numbers.
- **Name noise.** Dropped or duplicated legal suffixes (Pvt/Ltd/LLC/SARL), word transpositions, character typos ("Cmmplete"), accented Latin characters ("Léarning"), website-style names ("nexyxutilities.com"), and DBA names that differ completely while the address matches.

### 2.2 Solution Strategy
**Approach Type:** Blocking (reverse retrieval) + pairwise classifier + constrained assignment
**Core Innovation:** Exploiting the one-owner constraint. Retrieval runs from each S2/S3 record towards S1, and the final decision is a per-record argmax with a tuned threshold. This avoids assigning one S2/S3 record to multiple S1 entities. Candidate retrieval combines name and address evidence, using order-invariant name bigrams and balanced weighting of name and address tokens.

---

## 3. Candidate Generation (Blocking)
- **Normalization:** Unicode → ASCII transliteration (`unidecode`); expansion of name and address abbreviations; canonical state tokens (US and Indian states); removal of legal suffixes, including transliterated ones ("praaivett limittedd"); and a consonant "skeleton" of each name token (drop inner vowels, collapse doubled letters and aspirates), so that transliteration variants such as `ttreddrs`/`traders` → `trdrs` collide.
- **Blocking keys (one sparse TF-IDF vector per record):** name-skeleton unigrams (weight 1.0), **order-invariant** name-skeleton bigrams (weight 0.5), and address tokens (street numbers, street/locality/city words, state and postal tokens; weight 1.0). IDF comes from Source 1. Tokens with Source-1 document frequency above 4,000 are excluded from retrieval only.
- **Retrieval:** within each country, a cosine top-10 via multithreaded sparse matrix product (`sparse_dot_topn`). Candidates are then pruned to the top 6 per record with similarity ≥ 0.5 × the record's best similarity.
- **Candidate pairs generated (test):** 35,754,681 pairs for 1,732,544 S1 entities (about 21 per entity). The model scores exactly this set, and it is written to `candidate_pairs.tsv`.
- **Recall:** retrieval runs from every S2/S3 record, so each record can retrieve its owner even when an S1 entity has many matches. The final blocking design uses order-invariant name bigrams with weight 0.5, while name unigrams and address tokens each have weight 1.0. On held-out data, it retrieved **93.75%** of matched records.

---

## 4. Matching Model
**Features (38):**
- *Name:* rapidfuzz ratio, token-set, token-sort, partial ratio and Jaro–Winkler on the full name, on the legal-suffix-stripped core name and on the phonetic skeleton; exact-equality flags; first-token match; lengths.
- *Address:* ratio, token-set, token-sort, partial token-set and Jaro–Winkler; token-set on the digit tokens (house/plot/street numbers); exact digit-set match; postal-code match and presence; missing-address flag.
- *Context:* retrieval cosine, rank, gap to the query's best candidate and to the next candidate, number of candidates, non-Latin-script flag, source (S2/S3).

**Model type:** LightGBM (binary, 127 leaves, lr 0.08, 300 rounds), compared against a PyTorch MLP (3 hidden layers 256-256-128, BatchNorm, SiLU, dropout, OneCycle AdamW) and their average. LightGBM scored best on validation (0.9614 vs 0.9528 for the MLP and 0.9589 for the blend) and is used for the final predictions. Both models are trained from scratch; no pretrained or external models are used.

**Decision rule:** each S2/S3 record is assigned to its highest-probability S1 candidate when p ≥ t_country.
**Threshold selection method:** sweep of t on held-out S1 entities, maximizing the official macro F0.5 (per S1 entity, singletons included). The thresholds are India 0.325 and US 0.35; the global 0.35 is used for unseen countries such as France.

**Validation protocol:** 5% of S1 entities are held out. All their S2/S3 records are included, plus a random 10% sample of all other S2/S3 records, each blocked against the *full* S1 table so that neighbourhood density matches test conditions. False matches from sampled records are weighted by 1/0.10 to give an unbiased precision estimate. No record touching a validation entity is used for training.

---

## 5. Results & Error Analysis
| Final approach | Blocking recall | Validation macro F0.5 | Public leaderboard |
|---|---|---|---|
| Address-aware retrieval with order-invariant name bigrams, LightGBM pair scoring, and one-owner assignment | **93.75%** | **0.961 (US 0.974, India 0.942)** | **0.92** |

- **Common false positives:** the same address with different businesses (co-located firms, shared buildings in India), and generic names ("Primary Care", "Total Catering") in the same city.
- **Common false negatives:** the owner not retrieved by blocking (about 6% of matched records), mostly DBA/trade names that share no tokens with the S1 name combined with a partial address, and heavily abbreviated Indian addresses. The score is nearly flat across thresholds 0.2–0.5 (0.960–0.961), so the remaining headroom is in candidate recall rather than classification.

---

## 6. Conclusion
Reframing the task as one-owner assignment, with multilingual normalization, address-aware blocking and a strong pairwise classifier, gives macro F0.5 ≈ 0.96 on held-out data. It uses only 2 CPUs and 8 GB RAM, and no pretrained models. The key lessons: the ground-truth structure (each S2/S3 record has at most one owner) is worth more than model complexity, and error analysis of blocking misses gave the largest single gain (+0.04 F0.5).

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`: `normalize.py` (cleaning), `blocking.py` (retrieval), `features.py` (pair features), `pipeline.py` (entry point: `--stage train|infer|all`). See `README.md` for exact commands; `requirements.txt` pins the versions.

### B. Additional Results
Final test output: 5,853,789 matches; 1,639,293 S1 entities with at least one match; 93,251 predicted singletons (5.4%, compared with 5.6% singletons in training).

Threshold sweep (validation macro F0.5, LightGBM): 0.9602 @0.20, 0.9613 @0.30, 0.9613 @0.35, 0.9610 @0.40, 0.9599 @0.50.
