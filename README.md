# Business Entity Resolution — reproducible pipeline

CPU-only pipeline (developed and run on 2 vCPU / 8 GB RAM). No external data, APIs
or pretrained models are used: every model is trained from scratch on the provided
training data. Models: LightGBM (MIT) and a PyTorch MLP (BSD) — both far below 8B params.

## Layout
```
normalize.py   text normalisation (transliteration, abbreviations, states, legal suffixes)
blocking.py    sparse TF-IDF retrieval of top-K Source-1 candidates per S2/S3 record
features.py    38 pairwise similarity features (rapidfuzz, vectorised)
pipeline.py    train / validate / infer entry point
```

## Run
```bash
pip install -r requirements.txt
# DATA_DIR must contain train/ and test/ exactly as shipped (student_resource/dataset)
python pipeline.py --data DATA_DIR --work work --out output --stage train   # ~20 min
python pipeline.py --data DATA_DIR --work work --out output --stage infer   # ~45 min
python /path/to/student_resource/utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir DATA_DIR/test
```
`--stage all` runs both. The train stage prints the held-out macro F0.5 and writes
`work/meta.json` (chosen model, per-country thresholds, full threshold sweep),
`work/lgbm.txt` and `work/mlp.pkl`. The infer stage writes `output/candidate_pairs.tsv`
(exact candidate set scored by the model) and `output/matching_results.tsv`.

Key options (defaults used for the submission): `--val_frac 0.05 --rho 0.10
--k_block 10 --keep_k 6 --rel 0.5 --rounds 300 --mlp_epochs 1`.
