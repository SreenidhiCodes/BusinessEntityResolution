"""End-to-end Business Entity Resolution pipeline.

Stages
  prep   : read TSVs, normalise text (2 worker processes), cache as parquet
  train  : build training / validation query sets, block, featurise, fit
           LightGBM + a PyTorch MLP, tune the decision threshold on macro F0.5
  infer  : block + featurise + score the test set, write the two output TSVs

Usage:  python pipeline.py --data DATA_DIR --work WORK_DIR --out OUT_DIR [--stage all]
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import build_index, query_matrix, topk  # noqa: E402
from features import FEATURES, pair_features  # noqa: E402
from normalize import normalize_frame  # noqa: E402

NORM_COLS = ["n_full", "n_core", "n_skel", "a_norm", "a_dig", "postal", "nonascii"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ----------------------------------------------------------------------------- prep
def _norm_chunk(df):
    return normalize_frame(df)


def _raw_chunks(path, keep_ids, chunk=200_000):
    for df in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
                          quoting=3, engine="c", chunksize=chunk):
        df = df[["entity_id", "business_name", "business_address", "country"]]
        if keep_ids is not None:
            df = df[df["entity_id"].isin(keep_ids)]
        yield df.reset_index(drop=True)


def read_norm(path, cache, keep_ids=None, workers=2):
    """Streams the TSV in chunks, normalises in `workers` processes and writes an
    incremental parquet cache (bounded memory). `keep_ids` restricts to a subset."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not os.path.exists(cache):
        writer = None
        with Pool(workers) as p:
            for part in p.imap(_norm_chunk, _raw_chunks(path, keep_ids)):
                part = part.drop(columns=["business_name", "business_address"])
                part["nonascii"] = part["nonascii"].astype(np.int8)
                t = pa.Table.from_pandas(part, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(cache + ".tmp", t.schema)
                writer.write_table(t)
        writer.close()
        os.replace(cache + ".tmp", cache)
    df = pd.read_parquet(cache)
    if keep_ids is not None:
        df = df[df["entity_id"].isin(keep_ids)].reset_index(drop=True)
    return df


def id_num(ids):
    """'S2-12345' -> 12345 + 2e10 (compact int64 key)."""
    s = pd.Series(ids)
    pre = s.str.slice(1, 2).astype("int64").to_numpy()
    return s.str.slice(3).astype("int64").to_numpy() + pre * 10_000_000_000


def as_dict(df):
    return {c: (df[c].to_numpy() if c == "nonascii" else df[c].array) for c in NORM_COLS}


# ----------------------------------------------------------------------------- blocking
def block_country(s1c, qc, k, workers=2):
    """Returns pair arrays (qi, si, sim, rank) with indices local to s1c / qc."""
    X, vocab, colw = build_index(s1c["n_skel"].tolist(), s1c["a_norm"].tolist())
    XT = X.T.tocsr()
    Q = query_matrix(qc["n_skel"].tolist(), qc["a_norm"].tolist(), vocab, colw)
    res = topk(Q, XT, k, n_threads=workers)
    del X, XT, Q
    return res


def prune(qi, si, sim, rank, keep_k, rel):
    """Keeps at most keep_k per query and only candidates with sim >= rel * best."""
    best = np.zeros(qi.max() + 1 if len(qi) else 1, dtype=np.float32)
    np.maximum.at(best, qi, sim)
    m = (rank < keep_k) & (sim >= rel * best[qi])
    return qi[m], si[m], sim[m], rank[m]


# ----------------------------------------------------------------------------- scoring
class MLP:
    """Feed-forward neural network pair classifier (PyTorch, CPU)."""

    def __init__(self, n_in, hidden=256, epochs=4, lr=2e-3, bs=4096, seed=0):
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(2)
        nn = torch.nn
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1))
        self.epochs, self.lr, self.bs = epochs, lr, bs

    def _prep(self, X):
        X = X.astype(np.float32).copy()
        return (X - self.mu) / self.sd

    def fit(self, X, y):
        import torch
        self.mu = X.mean(0).astype(np.float32)
        self.sd = (X.std(0) + 1e-6).astype(np.float32)
        Xt = torch.from_numpy(self._prep(X))
        yt = torch.from_numpy(y.astype(np.float32))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        n = len(Xt)
        steps = self.epochs * ((n + self.bs - 1) // self.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        lossf = torch.nn.BCEWithLogitsLoss()
        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for b in range(0, n, self.bs):
                idx = perm[b:b + self.bs]
                opt.zero_grad()
                loss = lossf(self.net(Xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item() * len(idx)
            log(f"  MLP epoch {ep + 1}/{self.epochs} loss={tot / n:.4f}")
        return self

    def predict(self, X):
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(X), 200_000):
                xb = torch.from_numpy(self._prep(X[b:b + 200_000]))
                out.append(torch.sigmoid(self.net(xb).squeeze(1)).numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def assign(qi, si, p):
    """One-owner constraint: each query keeps only its highest-probability S1.
    Returns (q, s, p) arrays with one row per query."""
    order = np.lexsort((-p, qi))
    q2, s2, p2 = qi[order], si[order], p[order]
    first = np.r_[True, q2[1:] != q2[:-1]]
    return q2[first], s2[first], p2[first]


# ----------------------------------------------------------------------------- metric
def macro_f05(pred_sets, truth_sets, fp_weight=None):
    """pred_sets / truth_sets: dict s1 -> set(ids); fp_weight: dict id -> weight of a false match."""
    tot = 0.0
    for s1, truth in truth_sets.items():
        pred = pred_sets.get(s1, ())
        if not truth:
            tot += 1.0 if not pred else 0.0
            continue
        if not pred:
            continue
        tp = len(truth & pred)
        if tp == 0:
            continue
        fp = sum(fp_weight.get(x, 1.0) if fp_weight else 1.0 for x in pred - truth)
        prec, rec = tp / (tp + fp), tp / len(truth)
        tot += 1.25 * prec * rec / (0.25 * prec + rec)
    return tot / max(len(truth_sets), 1)


# ----------------------------------------------------------------------------- pair building
def build_pairs(s1, q, k_block, keep_k, rel, workers, tag):
    """Blocks every country present in q against s1 of the same country.
    Returns DataFrame-free arrays with GLOBAL row indices into s1 / q."""
    QI, SI, SIM, RK = [], [], [], []
    for country, qc in q.groupby("country", sort=False):
        s1_idx = np.flatnonzero(s1["country"].to_numpy() == country)
        if len(s1_idx) == 0:
            continue
        q_idx = qc.index.to_numpy()
        t = time.time()
        qi, si, sim, rk = block_country(s1.iloc[s1_idx], qc, k_block, workers)
        qi, si, sim, rk = prune(qi, si, sim, rk, keep_k, rel)
        QI.append(q_idx[qi]); SI.append(s1_idx[si]); SIM.append(sim); RK.append(rk)
        log(f"{tag} block {country}: S1={len(s1_idx)} Q={len(q_idx)} pairs={len(qi)} ({time.time() - t:.0f}s)")
        gc.collect()
    return (np.concatenate(QI), np.concatenate(SI), np.concatenate(SIM), np.concatenate(RK))


def featurize(s1, q, qi, si, sim, rk, workers, chunk=1_500_000):
    Qd, Sd = as_dict(q), as_dict(s1)
    is_s3 = q["entity_id"].str.startswith("S3-").to_numpy().astype(np.int8)
    out = []
    for b in range(0, len(qi), chunk):
        # keep whole query groups inside a chunk (pairs are sorted by query)
        out.append(pair_features(Qd, Sd, qi[b:b + chunk], si[b:b + chunk], sim[b:b + chunk],
                                 rk[b:b + chunk], is_s3, workers))
    return np.vstack(out)


def sort_by_query(qi, si, sim, rk):
    o = np.lexsort((rk, qi))
    return qi[o], si[o], sim[o], rk[o]


# ----------------------------------------------------------------------------- train
def train_stage(a):
    rng = np.random.default_rng(42)
    tr = os.path.join(a.data, "train")
    log("loading ground truth")
    gt = pd.read_csv(os.path.join(tr, "train_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False)
    ex = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    ex = ex[ex["m"].notna() & (ex["m"] != "")]
    gk = id_num(ex["m"].to_numpy()); gv = id_num(ex["source1_entity_id"].to_numpy())
    o = np.argsort(gk); gk, gv = gk[o], gv[o]
    del ex, gt; gc.collect()
    log(f"GT: {len(gk)} links")

    def owner_of(ids):
        k = id_num(ids)
        pos = np.clip(np.searchsorted(gk, k), 0, len(gk) - 1)
        return np.where(gk[pos] == k, gv[pos], -1)

    s1 = read_norm(os.path.join(tr, "train_source1.tsv"), os.path.join(a.work, "train_s1.parquet"))
    s1_num = id_num(s1["entity_id"].to_numpy())
    val_mask = rng.random(len(s1)) < a.val_frac
    val_nums = np.sort(s1_num[val_mask])
    qs = []
    for src in ("train_source2.tsv", "train_source3.tsv"):
        ids = pd.read_csv(os.path.join(tr, src), sep="\t", dtype=str, usecols=["entity_id"],
                          quoting=3)["entity_id"].to_numpy()
        own = owner_of(ids)
        is_val_owned = np.isin(own, val_nums)
        keep = is_val_owned | (rng.random(len(ids)) < a.rho)
        kept = set(ids[keep])
        full_cache = os.path.join(a.work, src.replace(".tsv", ".parquet"))
        cache = full_cache if os.path.exists(full_cache) else os.path.join(a.work, src.replace(".tsv", "_sub.parquet"))
        d = read_norm(os.path.join(tr, src), cache, keep_ids=kept, workers=a.workers)
        o2 = owner_of(d["entity_id"].to_numpy())
        d["owner"] = o2
        d["val_owned"] = np.isin(o2, val_nums)
        qs.append(d)
        del ids, own, kept; gc.collect()
        log(f"{src}: kept {len(d)} queries")
    q = pd.concat(qs, ignore_index=True)
    del qs; gc.collect()

    qi, si, sim, rk = build_pairs(s1, q, a.k_block, a.keep_k, a.rel, a.workers, "train")
    qi, si, sim, rk = sort_by_query(qi, si, sim, rk)
    q_owner = q["owner"].to_numpy()
    y = (q_owner[qi] == s1_num[si]).astype(np.int8)
    owned = q_owner != -1
    found = np.zeros(len(q), bool); found[qi[y == 1]] = True
    log(f"blocking recall (queries with owner): {found[owned].mean():.4f}; pairs={len(qi)}; pos={y.sum()}")

    t = time.time()
    F = featurize(s1, q, qi, si, sim, rk, a.workers)
    log(f"features {F.shape} in {time.time() - t:.0f}s")
    q_val = q["val_owned"].to_numpy()
    # training rows: queries not owned by validation entities and whose candidate list has no val S1
    s1_isval = val_mask
    involved = np.zeros(len(q), bool)
    involved[qi[q_val[qi] | s1_isval[si]]] = True
    pair_val = involved[qi]          # all pairs of queries touching validation
    trn = ~pair_val
    np.save(os.path.join(a.work, "F_train.npy"), F)

    import lightgbm as lgb
    log(f"LightGBM on {trn.sum()} pairs")
    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=a.workers, verbose=-1)
    idx_tr = np.flatnonzero(trn)
    # small early-stopping slice
    es = rng.random(len(idx_tr)) < 0.05
    dtr = lgb.Dataset(F[idx_tr[~es]], y[idx_tr[~es]], feature_name=FEATURES, free_raw_data=True)
    des = lgb.Dataset(F[idx_tr[es]], y[idx_tr[es]], reference=dtr)
    gbm = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[des],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])
    log("MLP")
    mlp = MLP(F.shape[1], epochs=a.mlp_epochs).fit(_mlp_x(F[idx_tr]), y[idx_tr])

    # ---------------- validation (macro F0.5 on held-out S1 entities)
    vm = pair_val
    p_gbm = gbm.predict(F[vm], num_threads=a.workers)
    p_mlp = mlp.predict(_mlp_x(F[vm]))
    q_ids = q["entity_id"].to_numpy()
    s1_ids = s1_num
    truth = {}
    for e in s1_ids[val_mask]:
        truth[e] = set()
    vq = np.flatnonzero(q_val)
    for qid, o in zip(q_ids[vq], q_owner[vq]):
        truth[o].add(qid)
    # FP weights: queries sampled only through the random rho-sample represent 1/rho queries
    fpw = {qid: (1.0 if v else 1.0 / a.rho) for qid, v in zip(q_ids, q_val)}
    countries = s1["country"].to_numpy()
    res = {}
    best = (-1, None, None)
    for name, p in [("lgbm", p_gbm), ("mlp", p_mlp), ("blend", 0.5 * p_gbm + 0.5 * p_mlp)]:
        aq, as_, ap = assign(qi[vm], si[vm], p)
        keep_val = s1_isval[as_]
        aq, as_, ap = aq[keep_val], as_[keep_val], ap[keep_val]
        for t_ in np.arange(0.2, 0.96, 0.05):
            m = ap >= t_
            pred = {}
            for qq, ss in zip(q_ids[aq[m]], s1_ids[as_[m]]):
                pred.setdefault(ss, set()).add(qq)
            f = macro_f05(pred, truth, fpw)
            res[f"{name}@{t_:.2f}"] = f
            if f > best[0]:
                best = (f, name, float(t_))
        log(f"{name}: best so far {best}")
    # per-country threshold for the chosen model
    f_all, model_name, thr = best
    p = {"lgbm": p_gbm, "mlp": p_mlp, "blend": 0.5 * p_gbm + 0.5 * p_mlp}[model_name]
    aq, as_, ap = assign(qi[vm], si[vm], p)
    keep_val = s1_isval[as_]
    aq, as_, ap = aq[keep_val], as_[keep_val], ap[keep_val]
    thresholds = {"_global": thr}
    for c in np.unique(countries[val_mask]):
        tc = {e: v for e, v in truth.items()}
        ents = set(s1_ids[val_mask & (countries == c)])
        tc = {e: v for e, v in truth.items() if e in ents}
        bc = (-1, thr)
        cm = countries[as_] == c
        for t_ in np.arange(0.2, 0.96, 0.025):
            m = cm & (ap >= t_)
            pred = {}
            for qq, ss in zip(q_ids[aq[m]], s1_ids[as_[m]]):
                pred.setdefault(ss, set()).add(qq)
            f = macro_f05(pred, tc, fpw)
            if f > bc[0]:
                bc = (f, float(t_))
        thresholds[c] = bc[1]
        log(f"country {c}: F0.5={bc[0]:.4f} at t={bc[1]:.3f} ({len(tc)} entities)")
    # overall with per-country thresholds
    tq = np.array([thresholds.get(c, thr) for c in countries[as_]])
    m = ap >= tq
    pred = {}
    for qq, ss in zip(q_ids[aq[m]], s1_ids[as_[m]]):
        pred.setdefault(ss, set()).add(qq)
    f_final = macro_f05(pred, truth, fpw)
    log(f"VALIDATION macro F0.5 = {f_final:.4f} (model={model_name}, thresholds={thresholds})")
    meta = dict(model=model_name, thresholds=thresholds, val_f05=f_final, sweep=res,
                args=vars(a), blocking_recall=float(found[owned].mean()))
    json.dump(meta, open(os.path.join(a.work, "meta.json"), "w"), indent=1)
    gbm.save_model(os.path.join(a.work, "lgbm.txt"))
    pickle.dump(mlp, open(os.path.join(a.work, "mlp.pkl"), "wb"))


def _mlp_x(F):
    X = F.copy()
    for c in ("len_nc_q", "len_nc_s", "len_a_q", "len_a_s", "q_ncand", "blk_rank"):
        j = FEATURES.index(c)
        X[:, j] = np.log1p(X[:, j])
    return X


# ----------------------------------------------------------------------------- infer
def infer_stage(a):
    import lightgbm as lgb
    te = os.path.join(a.data, "test")
    meta = json.load(open(os.path.join(a.work, "meta.json")))
    gbm = lgb.Booster(model_file=os.path.join(a.work, "lgbm.txt"))
    mlp = pickle.load(open(os.path.join(a.work, "mlp.pkl"), "rb"))
    model, thr = meta["model"], meta["thresholds"]
    s1 = read_norm(os.path.join(te, "test_source1.tsv"), os.path.join(a.work, "test_s1.parquet"))
    s1_ids = s1["entity_id"].to_numpy()
    countries = s1["country"].to_numpy()
    cand_s, cand_q, match_s, match_q = [], [], [], []
    import pyarrow.parquet as pq
    for src in ("test_source2.tsv", "test_source3.tsv"):
        qcache = os.path.join(a.work, src.replace(".tsv", ".parquet"))
        if not os.path.exists(qcache):
            read_norm(os.path.join(te, src), qcache)
            gc.collect()
        pf = pq.ParquetFile(qcache)
        ngroups = pf.num_row_groups
        old_file = os.path.join(a.work, f"scores_{src}.npz")
        parts = [None] if os.path.exists(old_file) else list(range(0, ngroups, a.groups_per_part))
        for part in parts:
            if part is None:
                score_file = old_file
                groups = list(range(ngroups))
            else:
                score_file = os.path.join(a.work, f"scores_{src}_{part}.npz")
                groups = list(range(part, min(part + a.groups_per_part, ngroups)))
            if os.path.exists(score_file):
                z = np.load(score_file)
                qi, si, P = z["qi"], z["si"], z["p"]
                q_num = id_num(pf.read_row_groups(groups, columns=["entity_id"]).column(0).to_pylist())
                log(f"{src} part {part}: resumed {len(qi)} scored pairs")
            else:
                q = pf.read_row_groups(groups).to_pandas()
                q["nonascii"] = q["nonascii"].astype(np.int8)
                q_num = id_num(q["entity_id"].to_numpy())
                qi, si, sim, rk = build_pairs(s1, q, a.k_block, a.keep_k, a.rel, a.workers, f"{src}[{part}]")
                qi, si, sim, rk = sort_by_query(qi, si, sim, rk)
                P = np.zeros(len(qi), np.float32)
                step = 2_000_000
                for b in range(0, len(qi), step):
                    e = min(b + step, len(qi))
                    F = featurize(s1, q, qi[b:e], si[b:e], sim[b:e], rk[b:e], a.workers)
                    pg = gbm.predict(F, num_threads=a.workers) if model in ("lgbm", "blend") else 0
                    pm = mlp.predict(_mlp_x(F)) if model in ("mlp", "blend") else 0
                    P[b:e] = pg if model == "lgbm" else (pm if model == "mlp" else 0.5 * pg + 0.5 * pm)
                    log(f"{src}[{part}]: scored {e}/{len(qi)}")
                    del F; gc.collect()
                np.savez(score_file, qi=qi, si=si, p=P)
                del q, sim, rk; gc.collect()
            cand_s.append(si.astype(np.int32)); cand_q.append(q_num[qi])
            aq, as_, ap = assign(qi, si, P)
            tq = np.array([thr.get(c, thr["_global"]) for c in countries[as_]], dtype=np.float32)
            m = ap >= tq
            match_s.append(as_[m].astype(np.int32)); match_q.append(q_num[aq[m]])
            log(f"{src} part {part}: {m.sum()} matches assigned")
            del qi, si, P, aq, as_, ap, q_num; gc.collect()
    os.makedirs(a.out, exist_ok=True)
    n_c = write_lists(os.path.join(a.out, "candidate_pairs.tsv"), "candidate_entity_ids",
                      s1_ids, np.concatenate(cand_s), np.concatenate(cand_q), sim_order=None)
    n_m = write_lists(os.path.join(a.out, "matching_results.tsv"), "matched_entity_ids",
                      s1_ids, np.concatenate(match_s), np.concatenate(match_q), sim_order=None)
    log(f"wrote outputs: {len(s1_ids)} S1 rows, {n_c} candidate pairs, {n_m} matches")


def num_to_id(n):
    return f"S{n // 10_000_000_000}-{n % 10_000_000_000}"


def write_lists(path, col, s1_ids, s_idx, q_num, sim_order=None):
    o = np.lexsort((q_num, s_idx))
    s_idx, q_num = s_idx[o], q_num[o]
    bounds = np.searchsorted(s_idx, np.arange(len(s1_ids) + 1))
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for i, e in enumerate(s1_ids):
            vals = np.unique(q_num[bounds[i]:bounds[i + 1]])
            n += len(vals)
            f.write(e + "\t" + ",".join(num_to_id(int(v)) for v in vals) + "\n")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing train/ and test/")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--stage", default="all", choices=["all", "train", "infer"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--rho", type=float, default=0.10)
    ap.add_argument("--k_block", type=int, default=10)
    ap.add_argument("--keep_k", type=int, default=6)
    ap.add_argument("--rel", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--mlp_epochs", type=int, default=1)
    ap.add_argument("--groups_per_part", type=int, default=7, help="parquet row groups (200k rows) per inference slice")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.stage in ("all", "train"):
        train_stage(a)
    if a.stage in ("all", "infer"):
        infer_stage(a)


if __name__ == "__main__":
    main()
