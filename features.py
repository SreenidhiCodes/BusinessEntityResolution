"""Pairwise features for (query S2/S3 record, candidate S1 record) pairs.

All string similarities are computed with rapidfuzz.process.cpdist, which
evaluates element-wise pairs in multi-threaded C++ (no Python loop per pair).
"""
import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

FEATURES = [
    "blk_sim", "blk_rank", "blk_gap_best", "blk_gap_next", "q_ncand",
    "nf_ratio", "nf_tset", "nf_partial", "nf_jw",
    "nc_ratio", "nc_tset", "nc_tsort", "nc_partial", "nc_jw", "nc_eq",
    "ns_ratio", "ns_tset", "ns_partial", "ns_eq",
    "a_ratio", "a_tset", "a_tsort", "a_partial_tset", "a_jw",
    "d_tset", "d_eq", "d_both", "postal_eq", "postal_both",
    "len_nc_q", "len_nc_s", "len_a_q", "len_a_s", "q_nonascii", "q_is_s3",
    "a_missing_q", "nc_first_eq", "s_blk_best",
]


def _cp(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32) / 100.0


def pair_features(Q, S, qi, si, sim, rank, is_s3, workers=2):
    """Q, S: dicts of numpy object arrays (n_full, n_core, n_skel, a_norm, a_dig, postal, nonascii).
    qi/si: int arrays of pair indices into Q and S. Returns float32 matrix [n_pairs x len(FEATURES)]."""
    n = len(qi)
    F = np.zeros((n, len(FEATURES)), dtype=np.float32)
    col = {f: i for i, f in enumerate(FEATURES)}

    # blocking-context features
    F[:, col["blk_sim"]] = sim
    F[:, col["blk_rank"]] = rank
    # per-query best / next similarities (pairs are grouped by query, sorted by rank)
    order = np.lexsort((rank, qi))
    qi_s, sim_s = qi[order], sim[order]
    starts = np.r_[0, np.flatnonzero(np.diff(qi_s)) + 1]
    counts = np.diff(np.r_[starts, len(qi_s)])
    best = np.repeat(sim_s[starts], counts)
    second = np.repeat(np.where(counts > 1, sim_s[np.minimum(starts + 1, len(sim_s) - 1)], 0.0), counts)
    inv = np.empty_like(order); inv[order] = np.arange(n)
    best, second, cnt = best[inv], second[inv], np.repeat(counts, counts)[inv]
    F[:, col["blk_gap_best"]] = best - sim
    F[:, col["blk_gap_next"]] = np.where(rank == 0, sim - second, 0.0)
    F[:, col["q_ncand"]] = cnt
    # best similarity the S1 record reaches among all its queries (competition on the S1 side)
    s_best = np.zeros(si.max() + 1 if n else 1, dtype=np.float32)
    np.maximum.at(s_best, si, sim)
    F[:, col["s_blk_best"]] = s_best[si] - sim

    def g(d, key, idx):
        return d[key].take(idx).tolist()

    qa, sa = g(Q, "n_full", qi), g(S, "n_full", si)
    F[:, col["nf_ratio"]] = _cp(qa, sa, fuzz.ratio, workers)
    F[:, col["nf_tset"]] = _cp(qa, sa, fuzz.token_set_ratio, workers)
    F[:, col["nf_partial"]] = _cp(qa, sa, fuzz.partial_ratio, workers)
    F[:, col["nf_jw"]] = process.cpdist(qa, sa, scorer=JaroWinkler.normalized_similarity, workers=workers, dtype=np.float32)
    del qa, sa
    qa, sa = g(Q, "n_core", qi), g(S, "n_core", si)
    F[:, col["nc_ratio"]] = _cp(qa, sa, fuzz.ratio, workers)
    F[:, col["nc_tset"]] = _cp(qa, sa, fuzz.token_set_ratio, workers)
    F[:, col["nc_tsort"]] = _cp(qa, sa, fuzz.token_sort_ratio, workers)
    F[:, col["nc_partial"]] = _cp(qa, sa, fuzz.partial_ratio, workers)
    F[:, col["nc_jw"]] = process.cpdist(qa, sa, scorer=JaroWinkler.normalized_similarity, workers=workers, dtype=np.float32)
    F[:, col["nc_eq"]] = [a == b for a, b in zip(qa, sa)]
    F[:, col["nc_first_eq"]] = [a.split(" ", 1)[0] == b.split(" ", 1)[0] for a, b in zip(qa, sa)]
    F[:, col["len_nc_q"]] = [len(a) for a in qa]
    F[:, col["len_nc_s"]] = [len(b) for b in sa]
    del qa, sa
    qa, sa = g(Q, "n_skel", qi), g(S, "n_skel", si)
    F[:, col["ns_ratio"]] = _cp(qa, sa, fuzz.ratio, workers)
    F[:, col["ns_tset"]] = _cp(qa, sa, fuzz.token_set_ratio, workers)
    F[:, col["ns_partial"]] = _cp(qa, sa, fuzz.partial_ratio, workers)
    F[:, col["ns_eq"]] = [a == b for a, b in zip(qa, sa)]
    del qa, sa
    qa, sa = g(Q, "a_norm", qi), g(S, "a_norm", si)
    F[:, col["a_ratio"]] = _cp(qa, sa, fuzz.ratio, workers)
    F[:, col["a_tset"]] = _cp(qa, sa, fuzz.token_set_ratio, workers)
    F[:, col["a_tsort"]] = _cp(qa, sa, fuzz.token_sort_ratio, workers)
    F[:, col["a_partial_tset"]] = _cp(qa, sa, fuzz.partial_token_set_ratio, workers)
    F[:, col["a_jw"]] = process.cpdist(qa, sa, scorer=JaroWinkler.normalized_similarity, workers=workers, dtype=np.float32)
    F[:, col["len_a_q"]] = [len(a) for a in qa]
    F[:, col["len_a_s"]] = [len(b) for b in sa]
    F[:, col["a_missing_q"]] = [len(a) == 0 for a in qa]
    del qa, sa
    qa, sa = g(Q, "a_dig", qi), g(S, "a_dig", si)
    F[:, col["d_tset"]] = _cp(qa, sa, fuzz.token_set_ratio, workers)
    F[:, col["d_eq"]] = [a == b and a != "" for a, b in zip(qa, sa)]
    F[:, col["d_both"]] = [a != "" and b != "" for a, b in zip(qa, sa)]
    del qa, sa
    qp = np.asarray(Q["postal"].take(qi).tolist(), dtype=object)
    sp_ = np.asarray(S["postal"].take(si).tolist(), dtype=object)
    F[:, col["postal_both"]] = (qp != "") & (sp_ != "")
    F[:, col["postal_eq"]] = (qp == sp_) & (qp != "")
    F[:, col["q_nonascii"]] = Q["nonascii"][qi]
    F[:, col["q_is_s3"]] = is_s3[qi] if isinstance(is_s3, np.ndarray) else is_s3
    return F
