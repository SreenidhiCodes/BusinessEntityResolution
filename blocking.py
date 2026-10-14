"""Candidate generation (blocking).

Reverse-direction retrieval: every Source-2/3 record ("query") retrieves its
top-K most similar Source-1 records within the same country, using a sparse
TF-IDF vector over
  * consonant-skeleton tokens of the core business name (legal suffixes removed),
  * name-skeleton token bigrams,
  * address tokens (street numbers, street / locality / city words, postal code).
Very frequent tokens (document frequency above `df_max` in Source 1) are dropped
from the retrieval index only - they carry little identity signal and dominate
the cost of the sparse product.

Because the ground truth shows every S2/S3 record belongs to at most one S1
entity, retrieving the best S1 records per query is the natural direction; the
S1-side candidate list (candidate_pairs.tsv) is the inversion of these lists.
"""
import numpy as np
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from normalize import ADDR_STOP


def record_tokens(skel, addr):
    nt = skel.split()
    toks = ["n" + t for t in nt if len(t) > 1 or t.isdigit()]
    toks += ["b" + "_".join(sorted((a, b))) for a, b in zip(nt, nt[1:])] if BIGRAM_SORTED else ["b" + a + "_" + b for a, b in zip(nt, nt[1:])]
    toks += ["a" + t for t in addr.split() if t not in ADDR_STOP and (len(t) > 1 or t.isdigit())]
    return toks


BIGRAM_SORTED = True
W = {"n": 1.0, "b": 0.5, "a": 1.0}


def _weights(tok):
    return W[tok[0]]


def build_index(s1_skel, s1_addr, df_max=4000):
    """Builds the S1 TF-IDF matrix. Returns (X [n1 x V] csr, vocab dict, idf array)."""
    vocab = {}
    rows, cols = [], []
    for i, (n, a) in enumerate(zip(s1_skel, s1_addr)):
        for t in set(record_tokens(n, a)):
            j = vocab.get(t)
            if j is None:
                j = len(vocab)
                vocab[t] = j
            rows.append(i)
            cols.append(j)
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    n1, V = len(s1_skel), len(vocab)
    df = np.bincount(cols, minlength=V)
    idf = np.log((n1 + 1) / (df + 1)) + 1.0
    w = np.ones(V, dtype=np.float32)
    for t, j in vocab.items():
        w[j] = _weights(t)
    colw = (idf * w).astype(np.float32)
    colw[df > df_max] = 0.0  # too common for retrieval
    X = sp.csr_matrix((colw[cols], (rows, cols)), shape=(n1, V), dtype=np.float32)
    X.eliminate_zeros()
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel()) + 1e-9
    X = sp.diags(1.0 / norms).dot(X).tocsr().astype(np.float32)
    return X, vocab, colw


def query_matrix(q_skel, q_addr, vocab, colw):
    rows, cols = [], []
    get = vocab.get
    for i, (n, a) in enumerate(zip(q_skel, q_addr)):
        for t in set(record_tokens(n, a)):
            j = get(t)
            if j is not None:
                rows.append(i)
                cols.append(j)
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    Q = sp.csr_matrix((colw[cols], (rows, cols)), shape=(len(q_skel), len(vocab)), dtype=np.float32)
    Q.eliminate_zeros()
    norms = np.sqrt(np.asarray(Q.multiply(Q).sum(axis=1)).ravel()) + 1e-9
    return sp.diags(1.0 / norms).dot(Q).tocsr().astype(np.float32)


def topk(Q, XT, k, threshold=0.05, n_threads=2, chunk=200_000):
    """Returns (q_idx, s1_idx, sim, rank) arrays for the top-k S1 per query."""
    qi, si, sv, rk = [], [], [], []
    for start in range(0, Q.shape[0], chunk):
        C = sp_matmul_topn(Q[start:start + chunk], XT, top_n=k, threshold=threshold,
                           sort=True, n_threads=n_threads).tocsr()
        counts = np.diff(C.indptr)
        q = np.repeat(np.arange(C.shape[0], dtype=np.int32), counts) + start
        r = (np.arange(C.nnz) - np.repeat(C.indptr[:-1], counts)).astype(np.int8)
        qi.append(q); si.append(C.indices.astype(np.int32)); sv.append(C.data.astype(np.float32)); rk.append(r)
    return (np.concatenate(qi), np.concatenate(si), np.concatenate(sv), np.concatenate(rk))
