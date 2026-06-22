"""TurboQuant-compressed vector store (Step 5, similarity path).

WHY THIS EXISTS (earned by an observed failure): `gitnexus doctor` showed the
LadybugDB VECTOR index is unavailable on this Windows platform — its semantic
search is brute-force exact-scan capped at 10k chunks, which does NOT reach the
100k-file target. GitNexus still gives us exact STRUCTURE (the KG) and a usable
embedding SOURCE. So the scalable SIMILARITY index is ours to build.

TurboQuant core idea (Google DeepMind, ICLR 2026):
  1. Apply a fixed RANDOM ORTHOGONAL ROTATION to every vector. For unit vectors
     this makes the coordinates approximately i.i.d. Gaussian — so a single,
     precomputed SCALAR quantizer per coordinate is near-optimal (no per-vector
     codebook training, fully online).
  2. Quantize each rotated coordinate to b bits against Gaussian-optimal levels.
  3. Search is ASYMMETRIC: keep the query full-precision, score against the
     de-quantized database codes (cheap, compressed RAM footprint), take top-N
     candidates, then RE-RANK those N with full-precision originals (fetched from
     disk / hung off the KG node). Structure stays exact; only similarity is lossy.

Pure numpy. Deterministic (seeded). No torch, no network.
"""
import numpy as np

# Gaussian-optimal (Lloyd-Max) reconstruction levels for a unit-variance normal,
# for 1..4 bits. Precomputed so quantization is a single vectorized lookup.
_LLOYD_LEVELS = {
    1: np.array([-0.7978845608, 0.7978845608]),
    2: np.array([-1.5104, -0.4528, 0.4528, 1.5104]),
    3: np.array([-2.1519, -1.3439, -0.7560, -0.2451,
                 0.2451, 0.7560, 1.3439, 2.1519]),
    4: np.array([-2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
                 0.1284, 0.3881, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326]),
}


def _random_rotation(d, seed):
    """A fixed random orthogonal d×d matrix (QR of a Gaussian matrix)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    # Fix sign ambiguity so the rotation is deterministic across numpy builds.
    return q * np.sign(np.diag(r))


class TurboQuantIndex:
    def __init__(self, dim, bits=2, seed=1234):
        if bits not in _LLOYD_LEVELS:
            raise ValueError("bits must be one of 1,2,3,4")
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.R = _random_rotation(dim, seed)
        self.levels = _LLOYD_LEVELS[bits]
        # Rotated unit-vector coords have std ~ 1/sqrt(dim); rescale the
        # unit-variance Lloyd levels to match before quantizing.
        self.scale = 1.0 / np.sqrt(dim)
        self._codes = None        # uint8 level indices, shape (n, dim)
        self._norms = None        # original L2 norms, shape (n,)
        self._originals = None    # full-precision vectors for re-rank (prod: on disk)

    def build(self, vectors):
        v = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(v, axis=1)
        norms[norms == 0] = 1.0
        units = v / norms[:, None]
        rot = units @ self.R                      # rotate -> ~i.i.d. Gaussian coords
        idx = self._quantize(rot)
        self._codes = idx.astype(np.uint8)
        self._norms = norms
        self._originals = v
        return self

    def _quantize(self, rot):
        # Nearest Gaussian-optimal level per coordinate (vectorized).
        scaled = rot / (self.scale)
        # |scaled - level| argmin over the small level set.
        diffs = np.abs(scaled[..., None] - self.levels[None, None, :])
        return diffs.argmin(axis=-1)

    def _dequantize(self, codes):
        return self.levels[codes] * self.scale     # back to rotated space

    def search(self, query, k=10, rerank_n=64):
        q = np.asarray(query, dtype=np.float64)
        qn = np.linalg.norm(q) or 1.0
        q_rot = (q / qn) @ self.R
        # Asymmetric scoring: full-precision query vs de-quantized DB codes.
        db = self._dequantize(self._codes)
        approx = db @ q_rot                          # cosine-ish on rotated units
        cand = np.argsort(-approx)[:max(rerank_n, k)]
        # Re-rank candidates with full-precision originals (exact cosine).
        o = self._originals[cand]
        exact = (o / np.linalg.norm(o, axis=1)[:, None]) @ (q / qn)
        order = cand[np.argsort(-exact)]
        return order[:k]

    def footprint_bytes(self):
        n = 0 if self._codes is None else self._codes.shape[0]
        return int(np.ceil(n * self.dim * self.bits / 8)) + n * 8  # codes + norms


def exact_topk(vectors, query, k=10):
    v = np.asarray(vectors, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    units = v / np.linalg.norm(v, axis=1)[:, None]
    sims = units @ (q / (np.linalg.norm(q) or 1.0))
    return np.argsort(-sims)[:k]


def recall_at_k(index, vectors, queries, k=10, rerank_n=64):
    hits = 0
    for q in queries:
        gold = set(exact_topk(vectors, q, k).tolist())
        got = set(index.search(q, k=k, rerank_n=rerank_n).tolist())
        hits += len(gold & got)
    return hits / (len(queries) * k)


def _clustered_corpus(n, dim, n_clusters=64, seed=7):
    """Realistic embedding geometry: unit vectors drawn around cluster centers
    (mimics how code/text embeddings live on a sphere in tight neighborhoods)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1)[:, None]
    assign = rng.integers(0, n_clusters, size=n)
    v = centers[assign] + 0.35 * rng.standard_normal((n, dim))
    return v / np.linalg.norm(v, axis=1)[:, None]


def _selftest():
    DIM, N, NQ = 384, 5000, 200
    corpus = _clustered_corpus(N, DIM)
    rng = np.random.default_rng(99)
    queries = corpus[rng.integers(0, N, NQ)] + 0.1 * rng.standard_normal((NQ, DIM))

    raw = N * DIM * 8
    print(f"corpus: {N} vectors x {DIM} dim  (uncompressed fp64 = {raw/1e6:.1f} MB)")
    print(f"{'bits':>4} {'recall@10':>10} {'compressed':>11} {'ratio':>7}")
    for bits in (1, 2, 3, 4):
        idx = TurboQuantIndex(DIM, bits=bits).build(corpus)
        r = recall_at_k(idx, corpus, queries, k=10, rerank_n=64)
        fp = idx.footprint_bytes()
        print(f"{bits:>4} {r:>10.3f} {fp/1e6:>9.2f}MB {raw/fp:>6.1f}x")

    # 100k-file extrapolation (assume ~6 code chunks/file -> 600k vectors).
    chunks = 600_000
    for bits in (2, 3):
        per = np.ceil(DIM * bits / 8) + 8
        print(f"100k files (~{chunks} chunks) @ {bits}-bit: "
              f"{chunks*per/1e6:.0f} MB compressed "
              f"(vs {chunks*DIM*4/1e9:.1f} GB fp32 raw)")


if __name__ == "__main__":
    _selftest()
