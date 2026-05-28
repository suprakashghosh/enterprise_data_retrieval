# Code Review Summary

**Scope**: Sub-Task 11 — Populate `relates_to` via Top-k Nearest Neighbors (`src/retrieval/similarity.py`, `tests/test_similarity.py`, `src/retrieval/__init__.py`)
**Overall risk**: Low
**Verdict**: Approve — no blocking issues; a few minor improvements recommended

---

The implementation is clean, well-structured, and correctly uses `argpartition` for an
O(n) partial sort per row to avoid full O(n log n) sorting. The sibling mask,
self-exclusion, threshold filtering, and immutability contract are all correctly
implemented. Test coverage (20/20 passing) is solid, covering the main paths
including edge cases.

---

## Findings

### [P2] Medium — Redundant second `min(k_search, n)` clamp

- **Location**: `src/retrieval/similarity.py:155-156`
- **Why it matters**: Line 155 already clamps to `n`:

  ```python
  k_search = min(top_k + 1 + (sibling_mask.sum(axis=1).max().item() or 0), n)
  k_search = min(k_search, n)  # can't ask for more than n
  ```

  The second `min(k_search, n)` on line 156 can never change the value —
  `k_search` is already guaranteed to be ≤ `n` from line 155.  The comment
  "can't ask for more than n" accurately describes the intent but the guard
  itself is dead code.  The subsequent `kth=min(k_search, n - 1)` on line 162
  is a separate, correct guard against passing `kth=n` (which would be
  out-of-bounds for a 0-indexed array).

- **Fix**: Remove line 156.  Keep the `kth=min(k_search, n - 1)` guard on
  line 162 as-is — that one is necessary.

---

### [P2] Medium — Dead `or 0` after `.item()` in `k_search` computation

- **Location**: `src/retrieval/similarity.py:155`
- **Why it matters**:

  ```python
  (sibling_mask.sum(axis=1).max().item() or 0)
  ```

  `sibling_mask.sum(axis=1).max()` always returns a `numpy.int64` scalar (the
  early return at line 137 guarantees `n ≥ 2` when this line is reached).
  `.item()` always converts it to a Python `int`.  Since the only falsy
  Python `int` is `0`, and `0 or 0` evaluates to `0`, the `or 0` is an
  identity operation — it can never change the value.

  While harmless, it signals either a misunderstanding of the numpy API or
  a leftover from a previous implementation.  Both are misleading to future
  readers.

- **Fix**: Remove `or 0`, leaving:

  ```python
  k_search = min(top_k + 1 + sibling_mask.sum(axis=1).max().item(), n)
  ```

---

### [P3] Low — Missing test for zero-norm embedding vectors

- **Location**: `src/retrieval/similarity.py:57-60`
- **Why it matters**: The code explicitly guards against zero-norm rows:

  ```python
  norms = np.where(norms == 0, 1.0, norms)
  ```

  This replaces a zero-norm vector with a unit-length zero vector, producing
  cosine similarity 0 with any other vector.  While the behavior is defensible
  (better than a division-by-zero or NaN), it has no test coverage.  If a
  future refactor removes this guard, zero-norm vectors would produce NaN or
  Inf values that silently propagate through `argpartition`.

- **Fix**: Add a test:

  ```python
  def test_zero_norm_vector(self):
      emb = np.array([[0.0, 0.0], [1.0, 0.0]])
      sim = compute_cosine_similarity_matrix(emb)
      np.testing.assert_allclose(sim[0, 1], 0.0)  # zero vector treated as 0-sim
      np.testing.assert_allclose(sim[0, 0], 0.0)  # zero vs zero → 0
  ```

---

### [P3] Low — No shape validation for 1D input in `compute_cosine_similarity_matrix`

- **Location**: `src/retrieval/similarity.py:56-57`
- **Why it matters**: The function's docstring specifies `(n, d)` shape, but if
  a caller passes a 1D array (e.g., `np.array([1.0, 2.0, 3.0])`), `np.linalg.norm`
  with `axis=1` on a 1D array will raise:

  ```
  numpy.exceptions.AxisError: axis 1 is out of bounds
  ```

  The error message is confusing — it doesn't tell the caller they passed a 1D
  array when a 2D array was expected.  The `embeddings.size == 0` guard on
  line 52 would not catch this (a 1D array of length 3 has `.size == 3`).

- **Fix**: Add an early guard:

  ```python
  if embeddings.ndim != 2:
      raise ValueError(
          f"Expected 2D (n, d) array, got {embeddings.ndim}D shape {embeddings.shape}."
      )
  ```

---

### [P3] Low — Scalability ceiling for large document sets

- **Location**: `src/retrieval/similarity.py:64, 81`
- **Why it matters**: Both the similarity matrix (float64 `n×n`) and the sibling
  mask (bool `n×n`) are fully materialized in memory.  For a corpus of ~70,000
  chunks (a typical large PDF collection), the similarity matrix alone is
  ~39 GB (70k² × 8 bytes), plus ~5 GB for the bool mask.

  This is fine for the intended "hundreds to low thousands" scale implied by
  the pure‑numpy design, but large-scale production users should be aware of
  the limit.  The module docstring explicitly calls out "Pure numpy — no
  HDBSCAN, no sklearn clustering" which sets appropriate expectations.

- **Fix**: No code change needed for now.  If scalability becomes a requirement
  later, consider batched FAISS or hnswlib approximate nearest‑neighbor search
  with an on‑the‑fly sibling check (avoiding the `n×n` matrix entirely).

---

### [P3] Low — `einsum` vs. direct matrix multiply

- **Location**: `src/retrieval/similarity.py:64`
- **Why it matters**: `np.einsum("ij,kj->ik", vecs, vecs, optimize=True)` is
  semantically equivalent to `vecs @ vecs.T`.  The `einsum` path with
  `optimize=True` invokes a cost‑based optimizer that typically delegates to
  BLAS for this contraction, but it adds a small dispatch overhead.  The
  comment says "cache-friendly via einsum" — however, BLAS `@` is highly
  optimized for cache locality on this exact operation.

  This is not a bug; both produce identical results.  But `vecs @ vecs.T` is:
  - More idiomatic for matrix multiplication
  - Guaranteed to hit BLAS (no optimizer dispatch)
  - More readable to anyone familiar with numpy

- **Fix**: Consider replacing with `vecs @ vecs.T` for clarity.  Not required.

---

## What the implementation gets right

The argpartition logic is **correct**.  Here is the verification:

1. **Buffer sizing**: `k_search = top_k + 1 + max_sibling_count`.  After
   filtering out self (1) and up to `max_sibling_count` siblings, at least
   `top_k` candidates remain — exactly what we need.

2. **argpartition correctness**: `np.argpartition(-sim, kth=k_search-1, axis=1)`
   guarantees that the first `k_search` indices per row correspond to the
   `k_search` largest similarities.  Because we request `k_search` with
   sufficient buffer, no valid candidate can "leak" past position `k_search-1`
   (candidates beyond that position have similarity ≤ the element at that
   position, and we already have ≥ `top_k` survivors with similarity ≥ that
   threshold).

3. **Sort within filter window**: After collecting unfiltered candidates, the
   code sorts by descending score — compensating for the fact that argpartition
   does not sort the first `k_search` elements.  This is efficient (sorting
   ~`k_search` items instead of `n`).

4. **Sibling mask symmetry**: `_build_sibling_mask` correctly sets both
   `mask[i,j]` and `mask[j,i]`, ensuring symmetric exclusion.

5. **Immutability**: `model_copy(update={"relates_to": top_ids})` produces new
   instances without mutating originals — consistent with the frozen model
   design.

6. **Test coverage**: 20 tests covering identity, orthogonal, opposite,
   auto‑normalization, empty/single‑vector, no‑sibling, shared‑ref, mixed‑ref,
   empty‑ref, empty‑chunks, single‑chunk, top‑k neighbors, self‑exclusion,
   sibling‑exclusion, threshold, determinism, immutability, shape mismatch, and
   sorted‑descending.

---

## Suggested Next Steps

- [ ] Remove redundant `min(k_search, n)` on line 156.
- [ ] Remove dead `or 0` on line 155.
- [ ] Add test for zero-norm embedding vectors.
- [ ] Optionally add 2D shape validation in `compute_cosine_similarity_matrix`.
- [ ] No blockers — safe to merge.
