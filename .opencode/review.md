# Code Review Summary

**Scope**: Sub-Task 9 — Cross-Reference Resolution  
**Overall risk**: Medium  
**Verdict**: Approve with comments  

---

## Findings

### [P1] High — Broken `__main__` block in `caption_extractor.py`

- **Location**: `src/utils/caption_extractor.py:298-309`
- **Why it matters**: Running `python src/utils/caption_extractor.py "Figure 1: Test"` prints `"f: i"` instead of `"figure: 1"`. The standalone CLI is completely broken by the return-type change.
- **Evidence**: `extract_caption_label` now returns a single string (e.g. `"figure 1"`), but the `__main__` block indexes into it as if it were a 2-tuple: `f"{result[0]}: {result[1]}"` produces `"f: i"`.
- **Fix**: Either restore tuple return and add a separate `extract_caption_label_str` wrapper, or update the `__main__` block to print the string directly. Also remove the dead commented-out test lines at 309–310.

---

### [P1] High — `find_all_caption_refs` may produce false-positive matches

- **Location**: `src/utils/caption_extractor.py:158-168` (`_FIND_RE` pattern)
- **Why it matters**: `_FIND_RE` starts with `\s*` (zero or more whitespace) and lacks a leading word-boundary `\b`. The regex can match short type names (e.g. `"fig"`, `"box"`, `"code"`, `"map"`) **mid-word** because `\s*` matches zero characters. This means text like `"reconfig1"` would produce a spurious `"figure 1"` match, and `"somecode 5"` could produce `"listing 5"`.
- **Evidence** (demonstration):
  - `"reconfig1"` — `fig` matches the `"fig"` alternation, `\s*\.?\s*` matches zero chars, `1` matches `\d+` → produces `"figure 1"`.
  - `"The box1 shipment arrived"` — `box` matches `"box"`, zero separators, `1` matches `\d+` → produces `"box 1"`.
- **Fix**: Add `\b` before the type alternation in `_FIND_RE`:
  ```python
  _FIND_RE = re.compile(
      rf"\b(?P<type>{_TYPE_ALTERNATION})"
      rf"\s*\.?\s*"
      ...
  )
  ```
  The anchored `_CAPTION_RE` does not need this because `^` + `\s*` already constrains it to the start-of-string.

---

### [P2] Medium — Duplicate `_SEPS` / `_SEP_RE` definitions

- **Location**: `src/utils/caption_extractor.py:107-117` and `122-132`
- **Why it matters**: Defines `_SEPS` and `_SEP_RE` twice identically. The second definition silently overwrites the first. While the values are identical (no functional bug), duplicate code is a maintenance hazard — someone updating only one copy would introduce a subtle divergence. Also creates a redundant section comment `# Separators` (line 122) without a matching heading above it.
- **Evidence**: Lines 107–117 and 122–132 are byte-for-byte identical.
- **Fix**: Delete lines 120–132 (the duplicate block + stale section comment).

---

### [P2] Medium — `_get_caption_for_item` duplicated across two modules

- **Location**: `src/chunking/visual_enricher.py:51-64` and `src/chunking/cross_reference_resolver.py:32-45`
- **Why it matters**: The exact same 14-line function is copy-pasted. Any bug fix or enhancement to caption extraction logic must be applied in two places, risking drift.
- **Evidence**: Both functions have identical logic (try `captions[0].cref`, fallback to `children[0].cref`, resolve via `text_lookup`).
- **Fix**: Extract into a shared utility, e.g. `src/chunking/_caption_utils.py`, and import it from both modules.

---

### [P2] Medium — `caption_number` field stores full label, not just the number

- **Location**: `src/chunking/models.py:163-170` (field docstring) vs usage in `docling_chunker.py:238-241` and `visual_enricher.py:253-255`
- **Why it matters**: The field documentation says `caption_number` stores "Extracted figure/table numbers from captions" with examples `['3.2', '1a', 'IV']`, implying bare numbers. However, both `docling_chunker.py` and `visual_enricher.py` now store the **full normalized label** (e.g. `"figure 3.2"`, `"figure 1a"`, `"table IV"`) via `extract_caption_label`. Any downstream code doing exact-match retrieval on this field (as the docstring advertises) would fail if it expects bare numbers.
- **Evidence**: `docling_chunker.py:238-241` does `cap_num = result` where `result` is `"figure 3.2"` (a string), and stores it in `caption_numbers` list. Previously this stored `"fig. 3.2"` — the label was always there, but the docstring was misleading. The normalization change makes this inconsistency more prominent.
- **Fix**: Either:
  - **(a)** Update the field docstring to accurately reflect that it stores the full normalized label, or
  - **(b)** Split `extract_caption_label` to return `(canonical_type, number)` and store only the number in `caption_number`, keeping `caption_text` for the full caption. The cross-reference resolver does not depend on `caption_number` at all (it builds its index from `pic_table_lookup` directly), so this would not break the resolver.

---

### [P3] Low — Dead code: dot-fallback in `extract_caption_label`

- **Location**: `src/utils/caption_extractor.py:206-213`
- **Why it matters**: The fallback logic that re-tries with `raw_type + "."` appended will never execute. The regex `_CAPTION_RE` is built from `_SORTED_TYPES`, which is derived from `VALID_IMAGE_TYPES.keys()`. Every string the regex can match as a `type` group is guaranteed to be a key in `VALID_IMAGE_TYPES` (after `.lower()`). The dot-fallback duplicates this guarantee for no benefit.
- **Evidence**: The regex alternation is `"|".join(re.escape(t) for t in _SORTED_TYPES)` where `_SORTED_TYPES = sorted(VALID_IMAGE_TYPES.keys(), ...)`. After `m.group("type").lower()`, the result is always a key of `VALID_IMAGE_TYPES`. The `canonical is None` branch is unreachable.
- **Fix**: Simplify to:
  ```python
  raw_type = m.group("type").lower()
  raw_number = _normalise_number(m.group("number"))
  canonical = VALID_IMAGE_TYPES.get(raw_type)
  if canonical is None:
      return None
  return f"{canonical} {raw_number}"
  ```
  Apply the same simplification to `find_all_caption_refs` (lines 254–259).

---

### [P3] Low — Missing direct unit tests for `find_all_caption_refs`

- **Location**: `tests/test_caption_extractor.py` (has no `find_all_caption_refs` tests)
- **Why it matters**: `find_all_caption_refs` is only tested indirectly through the cross-reference resolver tests, which use simple, short input texts. Edge cases like deduplication of duplicate labels, text with no matches, multiple matches of varying types, and the false-positive scenarios described above are never directly validated.
- **Evidence**: The test file only imports and tests `extract_caption_label`. There are zero parametrized test cases for `find_all_caption_refs`.
- **Fix**: Add a `TestFindAllCaptionRefs` class covering:
  - Empty text → `[]`
  - Text with no references → `[]`
  - Single reference → `["figure 1"]`
  - Multiple distinct references → `["figure 1", "table II"]`
  - Duplicate references (same label) → deduplicated to one entry
  - Abbreviation normalization (`"Fig. 3"` and `"Figure 3"` both → `"figure 3"`)
  - References in context (e.g. `"...as shown in Fig. 1, the results in Table II confirm..."`)

---

### [P3] Low — `_build_caption_index` silently overwrites duplicate labels

- **Location**: `src/chunking/cross_reference_resolver.py:82-84`
- **Why it matters**: If two different visual elements in the same document share the same canonical label (e.g. two unrelated "Figure 1" captions from different document sections), the index entry silently overwrites, losing the earlier reference. While this is acknowledged in the comment on line 82-83, a warning log would help diagnose unexpected resolution failures.
- **Evidence**: Line 84: `index[label] = chunk_id` — no check if `label` already exists.
- **Fix**: Add a debug or info log when overwriting:
  ```python
  if label in index and index[label] != chunk_id:
      _log.debug("Caption label '%s' overwritten: %s -> %s", label, index[label], chunk_id)
  index[label] = chunk_id
  ```

---

## Items Verified as Correct

| Concern | Assessment |
|---|---|
| `"Fig. 3"` → `"figure 3"` normalization | ✅ Correct — `_CAPTION_RE` matches `"Fig."`, `.lower()` → `"fig."`, `VALID_IMAGE_TYPES["fig."]` → `"figure"`, `_normalise_number("3")` → `"3"`, format → `"figure 3"`. |
| `model_copy` immutability | ✅ Correct — `ChunkMetadata` is `frozen=True`, `model_copy(update={...})` creates a new instance without mutating the original. Tests confirm original is unchanged. |
| Self-reference filtering | ✅ Correct — `target_id != cm.chunk_id` on line 138, tested explicitly in `test_self_reference_filtered`. |
| Deduplication | ✅ Correct — `set` for `target_ids`, `set` for `seen` labels, `sorted()` for deterministic output. |
| Empty `pic_table_lookup` early return | ✅ Correct — line 122-124 returns original chunks unmodified. |
| `_build_caption_index` skips items without caption text | ✅ Correct — lines 71-72, tested in `test_missing_caption_skipped`. |
| `_build_caption_index` skips non-caption text | ✅ Correct — `extract_caption_label` returns `None` for non-captions, tested in `test_label_not_in_caption_extractor_skipped`. |
| `_build_caption_index` skips orphan pics | ✅ Correct — `ref_to_chunk.get(ref)` is `None` for pics not in any chunk, tested in `test_pic_not_in_any_chunk_skipped`. |
| Imports/exports in `__init__.py` | ✅ Correct — `resolve_cross_references` imported and exported properly. |

---

## Suggested Next Steps

- [ ] Fix the `__main__` block in `caption_extractor.py` (P1)
- [ ] Add `\b` word boundary to `_FIND_RE` to prevent false positives (P1)
- [ ] Remove duplicate `_SEPS`/`_SEP_RE` block (P2)
- [ ] Extract shared `_get_caption_for_item` into a common utility module (P2)
- [ ] Update `caption_number` field docstring or refactor to store bare numbers (P2)
- [ ] Remove dead dot-fallback code in `extract_caption_label` and `find_all_caption_refs` (P3)
- [ ] Add direct unit tests for `find_all_caption_refs` (P3)
- [ ] Add debug log on duplicate label overwrite in `_build_caption_index` (P3)
- [ ] Re-run the full test suite (550+ tests) after fixes to confirm no regressions
