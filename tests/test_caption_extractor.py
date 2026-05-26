"""Tests for :mod:`src.utils.caption_extractor`."""

from __future__ import annotations

import pytest

from src.utils.caption_extractor import extract_caption_label


# ---------------------------------------------------------------------------
# Happy path — every supported type
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        # Figure variants
        ("Figure 1: Revenue by Quarter", "figure 1"),
        ("Figure 12. Analysis of variance", "figure 12"),
        ("Fig. 3: Experimental setup", "figure 3"),
        ("Fig 5. Results overview", "figure 5"),
        ("figure 10 — Comparison", "figure 10"),
        # Table
        ("Table 2: Summary of metrics", "table 2"),
        ("Table 7. Financial highlights", "table 7"),
        # Chart
        ("Chart 4: Market share trend", "chart 4"),
        # Graph
        ("Graph 6: Response time", "graph 6"),
        # Image / picture / photo
        ("Image 1: System architecture", "image 1"),
        ("Picture 3: Team photo", "image 3"),
        ("Photo 2. Product shot", "image 2"),
        ("Photograph 5. Aerial view", "image 5"),
        # Diagram
        ("Diagram 7: Process flow", "diagram 7"),
        # Illustration
        ("Illustration 8: Anatomy", "illustration 8"),
        # Schematic / drawing
        ("Schematic 9: Circuit", "schematic 9"),
        ("Drawing 11: Blueprint", "schematic 11"),
        # Exhibit
        ("Exhibit 12: Contract clause", "exhibit 12"),
        # Appendix
        ("Appendix A: Additional Data", "appendix A"),
        ("Appendix D Supplementary material", "appendix D"),
        # Box / panel
        ("Box 1: Key takeaways", "box 1"),
        ("Panel 2. Regional breakdown", "panel 2"),
        # Algorithm
        ("Algorithm 3: Gradient descent", "algorithm 3"),
        ("Algo 1: Sort", "algorithm 1"),
        ("Alg. 2: Merge step", "algorithm 2"),
        # Equation / formula
        ("Equation 5 — where x is...", "equation 5"),
        ("Eq. 4: Euler's identity", "equation 4"),
        ("Eq 7, continued", "equation 7"),
        ("Formula 9: Newton's second law", "equation 9"),
        ("Eqn 3: Diffusion model", "equation 3"),
        # Map
        ("Map 2: Geographic distribution", "map 2"),
        # Slide
        ("Slide 10: Agenda", "slide 10"),
        # Plate
        ("Plate 4: Micrograph", "plate 4"),
        # Scheme
        ("Scheme 1: Synthetic pathway", "scheme 1"),
        # Listing / code
        ("Listing 5: Main loop", "listing 5"),
        ("Code 3: Authentication flow", "listing 3"),
    ],
)
def test_valid_captions(text, expected):
    assert extract_caption_label(text) == expected


# ---------------------------------------------------------------------------
# Hierarchical / dotted numbers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Figure 1.1: Subplot", "figure 1.1"),
        ("Figure 2.3. Analysis", "figure 2.3"),
        ("Table 3.2.1: Deeply nested", "table 3.2.1"),
        ("Fig. 1.2.3.4 Results", "figure 1.2.3.4"),
        ("Chart 2.1 — Quarterly data", "chart 2.1"),
    ],
)
def test_hierarchical_numbers(text, expected):
    assert extract_caption_label(text) == expected


# ---------------------------------------------------------------------------
# Lettered suffixes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Figure 1a: Subfigure A", "figure 1a"),
        ("Figure 2B. Alternative view", "figure 2B"),
        ("Table 3c: Detail", "table 3c"),
        ("Photo 4a: Close-up", "image 4a"),
    ],
)
def test_lettered_numbers(text, expected):
    assert extract_caption_label(text) == expected


# ---------------------------------------------------------------------------
# Roman numerals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Table I: Constants", "table I"),
        ("Table II. Derived values", "table II"),
        ("Table III: Additional", "table III"),
        ("Table IV: Summary", "table IV"),
        ("Table V: Extended", "table V"),
        ("Table VI Results", "table VI"),
        ("Table VII Analysis", "table VII"),
        ("Table VIII Overview", "table VIII"),
        ("Table IX Parameters", "table IX"),
        ("Table X Final", "table X"),
        ("Table XI Extra", "table XI"),
        ("Table XII Bonus", "table XII"),
        ("Table XIII More", "table XIII"),
        ("Table XIV Last", "table XIV"),
    ],
)
def test_roman_numerals(text, expected):
    assert extract_caption_label(text) == expected


# ---------------------------------------------------------------------------
# Various separators
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        # Colon
        ("Figure 1: Revenue", "figure 1"),
        # Period
        ("Fig. 2. Results overview", "figure 2"),
        # Semicolon
        ("Table 3; Additional data", "table 3"),
        # Em-dash / en-dash
        ("Chart 4\u2014Market analysis", "chart 4"),
        ("Chart 4\u2013Market analysis", "chart 4"),
        # Plain dash
        ("Graph 5 - Trends", "graph 5"),
        # Bullet
        ("Figure 6 \u2022 Key findings", "figure 6"),
        # Middle dot
        ("Table 7 \u00b7 Summary", "table 7"),
        # Comma
        ("Eq. 8, continued", "equation 8"),
        # Closing paren (common in legal/financial docs)
        ("Exhibit 9) Final clause", "exhibit 9"),
        # Closing bracket
        ("Figure 10] References", "figure 10"),
        # Double space (implicit separator in poorly OCR'd text)
        ("Figure 11  Description goes here", "figure 11"),
        # Space + capital letter (no explicit separator)
        ("Table 12 Financial results for Q4", "table 12"),
        # Digit directly followed by capital letter
        ("Fig. 3A", "figure 3A"),
        # No separator at all (just the label)
        ("Figure 1", "figure 1"),
        ("Table 5", "table 5"),
        # Vertical bar / pipe
        ("Chart 10 | Market data", "chart 10"),
        # Double colon
        ("Algorithm 2: : : Gradient descent", "algorithm 2"),
    ],
)
def test_various_separators(text, expected):
    assert extract_caption_label(text) == expected


# ---------------------------------------------------------------------------
# Whitespace / OCR artifacts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        # Leading whitespace
        ("   Figure 1: Test", "figure 1"),
        ("\t\tFigure 1: Test", "figure 1"),
        ("\nFigure 1: Test", "figure 1"),
        # Multiple spaces between type and number
        ("Figure    1: Test", "figure 1"),
        ("Fig.     2. Results", "figure 2"),
        # Non-breaking space
        ("Figure\u00a01: Test", "figure 1"),
        # Trailing whitespace (should be irrelevant)
        ("Figure 1: Test   ", "figure 1"),
        # Mixed newlines and spaces in prefix
        (" \n Figure \n 3: Test", "figure 3"),
    ],
)
def test_whitespace_handling(text, expected):
    assert extract_caption_label(text) == expected


# ---------------------------------------------------------------------------
# Text that is NOT a valid caption label (should return None)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Revenue grew by 12% year-over-year.",
        "The following section describes our methodology.",
        "This is just a paragraph with the word Figure in it.",
        "Figure alone without a number should not match.",
        "Table",
        "1: Just a number with no type.",
        "Table: Missing number entirely.",
        # Lowercase "word" that happens to match a type (fig is not a caption
        # when it appears mid-sentence as "fig tree")
        "The fig tree grows rapidly Fig. 2: Caption later.",
        # Python function name — should not match "fig"
        "def fig(x): return x",
        # "chart" as a verb / common word
        "We chart the progress over time.",
        # "image" used as a common noun
        "The image quality was excellent Table 1: Results.",
        # "code" used as a common word
        "The source code is available online.",
        # "box" as a common word
        "The box contains the shipment.",
        # "listing" as a verb
        "After listing the items, Table 3 shows totals.",
        # Lowercase type with no number
        "figure: revenue chart",
    ],
)
def test_non_caption_text_returns_none(text):
    assert extract_caption_label(text) is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        # "Fig." with double dot should still work (strip trailing dot)
        ("Fig. 4. Experimental setup", "figure 4"),
        # Mixed case
        ("FIGURE 5: Test", "figure 5"),
        ("fig. 6: Test", "figure 6"),
        # "picture" → "image"
        ("Picture 7: Photograph", "image 7"),
        # "photo" → "image"
        ("Photo 8: Snapshot", "image 8"),
        # "photograph" → "image"
        ("Photograph 9: Landscape", "image 9"),
        # "drawing" → "schematic"
        ("Drawing 10: Blueprint", "schematic 10"),
        # "algo" → "algorithm"
        ("Algo 11: Sorting", "algorithm 11"),
        # "alg." → "algorithm"
        ("Alg. 12: Search", "algorithm 12"),
        # "eqn" → "equation"
        ("Eqn 13: Diffusion", "equation 13"),
        # "formula" → "equation"
        ("Formula 14: Quadratic", "equation 14"),
        # "code" → "listing"
        ("Code 15: Main", "listing 15"),
        # "listing" → "listing"
        ("Listing 16: Loop", "listing 16"),
        # Type-with-dot and number-with-dot (Fig. 1.1.)
        ("Fig. 1.1. Detailed view", "figure 1.1"),
        # Zero-leading hierarchical
        ("Figure 0.1: Preface", "figure 0.1"),
        # Large numbers
        ("Table 999: Overflow test", "table 999"),
        # Dash in number (range)
        ("Figure 1-2", "figure 1-2"),
    ],
)
def test_edge_cases(text, expected):
    assert extract_caption_label(text) == expected


# ---------------------------------------------------------------------------
# Deterministic / reproducible
# ---------------------------------------------------------------------------
def test_deterministic():
    assert extract_caption_label("Figure 1: Test") == extract_caption_label(
        "Figure 1: Test"
    )


# ---------------------------------------------------------------------------
# find_all_caption_refs
# ---------------------------------------------------------------------------
class TestFindAllCaptionRefs:
    """Tests for find_all_caption_refs — scanning text for caption references."""

    def test_empty_text(self):
        from src.utils.caption_extractor import find_all_caption_refs

        assert find_all_caption_refs("") == []
        assert find_all_caption_refs(None) == []  # type: ignore

    def test_no_references(self):
        from src.utils.caption_extractor import find_all_caption_refs

        assert find_all_caption_refs("This paragraph has no references.") == []

    def test_single_reference(self):
        from src.utils.caption_extractor import find_all_caption_refs

        refs = find_all_caption_refs("Please see Figure 1 for details.")
        assert refs == ["figure 1"]

    def test_multiple_distinct_references(self):
        from src.utils.caption_extractor import find_all_caption_refs

        refs = find_all_caption_refs("See Figure 1 and Table II for reference.")
        assert "figure 1" in refs
        assert "table II" in refs
        assert len(refs) == 2

    def test_deduplicate_references(self):
        from src.utils.caption_extractor import find_all_caption_refs

        refs = find_all_caption_refs("Figure 1 is shown. Also see Figure 1 again.")
        assert refs == ["figure 1"]

    def test_abbreviation_normalization(self):
        from src.utils.caption_extractor import find_all_caption_refs

        refs = find_all_caption_refs("See Fig. 3 and Figure 3 for confirmation.")
        assert refs == ["figure 3"]  # both normalize to the same label

    def test_context_references(self):
        from src.utils.caption_extractor import find_all_caption_refs

        text = "As shown in Fig. 1, the results in Table II confirm the hypothesis."
        refs = find_all_caption_refs(text)
        assert "figure 1" in refs
        assert "table II" in refs

    def test_no_false_positive_without_number(self):
        from src.utils.caption_extractor import find_all_caption_refs

        assert find_all_caption_refs("The figure shows the architecture.") == []
        assert find_all_caption_refs("A table display is provided.") == []
