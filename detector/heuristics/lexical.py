"""
Lexical heuristics for AI text detection.

Heuristics:
  1. Type-Token Ratio (TTR / CTTR)     — AI text is repetitive
  2. Hapax Legomena Ratio              — AI uses fewer unique words
  3. Rare Word Density                 — AI sticks to common vocabulary
  4. Mean Word Length                  — AI skews toward longer, formal words
"""

import math
import re
from collections import Counter

from data.wordlists import COMMON_WORDS


# ── helpers ─────────────────────────────────────────────────────────────────

def _alpha_words(words: list[str]) -> list[str]:
    """Return only alphabetic tokens, lowercased."""
    return [w.lower() for w in words if re.match(r"^[a-zA-Z]{2,}$", w)]


# ── individual metrics ───────────────────────────────────────────────────────

def compute_ttr(words: list[str]) -> tuple[float, str]:
    """
    Corrected Type-Token Ratio (CTTR).
    CTTR = unique_tokens / sqrt(2 * total_tokens)

    AI text → lower lexical diversity → lower CTTR → higher AI score.

    Returns (score 0-100, detail_string)
    """
    alpha = _alpha_words(words)
    n = len(alpha)
    if n < 10:
        return 0.0, f"too few words ({n}) for TTR"

    unique = len(set(alpha))
    cttr = unique / math.sqrt(2 * n) if n > 0 else 0

    # Expected CTTR range for human writing: 5–15
    # AI tends to produce CTTR 2–8
    if cttr < 4:
        score = 90.0
    elif cttr < 6:
        score = 70.0
    elif cttr < 9:
        score = 45.0
    elif cttr < 12:
        score = 25.0
    else:
        score = 10.0

    return score, f"CTTR={cttr:.2f} ({unique} unique / {n} total)"


def compute_hapax_ratio(words: list[str]) -> tuple[float, str]:
    """
    Hapax Legomena Ratio = words_appearing_exactly_once / total_words.

    Human writing has MORE hapax words (more variety).
    AI writing → fewer hapax → lower ratio → higher AI score.

    Returns (score 0-100, detail_string)
    """
    alpha = _alpha_words(words)
    n = len(alpha)
    if n < 20:
        return 0.0, f"too few words ({n}) for hapax"

    freq = Counter(alpha)
    hapax_count = sum(1 for c in freq.values() if c == 1)
    ratio = hapax_count / n

    # Human: ratio typically 0.35–0.60
    # AI: ratio typically 0.15–0.35
    if ratio < 0.15:
        score = 85.0
    elif ratio < 0.25:
        score = 70.0
    elif ratio < 0.35:
        score = 50.0
    elif ratio < 0.50:
        score = 25.0
    else:
        score = 5.0

    return score, f"hapax_ratio={ratio:.3f} ({hapax_count}/{n})"


def compute_rare_word_density(words: list[str]) -> tuple[float, str]:
    """
    Rare Word Density = words NOT in common vocabulary / total words.

    AI tends to use safe, high-frequency vocabulary.
    Very low rare-word density → more AI-like.

    Returns (score 0-100, detail_string)
    """
    alpha = _alpha_words(words)
    n = len(alpha)
    if n < 20:
        return 0.0, f"too few words ({n}) for rare word density"

    rare = sum(1 for w in alpha if w not in COMMON_WORDS)
    density = rare / n

    # Human: 0.40–0.65 rare words (lots of domain-specific / unique vocab)
    # AI: 0.20–0.40 (mostly safe common words)
    if density < 0.20:
        score = 85.0
    elif density < 0.30:
        score = 65.0
    elif density < 0.40:
        score = 40.0
    elif density < 0.55:
        score = 20.0
    else:
        score = 5.0

    return score, f"rare_density={density:.3f} ({rare}/{n} rare words)"


def compute_mean_word_length(words: list[str]) -> tuple[float, str]:
    """
    Mean Word Length.

    AI writing tends toward slightly longer, more formal vocabulary.
    Target zone: 5.5–7.0 characters per word on average.

    Returns (score 0-100, detail_string)
    """
    alpha = _alpha_words(words)
    n = len(alpha)
    if n < 10:
        return 0.0, f"too few words ({n}) for mean word length"

    mean_len = sum(len(w) for w in alpha) / n

    # AI zone: 5.2 – 6.8  (polished, slightly formal)
    # Human code: often shorter due to abbreviations
    # Very long (>7.5) or very short (<3.5) = unlikely AI
    if 5.0 <= mean_len <= 7.0:
        score = 60.0 + (1 - abs(mean_len - 6.0) / 1.0) * 20
        score = min(80.0, score)
    elif mean_len < 3.5 or mean_len > 8.5:
        score = 5.0
    else:
        score = 30.0

    return score, f"mean_word_len={mean_len:.2f}"


# ── main entry ───────────────────────────────────────────────────────────────

def analyze_lexical(
    words: list[str],
    lines: list[str],
) -> dict:
    """
    Run all lexical heuristics.

    Returns:
        {
          "score": float (0–100, weighted average),
          "flagged_lines": [(line_no, text, reason), ...],
          "details": {heuristic_name: detail_str},
        }
    """
    ttr_score,    ttr_detail    = compute_ttr(words)
    hapax_score,  hapax_detail  = compute_hapax_ratio(words)
    rare_score,   rare_detail   = compute_rare_word_density(words)
    wlen_score,   wlen_detail   = compute_mean_word_length(words)

    # Weighted composite (lexical is less reliable than pattern-based)
    weights = {
        "ttr":  0.35,
        "hapax": 0.30,
        "rare":  0.25,
        "wlen":  0.10,
    }
    composite = (
        weights["ttr"]   * ttr_score  +
        weights["hapax"] * hapax_score +
        weights["rare"]  * rare_score  +
        weights["wlen"]  * wlen_score
    )

    return {
        "score": round(composite, 2),
        "flagged_lines": [],          # lexical measures are file-level only
        "details": {
            "TTR / CTTR":       ttr_detail,
            "Hapax Legomena":   hapax_detail,
            "Rare Word Density": rare_detail,
            "Mean Word Length": wlen_detail,
        },
    }
