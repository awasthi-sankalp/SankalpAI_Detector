"""
Discourse / rhetoric heuristics for AI text detection.

Heuristics:
 11. AI Phrase Detection        — 200+ curated AI signature phrases
 12. Transition Word Overuse    — "furthermore", "moreover", etc.
 13. Hedge Phrase Ratio         — "it is worth noting", "notably", etc.
 14. Formula Pattern Matching   — intro/conclusion structural templates
 15. First-Person Avoidance     — AI avoids "I", "me", "my" naturally
"""

import re
from collections import defaultdict

from data.wordlists import (
    AI_SIGNATURE_PHRASES,
    HIGH_WEIGHT_PHRASES,
    TRANSITION_WORDS,
    HEDGE_PHRASES,
    FORMULA_PATTERNS,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return text.lower()


def _match_phrases(
    text_lower: str,
    phrases: list[str],
) -> list[str]:
    """Return all phrases that appear in text_lower."""
    return [p for p in phrases if p in text_lower]


# ── heuristic 11: AI signature phrases ──────────────────────────────────────

def compute_ai_phrase_score(
    text: str,
    lines: list[str],
) -> tuple[float, list[tuple[int, str, str]], str]:
    """
    Count AI signature phrase matches across the whole text.
    Uses per-100-words density for normalization.
    High-weight phrases (in HIGH_WEIGHT_PHRASES) count double.

    Returns (score 0-100, flagged_lines, detail)
    """
    text_lower = _normalize(text)
    word_count = max(1, len(text.split()))

    # count matches (double-weight for high-specificity)
    weighted_hits = 0
    matched_phrases: list[str] = []

    for phrase in AI_SIGNATURE_PHRASES:
        if phrase in text_lower:
            multiplier = 2 if phrase in HIGH_WEIGHT_PHRASES else 1
            weighted_hits += multiplier
            matched_phrases.append(phrase)

    density = (weighted_hits / word_count) * 100  # per 100 words

    # Scoring
    if density >= 3.0 or weighted_hits >= 10:
        score = 95.0
    elif density >= 2.0 or weighted_hits >= 7:
        score = 85.0
    elif density >= 1.0 or weighted_hits >= 4:
        score = 70.0
    elif density >= 0.5 or weighted_hits >= 2:
        score = 50.0
    elif weighted_hits >= 1:
        score = 30.0
    else:
        score = 0.0

    # Line-level attribution
    flagged: list[tuple[int, str, str]] = []
    if matched_phrases:
        phrase_set = set(matched_phrases)
        for i, line in enumerate(lines, 1):
            line_lower = _normalize(line)
            hits = [p for p in phrase_set if p in line_lower]
            if hits:
                reason = "AI phrase: " + " | ".join(f'"{p}"' for p in hits[:3])
                flagged.append((i, line.strip(), reason))

    detail = (
        f"{len(matched_phrases)} phrases matched, "
        f"weighted_hits={weighted_hits}, density={density:.3f}/100w"
    )
    return score, flagged, detail


# ── heuristic 12: transition word overuse ────────────────────────────────────

def compute_transition_ratio(
    sentences: list[str],
    lines: list[str],
) -> tuple[float, list[tuple[int, str, str]], str]:
    """
    Ratio of sentences that START with a transition word/phrase.
    AI overuses discourse markers as sentence openers.

    Returns (score 0-100, flagged_lines, detail)
    """
    if not sentences:
        return 0.0, [], "no sentences"

    flagged_set: dict[int, tuple[int, str, str]] = {}
    transition_sentences = 0

    for sent in sentences:
        sent_lower = sent.strip().lower()
        for tw in TRANSITION_WORDS:
            if sent_lower.startswith(tw):
                transition_sentences += 1
                break

    ratio = transition_sentences / len(sentences)

    # Human: 0.02–0.08; AI: 0.12–0.40
    if ratio > 0.35:
        score = 90.0
    elif ratio > 0.20:
        score = 70.0
    elif ratio > 0.12:
        score = 50.0
    elif ratio > 0.06:
        score = 25.0
    else:
        score = 5.0

    # Line-level attribution
    flagged: list[tuple[int, str, str]] = []
    if score > 30:
        for i, line in enumerate(lines, 1):
            line_lower = line.strip().lower()
            for tw in TRANSITION_WORDS:
                if line_lower.startswith(tw):
                    flagged.append((i, line.strip(), f'transition word opener: "{tw}"'))
                    break

    detail = f"transition_ratio={ratio:.3f} ({transition_sentences}/{len(sentences)} sentences)"
    return score, flagged[:8], detail


# ── heuristic 13: hedge phrase ratio ─────────────────────────────────────────

def compute_hedge_ratio(
    text: str,
    lines: list[str],
) -> tuple[float, list[tuple[int, str, str]], str]:
    """
    Count hedge phrases per 100 words.
    AI uses hedges to sound authoritative while remaining safe.

    Returns (score 0-100, flagged_lines, detail)
    """
    text_lower = _normalize(text)
    word_count = max(1, len(text.split()))

    matched: list[str] = []
    for hp in HEDGE_PHRASES:
        if hp in text_lower:
            matched.append(hp)

    density = (len(matched) / word_count) * 100

    if density >= 2.5 or len(matched) >= 8:
        score = 85.0
    elif density >= 1.5 or len(matched) >= 5:
        score = 65.0
    elif density >= 0.8 or len(matched) >= 3:
        score = 45.0
    elif len(matched) >= 1:
        score = 25.0
    else:
        score = 0.0

    flagged: list[tuple[int, str, str]] = []
    if matched:
        hedge_set = set(matched)
        for i, line in enumerate(lines, 1):
            line_lower = _normalize(line)
            hits = [h for h in hedge_set if h in line_lower]
            if hits:
                reason = "hedge phrase: " + " | ".join(f'"{h}"' for h in hits[:3])
                flagged.append((i, line.strip(), reason))

    detail = (
        f"{len(matched)} hedge phrases matched, "
        f"density={density:.3f}/100w"
    )
    return score, flagged[:6], detail


# ── heuristic 14: formula patterns ───────────────────────────────────────────

def compute_formula_patterns(
    lines: list[str],
) -> tuple[float, list[tuple[int, str, str]], str]:
    """
    Check for structural AI formulas: specific intro/conclusion templates.
    Each pattern match counts as a strong signal.

    Returns (score 0-100, flagged_lines, detail)
    """
    flagged: list[tuple[int, str, str]] = []
    total_matches = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in FORMULA_PATTERNS:
            if pattern.search(stripped):
                total_matches += 1
                flagged.append((i, stripped, f"structural formula: {pattern.pattern[:50]}"))
                break  # one match per line is enough

    n_lines = max(1, sum(1 for l in lines if l.strip()))
    density = total_matches / n_lines

    if total_matches >= 5 or density > 0.05:
        score = 85.0
    elif total_matches >= 3 or density > 0.025:
        score = 65.0
    elif total_matches >= 2:
        score = 45.0
    elif total_matches == 1:
        score = 25.0
    else:
        score = 0.0

    detail = f"{total_matches} formula matches in {n_lines} lines"
    return score, flagged[:6], detail


# ── heuristic 15: first-person avoidance ─────────────────────────────────────

_FIRST_PERSON_RE = re.compile(
    r"\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b"
)

def compute_first_person_avoidance(
    text: str,
    words: list[str],
) -> tuple[float, str]:
    """
    AI often avoids first-person pronouns entirely, or uses them awkwardly.
    Very low first-person ratio → potentially AI-generated.
    NOTE: code files naturally have low first-person; weight this lower.

    Returns (score 0-100, detail)
    """
    n = max(1, len(words))
    matches = len(_FIRST_PERSON_RE.findall(text))
    ratio = matches / n

    # For most text: 0.01–0.05 is normal for technical writing
    # Completely absent (0.0) is suspicious in prose / comments
    if ratio == 0.0:
        score = 55.0  # mild signal — could just be style
    elif ratio < 0.005:
        score = 40.0
    elif ratio < 0.015:
        score = 20.0
    else:
        score = 5.0

    return score, f"first_person_ratio={ratio:.4f} ({matches}/{n})"


# ── main entry ────────────────────────────────────────────────────────────────

def analyze_discourse(
    text: str,
    sentences: list[str],
    words: list[str],
    lines: list[str],
) -> dict:
    """
    Run all discourse heuristics.

    Returns:
        {
          "score": float (0–100),
          "flagged_lines": [(line_no, text, reason), ...],
          "details": {name: detail_str},
        }
    """
    ph_score, ph_flagged, ph_detail = compute_ai_phrase_score(text, lines)
    tr_score, tr_flagged, tr_detail = compute_transition_ratio(sentences, lines)
    hd_score, hd_flagged, hd_detail = compute_hedge_ratio(text, lines)
    fm_score, fm_flagged, fm_detail = compute_formula_patterns(lines)
    fp_score, fp_detail             = compute_first_person_avoidance(text, words)

    # Weights — phrase detection is the strongest signal
    weights = {
        "phrases":     0.45,
        "transitions": 0.20,
        "hedges":      0.15,
        "formulas":    0.12,
        "first_person": 0.08,
    }
    composite = (
        weights["phrases"]      * ph_score +
        weights["transitions"]  * tr_score +
        weights["hedges"]       * hd_score +
        weights["formulas"]     * fm_score +
        weights["first_person"] * fp_score
    )

    # Merge all flagged lines, dedup by line number (keep first occurrence)
    seen: set[int] = set()
    merged: list[tuple[int, str, str]] = []
    for flagged_list in (ph_flagged, tr_flagged, hd_flagged, fm_flagged):
        for entry in flagged_list:
            if entry[0] not in seen:
                seen.add(entry[0])
                merged.append(entry)

    merged.sort(key=lambda x: x[0])

    return {
        "score": round(composite, 2),
        "flagged_lines": merged,
        "details": {
            "AI Signature Phrases":   ph_detail,
            "Transition Word Ratio":  tr_detail,
            "Hedge Phrase Ratio":     hd_detail,
            "Formula Patterns":       fm_detail,
            "First-Person Avoidance": fp_detail,
        },
    }
