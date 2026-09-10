"""
Statistical / information-theoretic heuristics for AI text detection.

Heuristics:
 16. N-gram Repetition Score    — AI repeats 3-gram/4-gram patterns
 17. Readability Score          — AI clusters in FK grade 8–12
 18. Sentence Starter Diversity — AI over-repeats sentence openers
 19. MTLD Proxy                 — Measure of Textual Lexical Diversity
 20. Zipfian Deviation          — AI word-frequency deviates from human Zipf
"""

import math
import re
import statistics
from collections import Counter


# ── helpers ──────────────────────────────────────────────────────────────────

def _tokenize_simple(text: str) -> list[str]:
    """Simple word tokenizer (fallback, no NLTK)."""
    return re.findall(r"[a-zA-Z']{2,}", text)


def _sentences_from_text(text: str) -> list[str]:
    """Naive sentence splitter for fallback."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _syllable_count(word: str) -> int:
    """Rough syllable counter (regex-based, no CMUdict needed)."""
    word = word.lower()
    # remove silent e
    word = re.sub(r"e$", "", word)
    # count vowel groups
    vowels = re.findall(r"[aeiouy]+", word)
    count = len(vowels)
    return max(1, count)


# ── heuristic 16: n-gram repetition ─────────────────────────────────────────

def compute_ngram_repetition(words: list[str]) -> tuple[float, str]:
    """
    Proportion of 3-grams and 4-grams that appear more than once.

    AI text repeats phrase patterns; human text is more diverse.
    Score = weighted_repeated / total_ngrams

    Returns (score 0-100, detail)
    """
    lower = [w.lower() for w in words if re.match(r"^[a-zA-Z]{2,}$", w)]
    n = len(lower)

    if n < 15:
        return 0.0, "too few words for n-gram analysis"

    # Build 3-grams and 4-grams
    tri_counter:  Counter = Counter()
    quad_counter: Counter = Counter()

    for i in range(n - 2):
        tri_counter[tuple(lower[i:i+3])] += 1
    for i in range(n - 3):
        quad_counter[tuple(lower[i:i+4])] += 1

    tri_repeat  = sum(1 for c in tri_counter.values()  if c > 1)
    quad_repeat = sum(1 for c in quad_counter.values() if c > 1)
    total_tri   = len(tri_counter)
    total_quad  = len(quad_counter)

    if total_tri == 0:
        return 0.0, "no 3-grams"

    tri_ratio  = tri_repeat  / total_tri
    quad_ratio = quad_repeat / max(1, total_quad)
    combined   = (tri_ratio * 0.6) + (quad_ratio * 0.4)

    # Human: combined < 0.05; AI: combined 0.08–0.25
    if combined > 0.20:
        score = 90.0
    elif combined > 0.12:
        score = 70.0
    elif combined > 0.07:
        score = 50.0
    elif combined > 0.04:
        score = 25.0
    else:
        score = 5.0

    detail = (
        f"3-gram_repeat={tri_ratio:.3f} ({tri_repeat}/{total_tri}), "
        f"4-gram_repeat={quad_ratio:.3f} ({quad_repeat}/{total_quad}), "
        f"combined={combined:.3f}"
    )
    return score, detail


# ── heuristic 17: readability ─────────────────────────────────────────────────

def compute_readability(text: str, sentences: list[str]) -> tuple[float, str]:
    """
    Flesch-Kincaid Grade Level (manual implementation, no textstat needed).

    FK_GL = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

    AI typically writes at grade 8–13.
    Very low (<5) or very high (>16) grades are not typical of AI.

    Returns (score 0-100, detail)
    """
    words = _tokenize_simple(text)
    n_words = len(words)
    n_sents = max(1, len([s for s in sentences if s.strip()]))

    if n_words < 20 or n_sents < 2:
        return 0.0, f"too short (words={n_words}, sents={n_sents})"

    n_syllables = sum(_syllable_count(w) for w in words)

    asl = n_words / n_sents                  # avg sentence length
    asw = n_syllables / n_words              # avg syllables per word
    fk_gl = 0.39 * asl + 11.8 * asw - 15.59

    # Also compute Gunning Fog
    complex_words = sum(1 for w in words if _syllable_count(w) >= 3)
    gf = 0.4 * (asl + 100 * complex_words / n_words)

    # AI "sweet spot": FK 8–13, GF 10–15
    in_fk_zone  = 8 <= fk_gl <= 13
    in_gf_zone  = 9 <= gf    <= 16

    zone_matches = int(in_fk_zone) + int(in_gf_zone)

    if zone_matches == 2:
        score = 65.0
    elif zone_matches == 1:
        score = 35.0
    else:
        score = 10.0

    detail = f"FK_grade={fk_gl:.1f}, Gunning_Fog={gf:.1f}, syllables/word={asw:.2f}"
    return score, detail


# ── heuristic 18: sentence starter diversity ─────────────────────────────────

def compute_sentence_starter_diversity(
    sentences: list[str],
    lines: list[str],
) -> tuple[float, list[tuple[int, str, str]], str]:
    """
    TTR of first words of sentences.
    AI repeats the same starters: "The", "This", "In", "To", etc.

    Returns (score 0-100, flagged_lines, detail)
    """
    starters = []
    for s in sentences:
        words = s.strip().split()
        if words:
            starters.append(words[0].lower())

    n = len(starters)
    if n < 5:
        return 0.0, [], "too few sentences"

    unique = len(set(starters))
    ttr = unique / n

    # Find over-used starters
    freq = Counter(starters)
    overused = [(w, c) for w, c in freq.most_common(5) if c / n > 0.15]

    # Human: ttr typically 0.55–0.90
    # AI: 0.20–0.55 (e.g., "The" starts 30% of sentences)
    if ttr < 0.25:
        score = 85.0
    elif ttr < 0.40:
        score = 65.0
    elif ttr < 0.55:
        score = 40.0
    elif ttr < 0.70:
        score = 20.0
    else:
        score = 5.0

    # Flag first lines of paragraphs that use overused starters
    flagged: list[tuple[int, str, str]] = []
    if overused and score > 40:
        overused_words = {w for w, _ in overused}
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                continue
            first_word = stripped.split()[0].lower() if stripped.split() else ""
            if first_word in overused_words:
                count = freq[first_word]
                flagged.append((
                    i, stripped,
                    f'overused sentence starter: "{first_word}" ({count}×)'
                ))

    overused_str = ", ".join(f'"{w}"({c})' for w, c in overused) if overused else "none"
    detail = f"starter_ttr={ttr:.3f} ({unique}/{n}), overused={overused_str}"
    return score, flagged[:5], detail


# ── heuristic 19: MTLD proxy ─────────────────────────────────────────────────

def compute_mtld_proxy(words: list[str]) -> tuple[float, str]:
    """
    Moving Average Type-Token Ratio (MA-TTR) as a proxy for MTLD.
    Window of 50 words; compute TTR in each window, take mean.

    AI text: lower MA-TTR (less lexical diversity per window).

    Returns (score 0-100, detail)
    """
    alpha = [w.lower() for w in words if re.match(r"^[a-zA-Z]{2,}$", w)]
    n = len(alpha)
    window = 50

    if n < window:
        return 0.0, f"too few words ({n} < {window})"

    ttrs = []
    for i in range(n - window + 1):
        segment = alpha[i:i + window]
        ttrs.append(len(set(segment)) / window)

    ma_ttr = statistics.mean(ttrs)

    # Human: MA-TTR typically 0.65–0.80
    # AI: 0.50–0.65 (more repetitive within windows)
    if ma_ttr < 0.50:
        score = 85.0
    elif ma_ttr < 0.58:
        score = 65.0
    elif ma_ttr < 0.65:
        score = 45.0
    elif ma_ttr < 0.72:
        score = 20.0
    else:
        score = 5.0

    detail = f"MA-TTR={ma_ttr:.4f} (window={window}, samples={len(ttrs)})"
    return score, detail


# ── heuristic 20: Zipfian deviation ──────────────────────────────────────────

def compute_zipf_deviation(words: list[str]) -> tuple[float, str]:
    """
    Measure how closely word-frequency distribution follows Zipf's law
    (expected slope ≈ -1.0 in log-log rank-frequency space).

    AI text tends to follow Zipf more closely (less deviation) OR
    deviates in the tail (over-uses mid-frequency words).

    Metric: least-squares slope fit in log-log space.
    Human slope: typically -0.85 to -1.10
    AI slope: tends toward -0.70 to -0.85 (flatter tail)

    Returns (score 0-100, detail)
    """
    alpha = [w.lower() for w in words if re.match(r"^[a-zA-Z]{2,}$", w)]
    n = len(alpha)
    if n < 30:
        return 0.0, "too few words for Zipf analysis"

    freq = Counter(alpha)
    sorted_counts = sorted(freq.values(), reverse=True)

    # Fit log-log slope using least squares
    # log(freq) = slope * log(rank) + intercept
    log_ranks  = [math.log(r+1) for r in range(len(sorted_counts))]
    log_freqs  = [math.log(c)   for c in sorted_counts]

    n_pts = len(log_ranks)
    if n_pts < 5:
        return 0.0, "too few unique words"

    sum_x  = sum(log_ranks)
    sum_y  = sum(log_freqs)
    sum_xy = sum(x * y for x, y in zip(log_ranks, log_freqs))
    sum_x2 = sum(x * x for x in log_ranks)

    denom = n_pts * sum_x2 - sum_x ** 2
    if denom == 0:
        return 0.0, "degenerate Zipf fit"

    slope = (n_pts * sum_xy - sum_x * sum_y) / denom

    # Flatter slope (closer to 0) = fewer unique words in tail = AI
    # Human: slope ≈ -0.9 to -1.2
    # AI: slope ≈ -0.6 to -0.85
    if -0.75 < slope <= -0.60:
        score = 70.0
    elif -0.85 < slope <= -0.75:
        score = 45.0
    elif -0.95 < slope <= -0.85:
        score = 20.0
    else:
        score = 5.0

    detail = f"Zipf_slope={slope:.3f} (ideal=-1.0, AI zone≈-0.6 to -0.85)"
    return score, detail


# ── main entry ────────────────────────────────────────────────────────────────

def analyze_statistical(
    text: str,
    words: list[str],
    sentences: list[str],
    lines: list[str],
) -> dict:
    """
    Run all statistical heuristics.

    Returns:
        {
          "score": float (0–100),
          "flagged_lines": [(line_no, text, reason), ...],
          "details": {name: detail_str},
        }
    """
    ng_score,  ng_detail                   = compute_ngram_repetition(words)
    rd_score,  rd_detail                   = compute_readability(text, sentences)
    ss_score,  ss_flagged, ss_detail       = compute_sentence_starter_diversity(sentences, lines)
    mt_score,  mt_detail                   = compute_mtld_proxy(words)
    zp_score,  zp_detail                   = compute_zipf_deviation(words)

    weights = {
        "ngram":    0.28,
        "readability": 0.15,
        "starters": 0.20,
        "mtld":     0.22,
        "zipf":     0.15,
    }
    composite = (
        weights["ngram"]       * ng_score +
        weights["readability"] * rd_score +
        weights["starters"]    * ss_score +
        weights["mtld"]        * mt_score +
        weights["zipf"]        * zp_score
    )

    return {
        "score": round(composite, 2),
        "flagged_lines": ss_flagged,   # only starters produce per-line output
        "details": {
            "N-gram Repetition":            ng_detail,
            "Readability (FK / GF)":        rd_detail,
            "Sentence Starter Diversity":   ss_detail,
            "MTLD Proxy (MA-TTR)":          mt_detail,
            "Zipfian Deviation":            zp_detail,
        },
    }
