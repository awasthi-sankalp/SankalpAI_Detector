"""
Syntactic / structural heuristics for AI text detection.

Heuristics:
  5. Sentence Length Uniformity  — AI produces eerily uniform sentence lengths
  6. Burstiness Score            — human text is bursty; AI is not
  7. Paragraph Length Uniformity — AI generates paragraphs of similar size
  8. Passive Voice Ratio         — AI overuses passive constructions
  9. Adverb Density              — AI over-uses -ly adverbs
 10. Punctuation Entropy         — AI uses patterned, low-entropy punctuation
"""

import math
import re
import statistics
from collections import Counter

from data.wordlists import NON_ADVERB_LY_WORDS

# ── passive-voice regex ──────────────────────────────────────────────────────
# Matches:  "is/are/was/were/been/being + (opt. adverb) + past-participle"
_PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|been|being|get|gets|got|getting)\s+"
    r"(?:\w+ly\s+)?"          # optional adverb
    r"([a-z]+ed|[a-z]+en|[a-z]+t)\b",  # past participle heuristic
    re.IGNORECASE,
)

# ── adverb regex ─────────────────────────────────────────────────────────────
_ADVERB_RE = re.compile(r"\b([a-zA-Z]{4,}ly)\b")


# ── helpers ──────────────────────────────────────────────────────────────────

def _sentence_lengths(sentences: list[str]) -> list[int]:
    """Word-count of each sentence."""
    return [len(s.split()) for s in sentences if s.strip()]


def _cov(values: list[float]) -> float:
    """Coefficient of variation = std / mean (0 if mean==0)."""
    if len(values) < 2:
        return 0.0
    mu = statistics.mean(values)
    if mu == 0:
        return 0.0
    return statistics.stdev(values) / mu


def _entropy(counter: Counter) -> float:
    """Shannon entropy of a frequency distribution."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


# ── individual heuristics ─────────────────────────────────────────────────────

def compute_sentence_uniformity(sentences: list[str]) -> tuple[float, str]:
    """
    Coefficient of Variation of sentence lengths.
    Low CoV → sentences all about the same length → AI-like.

    Returns (score 0-100, detail)
    """
    lengths = _sentence_lengths(sentences)
    if len(lengths) < 3:
        return 0.0, "too few sentences"

    cov = _cov([float(x) for x in lengths])
    mean_l = statistics.mean(lengths)

    # Human CoV: typically 0.45–0.90
    # AI CoV: typically 0.15–0.45
    if cov < 0.20:
        score = 90.0
    elif cov < 0.35:
        score = 70.0
    elif cov < 0.50:
        score = 45.0
    elif cov < 0.70:
        score = 20.0
    else:
        score = 5.0

    return score, f"CoV={cov:.3f}, mean_sentence_len={mean_l:.1f}w"


def compute_burstiness(sentences: list[str]) -> tuple[float, str]:
    """
    Burstiness B = (σ - μ) / (σ + μ)   (Goh & Barabasi 2008)

    B ≈ +1  → very bursty (human)
    B ≈  0  → neutral / Poisson
    B ≈ -1  → very regular (AI)

    Returns (score 0-100, detail)
    """
    lengths = _sentence_lengths(sentences)
    if len(lengths) < 5:
        return 0.0, "too few sentences for burstiness"

    mu  = statistics.mean(lengths)
    sig = statistics.stdev(lengths)
    denom = sig + mu
    if denom == 0:
        return 50.0, "B=undefined"

    B = (sig - mu) / denom

    # B < -0.3 → AI-like; B > 0.2 → human-like
    if B < -0.40:
        score = 90.0
    elif B < -0.20:
        score = 70.0
    elif B < 0.00:
        score = 50.0
    elif B < 0.20:
        score = 30.0
    else:
        score = 10.0

    return score, f"B={B:.3f}"


def compute_paragraph_uniformity(paragraphs: list[str]) -> tuple[float, str]:
    """
    CoV of paragraph lengths (in words).
    AI writes paragraphs of similar length.

    Returns (score 0-100, detail)
    """
    lengths = [len(p.split()) for p in paragraphs if len(p.split()) > 3]
    if len(lengths) < 3:
        return 0.0, "too few paragraphs"

    cov = _cov([float(x) for x in lengths])

    if cov < 0.20:
        score = 85.0
    elif cov < 0.40:
        score = 65.0
    elif cov < 0.60:
        score = 40.0
    else:
        score = 10.0

    return score, f"para_CoV={cov:.3f} ({len(lengths)} paragraphs)"


def compute_passive_voice(
    lines: list[str],
) -> tuple[float, list[tuple[int, str, str]], str]:
    """
    Passive voice ratio = passive_sentences / total_sentences.
    AI heavily favors passive constructions.

    Returns (score 0-100, flagged_lines, detail)
    """
    flagged: list[tuple[int, str, str]] = []
    passive_count = 0
    total = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        total += 1
        if _PASSIVE_RE.search(stripped):
            passive_count += 1
            flagged.append((i, stripped, "passive voice detected"))

    if total == 0:
        return 0.0, [], "no lines"

    ratio = passive_count / total

    # Human prose: 0.05–0.15; AI: 0.20–0.45
    if ratio > 0.40:
        score = 85.0
    elif ratio > 0.25:
        score = 65.0
    elif ratio > 0.15:
        score = 40.0
    elif ratio > 0.08:
        score = 20.0
    else:
        score = 5.0

    # Only keep the first 5 flagged lines to avoid noise
    return score, flagged[:5], f"passive_ratio={ratio:.3f} ({passive_count}/{total})"


def compute_adverb_density(
    words: list[str],
    lines: list[str],
) -> tuple[float, list[tuple[int, str, str]], str]:
    """
    Adverb density = -ly adverbs / total word count.
    AI overuses adverbs (especially intensifier adverbs).

    Returns (score 0-100, flagged_lines, detail)
    """
    alpha_words = [w.lower() for w in words if re.match(r"^[a-zA-Z]+$", w)]
    n = len(alpha_words)
    if n < 10:
        return 0.0, [], "too few words"

    adverbs = [w for w in alpha_words
               if w.endswith("ly") and w not in NON_ADVERB_LY_WORDS and len(w) > 4]
    density = len(adverbs) / n

    # Human: 0.01–0.04; AI: 0.05–0.12
    if density > 0.10:
        score = 85.0
    elif density > 0.07:
        score = 65.0
    elif density > 0.05:
        score = 45.0
    elif density > 0.03:
        score = 25.0
    else:
        score = 5.0

    # Flag lines with 3+ adverbs
    flagged: list[tuple[int, str, str]] = []
    if score > 40:
        for i, line in enumerate(lines, 1):
            matches = [m.group() for m in _ADVERB_RE.finditer(line)
                       if m.group().lower() not in NON_ADVERB_LY_WORDS]
            if len(matches) >= 3:
                flagged.append((i, line.strip(),
                                f"high adverb density: {', '.join(matches[:5])}"))

    return score, flagged[:5], f"adverb_density={density:.4f} ({len(adverbs)}/{n})"


def compute_punctuation_entropy(text: str) -> tuple[float, str]:
    """
    Shannon entropy of punctuation character usage.
    AI punctuation is patterned → lower entropy.
    Rich human punctuation → higher entropy.

    Returns (score 0-100, detail)
    """
    puncts = [c for c in text if c in r'.,;:!?-—–()[]{}"\'/\\']
    if len(puncts) < 10:
        return 0.0, "too few punctuation marks"

    freq = Counter(puncts)
    H = _entropy(freq)

    # Typical human entropy: 1.8–3.2 bits
    # AI: 1.0–2.0 bits (mostly commas and periods)
    if H < 1.0:
        score = 80.0
    elif H < 1.5:
        score = 60.0
    elif H < 2.0:
        score = 40.0
    elif H < 2.5:
        score = 20.0
    else:
        score = 5.0

    top = ", ".join(f"'{c}':{n}" for c, n in freq.most_common(4))
    return score, f"punct_entropy={H:.3f} bits ({top})"


# ── main entry ────────────────────────────────────────────────────────────────

def analyze_syntactic(
    text: str,
    sentences: list[str],
    paragraphs: list[str],
    words: list[str],
    lines: list[str],
) -> dict:
    """
    Run all syntactic heuristics.

    Returns:
        {
          "score": float (0–100),
          "flagged_lines": [(line_no, text, reason), ...],
          "details": {name: detail_str},
        }
    """
    su_score,  su_detail  = compute_sentence_uniformity(sentences)
    bu_score,  bu_detail  = compute_burstiness(sentences)
    pu_score,  pu_detail  = compute_paragraph_uniformity(paragraphs)
    pa_score,  pa_flagged, pa_detail = compute_passive_voice(lines)
    ad_score,  ad_flagged, ad_detail = compute_adverb_density(words, lines)
    pe_score,  pe_detail  = compute_punctuation_entropy(text)

    weights = {
        "sent_uniform": 0.25,
        "burstiness":   0.20,
        "para_uniform": 0.15,
        "passive":      0.18,
        "adverb":       0.12,
        "punct_entropy": 0.10,
    }
    composite = (
        weights["sent_uniform"]  * su_score +
        weights["burstiness"]    * bu_score +
        weights["para_uniform"]  * pu_score +
        weights["passive"]       * pa_score +
        weights["adverb"]        * ad_score +
        weights["punct_entropy"] * pe_score
    )

    flagged = pa_flagged + ad_flagged

    return {
        "score": round(composite, 2),
        "flagged_lines": flagged,
        "details": {
            "Sentence Uniformity (CoV)": su_detail,
            "Burstiness (B)":            bu_detail,
            "Paragraph Uniformity":      pu_detail,
            "Passive Voice Ratio":       pa_detail,
            "Adverb Density":            ad_detail,
            "Punctuation Entropy":       pe_detail,
        },
    }
