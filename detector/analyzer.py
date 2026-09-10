"""
Analysis orchestrator.

For each file:
  1. Read text (try UTF-8 → UTF-8-BOM → latin-1 → ignore errors)
  2. Tokenize into words / sentences / paragraphs / lines
  3. Run all four heuristic modules
  4. Aggregate score and flagged lines
  5. Return result dict (or None if file cannot be read / is empty)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from detector.heuristics.lexical      import analyze_lexical
from detector.heuristics.syntactic    import analyze_syntactic
from detector.heuristics.discourse    import analyze_discourse
from detector.heuristics.statistical  import analyze_statistical
from detector.heuristics.code_structure import analyze_code_structure
from detector.scorer import compute_composite_score


# ── text reader ───────────────────────────────────────────────────────────────

_ENCODINGS = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
_MAX_BYTES  = 512_000   # analyse at most 512 KB per file


def _read_text(filepath: Path) -> str | None:
    """
    Try several encodings; return decoded text or None on failure.
    Caps at _MAX_BYTES so huge generated files don't stall the tool.
    """
    for enc in _ENCODINGS:
        try:
            with open(filepath, "rb") as f:
                raw = f.read(_MAX_BYTES)
            text = raw.decode(enc, errors="strict")
            return text
        except (UnicodeDecodeError, LookupError):
            continue

    # Last resort: decode with replacement characters
    try:
        with open(filepath, "rb") as f:
            raw = f.read(_MAX_BYTES)
        return raw.decode("utf-8", errors="replace")
    except OSError:
        return None


# ── tokenizers ────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Sentence splitter: tries NLTK punkt, falls back to regex.
    """
    try:
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            # punkt not downloaded yet — use regex fallback
            pass
    except ImportError:
        pass
    # Regex fallback: split on sentence-ending punctuation followed by space/newline
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


def _split_words(text: str) -> list[str]:
    """
    Word tokenizer: tries NLTK, falls back to regex.
    """
    try:
        import nltk
        try:
            return nltk.word_tokenize(text)
        except LookupError:
            pass
    except ImportError:
        pass
    return re.findall(r"[a-zA-Z']{1,}", text)


def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_lines(text: str) -> list[str]:
    """Split into individual lines."""
    return text.splitlines()


# ── merge + deduplicate flagged lines ────────────────────────────────────────

def _merge_flagged(
    *flagged_lists: list[tuple[int, str, str]],
    max_lines: int = 20,
) -> list[tuple[int, str, str]]:
    """
    Merge multiple lists of (line_no, text, reason).
    Deduplicate by line number (keep first reason seen).
    Sort by line number. Cap at max_lines.
    """
    seen: dict[int, tuple[int, str, str]] = {}
    for lst in flagged_lists:
        for entry in lst:
            lineno = entry[0]
            if lineno not in seen:
                seen[lineno] = entry

    sorted_entries = sorted(seen.values(), key=lambda x: x[0])
    return sorted_entries[:max_lines]


# ── main class ────────────────────────────────────────────────────────────────

class Analyzer:
    """
    Stateless analysis orchestrator.
    Call `analyze(filepath)` for each file; it returns a result dict or None.
    """

    def analyze(self, filepath: Path) -> dict | None:
        """
        Analyse a single file.

        Returns:
            {
              "score":            float (0–100 composite AI score),
              "grade":            str   ("VERY HIGH" / "HIGH" / etc.),
              "flagged_lines":    [(line_no, text, reason), ...],
              "heuristic_scores": {category: score, ...},
              "heuristic_details":{category: {heuristic: detail}},
              "word_count":       int,
              "line_count":       int,
              "too_short":        bool,
            }
            or None if the file cannot be read / is empty.
        """
        # ── 1. Read text ──────────────────────────────────────────────────────
        text = _read_text(filepath)
        if text is None:
            return None

        text = text.strip()
        if not text:
            return None

        # ── 2. Tokenize ───────────────────────────────────────────────────────
        lines      = _split_lines(text)
        words      = _split_words(text)
        sentences  = _split_sentences(text)
        paragraphs = _split_paragraphs(text)

        word_count = len([w for w in words if re.match(r"[a-zA-Z]", w)])
        too_short  = word_count < 30

        # -- 3. Run heuristic modules ------------------------------------------------
        lex_result   = analyze_lexical(words, lines)
        syn_result   = analyze_syntactic(text, sentences, paragraphs, words, lines)
        dis_result   = analyze_discourse(text, sentences, words, lines)
        stat_result  = analyze_statistical(text, words, sentences, lines)
        code_result  = analyze_code_structure(text, lines, filepath.suffix)
        
        from detector.heuristics.line_analyzer import analyze_lines_deep
        line_score, line_flagged = analyze_lines_deep(lines, filepath.suffix)
        line_result = {
            "score": line_score,
            "flagged_lines": line_flagged,
            "details": {"Line Density": f"net AI density mapped to {line_score}/100"}
        }

        # -- 4. Aggregate score -------------------------------------------------------
        ext = filepath.suffix
        heuristic_results = {
            "discourse":      dis_result,
            "syntactic":      syn_result,
            "statistical":    stat_result,
            "lexical":        lex_result,
            "code_structure": code_result,
            "line_analyzer":  line_result,
        }
        composite = compute_composite_score(heuristic_results, ext)

        # -- 5. Merge flagged lines ---------------------------------------------------
        flagged = _merge_flagged(
            dis_result["flagged_lines"],
            syn_result["flagged_lines"],
            stat_result["flagged_lines"],
            code_result["flagged_lines"],
            line_result["flagged_lines"],
        )

        # If score is high but no lines were flagged, pick the 3 longest lines.
        if composite >= 45 and not flagged:
            candidates = sorted(
                [(i + 1, l.strip()) for i, l in enumerate(lines)
                 if len(l.strip()) > 20],
                key=lambda x: -len(x[1]),
            )
            flagged = [(ln, txt, "statistical pattern") for ln, txt in candidates[:3]]

        from detector.scorer import grade_score
        return {
            "score":             composite,
            "grade":             grade_score(composite),
            "flagged_lines":     flagged,
            "heuristic_scores":  {k: round(v["score"], 1) for k, v in heuristic_results.items()},
            "heuristic_details": {k: v["details"] for k, v in heuristic_results.items()},
            "word_count":        word_count,
            "line_count":        len(lines),
            "too_short":         too_short,
        }

