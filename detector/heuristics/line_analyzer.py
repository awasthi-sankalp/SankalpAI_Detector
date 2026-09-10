"""
Line-by-line deep heuristic analyzer.
Evaluates every single line in a file individually to determine how AI-like it is,
preventing false positives on well-structured human code by looking for specific
micro-patterns that AI relies on but human engineers naturally avoid.
"""

import re
from typing import List, Tuple, Dict, Any
from detector.heuristics.code_structure import AI_IDENTIFIER_VOCAB, _split_identifier_tokens

# ---------------------------------------------------------------------------
# Pre-compiled Regex Patterns for Line-by-Line Analysis
# ---------------------------------------------------------------------------

# AI loves perfectly punctuated, full-sentence inline comments.
# Humans usually write: # load model
# AI writes:          # Load the pre-trained model for inference.
_PERFECT_COMMENT_RE = re.compile(r'^\s*#\s+[A-Z][^\n]{10,}[.!?]\s*$')

# AI loves redundant "Step" comments
_STEP_COMMENT_RE = re.compile(r'^\s*#\s*(?:Step \d+:|First,|Next,|Finally,|Then,)\s+[A-Z]')

# AI generic exception handling
_GENERIC_EXCEPT_RE = re.compile(r'^\s*except\s+Exception\s+as\s+[a-zA-Z0-9_]+\s*:')

# Human "lazy" comments (TODOs, lowercase, no punctuation, short)
_HUMAN_COMMENT_RE = re.compile(r'^\s*#\s*(?:todo|fixme|hack|note|review)\b', re.IGNORECASE)
_LAZY_COMMENT_RE = re.compile(r'^\s*#\s+[a-z][^.!?]{2,40}\s*$')

# ---------------------------------------------------------------------------
# Line Analyzer
# ---------------------------------------------------------------------------

def analyze_lines_deep(lines: List[str], file_ext: str) -> Tuple[float, List[Tuple[int, str, str]]]:
    """
    Analyzes a file line-by-line.
    Returns:
        score: A 0-100 score based on the density of highly AI-like lines.
        flagged_lines: A list of (line_num, text, reason) for the most suspicious lines.
    """
    if not lines:
        return 0.0, []

    ai_line_scores = []
    suspicious_lines = []

    # Keep track of previous lines for contextual rules (like comment -> code redundancy)
    prev_line = ""
    prev_is_comment = False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
            
        line_ai_score = 0
        reasons = []

        is_comment = stripped.startswith("#") or stripped.startswith("//")
        
        # 1. Perfect Comment Formatting
        if _PERFECT_COMMENT_RE.match(stripped):
            line_ai_score += 30
            reasons.append("perfectly formatted/punctuated inline comment")
            
        if _STEP_COMMENT_RE.match(stripped):
            line_ai_score += 50
            reasons.append("AI step-by-step reasoning comment")

        # 2. Human Comment Deductions (rewarding human laziness)
        if _HUMAN_COMMENT_RE.match(stripped) or _LAZY_COMMENT_RE.match(stripped):
            line_ai_score -= 50  # Highly human

        # 3. Generic Exception Handling
        if _GENERIC_EXCEPT_RE.match(stripped):
            line_ai_score += 25
            reasons.append("generic exception handler")

        # 4. Redundant Comment-to-Code mapping
        if not is_comment and prev_is_comment and prev_line:
            # Check if the comment literally just says what the code does
            comment_words = set(re.findall(r'[a-zA-Z]+', prev_line.lower()))
            code_words = set(re.findall(r'[a-zA-Z]+', stripped.lower()))
            # If they share > 60% vocabulary (ignoring short words)
            meaningful_comment = {w for w in comment_words if len(w) > 3}
            if meaningful_comment:
                overlap = len(meaningful_comment.intersection(code_words)) / len(meaningful_comment)
                if overlap > 0.6:
                    line_ai_score += 40
                    reasons.append("redundant comment describing exact code syntax")

        # 5. Over-reliance on "data", "result", "info" generic AI identifiers in assignments
        # e.g., processed_data_result = ...
        if "=" in stripped and not is_comment:
            left_side = stripped.split("=")[0].strip()
            # If the left side is a single variable assignment
            if re.match(r'^[a-zA-Z_]+$', left_side):
                tokens = _split_identifier_tokens(left_side)
                ai_words = [t for t in tokens if t in AI_IDENTIFIER_VOCAB]
                if len(ai_words) >= 2 and len(left_side) > 12:
                    line_ai_score += 35
                    reasons.append(f"AI-favored variable structure ({left_side})")

        # 6. Unnecessary type casting / verbosity
        if "str(str(" in stripped or "list(list(" in stripped:
            line_ai_score += 40
            reasons.append("redundant type casting hallucination")

        if line_ai_score > 0:
            ai_line_scores.append(line_ai_score)
            if line_ai_score >= 30:
                suspicious_lines.append((i, stripped, " | ".join(reasons)))
        elif line_ai_score < 0:
            # Record negative scores to offset AI scores
            ai_line_scores.append(line_ai_score)

        prev_line = stripped
        prev_is_comment = is_comment

    # Aggregate file score based on line scores
    if not ai_line_scores:
        return 0.0, []

    total_positive = sum(s for s in ai_line_scores if s > 0)
    total_negative = abs(sum(s for s in ai_line_scores if s < 0))
    
    net_score = total_positive - total_negative
    
    # Normalize by file length (a net score of ~100 per 50 lines is highly AI)
    # We use a logarithmic scale to map to 0-100
    line_count = len(lines)
    density = net_score / max(1, line_count)
    
    # Density mappings:
    # Human: < 0.5
    # Mixed: 1.0 - 2.0
    # AI: > 3.0
    if density <= 0:
        final_score = 0.0
    elif density >= 4.0:
        final_score = 100.0
    else:
        final_score = (density / 4.0) * 100.0

    return round(final_score, 2), suspicious_lines
