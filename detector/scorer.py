"""
Weighted scoring engine for combining heuristic module results.

Category weights (sum = 1.0):
  discourse   0.45  ← strongest signal (AI phrase matches, transitions, hedges)
  syntactic   0.25  ← structural patterns (uniformity, passive, adverbs)
  statistical 0.18  ← information-theoretic (n-gram, Zipf, MTLD)
  lexical     0.12  ← vocabulary (TTR, hapax, rare words)

Code files get a modifier: statistical and syntactic weights are reduced
slightly because code token distributions differ from prose.
"""

from __future__ import annotations

# File extensions considered "code" (vs. prose / markup)
CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift",
    ".kt", ".kts", ".scala", ".r", ".m", ".sh", ".bash",
    ".ps1", ".psm1", ".lua", ".pl", ".pm", ".ex", ".exs",
    ".clj", ".cljs", ".elm", ".hs", ".ml", ".mli",
    ".fs", ".fsx", ".vb", ".asm", ".s", ".dart",
    ".groovy", ".gradle", ".tf", ".hcl",
})

# Category weights for PROSE files (text, markdown, etc.)
PROSE_WEIGHTS = {
    "discourse":      0.40,
    "syntactic":      0.22,
    "statistical":    0.16,
    "lexical":        0.12,
    "code_structure": 0.10,   # still useful for inline code blocks
}

# Category weights for CODE files
CODE_WEIGHTS = {
    "discourse":      0.45,   # AI phrases in comments/docstrings
    "code_structure": 0.30,   # identifier names, typing, docstrings -- strongest code signal
    "syntactic":      0.12,
    "statistical":    0.08,
    "lexical":        0.05,
}


def is_code_file(extension: str) -> bool:
    return extension.lower() in CODE_EXTENSIONS


def compute_composite_score(
    heuristic_results: dict[str, dict],
    file_extension: str = "",
) -> float:
    """
    Combine category scores into a single 0–100 composite score.

    Args:
        heuristic_results: {
            "discourse":   {"score": float, ...},
            "syntactic":   {"score": float, ...},
            "statistical": {"score": float, ...},
            "lexical":     {"score": float, ...},
            "code_structure": {"score": float, ...},
        }
        file_extension: e.g. ".py", ".txt"

    Returns:
        Composite score 0–100 (higher = more AI-like).
    """
    weights = CODE_WEIGHTS.copy() if is_code_file(file_extension) else PROSE_WEIGHTS.copy()

    # Dynamic re-weighting for comment-stripped code
    if is_code_file(file_extension):
        discourse_score = heuristic_results.get("discourse", {}).get("score", 0.0)
        code_score = heuristic_results.get("code_structure", {}).get("score", 0.0)
        
        # If discourse is very low (e.g. comments stripped) but structure is high,
        # shift reliance away from discourse and onto structure.
        if discourse_score < 40 and code_score > 40:
            weights["discourse"] -= 0.45
            weights["code_structure"] += 0.45

    composite = 0.0
    for category, weight in weights.items():
        result = heuristic_results.get(category, {})
        cat_score = result.get("score", 0.0)
        composite += weight * cat_score

    return round(composite, 2)


def get_threshold(sensitivity: str) -> int:
    """Map sensitivity level to score threshold (files above this are flagged)."""
    return {"low": 72, "medium": 55, "high": 42}.get(sensitivity, 55)


def grade_score(score: float) -> str:
    """Human-readable risk label for a composite score."""
    if score >= 85:
        return "VERY HIGH"
    elif score >= 70:
        return "HIGH"
    elif score >= 55:
        return "MEDIUM"
    elif score >= 40:
        return "LOW"
    else:
        return "MINIMAL"
