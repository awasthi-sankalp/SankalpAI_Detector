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
    "code_structure": 0.10,
}

# Category weights for standard CODE files (has comments/docstrings)
CODE_WEIGHTS = {
    "discourse":      0.45,
    "code_structure": 0.30,
    "syntactic":      0.12,
    "statistical":    0.08,
    "lexical":        0.05,
}

# Category weights for TERSE CODE files (no comments, bare code)
# Since discourse is 0, we distribute its weight to structural metrics
TERSE_CODE_WEIGHTS = {
    "code_structure": 0.55,   # Dominant signal: variables, typing, naming
    "syntactic":      0.20,   # Line uniformity, burstiness
    "statistical":    0.15,   # Token repetition
    "lexical":        0.10,
    "discourse":      0.00,   # Ignored
}


def is_code_file(extension: str) -> bool:
    return extension.lower() in CODE_EXTENSIONS


def compute_composite_score(
    heuristic_results: dict[str, dict],
    file_extension: str = "",
) -> float:
    """
    Calculate the weighted composite score based on file extension and
    heuristic performance.
    """
    is_code = is_code_file(file_extension)
    
    if is_code:
        discourse_score = heuristic_results.get("discourse", {}).get("score", 0.0)
        # If the file has almost no discourse signals (e.g., no comments, no AI phrases)
        # it is "terse code" and needs a specialized structural scoring profile.
        if discourse_score <= 15.0:
            weights = TERSE_CODE_WEIGHTS.copy()
        else:
            weights = CODE_WEIGHTS.copy()
            # Dynamic re-weighting for comment-stripped code that still has some discourse
            code_score = heuristic_results.get("code_structure", {}).get("score", 0.0)
            if discourse_score < 40 and code_score > 40:
                weights["discourse"] -= 0.20
                weights["code_structure"] += 0.20
    else:
        weights = PROSE_WEIGHTS.copy()

    composite = 0.0
    for category, weight in weights.items():
        result = heuristic_results.get(category, {})
        cat_score = result.get("score", 0.0)
        composite += weight * cat_score

    # Boost score slightly if code is completely perfectly typed but undocumented
    # (A very common AI pattern for generated scripts)
    if is_code:
        code_details = heuristic_results.get("code_structure", {}).get("details", {})
        ta_density = code_details.get("Type Annotation Density", "")
        ds_coverage = code_details.get("Docstring Coverage", "")
        if "density=1.0" in ta_density and "coverage=0.0" in ds_coverage:
            composite = min(100.0, composite * 1.15)  # 15% boost

    return round(composite, 2)


def get_threshold(sensitivity: str) -> float:
    if sensitivity == "low":
        return 72.0
    elif sensitivity == "medium":
        return 55.0
    elif sensitivity == "high":
        return 42.0
    elif sensitivity == "paranoid":
        return 25.0
    return 55.0


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
