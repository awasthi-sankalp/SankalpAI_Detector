"""
Code structure heuristics for AI text detection.
Analyzes variable names, function names, typing patterns, and
structural conventions -- all visible EVEN when comments are removed.

AI-generated code has very specific fingerprints in its code structure:
- Overly descriptive, verbose identifier names
- Near-zero use of short human-style names (x, tmp, buf, n, ret...)
- Perfect naming convention consistency (never mixes snake_case / camelCase)
- Obsessive type annotations on every function
- Docstring on every single function/class
- Excessive try/except coverage
- Repetitive "AI vocabulary" words in identifiers
  (process_, handle_, manage_, _result, _data, _config...)
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

# ---------------------------------------------------------------------------
# AI vocabulary: words commonly embedded in AI-generated identifier names
# ---------------------------------------------------------------------------
AI_IDENTIFIER_VOCAB = frozenset({
    # Action-verb prefixes AI loves
    "process", "handle", "manage", "transform", "validate", "generate",
    "initialize", "configure", "calculate", "compute", "extract", "retrieve",
    "fetch", "perform", "execute", "build", "prepare", "setup",
    "check", "verify", "ensure", "convert", "parse", "format", "render",
    "update", "create", "remove", "cleanup", "teardown", "initialize",
    # Noun suffixes AI appends everywhere
    "result", "config", "output", "manager", "handler", "processor",
    "transformer", "validator", "helper", "utility", "service", "provider",
    "factory", "builder", "controller", "repository", "adapter",
    "response", "payload", "content", "status", "context", "resource",
    "instance", "environment", "parameter", "argument",
    # Over-qualified modifiers AI attaches
    "processed", "transformed", "validated", "configured", "initialized",
    "computed", "extracted", "retrieved", "generated", "formatted",
    "current", "target", "source", "destination", "temporary",
    "intermediate", "initial", "final", "default", "custom",
    "specified", "provided", "given", "expected", "actual",
    "converted", "parsed", "cleaned", "normalized", "sanitized",
})

# Human short-form names that AI almost NEVER uses (except i,j,k loop vars)
HUMAN_SHORT_NAMES = frozenset({
    "x", "y", "z", "n", "m", "k", "c", "s", "f", "g", "h", "t", "u", "v",
    "a", "b", "d", "e", "l", "r", "p", "q", "w",
    "tmp", "buf", "ptr", "arr", "num", "idx", "pos", "cnt", "ret",
    "err", "obj", "fn", "cb", "ev", "op", "ch", "fp", "fd",
    "ok", "io", "db", "rx", "tx",
})

# Language keywords / builtins to exclude from identifier analysis
_KEYWORDS = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield", "self", "cls", "super", "print", "len",
    "range", "type", "list", "dict", "tuple", "set", "str", "int", "float",
    "bool", "bytes", "object", "open", "input", "iter", "next", "map",
    "filter", "zip", "enumerate", "sorted", "reversed", "max", "min",
    "sum", "abs", "repr", "format", "vars", "dir", "id", "hash", "hex",
    "isinstance", "hasattr", "getattr", "setattr", "all", "any",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "OSError", "FileNotFoundError",
    "var", "let", "const", "function", "new", "this", "null",
    "undefined", "console", "window", "document", "module", "exports",
    "require", "Promise", "Array", "Object", "String", "Number",
    "public", "private", "protected", "void", "static", "override",
    "virtual", "abstract", "interface", "struct", "enum", "final",
    "implements", "extends", "package", "namespace", "using", "include",
    "os", "re", "sys", "io", "json", "math", "time", "datetime",
    "logging", "unittest", "pytest", "threading", "subprocess",
    "switch", "case", "default", "do", "typeof", "instanceof", "delete",
    "throw", "catch", "char", "byte", "short", "long", "double",
    "main", "args", "auto", "nullptr", "template", "typename",
})

# Regex: strip string literals before identifier extraction
_STR_TRIPLE_DQ = re.compile(r'""".*?"""', re.DOTALL)
_STR_TRIPLE_SQ = re.compile(r"'''.*?'''", re.DOTALL)
_STR_DQ        = re.compile(r'"(?:[^"\\]|\\.)*"')
_STR_SQ        = re.compile(r"'(?:[^'\\]|\\.)*'")
_STR_BACKTICK  = re.compile(r'`(?:[^`\\]|\\.)*`')

# Comment removal
_COMMENT_HASH  = re.compile(r'#.*$', re.MULTILINE)
_COMMENT_SLASH = re.compile(r'//.*$', re.MULTILINE)
_COMMENT_BLOCK = re.compile(r'/\*.*?\*/', re.DOTALL)

# Identifier pattern
_IDENT_RE = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b')

# camelCase token splitter
_CAMEL_RE = re.compile(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]|\d+')

# Python-specific patterns
_PY_FUNC_RE     = re.compile(r'def\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*(?:->[^:]+)?:', re.MULTILINE)
_PY_CLASS_RE    = re.compile(r'class\s+([A-Za-z]\w*)', re.MULTILINE)
_PY_DOCSTR_RE   = re.compile(r'(?:def|class)\s+\w[^:]*:\s*\n\s*"""', re.MULTILINE)
_PY_TYPEHINT_RE = re.compile(r'def\s+\w+\([^)]*:\s*\w', re.MULTILINE)
_PY_TRYEXCEPT_RE = re.compile(r'\btry\s*:', re.MULTILINE)

# JS/TS patterns
_JS_FUNC_RE = re.compile(
    r'(?:function\s+([a-zA-Z_$]\w*)|(?:const|let|var)\s+([a-zA-Z_$]\w*)\s*=\s*(?:async\s+)?(?:function|\())',
    re.MULTILINE,
)
_JS_CLASS_RE = re.compile(r'class\s+([A-Za-z]\w*)', re.MULTILINE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_strings_and_comments(text: str) -> str:
    """Remove string literals and comments, leaving only code tokens."""
    t = _STR_TRIPLE_DQ.sub("''", text)
    t = _STR_TRIPLE_SQ.sub("''", t)
    t = _STR_DQ.sub("''", t)
    t = _STR_SQ.sub("''", t)
    t = _STR_BACKTICK.sub("``", t)
    t = _COMMENT_BLOCK.sub("", t)
    t = _COMMENT_HASH.sub("", t)
    t = _COMMENT_SLASH.sub("", t)
    return t


def _split_identifier_tokens(name: str) -> list[str]:
    """Split snake_case and CamelCase identifiers into component words."""
    if "_" in name:
        parts = [p for p in name.split("_") if p]
    else:
        parts = _CAMEL_RE.findall(name)
    return [p.lower() for p in parts if len(p) >= 2]


def _extract_identifiers(text: str) -> list[str]:
    """Extract all identifiers from code (with strings/comments stripped)."""
    code = _strip_strings_and_comments(text)
    all_idents = _IDENT_RE.findall(code)
    return [ident for ident in all_idents if ident not in _KEYWORDS]


# ---------------------------------------------------------------------------
# Heuristic: identifier length distribution
# ---------------------------------------------------------------------------

def compute_identifier_length(
    identifiers: list[str],
    lines: list[str],
) -> tuple[float, list[tuple[int, str, str]], str]:
    """
    Average identifier length.
    AI generates very long, descriptive names (avg 12-20 chars).
    Human code uses shorter names (avg 4-8 chars).

    Returns (score 0-100, flagged_lines, detail)
    """
    if len(identifiers) < 10:
        return 0.0, [], "too few identifiers"

    lengths = [len(i) for i in identifiers]
    mean_len = statistics.mean(lengths)
    long_idents = [i for i in identifiers if len(i) >= 15]

    # Human avg: 4-8; AI avg: 9-20
    if mean_len >= 13:
        score = 95.0
    elif mean_len >= 10:
        score = 75.0
    elif mean_len >= 8:
        score = 50.0
    elif mean_len >= 6:
        score = 25.0
    else:
        score = 5.0

    # Flag lines containing very long identifiers
    flagged: list[tuple[int, str, str]] = []
    if long_idents and score >= 40:
        long_set = set(long_idents[:20])
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            for ident in long_set:
                if ident in stripped:
                    flagged.append((
                        i, stripped,
                        f"long identifier: '{ident}' ({len(ident)} chars)"
                    ))
                    break

    detail = (
        f"mean_identifier_len={mean_len:.1f}, "
        f"longest={max(lengths) if lengths else 0}, "
        f"long_idents(>=15)={len(long_idents)}/{len(identifiers)}"
    )
    return score, flagged[:5], detail


# ---------------------------------------------------------------------------
# Heuristic: AI vocabulary in identifier names
# ---------------------------------------------------------------------------

def compute_ai_vocab_in_identifiers(
    identifiers: list[str],
    lines: list[str],
) -> tuple[float, list[tuple[int, str, str]], str]:
    """
    Check what proportion of identifiers contain AI-vocabulary words.
    AI code is full of '_handler', '_processor', '_result', '_config', etc.

    Returns (score 0-100, flagged_lines, detail)
    """
    if len(identifiers) < 5:
        return 0.0, [], "too few identifiers"

    ai_hit_idents: list[str] = []
    for ident in identifiers:
        tokens = _split_identifier_tokens(ident)
        if any(t in AI_IDENTIFIER_VOCAB for t in tokens):
            ai_hit_idents.append(ident)

    ratio = len(ai_hit_idents) / len(identifiers)

    # Human: 0.05-0.15; AI: 0.25-0.60
    if ratio >= 0.40:
        score = 95.0
    elif ratio >= 0.25:
        score = 75.0
    elif ratio >= 0.15:
        score = 50.0
    elif ratio >= 0.08:
        score = 25.0
    else:
        score = 5.0

    # Flag lines with AI-vocabulary identifiers
    flagged: list[tuple[int, str, str]] = []
    if ai_hit_idents and score >= 40:
        ai_set = set(ai_hit_idents[:30])
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            hits = [ident for ident in ai_set if ident in stripped]
            if hits:
                words = [t for ident in hits for t in _split_identifier_tokens(ident)
                         if t in AI_IDENTIFIER_VOCAB]
                flagged.append((
                    i, stripped,
                    f"AI-vocabulary identifier: {', '.join(repr(h) for h in hits[:3])} "
                    f"(AI words: {', '.join(set(words[:4]))})"
                ))

    top_ai = ", ".join(f"'{i}'" for i in ai_hit_idents[:5])
    detail = (
        f"ai_vocab_ratio={ratio:.3f} ({len(ai_hit_idents)}/{len(identifiers)}), "
        f"examples: {top_ai}"
    )
    return score, flagged[:5], detail


# ---------------------------------------------------------------------------
# Heuristic: short-name deficit
# ---------------------------------------------------------------------------

def compute_short_name_deficit(identifiers: list[str]) -> tuple[float, str]:
    """
    AI almost never uses short abbreviation names (tmp, buf, ptr, x, n, etc.)
    unless they are standard loop counters (i, j, k, idx).
    Very low short-name usage = suspicious.

    Returns (score 0-100, detail)
    """
    if len(identifiers) < 10:
        return 0.0, "too few identifiers"

    # Count identifiers that are 1-3 chars or in HUMAN_SHORT_NAMES
    short_count = sum(
        1 for ident in identifiers
        if len(ident) <= 3 or ident.lower() in HUMAN_SHORT_NAMES
    )
    ratio = short_count / len(identifiers)

    # Human: 0.10-0.30 short names; AI: 0.00-0.05
    if ratio < 0.02:
        score = 85.0
    elif ratio < 0.05:
        score = 65.0
    elif ratio < 0.10:
        score = 40.0
    elif ratio < 0.18:
        score = 20.0
    else:
        score = 5.0

    detail = f"short_name_ratio={ratio:.3f} ({short_count}/{len(identifiers)})"
    return score, detail


# ---------------------------------------------------------------------------
# Heuristic: naming convention consistency
# ---------------------------------------------------------------------------

def compute_naming_consistency(identifiers: list[str]) -> tuple[float, str]:
    """
    AI is perfectly consistent: 100% snake_case in Python, 100% camelCase in JS.
    Humans mix conventions slightly.
    Perfect consistency (>95%) = slightly suspicious.

    Returns (score 0-100, detail)
    """
    if len(identifiers) < 10:
        return 0.0, "too few identifiers"

    snake  = sum(1 for i in identifiers if "_" in i and i == i.lower())
    camel  = sum(1 for i in identifiers if "_" not in i and any(c.isupper() for c in i[1:]))
    total  = len(identifiers)

    dominant = max(snake, camel)
    consistency = dominant / total if total > 0 else 0

    # AI: consistency >= 0.90; Human: 0.70-0.85 (some mixing / abbreviations)
    if consistency >= 0.95:
        score = 75.0
    elif consistency >= 0.90:
        score = 50.0
    elif consistency >= 0.80:
        score = 25.0
    else:
        score = 5.0

    style = "snake_case" if snake > camel else "camelCase"
    detail = f"naming_consistency={consistency:.3f} ({dominant}/{total} {style})"
    return score, detail


# ---------------------------------------------------------------------------
# Heuristic: docstring coverage
# ---------------------------------------------------------------------------

def compute_docstring_coverage(text: str) -> tuple[float, str]:
    """
    AI writes a docstring for EVERY function and class.
    Count function/class definitions vs how many have an immediately following docstring.

    Returns (score 0-100, detail)
    """
    # Count all def / class blocks
    func_count  = len(re.findall(r'^\s*def\s+\w+', text, re.MULTILINE))
    class_count = len(re.findall(r'^\s*class\s+\w+', text, re.MULTILINE))
    total = func_count + class_count

    if total < 2:
        return 0.0, f"too few functions/classes ({total})"

    # Count those immediately followed by a docstring
    doc_count = len(_PY_DOCSTR_RE.findall(text))

    coverage = doc_count / total

    # Human: 0.10-0.40; AI: 0.70-1.00
    if coverage >= 0.90:
        score = 90.0
    elif coverage >= 0.70:
        score = 70.0
    elif coverage >= 0.50:
        score = 45.0
    elif coverage >= 0.30:
        score = 20.0
    else:
        score = 5.0

    detail = f"docstring_coverage={coverage:.3f} ({doc_count}/{total} funcs/classes)"
    return score, detail


# ---------------------------------------------------------------------------
# Heuristic: error handling over-engineering
# ---------------------------------------------------------------------------

def compute_error_handling_density(text: str, lines: list[str]) -> tuple[float, list[tuple[int, str, str]], str]:
    """
    AI wraps almost everything in try/except.
    High try/except density relative to total lines = AI pattern.

    Returns (score 0-100, flagged_lines, detail)
    """
    total_lines = max(1, len([l for l in lines if l.strip()]))
    try_count = len(_PY_TRYEXCEPT_RE.findall(text))

    # Also check JS-style try
    js_try = len(re.findall(r'\btry\s*\{', text))
    total_try = try_count + js_try

    density = total_try / total_lines * 100  # per 100 lines

    # Human: 0-2 per 100 lines; AI: 4-15 per 100 lines
    if density >= 8:
        score = 85.0
    elif density >= 5:
        score = 65.0
    elif density >= 3:
        score = 40.0
    elif density >= 1.5:
        score = 20.0
    else:
        score = 5.0

    # Flag try: lines
    flagged: list[tuple[int, str, str]] = []
    if score >= 40:
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if re.match(r'^try\s*[:{]', stripped):
                flagged.append((i, stripped, "AI-style excessive try/except block"))

    detail = f"try_except_density={density:.2f}/100L ({total_try} blocks in {total_lines} lines)"
    return score, flagged[:4], detail


# ---------------------------------------------------------------------------
# Heuristic: type annotation density (Python)
# ---------------------------------------------------------------------------

def compute_type_annotation_density(text: str) -> tuple[float, str]:
    """
    AI annotates EVERY parameter and return type.
    Count: functions with any type hint / total functions.

    Returns (score 0-100, detail)
    """
    all_funcs = _PY_FUNC_RE.findall(text)
    if len(all_funcs) < 2:
        return 0.0, f"too few functions ({len(all_funcs)})"

    # Count functions that have at least one : type annotation in params or ->
    annotated = 0
    for m in re.finditer(
        r'def\s+\w+\(([^)]*)\)\s*(?P<ret>->[^:]+)?:',
        text, re.MULTILINE
    ):
        params = m.group(1)
        ret    = m.group("ret") or ""
        if ":" in params or ret.strip():
            annotated += 1

    total_funcs = len(all_funcs)
    density = annotated / total_funcs if total_funcs > 0 else 0

    # Human: 0.0-0.30; AI: 0.70-1.00
    if density >= 0.90:
        score = 85.0
    elif density >= 0.70:
        score = 65.0
    elif density >= 0.50:
        score = 40.0
    elif density >= 0.30:
        score = 15.0
    else:
        score = 0.0

    detail = f"type_annotation_density={density:.3f} ({annotated}/{total_funcs} functions)"
    return score, detail


# ---------------------------------------------------------------------------
# Heuristic: code line length uniformity
# ---------------------------------------------------------------------------

def compute_code_line_uniformity(lines: list[str]) -> tuple[float, str]:
    """
    AI code has eerily uniform non-empty line lengths.
    Similar to sentence uniformity but applied to code lines.

    Returns (score 0-100, detail)
    """
    code_lines = [len(l) for l in lines if l.strip() and not l.strip().startswith(("#", "//", "/*", "*"))]
    if len(code_lines) < 10:
        return 0.0, "too few code lines"

    mean_len = statistics.mean(code_lines)
    std_len  = statistics.stdev(code_lines) if len(code_lines) > 1 else 0
    cov      = std_len / mean_len if mean_len > 0 else 0

    # Human CoV: 0.50-0.90; AI: 0.25-0.50
    if cov < 0.30:
        score = 80.0
    elif cov < 0.40:
        score = 60.0
    elif cov < 0.55:
        score = 35.0
    else:
        score = 5.0

    detail = f"code_line_len_CoV={cov:.3f} (mean={mean_len:.1f}, std={std_len:.1f})"
    return score, detail


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def analyze_code_structure(
    text: str,
    lines: list[str],
    file_ext: str = "",
) -> dict:
    """
    Run all code-structure heuristics.
    These work even if ALL comments are removed.

    Returns:
        {
          "score": float (0-100),
          "flagged_lines": [(line_no, text, reason), ...],
          "details": {name: detail_str},
        }
    """
    identifiers = _extract_identifiers(text)

    il_score, il_flagged, il_detail  = compute_identifier_length(identifiers, lines)
    av_score, av_flagged, av_detail  = compute_ai_vocab_in_identifiers(identifiers, lines)
    sn_score, sn_detail              = compute_short_name_deficit(identifiers)
    nc_score, nc_detail              = compute_naming_consistency(identifiers)
    ds_score, ds_detail              = compute_docstring_coverage(text)
    eh_score, eh_flagged, eh_detail  = compute_error_handling_density(text, lines)
    ta_score, ta_detail              = compute_type_annotation_density(text)
    cl_score, cl_detail              = compute_code_line_uniformity(lines)

    weights = {
        "identifier_length":   0.25,
        "ai_vocab":            0.25,
        "short_name_deficit":  0.10,
        "naming_consistency":  0.05,
        "docstring_coverage":  0.15,
        "error_handling":      0.05,
        "type_annotations":    0.10,
        "line_uniformity":     0.05,
    }
    composite = (
        weights["identifier_length"]  * il_score +
        weights["ai_vocab"]           * av_score +
        weights["short_name_deficit"] * sn_score +
        weights["naming_consistency"] * nc_score +
        weights["docstring_coverage"] * ds_score +
        weights["error_handling"]     * eh_score +
        weights["type_annotations"]   * ta_score +
        weights["line_uniformity"]    * cl_score
    )

    # Merge flagged lines
    seen: set[int] = set()
    merged: list[tuple[int, str, str]] = []
    for lst in (il_flagged, av_flagged, eh_flagged):
        for entry in lst:
            if entry[0] not in seen:
                seen.add(entry[0])
                merged.append(entry)
    merged.sort(key=lambda x: x[0])

    return {
        "score": round(composite, 2),
        "flagged_lines": merged,
        "details": {
            "Identifier Length":      il_detail,
            "AI Vocab in Identifiers": av_detail,
            "Short-Name Deficit":     sn_detail,
            "Naming Consistency":     nc_detail,
            "Docstring Coverage":     ds_detail,
            "Error Handling Density": eh_detail,
            "Type Annotation Density": ta_detail,
            "Code Line Uniformity":   cl_detail,
        },
    }
