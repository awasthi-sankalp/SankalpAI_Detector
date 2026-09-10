"""
Terminal reporter -- streaming, coloured output.

Output contract:
  - Nothing is printed for clean files.
  - For AI-detected files, print immediately (streaming):

  [AI DETECTED] relative/path/to/file.py  (Score: 84/100 | HIGH)
    - L3:  "It is worth noting that the implementation delves into..."
    - L11: "Furthermore, this function is designed to..."
    - L24: "In conclusion, this approach provides..."

  - --verbose adds a breakdown of per-category scores and details.
  - A summary line is printed at the end.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# -- colour support (colorama for Windows ANSI; graceful fallback) -------------

try:
    from colorama import init as _colorama_init, Fore, Style
    _colorama_init(autoreset=True)
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False

    class _Dummy:
        def __getattr__(self, _): return ""

    Fore   = _Dummy()
    Style  = _Dummy()


# -- helpers ------------------------------------------------------------------

def _c(text: str, *codes) -> str:
    """Wrap text in ANSI codes (no-op if colorama not installed)."""
    if not _HAS_COLOR:
        return text
    prefix = "".join(codes)
    return f"{prefix}{text}{Style.RESET_ALL}"


def _truncate(text: str, max_len: int = 110) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def _score_color(score: float):
    """Return colour code based on score severity."""
    if score >= 80:
        return Fore.RED
    elif score >= 65:
        return Fore.YELLOW
    elif score >= 50:
        return Fore.CYAN
    return Fore.WHITE


def _score_bar(score: float, width: int = 12) -> str:
    """Return a small ASCII progress bar, e.g. [########....]"""
    filled = round(score / 100 * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Big ASCII-art welcome banner (shown once at startup)
# ---------------------------------------------------------------------------

_BANNER = r"""
   ____              _             _            _    ___
  / ___|  __ _ _ __ | | ____ _| |_ __    / \  |_ _|
  \___ \ / _` | '_ \| |/ / _` | | '_ \  / _ \  | |
   ___) | (_| | | | |   < (_| | | |_) |/ ___ \ | |
  |____/ \__,_|_| |_|_|\_\__,_|_| .__//_/   \_\___|
                                   |_|

   ____       _            _
  |  _ \  ___| |_ ___  ___| |_ ___  _ __
  | | | |/ _ \ __/ _ \/ __| __/ _ \| '__|
  | |_| |  __/ ||  __/ (__| || (_) | |
  |____/ \___|\__\___|\___|\__\___/|_|

"""


def print_welcome() -> None:
    """Print the big blue welcome banner (called once at startup)."""
    try:
        import pyfiglet
        top = pyfiglet.figlet_format("SankalpAI", font="big")
        bot = pyfiglet.figlet_format("Detector",  font="big")
        art = top + bot
    except ImportError:
        art = _BANNER

    print()
    for line in art.splitlines():
        print(_c(line, Fore.BLUE, Style.BRIGHT))
    print(_c(
        "          Local AI-Generated Text Detector  |  28 Heuristics  |  No Models  |  No Internet",
        Fore.CYAN,
    ))
    print()


# -- main reporter class -------------------------------------------------------

class Reporter:

    def print_header(
        self,
        root: Path,
        exclude: list[str],
        sensitivity: str,
        threshold: int,
    ) -> None:
        print()
        print(_c("=" * 72, Fore.CYAN))
        print(_c("  AI TEXT DETECTOR  --  Local, no models, no internet", Fore.CYAN, Style.BRIGHT))
        print(_c("=" * 72, Fore.CYAN))
        print(f"  Folder      : {root}")
        if exclude:
            print(f"  Excluded    : {', '.join(exclude)}")
        print(f"  Sensitivity : {sensitivity.upper()}  (threshold >= {threshold}/100)")
        print(_c("-" * 72, Fore.CYAN))
        print()

    # -- per-file result -------------------------------------------------------

    def print_result(
        self,
        filepath: Path,
        result: dict,
        root: Path,
        verbose: bool = False,
    ) -> None:
        """Print a detected-AI result immediately (streaming)."""
        score = result["score"]
        grade = result["grade"]
        wc    = result["word_count"]
        lc    = result["line_count"]
        short_note = "  [short file]" if result.get("too_short") else ""

        # Relative path for cleaner display
        try:
            rel = filepath.relative_to(root)
        except ValueError:
            rel = filepath

        col = _score_color(score)

        # -- header line -------------------------------------------------------
        print(
            _c("[AI DETECTED]", col, Style.BRIGHT)
            + "  "
            + _c(str(rel), Style.BRIGHT)
            + "  "
            + _c(f"(Score: {score:.0f}/100 | {grade})", col)
            + _c(f"  [{wc}w / {lc}L]{short_note}", Fore.WHITE)
        )

        # -- verbose: per-category scores -------------------------------------
        if verbose:
            print(_c("  +-- Heuristic Breakdown", Fore.CYAN))
            cat_scores  = result.get("heuristic_scores",  {})
            cat_details = result.get("heuristic_details", {})

            category_display = {
                "discourse":      "Discourse / Rhetoric",
                "code_structure": "Code Structure / Names",
                "syntactic":      "Syntactic / Structure",
                "statistical":    "Statistical / Info",
                "lexical":        "Lexical / Vocabulary",
            }
            for cat, display_name in category_display.items():
                cscore = cat_scores.get(cat, 0.0)
                bar    = _score_bar(cscore)
                print(
                    _c(f"  |  {display_name:<24}", Fore.CYAN)
                    + f" {bar}  "
                    + _c(f"{cscore:5.1f}/100", _score_color(cscore))
                )
                if cat in cat_details:
                    for hname, hdetail in cat_details[cat].items():
                        print(_c(f"  |    . {hname}: {hdetail}", Fore.WHITE))

            print(_c("  +--", Fore.CYAN))

        # -- flagged lines -----------------------------------------------------
        flagged = result.get("flagged_lines", [])
        if flagged:
            for lineno, text, reason in flagged:
                display_text = _truncate(text, 100)
                print(
                    "    "
                    + _c("-", Fore.YELLOW)
                    + "  "
                    + _c(f"L{lineno}:", Fore.YELLOW, Style.BRIGHT)
                    + "  "
                    + _c(f'"{display_text}"', Fore.WHITE)
                )
                print(_c(f"         -> {reason}", Fore.WHITE))
        else:
            print(_c(
                "    (no specific lines flagged -- statistical signal only)",
                Fore.WHITE,
            ))

        print()   # blank line between files for readability

    # -- final summary ---------------------------------------------------------

    def print_summary(self, total: int, flagged: int) -> None:
        clean = total - flagged
        print(_c("-" * 72, Fore.CYAN))
        print(
            _c("  SCAN COMPLETE", Fore.CYAN, Style.BRIGHT)
            + f"  |  {total} files scanned"
            + "  |  "
            + _c(f"{flagged} flagged", Fore.RED if flagged else Fore.GREEN, Style.BRIGHT)
            + "  |  "
            + _c(f"{clean} clean", Fore.GREEN)
        )
        print(_c("=" * 72, Fore.CYAN))
        print()

    def print_skipped(self, filepath: Path, reason: str) -> None:
        """Optional: print skipped files (only used if --debug is added later)."""
        pass
