import sys as _sys, io as _io
# Force UTF-8 output early so Unicode/box-drawing chars work on Windows cp1252
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""
SankalpAI_Detector.py -- Main CLI entry point for the Local AI Text Detector.

Usage:
    python SankalpAI_Detector.py --folder PATH [options]

Options:
    --folder / -f   PATH        Root folder to scan (required)
    --exclude / -e  NAME ...    Folder or file names to exclude
    --sensitivity / -s LEVEL    low | medium (default) | high
    --ext           EXT ...     Only scan these extensions (e.g. .py .js)
    --verbose / -v              Show per-heuristic score breakdown
    --no-color                  Disable coloured output

Examples:
    python SankalpAI_Detector.py --folder .
    python SankalpAI_Detector.py --folder C:\\MyProject --exclude node_modules .git dist __pycache__
    python SankalpAI_Detector.py --folder . --sensitivity high --verbose
    python SankalpAI_Detector.py --folder . --ext .py .js .ts --exclude __pycache__ venv
"""

import argparse
import sys
import os
from pathlib import Path


# -- NLTK auto-setup (silent, runs once) ---------------------------------------

def _ensure_nltk() -> None:
    """Download required NLTK data if not already present (silent)."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            print("  [setup] Downloading NLTK punkt tokenizer (one-time)...", end="", flush=True)
            nltk.download("punkt",     quiet=True)
            nltk.download("punkt_tab", quiet=True)
            print(" done.")
    except ImportError:
        pass   # NLTK not installed -- fallback tokenizers will be used


# -- argument parser -----------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="SankalpAI_Detector",
        description=(
            "Local AI-generated text detector -- "
            "no models, no APIs, no internet required.\n"
            "Scans files using 28 statistical, linguistic, stylometric, and "
            "code-structure heuristics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
sensitivity levels:
  low     flag files scoring >= 72  (fewer false positives)
  medium  flag files scoring >= 55  (balanced -- DEFAULT)
  high    flag files scoring >= 42  (catches more, some noise)

examples:
  python SankalpAI_Detector.py --folder .
  python SankalpAI_Detector.py --folder C:\\MyProject --exclude node_modules .git dist build
  python SankalpAI_Detector.py --folder . --sensitivity high --verbose
  python SankalpAI_Detector.py --folder . --ext .py .js .ts --exclude __pycache__ venv
""",
    )

    parser.add_argument(
        "--folder", "-f",
        required=True,
        metavar="PATH",
        help="Root folder to scan (required)",
    )
    parser.add_argument(
        "--exclude", "-e",
        nargs="*",
        default=[],
        metavar="NAME",
        help=(
            "Folder / file names to exclude at any depth. "
            "e.g. --exclude node_modules .git dist __pycache__ venv"
        ),
    )
    parser.add_argument(
        "--sensitivity", "-s",
        choices=["low", "medium", "high", "paranoid"],
        default="medium",
        metavar="LEVEL",
        help="Detection sensitivity: low | medium (default) | high | paranoid",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=None,
        metavar="EXT",
        help=(
            "Only scan files with these extensions. "
            "e.g. --ext .py .js .txt .md  "
            "(default: all non-binary files)"
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-heuristic score breakdown for each detected file",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable coloured terminal output",
    )

    return parser


# -- main ----------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    # -- Disable colour if requested -------------------------------------------
    if args.no_color:
        os.environ["NO_COLOR"] = "1"
        try:
            import colorama
            colorama.deinit()
        except ImportError:
            pass

    # -- Validate folder --------------------------------------------------------
    folder = Path(args.folder).resolve()
    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder}", file=sys.stderr)
        return 1
    if not folder.is_dir():
        print(f"[ERROR] Path is not a directory: {folder}", file=sys.stderr)
        return 1

    # -- NLTK silent setup ------------------------------------------------------
    _ensure_nltk()

    # -- Import detector modules (after NLTK setup) ----------------------------
    from detector.scanner  import Scanner
    from detector.analyzer import Analyzer
    from detector.reporter import Reporter, print_welcome
    from detector.scorer   import get_threshold

    # -- Welcome banner (big blue text) ----------------------------------------
    print_welcome()

    threshold = get_threshold(args.sensitivity)

    reporter = Reporter()
    reporter.print_header(folder, args.exclude, args.sensitivity, threshold)

    scanner  = Scanner(folder, exclude=args.exclude, extensions=args.ext)
    analyzer = Analyzer()

    total_files   = 0
    flagged_files = 0

    try:
        for filepath in scanner.walk():
            total_files += 1

            # Status ticker (overwrite same line)
            print(f"\r  Scanning: {filepath.name[:60]:<60}", end="", flush=True)

            result = analyzer.analyze(filepath)

            if result is None:
                continue   # unreadable / empty -- skip silently

            if result["score"] >= threshold:
                # Clear the ticker line before printing result
                print("\r" + " " * 70 + "\r", end="")
                flagged_files += 1
                reporter.print_result(
                    filepath, result, folder,
                    verbose=args.verbose,
                )

    except KeyboardInterrupt:
        print("\r" + " " * 70 + "\r", end="")
        print("\n  [interrupted by user]\n")

    # Clear ticker on normal exit
    print("\r" + " " * 70 + "\r", end="")

    reporter.print_summary(total_files, flagged_files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
