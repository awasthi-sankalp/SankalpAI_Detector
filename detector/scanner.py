"""
Directory scanner with exclusion support and binary-file detection.
Yields Path objects for every file that should be analyzed.
"""

from __future__ import annotations

import os
from pathlib import Path


# Extensions we will NEVER attempt to read as text
# (common binary / media / compiled formats)
_BINARY_EXTENSIONS = frozenset({
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".ico", ".svg", ".heic", ".heif", ".raw", ".psd",
    # Audio / Video
    ".mp3", ".mp4", ".wav", ".flac", ".aac", ".ogg", ".wma",
    ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    # Archives / compressed
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tgz",
    # Compiled / binary
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".lib",
    ".pyc", ".pyo", ".class", ".jar", ".war", ".ear",
    ".wasm", ".beam",
    # Databases / data blobs
    ".db", ".sqlite", ".sqlite3", ".mdb", ".accdb",
    ".bin", ".dat", ".iso", ".img",
    # Fonts / documents (binary)
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Package lock files / generated artifacts (usually not AI-generated)
    ".lock", ".sum",
})


class Scanner:
    """
    Walk a directory tree, applying name-based exclusions, and yield
    Path objects for all readable (non-binary) files.

    Args:
        root:       Root directory Path.
        exclude:    List of folder/file names (or name patterns) to skip.
                    Comparison is case-insensitive on Windows.
        extensions: If given, only files with these extensions are yielded.
                    e.g. [".py", ".js"]
    """

    def __init__(
        self,
        root: Path,
        exclude: list[str] | None = None,
        extensions: list[str] | None = None,
    ) -> None:
        self.root = root
        # Normalise exclusion names to lowercase for case-insensitive match
        self._exclude: set[str] = {e.lstrip(os.sep).lower() for e in (exclude or [])}
        # Normalise extensions (ensure leading dot, lowercase)
        if extensions:
            self._extensions: set[str] | None = {
                (e if e.startswith(".") else f".{e}").lower()
                for e in extensions
            }
        else:
            self._extensions = None

    # ── internals ─────────────────────────────────────────────────────────────

    def _is_excluded(self, path: Path) -> bool:
        """
        Return True if any component of the path (relative to root)
        matches an exclusion name.
        """
        if not self._exclude:
            return False
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            rel = path

        for part in rel.parts:
            if part.lower() in self._exclude:
                return True
        return False

    def _is_likely_binary(self, path: Path) -> bool:
        """Quick binary check: known extension OR null bytes in first 8 KB."""
        if path.suffix.lower() in _BINARY_EXTENSIONS:
            return True
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
            return b"\x00" in chunk
        except OSError:
            return True

    def _extension_ok(self, path: Path) -> bool:
        """Return True if we should scan this extension."""
        if self._extensions is None:
            return True
        return path.suffix.lower() in self._extensions

    # ── public API ────────────────────────────────────────────────────────────

    def walk(self):
        """
        Yield Path objects for every file that passes all filters.
        Directories that match exclusion rules are pruned entirely.
        """
        for dirpath, dirnames, filenames in os.walk(self.root):
            current_dir = Path(dirpath)

            # Prune excluded sub-directories in-place so os.walk skips them
            dirnames[:] = [
                d for d in dirnames
                if not self._is_excluded(current_dir / d)
            ]

            for fname in filenames:
                filepath = current_dir / fname

                if self._is_excluded(filepath):
                    continue
                if not self._extension_ok(filepath):
                    continue
                if self._is_likely_binary(filepath):
                    continue

                yield filepath
