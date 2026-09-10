"""
setup_nltk.py — One-time NLTK data downloader.

Run this once before first use if you want to pre-download NLTK punkt:
    python setup_nltk.py

The main detector (detect.py) also does this automatically on first run.
"""
import sys


def main() -> None:
    try:
        import nltk
    except ImportError:
        print("[ERROR] nltk is not installed.")
        print("        Run: pip install nltk")
        sys.exit(1)

    datasets = [
        ("tokenizers/punkt",     "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    print("Checking NLTK data…")
    for lookup_path, download_id in datasets:
        try:
            nltk.data.find(lookup_path)
            print(f"  ✓  {download_id} already present")
        except LookupError:
            print(f"  ↓  Downloading {download_id}…", end="", flush=True)
            nltk.download(download_id, quiet=True)
            print(" done.")

    print("\nAll NLTK data ready. You can now run: python detect.py --help")


if __name__ == "__main__":
    main()
