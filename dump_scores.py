import sys
from pathlib import Path
from detector.analyzer import Analyzer

a = Analyzer()
folder = Path(r'D:\SatLITE AI\Py')
for f in folder.rglob('*.py'):
    r = a.analyze(f)
    if r:
        print(f"\n{f.name}: Score = {r['score']}")
        print(f"  Heuristics: {r['heuristic_scores']}")
        print(f"  Words: {r['word_count']}, Lines: {r['line_count']}")
        for k, v in r['heuristic_details'].get('code_structure', {}).items():
            print(f"  {k}: {v}")
