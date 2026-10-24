import sys
from pathlib import Path

# Make ``crux`` importable without installing the package (python -m unittest discover -s tests -t .).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
