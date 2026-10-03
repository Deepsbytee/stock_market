"""
conftest.py
===========
Ensures the repo root is on sys.path so that `import backend.features`
resolves correctly from any test file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
