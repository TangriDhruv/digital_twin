"""
conftest.py
-----------
Adds the backend directory to sys.path so tests can import
backend modules (chunkers, pii_redactor, prompt, etc.) directly.
"""

import sys
from pathlib import Path

# Make `import chunkers`, `import pii_redactor`, etc. work from tests/
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
