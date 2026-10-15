"""Keep the test suite out of the real storage directory.

The app resolves every asset path from ``settings``: the SQLite database
(``storage/platform.db``), models, datasets, projects, experiments, reports
and the Chroma store. ``settings`` is built once at import time from
environment variables, which take priority over the values in ``.env``.

Setting them here — before any ``app`` module is imported — points the whole
suite at a throwaway directory that is wiped at the start of each session,
so running the tests no longer adds projects, models or datasets to the real
database.
"""
import os
import shutil
import tempfile
from pathlib import Path

_TEST_STORAGE = Path(tempfile.gettempdir()) / "agentic_ml_platform_pytest_storage"

# Remove first: an absolute STORAGE_DIR replaces the base directory, and a
# stale directory would leak rows between runs.
if _TEST_STORAGE.exists():
    shutil.rmtree(_TEST_STORAGE, ignore_errors=True)

os.environ["STORAGE_DIR"] = str(_TEST_STORAGE)
os.environ["CHROMA_PERSIST_DIR"] = str(_TEST_STORAGE / "chromadb")
