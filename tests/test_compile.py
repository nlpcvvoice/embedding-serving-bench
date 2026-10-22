"""Sanity check: every benchmark script compiles.

The scripts hit live endpoints at import time, so they are compile-checked
rather than imported. Run: python3 tests/test_compile.py
"""
import py_compile
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def test_all_scripts_compile():
    scripts = sorted(SRC.glob("*.py"))
    assert scripts, f"no scripts found in {SRC}"
    for script in scripts:
        py_compile.compile(str(script), doraise=True)


if __name__ == "__main__":
    test_all_scripts_compile()
    print("OK: all scripts compile")
    sys.exit(0)
