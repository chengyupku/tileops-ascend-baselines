#!/usr/bin/env python3
"""Rewrite tests/golden/ from the fixture snapshot.

Run after a deliberate change to what the pages say, and read the diff: it is
the change, stated in the product rather than in the code.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from test_renderer import GOLDEN, PROSE, default_locale, render  # noqa: E402

if __name__ == "__main__":
    tmp = os.path.join(os.path.dirname(GOLDEN), ".golden-tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    # Only the default locale is held: see test_renderer.LOCALE_SUFFIXES.
    pages, _ = render(tmp)
    pages = default_locale(pages)
    shutil.rmtree(GOLDEN, ignore_errors=True)
    os.makedirs(GOLDEN)
    written = 0
    for name, text in pages.items():
        if name in PROSE:
            continue
        with open(os.path.join(GOLDEN, name), "w", encoding="utf-8") as f:
            f.write(text)
        written += 1
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {written} pages to {GOLDEN}")
