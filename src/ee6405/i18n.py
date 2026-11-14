"""Interface language: Chinese by default, English with ``EE6405_UI_LANG=en``.

Each interface string is written once, next to the code that shows it, as
``tr("中文", "English")``. Two languages do not justify a catalogue file, and
keeping both versions side by side makes a missing translation obvious in review.
"""

from __future__ import annotations

import os

LANG = (os.getenv("EE6405_UI_LANG") or "zh").strip().lower()
ENGLISH = LANG.startswith("en")


def tr(zh: str, en: str) -> str:
    """The string for the configured interface language."""
    return en if ENGLISH else zh
