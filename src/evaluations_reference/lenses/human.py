"""Lens 3: human-grade.

Prompts for a 1-5 rating per test case in interactive CLI mode. Auto-skips when
stdin is not a TTY or when ``SKIP_HUMAN_EVAL=1`` is set; a skipped rating is
recorded as ``None`` with a note rather than blocking the run.
"""

from __future__ import annotations

import os
import sys

from evaluations_reference.models import TestCase


def should_skip() -> tuple[bool, str]:
    """Return ``(skip, reason)`` for whether human grading can run here."""
    if os.environ.get("SKIP_HUMAN_EVAL") == "1":
        return True, "SKIP_HUMAN_EVAL=1"
    if not sys.stdin.isatty():
        return True, "stdin is not a TTY"
    return False, ""


def prompt_rating(case: TestCase, output: str) -> int:
    """Prompt the grader for a 1-5 rating, re-asking until a valid one is given."""
    print("\n" + "=" * 60, file=sys.stderr)
    print(f"Input:  {case.input}", file=sys.stderr)
    if case.expected is not None:
        print(f"Expected: {case.expected}", file=sys.stderr)
    print(f"Output: {output}", file=sys.stderr)
    while True:
        raw = input("Rate 1-5 (1=poor, 5=excellent): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("Please enter a whole number from 1 to 5.", file=sys.stderr)
