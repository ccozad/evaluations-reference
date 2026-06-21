"""Shared pytest fixtures and environment setup.

Force the human-grade lens to auto-skip during tests so the suite never blocks
on stdin, regardless of how the test runner is invoked.
"""

import os

os.environ["SKIP_HUMAN_EVAL"] = "1"
