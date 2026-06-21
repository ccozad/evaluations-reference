"""Placeholder test so the suite is non-empty until M1 lands real coverage."""

from evaluations_reference import __version__


def test_package_imports() -> None:
    assert __version__ == "0.0.0"
