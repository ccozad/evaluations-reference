"""The `evals` CLI loads a local .env at startup."""

import os
from pathlib import Path

import pytest

from evaluations_reference.cli import main
from evaluations_reference.models import EvalDataset, Lens, LensConfig, TestCase


def test_cli_loads_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A sentinel var (not a real key) so the test never depends on ANTHROPIC_API_KEY.
    monkeypatch.delenv("EVALS_DOTENV_SENTINEL", raising=False)
    (tmp_path / ".env").write_text("EVALS_DOTENV_SENTINEL=loaded\n")

    ds_path = tmp_path / "ds.json"
    EvalDataset(
        name="d", lens_config=LensConfig(enabled=[Lens.CODE]), test_cases=[TestCase(input="q")]
    ).save(ds_path)

    # load_dotenv() searches the current working directory and its parents.
    monkeypatch.chdir(tmp_path)
    assert main(["describe", str(ds_path)]) == 0
    assert os.environ.get("EVALS_DOTENV_SENTINEL") == "loaded"
