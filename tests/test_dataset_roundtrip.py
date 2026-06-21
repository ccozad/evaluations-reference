"""Dataset roundtrip: load -> save -> load produces identical data, for JSON and YAML."""

from pathlib import Path

from evaluations_reference.models import EvalDataset

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_dataset.json"


def test_json_roundtrip(tmp_path: Path) -> None:
    original = EvalDataset.load(EXAMPLE)
    out = tmp_path / "ds.json"
    original.save(out)
    reloaded = EvalDataset.load(out)
    assert reloaded == original


def test_yaml_roundtrip(tmp_path: Path) -> None:
    original = EvalDataset.load(EXAMPLE)
    out = tmp_path / "ds.yaml"
    original.save(out)
    reloaded = EvalDataset.load(out)
    assert reloaded == original


def test_cross_format_equivalent(tmp_path: Path) -> None:
    original = EvalDataset.load(EXAMPLE)
    json_out = tmp_path / "ds.json"
    yaml_out = tmp_path / "ds.yml"
    original.save(json_out)
    original.save(yaml_out)
    assert EvalDataset.load(json_out) == EvalDataset.load(yaml_out)
