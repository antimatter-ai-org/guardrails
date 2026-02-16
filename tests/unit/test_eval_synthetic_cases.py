from __future__ import annotations

from pathlib import Path

from app.eval.synthetic_cases import default_synthetic_case_dir, list_local_cases, load_local_case


def test_default_synthetic_case_dir_points_to_local_namespace() -> None:
    assert default_synthetic_case_dir() == Path(".local") / "synthetic_cases"


def test_list_and_load_local_cases(tmp_path: Path) -> None:
    root = tmp_path / "synthetic_cases"
    root.mkdir(parents=True)
    case_path = root / "case_a.txt"
    case_path.write_text("hello", encoding="utf-8")

    names = list_local_cases(root)
    assert names == ["case_a.txt"]
    assert load_local_case("case_a.txt", base_dir=root) == "hello"
