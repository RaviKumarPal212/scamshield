from pathlib import Path


def test_required_project_documents_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").is_file()
    assert (root / "docs" / "project_requirements.md").is_file()
    assert (root / "docs" / "data_sources.md").is_file()
    assert (root / "docs" / "evaluation_plan.md").is_file()
    assert (root / "scripts" / "check_environment.py").is_file()
