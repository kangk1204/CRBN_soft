from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def iter_text_files():
    skip_dirs = {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}
    for path in ROOT.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.is_file() and path.suffix in {".md", ".py", ".toml", ".yml", ".yaml", ".cff", ".txt"}:
            yield path


def test_public_repo_contains_no_private_or_draft_dirs():
    banned_paths = {
        ".omx",
        "archive",
        "docs",
        "fig" + "ures",
        "manu" + "script",
    }
    present = {path.name for path in ROOT.iterdir() if path.name in banned_paths}
    assert present == set()


def test_public_repo_text_has_no_private_or_draft_terms():
    terms = [
        "manu" + "script",
        "jour" + "nal",
        "sub" + "mission",
        "cover " + "letter",
        "review" + "er",
    ]
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(term) for term in terms) + r")\b",
        re.IGNORECASE,
    )
    offenders = []
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{line_no}: {line}")
    assert offenders == []


def test_primary_mode_workflow_does_not_require_render_inputs():
    source = (ROOT / "scripts" / "reproduce_modes.py").read_text(encoding="utf-8")
    assert "render/open_8cvp.pdb" not in source
    assert 'label == "8CVP"' in source
