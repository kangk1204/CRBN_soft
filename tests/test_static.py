from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}
BANNED_PATH_TERMS = (
    "manu" + "script",
    "review" + "er",
    "sub" + "mission",
    "cover_" + "letter",
    "cover-" + "letter",
)


def iter_text_files():
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.is_file() and not path.is_symlink():
            try:
                path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            yield path


def forbidden_release_path(path: Path) -> bool:
    normalized = path.as_posix().lower()
    banned_components = {
        ".omx",
        "archive",
        "docs",
        "fig" + "ures",
        "manu" + "script",
    }
    return any(part.lower() in banned_components for part in path.parts) or any(
        term in normalized for term in BANNED_PATH_TERMS
    )


def test_public_repo_contains_no_private_or_draft_paths_or_symlinks():
    offenders = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.is_symlink() or forbidden_release_path(relative):
            offenders.append(relative.as_posix())
    assert offenders == []


def test_private_path_filter_catches_nested_and_binary_names():
    assert forbidden_release_path(Path("data/nested/manu" + "script/draft.bin"))
    assert forbidden_release_path(Path("data/review" + "er_notes.pdf"))
    assert not forbidden_release_path(Path("data/crbn_ensemble.ens.npz"))


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


def test_main_figure_builders_prepare_output_directories():
    for name in (
        "build_fig1.py",
        "build_fig2.py",
        "build_fig3.py",
        "build_fig4.py",
        "build_fig5_robustness.py",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "prepare_figure_dirs()" in source, name


def test_fig1_uses_the_generated_window_sensitivity_key():
    source = (ROOT / "scripts" / "build_fig1.py").read_text(encoding="utf-8")
    assert "a_paper_rule" in source
    assert "a_primaryer_rule" not in source


def test_fig4_declares_its_required_prepared_panel():
    source = (ROOT / "scripts" / "build_fig4.py").read_text(encoding="utf-8")
    assert "require_prepared_panel" in source
    assert "pymol -cq scripts/render_fig4_pocket.py" in source


def test_ci_runs_python311_lint_compile_and_tests():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'python-version: "3.11"' in workflow
    assert "python -m compileall -q scripts tests" in workflow
    assert "python -m ruff check ." in workflow
    assert "python -m pytest -q" in workflow
    assert "cffconvert --validate --infile CITATION.cff" in workflow
    assert "actions/checkout@v" not in workflow
    assert "actions/setup-python@v" not in workflow
    assert "persist-credentials: false" in workflow
    assert "timeout-minutes: 30" in workflow


def test_reproduce_tensor_does_not_hide_malformed_curation_json():
    source = (ROOT / "scripts" / "reproduce_tensor.py").read_text(encoding="utf-8")
    chain_loader = source.split("# Recorded primary-chain overrides", 1)[1].split(
        "# Committed RCSB entity metadata", 1
    )[0]
    metadata_loader = source.split("# Committed RCSB entity metadata", 1)[1].split(
        "def fetch_cif", 1
    )[0]
    assert "except FileNotFoundError:" in chain_loader
    assert "except Exception" not in chain_loader
    assert "except FileNotFoundError:" in metadata_loader
    assert "except Exception" not in metadata_loader
