from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}


def tracked_release_paths() -> list[Path]:
    top_level = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert Path(top_level).resolve() == ROOT.resolve()

    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "."],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    tracked = [
        Path(name)
        for name in result.stdout.decode("utf-8").split("\0")
        if name
    ]
    assert tracked
    return tracked


def forbidden_release_path(path: Path) -> bool:
    banned_components = {".omx", "archive", "docs", "draft", "figures", "internal"}
    return any(part.casefold() in banned_components for part in path.parts)


def test_public_repo_contains_no_private_or_draft_paths_or_symlinks():
    offenders = []
    tracked = tracked_release_paths()
    for relative in tracked:
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if (ROOT / relative).is_symlink() or forbidden_release_path(relative):
            offenders.append(relative.as_posix())
    assert offenders == []


def test_tracked_release_surface_is_text_only_and_code_scoped():
    tracked = tracked_release_paths()
    allowed_roots = {".github", "data", "scripts", "tests"}
    allowed_root_files = {
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
        "README.md",
        "SECURITY.md",
        "environment.yml",
        "pytest.ini",
        "ruff.toml",
    }
    allowed_data = {
        Path("data/curation_study_groups.csv"),
        Path("data/curation_study_overrides.csv"),
    }
    binary_suffixes = {
        ".doc",
        ".docx",
        ".jpeg",
        ".jpg",
        ".pdf",
        ".png",
        ".pptx",
        ".rtf",
        ".svg",
        ".xls",
        ".xlsx",
        ".zip",
    }
    offenders = []
    for relative in tracked:
        if len(relative.parts) == 1:
            if relative.as_posix() not in allowed_root_files:
                offenders.append(f"unexpected root file: {relative}")
        elif relative.parts[0] not in allowed_roots:
            offenders.append(f"unexpected tracked area: {relative}")
        if relative.parts[0] == "data" and relative not in allowed_data:
            offenders.append(f"unexpected tracked data file: {relative}")
        if relative.suffix.lower() in binary_suffixes:
            offenders.append(f"binary-like suffix: {relative}")
        raw = (ROOT / relative).read_bytes()
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            offenders.append(f"non-UTF-8 tracked file: {relative}")
        if b"\0" in raw:
            offenders.append(f"NUL byte in tracked file: {relative}")
    assert offenders == []


def test_primary_mode_workflow_does_not_require_render_inputs():
    source = (ROOT / "scripts" / "reproduce_modes.py").read_text(encoding="utf-8")
    assert "render/open_8cvp.pdb" not in source
    assert 'label == "8CVP"' in source


def test_main_figure_builders_use_root_relative_deterministic_exports():
    for name in (
        "build_fig1.py",
        "build_fig2.py",
        "build_fig3.py",
        "build_fig4.py",
        "build_fig5_robustness.py",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'ROOT = Path(__file__).resolve().parents[1]' in source, name
        assert "save_figure_set" in source, name


def test_fig1_uses_the_generated_window_sensitivity_key():
    source = (ROOT / "scripts" / "build_fig1.py").read_text(encoding="utf-8")
    assert "a_paper_rule" in source
    assert "a_primaryer_rule" not in source
    assert "normalized_band = (raw_band - closed_mean) / (open_mean - closed_mean)" in source


def test_fig4_declares_its_required_prepared_panel():
    source = (ROOT / "scripts" / "build_fig4.py").read_text(encoding="utf-8")
    assert "FROZEN_STRUCTURE_SHA256" in source
    assert "_verify_structure_input()" in source


def test_source_data_exporter_and_shared_style_are_public_code():
    exporter = ROOT / "scripts" / "export_figure_source_data.py"
    style = ROOT / "scripts" / "figure_style.py"
    assert exporter.is_file()
    assert style.is_file()
    source = exporter.read_text(encoding="utf-8")
    assert all(f"export_fig{index}()" in source for index in range(1, 6))
    assert 'plt.style.use(["science", "no-latex"])' in style.read_text(encoding="utf-8")


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
