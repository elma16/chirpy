#!/usr/bin/env python3
from pathlib import Path


from chirpy.utils.paths import find_repo_root, detect_root
from chirpy.utils.progress import Progress, ProgressConfig


def test_find_repo_root_simple(tmp_path: Path):
    """
    find_repo_root should climb up until it finds a directory
    containing the marker path.
    """
    # Make a fake tree:
    # tmp_path/
    #   project/
    #     src/chirpy/
    root = tmp_path / "project"
    marker_dir = root / "src" / "chirpy"
    marker_dir.mkdir(parents=True)

    start = marker_dir / "deeper" / "nested"
    start.mkdir(parents=True)

    found = find_repo_root(start, marker="src/chirpy")
    assert found == root


def test_detect_root_env_override(monkeypatch, tmp_path: Path):
    """
    CHIRPY_ROOT should override everything else in detect_root().
    """
    monkeypatch.setenv("CHIRPY_ROOT", str(tmp_path))
    root = detect_root()
    assert root == tmp_path


def test_detect_root_notebook_fallback(monkeypatch, tmp_path: Path):
    """
    When CHIRPY_ROOT is unset and no __file__ in globals(),
    detect_root should fall back to walking up from cwd.
    """
    monkeypatch.delenv("CHIRPY_ROOT", raising=False)

    # Force no __file__ in globals of chirpy.utils.paths by re-importing
    # in a clean module namespace.
    import chirpy.utils.paths as paths_mod

    # Simulate REPL / notebook by removing __file__ from module globals
    if "__file__" in paths_mod.__dict__:
        del paths_mod.__dict__["__file__"]

    # Make a fake tree: cwd/ src/chirpy exists → root=cwd
    cwd = tmp_path / "work"
    cwd.mkdir(parents=True)
    (cwd / "src" / "chirpy").mkdir(parents=True)
    monkeypatch.chdir(cwd)

    root = paths_mod.detect_root()
    assert root == cwd


def test_progress_iter_no_tty(monkeypatch):
    """
    Progress.iter should degrade to a plain iterator when tqdm is unavailable
    or TTY is not present.
    """
    # Force _tty_available to return False
    monkeypatch.setenv("PYTEST_DISABLE_TTY", "1")

    cfg = ProgressConfig(enabled=True, backend="tqdm")
    prog = Progress(cfg)
    items = list(prog.iter(range(3), total=3, desc="test"))
    assert items == [0, 1, 2]


def test_progress_task_noop_when_disabled():
    """
    Progress.task should yield a no-op update function even when tqdm is not
    available; calling update(n) should not raise.
    """
    prog = Progress(ProgressConfig(enabled=False, backend="none"))
    with prog.task(total=5, desc="phase") as update:
        # update should be callable and harmless
        update(1)
        update(4)
