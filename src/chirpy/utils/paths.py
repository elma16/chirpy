from pathlib import Path
import os
import shutil


def find_repo_root(start: Path, marker: str = "src/chirpy") -> Path:
    """Climb upward from `start` until a directory containing `marker` exists."""
    for parent in [start] + list(start.parents):
        if (parent / marker).exists():
            return parent
    return start  # fallback if marker not found


def detect_root() -> Path:
    # 1. Env var override
    if "CHIRPY_ROOT" in os.environ:
        return Path(os.environ["CHIRPY_ROOT"]).expanduser()

    # 2. Colab
    if "COLAB_GPU" in os.environ:
        return Path("/content/drive/MyDrive/chirpy")

    # 3. Script
    if "__file__" in globals():
        return find_repo_root(Path(__file__).resolve())

    # 4. Notebook / REPL → start from CWD
    return find_repo_root(Path.cwd())


def resolve_kwave_binary() -> Path | None:
    """
    Try to locate a usable k-Wave binary.

    Resolution order
    ----------------
    1) Env var `CHIRPY_KWAVE_BIN`
    2) Any `kspaceFirstOrder-OMP` available on PATH
    3) None (caller can still supply a path explicitly)

    Note
    ----
    On macOS with the patched absorption build, set `CHIRPY_KWAVE_BIN` to
    your compiled `kspaceFirstOrder-OMP` path (see README for build steps).
    """
    env = os.environ.get("CHIRPY_KWAVE_BIN")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p

    which = shutil.which("kspaceFirstOrder-OMP")
    if which:
        return Path(which).expanduser()

    return None
