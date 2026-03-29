import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import pytest

# Ensure the in-tree package is imported ahead of any installed copy
REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from chirpy.utils.paths import resolve_kwave_binary


def _kwave_binary_startup_issue(binary_path: Path) -> str | None:
    try:
        proc = subprocess.run(
            [str(binary_path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError as exc:
        return f"k-Wave binary is not runnable: {exc}"
    except subprocess.TimeoutExpired:
        return None

    stderr = proc.stderr or ""
    if "Library not loaded" in stderr and "libhdf5.310.dylib" in stderr:
        return (
            "skipping k-Wave C++ integration tests on macOS because the binary "
            "hits the known k-wave-python HDF5 mismatch "
            "(issue #661: missing libhdf5.310.dylib)"
        )
    return None


@pytest.fixture(scope="session")
def kwave_bin():
    p = os.environ.get("CHIRPY_KWAVE_BIN") or resolve_kwave_binary()
    if not p:
        pytest.skip("CHIRPY_KWAVE_BIN not set; skipping k-Wave integration tests")
    p = Path(p)
    if not p.is_file():
        pytest.skip("CHIRPY_KWAVE_BIN not set; skipping k-Wave integration tests")
    # Always work from a writable temp copy to avoid chmod issues in kwave executor
    tmp_dir = Path(tempfile.mkdtemp(prefix="chirpy_kwave_"))
    tmp_bin = tmp_dir / p.name
    shutil.copy2(p, tmp_bin)
    tmp_bin.chmod(tmp_bin.stat().st_mode | 0o755)
    startup_issue = _kwave_binary_startup_issue(tmp_bin)
    if startup_issue:
        pytest.skip(startup_issue)
    return tmp_bin


@pytest.fixture
def no_custom_kwave_binary(monkeypatch):
    monkeypatch.delenv("CHIRPY_KWAVE_BIN", raising=False)


@pytest.fixture(scope="session")
def installed_kwave_cpp_binary():
    kwave = pytest.importorskip("kwave")
    binary_name = "kspaceFirstOrder-OMP.exe" if sys.platform.startswith("win") else "kspaceFirstOrder-OMP"
    binary_path = Path(kwave.BINARY_PATH) / binary_name
    if not binary_path.is_file():
        pytest.skip(f"Installed k-Wave C++ binary not found at {binary_path}")
    startup_issue = _kwave_binary_startup_issue(binary_path)
    if startup_issue:
        pytest.skip(startup_issue)
    return binary_path


@pytest.fixture(scope="session")
def tiny_grid():
    from chirpy.geometry import ImageGrid2D

    return ImageGrid2D(nx=64, ny=64, dx=1e-3)


@pytest.fixture(scope="session")
def ring8(tiny_grid):
    from chirpy.geometry import TransducerArray2D

    return TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=8, r=None)


@pytest.fixture(scope="session")
def ring32(tiny_grid):
    from chirpy.geometry import TransducerArray2D

    return TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=32, r=None)


@pytest.fixture(scope="session")
def gaussian_pulse():
    from chirpy.signals import GaussianModulatedPulse

    return GaussianModulatedPulse(f0=3e5, frac_bw=0.75, amp=1.0)


@pytest.fixture(scope="session")
def record_time(tiny_grid):
    c0 = 1500.0
    width = tiny_grid.extent[1] - tiny_grid.extent[0]
    return 1.2 * width / c0


@pytest.fixture(scope="session")
def c0():
    return 1500.0
