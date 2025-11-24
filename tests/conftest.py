import os
import sys
import shutil
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
    return tmp_bin


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
