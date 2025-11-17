import sys
from datetime import datetime
from pathlib import Path

# -- Path setup -----------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "chirpy"
author = "Wei Liao, Elliott MacNeil"
year = datetime.now().year
copyright = f"{year}"

# -- General config -------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc.typehints",
]

autosummary_generate = True
autodoc_typehints = "description"
autoclass_content = "class"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = []

# -- HTML output ----------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
