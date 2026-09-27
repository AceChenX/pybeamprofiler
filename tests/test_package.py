"""The package as installed: its metadata and what importing it costs."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys

import pytest

import pybeamprofiler


def test_version_matches_the_installed_metadata():
    """``__version__`` is written by hand next to pyproject's version."""
    assert pybeamprofiler.__version__ == importlib.metadata.version("pybeamprofiler")


def test_importing_the_package_does_not_load_the_gui():
    """Dash and Flask load only when ``create_app`` is first used."""
    code = (
        "import sys, pybeamprofiler; "
        "assert 'dash' not in sys.modules, 'dash imported eagerly'; "
        "app_factory = pybeamprofiler.create_app; "
        "assert 'dash' in sys.modules; "
        "print(app_factory.__module__)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "pybeamprofiler.dash_app"


def test_unknown_attributes_still_raise():
    with pytest.raises(AttributeError, match="not_a_thing"):
        getattr(pybeamprofiler, "not_a_thing")
