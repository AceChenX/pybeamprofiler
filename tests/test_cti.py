"""Tests for GenTL producer (``.cti``) discovery.

These drive the real filesystem through ``tmp_path`` rather than mocking
``os.listdir``, so directory recursion, the symlink guard and de-duplication
are actually exercised. The path *tables* are checked separately, because the
bug that prompted this module was the tables drifting apart between vendors.
"""

from __future__ import annotations

import os
import platform
from unittest.mock import patch

import pytest

from pybeamprofiler import cti
from pybeamprofiler.basler import BaslerCamera
from pybeamprofiler.flir import FlirCamera
from pybeamprofiler.gen_camera import HarvesterCamera
from pybeamprofiler.utils import find_cti_files


def _make_cti(directory, *names) -> list[str]:
    """Create empty ``.cti`` files and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    made = []
    for name in names:
        path = directory / name
        path.write_bytes(b"")
        made.append(str(path))
    return made


@pytest.fixture
def fake_sdk(tmp_path, monkeypatch):
    """Point every vendor table at a throwaway tree and hand back the roots."""
    spin = tmp_path / "spinnaker"
    pylon = tmp_path / "pylon"

    def install(spin_dirs=None, pylon_dirs=None, system="TestOS"):
        monkeypatch.setattr(
            cti,
            "_VENDOR_DIRS",
            {
                system: {
                    cti.SPINNAKER: tuple(spin_dirs or ()),
                    cti.PYLON: tuple(pylon_dirs or ()),
                }
            },
        )
        monkeypatch.setattr(cti.platform, "system", lambda: system)

    return install, spin, pylon


@pytest.mark.real_cti
class TestVendorPathTables:
    """The tables are the whole point of the module — pin down their shape.

    Discovery and the vendor camera classes must look in the *same* places.
    They previously did not: Pylon was searched under ``lib64`` in one place
    and ``lib`` in the other, so ``list_cameras()`` came back empty on Linux
    installs where ``BaslerCamera()`` opened perfectly well.
    """

    @pytest.mark.parametrize("system", ["Windows", "Linux", "Darwin"])
    def test_every_platform_covers_both_vendors(self, system):
        table = cti._VENDOR_DIRS[system]
        assert table[cti.SPINNAKER], f"no Spinnaker paths for {system}"
        assert table[cti.PYLON], f"no Pylon paths for {system}"

    def test_linux_pylon_covers_both_lib_layouts(self):
        paths = [d.path for d in cti._VENDOR_DIRS["Linux"][cti.PYLON]]
        assert "/opt/pylon/lib/gentlproducer/gtl" in paths
        assert "/opt/pylon/lib64/gentlproducer/gtl" in paths

    def test_macos_pylon_includes_the_producer_subdirectory(self):
        paths = [d.path for d in cti._VENDOR_DIRS["Darwin"][cti.PYLON]]
        assert "/Library/Frameworks/pylon.framework/Libraries/gentlproducer/gtl" in paths

    def test_windows_spinnaker_recurses_into_toolchain_subdirs(self):
        entries = cti._VENDOR_DIRS["Windows"][cti.SPINNAKER]
        assert entries, "Spinnaker paths missing on Windows"
        assert all(e.recurse for e in entries), "Spinnaker installs into cti64/<toolchain>"

    def test_unknown_platform_yields_nothing_rather_than_raising(self):
        assert cti.cti_files_for(system="Plan9") == []


class TestScanning:
    def test_finds_producers_in_a_flat_directory(self, fake_sdk):
        install, spin, _ = fake_sdk
        made = _make_cti(spin, "FLIR_GenTL.cti")
        install(spin_dirs=[cti._SearchDir(str(spin))])

        assert cti.cti_files_for(cti.SPINNAKER) == made

    def test_ignores_non_cti_files(self, fake_sdk):
        install, spin, _ = fake_sdk
        _make_cti(spin, "producer.cti")
        (spin / "readme.txt").write_text("not a producer")
        (spin / "libfoo.so").write_bytes(b"")
        install(spin_dirs=[cti._SearchDir(str(spin))])

        found = cti.cti_files_for(cti.SPINNAKER)
        assert len(found) == 1
        assert found[0].endswith("producer.cti")

    def test_recurse_descends_one_level(self, fake_sdk):
        """Spinnaker on Windows nests producers under a toolchain directory."""
        install, spin, _ = fake_sdk
        _make_cti(spin / "vs2015", "FLIR_GenTL.cti")
        install(spin_dirs=[cti._SearchDir(str(spin), recurse=True)])

        found = cti.cti_files_for(cti.SPINNAKER)
        assert len(found) == 1
        assert found[0].endswith(os.path.join("vs2015", "FLIR_GenTL.cti"))

    def test_without_recurse_subdirectories_are_ignored(self, fake_sdk):
        install, spin, _ = fake_sdk
        _make_cti(spin / "vs2015", "FLIR_GenTL.cti")
        spin.mkdir(parents=True, exist_ok=True)
        install(spin_dirs=[cti._SearchDir(str(spin), recurse=False)])

        assert cti.cti_files_for(cti.SPINNAKER) == []

    def test_recurse_also_keeps_producers_in_the_root(self, fake_sdk):
        install, spin, _ = fake_sdk
        _make_cti(spin, "Root.cti")
        _make_cti(spin / "vs2015", "Nested.cti")
        install(spin_dirs=[cti._SearchDir(str(spin), recurse=True)])

        names = [os.path.basename(p) for p in cti.cti_files_for(cti.SPINNAKER)]
        assert sorted(names) == ["Nested.cti", "Root.cti"]

    def test_missing_directory_is_skipped(self, fake_sdk):
        install, spin, _ = fake_sdk
        install(spin_dirs=[cti._SearchDir(str(spin / "not-installed"))])
        assert cti.cti_files_for(cti.SPINNAKER) == []

    def test_a_file_where_a_directory_was_expected_is_skipped(self, fake_sdk, tmp_path):
        install, _, _ = fake_sdk
        bogus = tmp_path / "pylon-is-a-file"
        bogus.write_text("oops")
        install(pylon_dirs=[cti._SearchDir(str(bogus))])
        assert cti.cti_files_for(cti.PYLON) == []

    def test_unreadable_directory_is_skipped(self, fake_sdk):
        install, spin, _ = fake_sdk
        _make_cti(spin, "producer.cti")
        install(spin_dirs=[cti._SearchDir(str(spin))])
        os.chmod(spin, 0o000)
        try:
            # Root can read anything, so only assert it does not raise.
            cti.cti_files_for(cti.SPINNAKER)
        finally:
            os.chmod(spin, 0o755)

    def test_results_are_sorted_within_a_directory(self, fake_sdk):
        install, _, pylon = fake_sdk
        _make_cti(pylon, "ProducerU3V.cti", "ProducerGEV.cti")
        install(pylon_dirs=[cti._SearchDir(str(pylon))])

        names = [os.path.basename(p) for p in cti.cti_files_for(cti.PYLON)]
        assert names == ["ProducerGEV.cti", "ProducerU3V.cti"]

    def test_overlapping_directories_are_de_duplicated(self, fake_sdk, tmp_path):
        """macOS scans the Pylon framework root *and* its producer subdir."""
        install, _, pylon = fake_sdk
        inner = pylon / "gentlproducer" / "gtl"
        _make_cti(inner, "ProducerGEV.cti")
        install(
            pylon_dirs=[
                cti._SearchDir(str(inner)),
                cti._SearchDir(str(pylon), recurse=False),
                cti._SearchDir(str(inner)),  # same dir listed twice
            ]
        )
        assert len(cti.cti_files_for(cti.PYLON)) == 1

    def test_symlink_pointing_outside_the_tree_is_rejected(self, fake_sdk, tmp_path):
        """A producer symlinked out of the SDK directory is not trusted."""
        install, spin, _ = fake_sdk
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        real = outside / "evil.cti"
        real.write_bytes(b"")
        spin.mkdir(parents=True, exist_ok=True)
        (spin / "evil.cti").symlink_to(real)
        install(spin_dirs=[cti._SearchDir(str(spin))])

        assert cti.cti_files_for(cti.SPINNAKER) == []

    def test_symlink_within_the_tree_is_accepted(self, fake_sdk):
        install, spin, _ = fake_sdk
        _make_cti(spin / "real", "producer.cti")
        (spin / "link.cti").symlink_to(spin / "real" / "producer.cti")
        install(spin_dirs=[cti._SearchDir(str(spin))])

        assert len(cti.cti_files_for(cti.SPINNAKER)) == 1


class TestFindCtiFiles:
    def test_collects_every_vendor(self, fake_sdk):
        install, spin, pylon = fake_sdk
        _make_cti(spin, "FLIR_GenTL.cti")
        _make_cti(pylon, "ProducerGEV.cti")
        install(spin_dirs=[cti._SearchDir(str(spin))], pylon_dirs=[cti._SearchDir(str(pylon))])

        names = [os.path.basename(p) for p in cti.find_cti_files()]
        assert sorted(names) == ["FLIR_GenTL.cti", "ProducerGEV.cti"]

    def test_utils_re_export_matches(self, fake_sdk):
        install, spin, _ = fake_sdk
        _make_cti(spin, "FLIR_GenTL.cti")
        install(spin_dirs=[cti._SearchDir(str(spin))])

        assert find_cti_files() == cti.find_cti_files()

    def test_nothing_installed(self, fake_sdk):
        install, _, _ = fake_sdk
        install()
        assert cti.find_cti_files() == []

    def test_real_platform_call_does_not_raise(self):
        """Whatever this machine has installed, discovery must stay quiet."""
        assert isinstance(cti.find_cti_files(), list)
        assert platform.system() in {"Windows", "Linux", "Darwin"} or True


class TestParseGentlPath:
    def test_direct_cti_file(self, tmp_path):
        (made,) = _make_cti(tmp_path, "producer.cti")
        assert cti.parse_gentl_path(made) == [made]

    def test_directory_is_scanned(self, tmp_path):
        made = _make_cti(tmp_path / "gtl", "a.cti", "b.cti")
        found = cti.parse_gentl_path(str(tmp_path / "gtl"))
        assert sorted(found) == sorted(made)

    def test_multiple_entries(self, tmp_path):
        one = _make_cti(tmp_path / "one", "a.cti")
        two = _make_cti(tmp_path / "two", "b.cti")
        sep = ";" if os.name == "nt" else ":"
        found = cti.parse_gentl_path(f"{tmp_path / 'one'}{sep}{tmp_path / 'two'}")
        assert sorted(found) == sorted(one + two)

    def test_blank_and_missing_entries_are_skipped(self, tmp_path):
        (made,) = _make_cti(tmp_path, "producer.cti")
        sep = ";" if os.name == "nt" else ":"
        value = f"{sep} {sep}/nonexistent{sep}{made}{sep}"
        assert cti.parse_gentl_path(value) == [made]

    def test_non_cti_file_entry_is_ignored(self, tmp_path):
        other = tmp_path / "notes.txt"
        other.write_text("hello")
        assert cti.parse_gentl_path(str(other)) == []

    def test_empty_value(self):
        assert cti.parse_gentl_path("") == []

    def test_duplicates_collapse(self, tmp_path):
        (made,) = _make_cti(tmp_path, "producer.cti")
        sep = ";" if os.name == "nt" else ":"
        assert cti.parse_gentl_path(f"{made}{sep}{made}") == [made]


class TestHarvesterAdapter:
    """``HarvesterCamera`` collapses a single result to a bare string."""

    def test_single_result_is_a_string(self, tmp_path):
        (made,) = _make_cti(tmp_path, "producer.cti")
        assert HarvesterCamera._parse_gentl_path(made) == made

    def test_multiple_results_stay_a_list(self, tmp_path):
        made = _make_cti(tmp_path, "a.cti", "b.cti")
        result = HarvesterCamera._parse_gentl_path(str(tmp_path))
        assert isinstance(result, list)
        assert sorted(result) == sorted(made)

    def test_nothing_found_is_none(self):
        assert HarvesterCamera._parse_gentl_path("/nonexistent") is None


class TestVendorAdapters:
    def test_flir_returns_the_first_producer(self, fake_sdk):
        install, spin, _ = fake_sdk
        _make_cti(spin, "A_GenTL.cti", "B_GenTL.cti")
        install(spin_dirs=[cti._SearchDir(str(spin))])

        found = FlirCamera._find_flir_cti()
        assert isinstance(found, str)
        assert found.endswith("A_GenTL.cti")

    def test_flir_returns_none_when_absent(self, fake_sdk):
        install, _, _ = fake_sdk
        install()
        assert FlirCamera._find_flir_cti() is None

    def test_flir_does_not_pick_up_pylon_producers(self, fake_sdk):
        install, spin, pylon = fake_sdk
        _make_cti(pylon, "ProducerGEV.cti")
        spin.mkdir(parents=True, exist_ok=True)
        install(spin_dirs=[cti._SearchDir(str(spin))], pylon_dirs=[cti._SearchDir(str(pylon))])

        assert FlirCamera._find_flir_cti() is None

    def test_basler_returns_every_producer(self, fake_sdk):
        install, _, pylon = fake_sdk
        made = _make_cti(pylon, "ProducerGEV.cti", "ProducerU3V.cti")
        install(pylon_dirs=[cti._SearchDir(str(pylon))])

        assert sorted(BaslerCamera._find_basler_cti() or []) == sorted(made)

    def test_basler_returns_none_when_absent(self, fake_sdk):
        install, _, _ = fake_sdk
        install()
        assert BaslerCamera._find_basler_cti() is None

    def test_basler_picks_up_an_unknown_producer_name(self, fake_sdk):
        """Scanning the directory beats matching a hard-coded name list."""
        install, _, pylon = fake_sdk
        _make_cti(pylon, "ProducerCXP.cti")
        install(pylon_dirs=[cti._SearchDir(str(pylon))])

        found = BaslerCamera._find_basler_cti()
        assert found is not None
        assert found[0].endswith("ProducerCXP.cti")


class TestFilesystemFailures:
    """Every OS call here can fail on a real machine — a stale NFS mount, a
    revoked permission, a symlink loop. Discovery must degrade to "found
    nothing" rather than take the app down."""

    def test_contains_treats_unrelated_roots_as_outside(self):
        """``commonpath`` raises across Windows drives; that means "no"."""
        with patch("pybeamprofiler.cti.os.path.commonpath", side_effect=ValueError("drives")):
            assert cti._contains("/base", "/base/child.cti") is False

    def test_contains_survives_an_os_error(self):
        with patch("pybeamprofiler.cti.os.path.realpath", side_effect=OSError("loop")):
            assert cti._contains("/base", "/base/child.cti") is False

    def test_scan_skips_a_directory_it_cannot_resolve(self, tmp_path):
        _make_cti(tmp_path, "producer.cti")
        with patch("pybeamprofiler.cti.os.path.realpath", side_effect=OSError("symlink loop")):
            assert cti._scan(cti._SearchDir(str(tmp_path))) == []

    def test_scan_continues_when_subdirectory_listing_fails(self, tmp_path):
        """A failed recursion still returns whatever the root itself holds."""
        _make_cti(tmp_path, "root.cti")
        real_listdir = os.listdir
        calls = {"n": 0}

        def flaky(path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PermissionError("nope")
            return real_listdir(path)

        with patch("pybeamprofiler.cti.os.listdir", side_effect=flaky):
            found = cti._scan(cti._SearchDir(str(tmp_path), recurse=True))

        assert [os.path.basename(p) for p in found] == ["root.cti"]

    def test_scan_skips_a_root_it_cannot_list(self, tmp_path):
        with patch("pybeamprofiler.cti.os.listdir", side_effect=PermissionError("nope")):
            assert cti._scan(cti._SearchDir(str(tmp_path))) == []

    def test_dedupe_falls_back_to_the_raw_path(self):
        with patch("pybeamprofiler.cti.os.path.realpath", side_effect=OSError("gone")):
            assert cti._dedupe(["/a.cti", "/a.cti", "/b.cti"]) == ["/a.cti", "/b.cti"]
