"""Locating GenTL producer (``.cti``) files across vendors and platforms.

One table, one search. FLIR, Basler and the generic camera-discovery helper
used to carry three separate copies of these paths, and they had drifted:
the discovery helper looked for Pylon under ``lib64`` on Linux while the
Basler class looked under ``lib``, and it never descended into Spinnaker's
versioned ``cti64\\vs2015``-style subdirectories on Windows. The result was
that ``list_cameras()`` reported nothing on machines where opening the
camera directly worked fine.

The vendor split only decides *where* to look — a ``.cti`` is a ``.cti``, and
Harvesters is happy to load several at once.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SPINNAKER = "spinnaker"
PYLON = "pylon"
VENDORS = (SPINNAKER, PYLON)


@dataclass(frozen=True)
class _SearchDir:
    """A directory to scan, and whether its producers sit one level down.

    Spinnaker on Windows installs into a per-toolchain subdirectory
    (``cti64\\vs2015``), so that root needs one level of recursion; every
    other location holds the ``.cti`` files directly.

    ``root`` is the SDK installation the directory belongs to. A ``.cti``
    that is a symlink is followed only if it stays inside it -- a versioned
    SDK may link ``gtl/Producer.cti`` to ``../Producer-7.4.cti`` -- and one
    pointing anywhere else is ignored. Without a root, the scanned directory
    itself is the limit.
    """

    path: str
    recurse: bool = False
    root: str | None = None


# Ordered most- to least-specific: a hit in a versioned SDK directory should
# win over a catch-all like ``/usr/local/lib``.
_VENDOR_DIRS: dict[str, dict[str, tuple[_SearchDir, ...]]] = {
    "Windows": {
        SPINNAKER: (
            _SearchDir(
                r"C:\Program Files\Teledyne\Spinnaker\cti64",
                recurse=True,
                root=r"C:\Program Files\Teledyne\Spinnaker",
            ),
            _SearchDir(
                r"C:\Program Files\FLIR Systems\Spinnaker\cti64",
                recurse=True,
                root=r"C:\Program Files\FLIR Systems\Spinnaker",
            ),
        ),
        PYLON: tuple(
            _SearchDir(
                rf"C:\Program Files\Basler\pylon {v}\Runtime\x64",
                root=rf"C:\Program Files\Basler\pylon {v}",
            )
            for v in ("8", "7", "6", "5")
        ),
    },
    "Linux": {
        SPINNAKER: (_SearchDir("/opt/spinnaker/lib/flir-gentl", root="/opt/spinnaker"),),
        PYLON: (
            # ``/opt/pylon`` is a symlink to the newest install; both lib and
            # lib64 layouts exist in the wild depending on SDK vintage.
            _SearchDir("/opt/pylon/lib/gentlproducer/gtl", root="/opt/pylon"),
            _SearchDir("/opt/pylon/lib64/gentlproducer/gtl", root="/opt/pylon"),
            _SearchDir("/opt/pylon5/lib/gentlproducer/gtl", root="/opt/pylon5"),
            _SearchDir("/opt/pylon5/lib64/gentlproducer/gtl", root="/opt/pylon5"),
        ),
    },
    "Darwin": {
        SPINNAKER: (
            _SearchDir("/usr/local/lib/spinnaker-gentl", root="/usr/local/lib"),
            _SearchDir(
                "/Library/Application Support/FLIR/Spinnaker/lib",
                root="/Library/Application Support/FLIR/Spinnaker",
            ),
            # Broad fallback for hand-installed producers. Last, so a real SDK
            # directory always wins.
            _SearchDir("/usr/local/lib"),
        ),
        PYLON: (
            _SearchDir(
                "/Library/Frameworks/pylon.framework/Libraries/gentlproducer/gtl",
                root="/Library/Frameworks/pylon.framework",
            ),
            _SearchDir(
                "/Library/Frameworks/pylon.framework/Libraries",
                root="/Library/Frameworks/pylon.framework",
            ),
        ),
    },
}


def _contains(parent: str, child: str) -> bool:
    """Is *child* inside *parent* once symlinks are resolved?

    Guards against a symlink in an SDK directory pointing somewhere else
    entirely. ``commonpath`` raises for paths that share no root at all
    (different Windows drives), which is itself a "no" rather than an error.
    """
    try:
        return os.path.commonpath([os.path.realpath(child), parent]) == parent
    except (ValueError, OSError):
        return False


def _scan(entry: _SearchDir) -> list[str]:
    """Return the ``.cti`` files in one search directory, sorted by name."""
    base = entry.path
    if not os.path.isdir(base):
        return []
    try:
        base = os.path.realpath(base)
        limit = os.path.realpath(entry.root) if entry.root else base
    except (OSError, ValueError) as e:
        logger.debug("Could not resolve path %s: %s", base, e)
        return []
    if not _contains(limit, base):
        limit = base  # a root that does not contain the directory is a table mistake

    roots = [base]
    if entry.recurse:
        try:
            roots += [
                os.path.join(base, name)
                for name in sorted(os.listdir(base))
                if os.path.isdir(os.path.join(base, name))
            ]
        except OSError as e:
            logger.debug("Could not list directory %s: %s", base, e)

    found: list[str] = []
    for root in roots:
        try:
            names = sorted(os.listdir(root))
        except OSError as e:
            logger.debug("Could not list directory %s: %s", root, e)
            continue
        for name in names:
            if not name.endswith(".cti"):
                continue
            full = os.path.join(root, name)
            if _contains(limit, full):
                found.append(full)
            else:
                logger.debug("Ignoring %s: it links outside %s", full, limit)
    return found


def _dedupe(paths: list[str]) -> list[str]:
    """Drop repeats (by resolved path) while preserving discovery order.

    Several search directories overlap by design — on macOS the Pylon
    framework root is scanned as well as its ``gentlproducer/gtl`` subdir —
    so the same producer can legitimately turn up twice.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        try:
            key = os.path.realpath(path)
        except (OSError, ValueError):
            key = path
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def cti_files_for(vendor: str | None = None, *, system: str | None = None) -> list[str]:
    """Find GenTL producers for one vendor, or all of them.

    Args:
        vendor: :data:`SPINNAKER`, :data:`PYLON`, or ``None`` for every vendor.
        system: Platform name to search paths for; defaults to the running
            platform. Mostly here so tests can cover all three.

    Returns:
        Absolute ``.cti`` paths, most-specific location first, de-duplicated.
    """
    system = system or platform.system()
    by_vendor = _VENDOR_DIRS.get(system, {})
    vendors = (vendor,) if vendor is not None else VENDORS

    found: list[str] = []
    for name in vendors:
        for entry in by_vendor.get(name, ()):
            found.extend(_scan(entry))
    return _dedupe(found)


def find_cti_files() -> list[str]:
    """Find every GenTL producer installed on this machine.

    The vendor SDK locations come first, then whatever
    ``GENICAM_GENTL64_PATH`` lists. The variable is the GenTL standard's way
    of announcing a producer, and a camera opened through it by
    :class:`~pybeamprofiler.flir.FlirCamera` or
    :class:`~pybeamprofiler.basler.BaslerCamera` has to be found again by
    discovery -- otherwise switching away from it in the GUI is a one-way
    trip.

    Returns:
        Absolute ``.cti`` file paths (empty if no SDK is installed).
    """
    return _dedupe(cti_files_for() + parse_gentl_path(os.environ.get("GENICAM_GENTL64_PATH", "")))


def parse_gentl_path(gentl_path: str) -> list[str]:
    """Expand a ``GENICAM_GENTL64_PATH`` value into concrete ``.cti`` paths.

    Accepts the platform-separated list the GenICam standard defines, where
    each entry is either a directory to scan or a ``.cti`` file itself.

    Args:
        gentl_path: Raw environment variable value.

    Returns:
        Existing ``.cti`` paths, de-duplicated; empty if none resolve.
    """
    separator = ";" if os.name == "nt" else ":"
    found: list[str] = []

    for raw in gentl_path.split(separator):
        path = raw.strip()
        # Quotes around an entry are part of how it was typed, not of the path.
        if len(path) >= 2 and path[0] == path[-1] and path[0] in "\"'":
            path = path[1:-1].strip()
        if not path or not os.path.exists(path):
            continue
        if os.path.isdir(path):
            found.extend(_scan(_SearchDir(path)))
        elif path.endswith(".cti"):
            found.append(path)

    return _dedupe(found)
