"""Release gate: every registered tool must have a corresponding icon.

The MCP ``Icon`` spec requires ``https://`` or ``data:`` URIs, and the
loader at ``parkour_mcp/__init__.py#_load_tool_icon`` resolves a tool
key through ``_ICON_FILES`` to a ``parkour_mcp/assets/icons/<filename>.svg``
file. When a tool is registered but lacks an ``_ICON_FILES`` entry —
or its entry references a missing file — the loader silently returns
``None`` and the tool ships without an icon. This test fails CI when
that gap opens.

Also verifies the regeneration registry in ``scripts/generate_icons.py``
includes a ``Glyph`` row for every icon so ``just icons`` reproduces
the SVG set deterministically. A new icon SVG without a glyph row
would slip past lint and only be caught the next time the script ran.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from parkour_mcp import _ALWAYS_ON_TOOLS, _ICON_FILES, _OPTIONAL_TOOLS

REPO_ROOT = Path(__file__).resolve().parent.parent
ICONS_DIR = REPO_ROOT / "parkour_mcp" / "assets" / "icons"


def _registered_tool_keys() -> set[str]:
    keys: set[str] = {name for name, _ in _ALWAYS_ON_TOOLS}
    keys.update(_OPTIONAL_TOOLS)
    return keys


def _glyph_filenames_from_generator() -> set[str]:
    """Load ``scripts/generate_icons.py``'s GLYPHS without running main()."""
    script_path = REPO_ROOT / "scripts" / "generate_icons.py"
    spec = importlib.util.spec_from_file_location(
        "_test_icons_generate", script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_test_icons_generate"] = module
    spec.loader.exec_module(module)
    return {g.filename for g in module.GLYPHS}


class TestIconCoverage:
    def test_every_tool_has_icon_files_entry(self):
        registered = _registered_tool_keys()
        missing = registered - set(_ICON_FILES)
        assert not missing, (
            f"Tools registered without an _ICON_FILES entry: "
            f"{sorted(missing)}. Add a row to "
            f"parkour_mcp/__init__.py:_ICON_FILES."
        )

    def test_no_orphan_icon_files_entries(self):
        # Every _ICON_FILES key should map to a registered tool. An
        # orphan entry implies the tool was renamed or removed without
        # cleanup.
        registered = _registered_tool_keys()
        orphans = set(_ICON_FILES) - registered
        assert not orphans, (
            f"_ICON_FILES has entries for unregistered tools: "
            f"{sorted(orphans)}. Remove them from "
            f"parkour_mcp/__init__.py:_ICON_FILES."
        )

    def test_every_icon_file_exists_on_disk(self):
        missing = []
        for tool_key, filename in _ICON_FILES.items():
            svg_path = ICONS_DIR / f"{filename}.svg"
            if not svg_path.is_file():
                missing.append(f"{tool_key} → {svg_path.name}")
        assert not missing, (
            f"_ICON_FILES references missing SVGs: {missing}. "
            f"Run `just icons` (or `python3 scripts/generate_icons.py`) "
            f"to regenerate."
        )

    def test_server_icon_exists(self):
        # The server-level icon is referenced by _SERVER_ICON_FILE in
        # __init__.py and must be present alongside the tool icons.
        from parkour_mcp import _SERVER_ICON_FILE
        path = ICONS_DIR / f"{_SERVER_ICON_FILE}.svg"
        assert path.is_file(), (
            f"Server icon missing: {path.name}. "
            f"Run `just icons` to regenerate."
        )


class TestIconRegenerationRegistry:
    def test_every_icon_has_a_glyph_row(self):
        glyph_names = _glyph_filenames_from_generator()
        # Every filename referenced by _ICON_FILES (plus the server icon)
        # must have a corresponding Glyph row so the generator can
        # reproduce the SVG.
        from parkour_mcp import _SERVER_ICON_FILE
        required = set(_ICON_FILES.values()) | {_SERVER_ICON_FILE}
        missing = required - glyph_names
        assert not missing, (
            f"Icons missing from scripts/generate_icons.py:GLYPHS: "
            f"{sorted(missing)}. Add a Glyph(...) row so `just icons` "
            f"can regenerate them."
        )

    def test_no_orphan_glyph_rows(self):
        # Every Glyph row should correspond to a referenced icon. Orphan
        # rows imply the icon was removed but the generator still tries
        # to produce its SVG (clutter, not correctness).
        from parkour_mcp import _SERVER_ICON_FILE
        required = set(_ICON_FILES.values()) | {_SERVER_ICON_FILE}
        glyph_names = _glyph_filenames_from_generator()
        orphans = glyph_names - required
        assert not orphans, (
            f"scripts/generate_icons.py:GLYPHS has entries unused by "
            f"_ICON_FILES: {sorted(orphans)}. Remove the rows or wire "
            f"the icon up in parkour_mcp/__init__.py."
        )
