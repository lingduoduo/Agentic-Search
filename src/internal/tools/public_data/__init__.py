"""Keyless public data-source tools (weather, markets, reference, geo).

Each theme module exposes ``build_*_tool()`` factories returning FunctionTools.
``public_data_tools()`` collects them; see ``knowledge_base.seed_tools``.
"""

from __future__ import annotations
