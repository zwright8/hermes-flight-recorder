"""Side-effect-free Python module discovery."""

from __future__ import annotations

import importlib.machinery
import importlib.util


def module_available_without_import(module: str) -> bool:
    """Return whether a module is discoverable without importing parent packages."""
    parts = module.strip().split(".")
    if not parts or any(not part or not part.isidentifier() for part in parts):
        return False
    try:
        if len(parts) == 1:
            return importlib.util.find_spec(parts[0]) is not None

        spec = importlib.machinery.PathFinder.find_spec(parts[0])
        if spec is None:
            return False
        for index in range(1, len(parts)):
            locations = spec.submodule_search_locations
            if locations is None:
                return False
            qualified_name = ".".join(parts[: index + 1])
            spec = importlib.machinery.PathFinder.find_spec(qualified_name, list(locations))
            if spec is None:
                return False
        return True
    except (ImportError, AttributeError, TypeError, ValueError):
        return False
