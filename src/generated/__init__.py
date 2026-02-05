"""Auto-generated modules created by the Meta-Agent self-improvement system.

This package dynamically discovers and imports all generated Python modules
so they can be accessed via `from src.generated import <module_name>`.
"""

import importlib
import os
import logging

logger = logging.getLogger(__name__)

__all__: list[str] = []

_package_dir = os.path.dirname(__file__)

for _filename in sorted(os.listdir(_package_dir)):
    if _filename.endswith(".py") and _filename != "__init__.py":
        _module_name = _filename[:-3]
        try:
            _module = importlib.import_module(f".{_module_name}", __name__)
            globals()[_module_name] = _module
            __all__.append(_module_name)
        except Exception as e:
            logger.warning("Failed to import generated module %s: %s", _module_name, e)
