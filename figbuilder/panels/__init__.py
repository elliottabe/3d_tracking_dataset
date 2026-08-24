"""Panel type registry. Importing this package registers all built-in types."""
from __future__ import annotations

from figbuilder.panels import courtship, generic  # noqa: F401
from figbuilder.panels.base import (  # noqa: F401
    PanelType, get_panel_type, list_panel_types, register,
)
