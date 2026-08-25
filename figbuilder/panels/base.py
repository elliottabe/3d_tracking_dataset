"""Panel type registry.

A panel type owns three things: a JSON schema (which drives the front-end
properties form), a list of the bundle datasets it needs, and a `draw` method
that renders into one matplotlib axes. Adding a panel type is Python-only —
the UI is generated from the schema.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import matplotlib.pyplot as plt

from figbuilder.cosmetics import SHARED_SCHEMA


class PanelType:
    """Base class for all panel types."""

    id: str = ""
    label: str = ""
    #: Names of bundle datasets this type requires in `data`.
    needs: List[str] = []
    #: JSON Schema for the `spec` object; drives the properties UI.
    schema: Dict[str, Any] = {"type": "object", "properties": {}}
    #: matplotlib projection this panel needs, e.g. "polar". None = rectilinear.
    projection: Optional[str] = None

    def draw(self, ax: plt.Axes, data: Dict[str, Any], spec: Dict[str, Any]) -> None:
        raise NotImplementedError


_REGISTRY: Dict[str, PanelType] = {}


def register(cls: Type[PanelType]) -> Type[PanelType]:
    """Class decorator: add a panel type to the registry."""
    if not cls.id:
        raise ValueError(f"{cls.__name__} must set a non-empty `id`")
    _REGISTRY[cls.id] = cls()
    return cls


def get_panel_type(type_id: str) -> PanelType:
    try:
        return _REGISTRY[type_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"unknown panel type {type_id!r}; known types: {known}") from None


def full_schema(ptype: PanelType) -> Dict[str, Any]:
    """A panel type's own schema plus the shared cosmetic options.

    Kept separate from `PanelType.schema` so a panel type still declares only
    what is genuinely its own; the shared block is merged in at the one place
    the UI reads.
    """
    schema = dict(ptype.schema)
    schema["properties"] = {**SHARED_SCHEMA, **schema.get("properties", {})}
    return schema


def list_panel_types() -> List[Dict[str, Any]]:
    return [{"id": p.id, "label": p.label, "needs": list(p.needs),
             "schema": full_schema(p), "projection": p.projection}
            for p in sorted(_REGISTRY.values(), key=lambda p: p.id)]
