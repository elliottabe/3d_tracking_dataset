from jarvis_jax.config import ViTPoseConfig
__all__ = ["ViTPoseConfig", "ViTPose"]

def __getattr__(name):
    if name == "ViTPose":
        from jarvis_jax.models.vitpose import ViTPose
        return ViTPose
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
