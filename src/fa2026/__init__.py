"""fa2026 — CMF data generators for Forum Acusticum 2026."""

__all__ = ["CMFDataGenerator", "vogel_subarray"]


def __getattr__(name):
    if name == "CMFDataGenerator":
        from .generators.generator import CMFDataGenerator

        return CMFDataGenerator
    if name == "vogel_subarray":
        from .physical import vogel_subarray

        return vogel_subarray
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
