"""Shared CLI utilities for training and export scripts."""


def coerce(value: str):
    """Best-effort cast of a CLI string to int / float / bool."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_extra_args(extra: list[str]) -> dict:
    """Parse unknown argparse tokens into a kwargs dict.

    Supports both ``--key value`` and ``--key=value`` forms.
    Dashes in key names are converted to underscores so they map
    directly to Ultralytics keyword arguments.
    """
    kwargs: dict = {}
    key = None
    for token in extra:
        if token.startswith("--"):
            if "=" in token:
                k, v = token.lstrip("-").split("=", 1)
                kwargs[k.replace("-", "_")] = coerce(v)
            else:
                key = token.lstrip("-").replace("-", "_")
        elif key is not None:
            kwargs[key] = coerce(token)
            key = None
    return kwargs
