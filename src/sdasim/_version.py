try:
    from importlib.metadata import version

    __version__ = version("sdasim")
except Exception:
    __version__ = "0.1.0.dev0"
