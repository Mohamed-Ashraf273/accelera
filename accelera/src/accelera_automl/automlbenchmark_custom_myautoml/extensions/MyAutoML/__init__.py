def version():
    return "local"


def setup(*args, **kwargs):
    return None


def run(*args, **kwargs):
    from .exec import run

    return run(*args, **kwargs)
