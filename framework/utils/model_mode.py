from contextlib import contextmanager


@contextmanager
def temporary_eval(module):
    """Disable dropout temporarily, preserving frozen/mixed submodule modes."""
    modes = [(child, child.training) for child in module.modules()]
    try:
        module.eval()
        yield module
    finally:
        for child, training in modes:
            child.training = training
