

__all__ = [
    'LogUnit',
]


class LogUnit(object):

    def __init__(self, *, verbose: bool = True):
        self.verbose = verbose

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
