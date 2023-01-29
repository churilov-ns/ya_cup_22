from dataclasses import dataclass


@dataclass
class KNNResult(object):
    track: str
    inv_dist: float = 0.0
