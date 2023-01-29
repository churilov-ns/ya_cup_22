import abc
from typing import List, Dict, Any

from ya_cup_2022.df import dto


__all__ = [
    'Sampler',
]


class Sampler(abc.ABC):

    @property
    @abc.abstractmethod
    def add_features(self) -> List[dto.Feature]:
        raise NotImplemented

    @abc.abstractmethod
    def get_sample(
        self,
        tracks: List[str],
        count: int,
        **kwargs,
    ) -> List[dto.SampleItem]:
        raise NotImplemented

    @abc.abstractmethod
    def get_add_features(
        self,
        track: str,
        tracks: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        raise NotImplemented
