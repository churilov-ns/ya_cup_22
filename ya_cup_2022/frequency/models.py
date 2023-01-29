import abc
from collections import defaultdict, deque
from typing import Optional, Generator

from ya_cup_2022.utils import LogUnit
from ya_cup_2022.frequency import dto


__all__ = [
    'FrequencyModel',
    'AbsFrequencyModel',
    'RelFrequencyModel',
]


class FrequencyModel(abc.ABC, LogUnit):

    def __init__(self, *, normalized: bool = True, verbose: bool = True):
        super().__init__(verbose=verbose)
        self.normalized = normalized

    def fit(self, *files: str):
        self.log('Starting fit process')
        for file in files:
            self.partial_fit(file)
        self.log('Fit process finished')

    def partial_fit(self, filename: str):
        self.log(f'Analyzing file: {filename}')
        with open(filename, 'rt') as f:
            for line in f:
                self._update(line.strip())

    @abc.abstractmethod
    def predict(self, *args) -> float:
        raise NotImplemented

    @abc.abstractmethod
    def predict_top(self, *args) -> Generator[dto.TrackFrequency, None, None]:
        raise NotImplemented

    @abc.abstractmethod
    def save_model(self, filename: str):
        raise NotImplemented

    @classmethod
    @abc.abstractmethod
    def load_model(
        cls,
        filename: str,
        cutoff: Optional[int] = None,
        **model_params,
    ) -> 'FrequencyModel':
        raise NotImplemented

    @abc.abstractmethod
    def _update(self, stripped_line: str):
        raise NotImplemented


class AbsFrequencyModel(FrequencyModel):

    def __init__(self, *, normalized: bool = True, verbose: bool = True):
        super().__init__(normalized=normalized, verbose=verbose)
        self.data = dto.FrequencyModelData()

    def predict(self, track: str) -> float:
        return self.data.get(track, normalized=self.normalized)

    def predict_top(
        self,
        cutoff: Optional[int] = None,
    ) -> Generator[dto.TrackFrequency, None, None]:
        return self.data.get_top(cutoff, normalized=self.normalized)

    def save_model(self, filename: str):
        with open(filename, 'wt') as f:
            f.write(self.data.to_string())

    @classmethod
    def load_model(
        cls,
        filename: str,
        cutoff: Optional[int] = None,
        **model_params,
    ) -> 'FrequencyModel':
        with open(filename, 'rt') as f:
            model = AbsFrequencyModel(**model_params)
            model.data = dto.FrequencyModelData.from_string(
                f.read().strip(), cutoff)
            return model

    def _update(self, stripped_line: str):
        for track in stripped_line.split(' '):
            self.data.add(track)


class RelFrequencyModel(FrequencyModel):

    def __init__(
        self, *,
        skip: int = 0,
        normalized: bool = True,
        verbose: bool = True,
    ):
        super().__init__(normalized=normalized, verbose=verbose)
        self.data = defaultdict(dto.FrequencyModelData)
        self.skip = skip

    def predict(self, track_1: str, track_2: str) -> float:
        return self.data[track_1].get(track_2, normalized=self.normalized)

    def predict_top(
        self,
        track: str,
        cutoff: Optional[int] = None,
    ) -> Generator[dto.TrackFrequency, None, None]:
        return self.data[track].get_top(cutoff, normalized=self.normalized)

    def save_model(self, filename: str):
        with open(filename, 'wt') as f:
            for track, data in self.data.items():
                f.write(f'{track}~{data.to_string()}\n')

    @classmethod
    def load_model(
        cls,
        filename: str,
        cutoff: Optional[int] = None,
        **model_params,
    ) -> 'FrequencyModel':
        with open(filename, 'rt') as f:
            model = RelFrequencyModel(**model_params)
            for line in f:
                track, data = line.strip().split('~')
                model.data[track] = dto.FrequencyModelData.from_string(
                    data, cutoff)
            return model

    def _update(self, stripped_line: str):
        tracks = list(stripped_line.split(' '))
        for i in range(len(tracks)):
            j = i - 1 - self.skip
            if j >= 0:
                self.data[tracks[j]].add(tracks[i])
