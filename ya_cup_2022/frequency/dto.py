from typing import Generator, Optional, Dict
from collections import Counter
from dataclasses import dataclass, field


__all__ = [
    'TrackFrequency',
    'WeightFrequency',
    'CombinedTrackFrequency',
    'FrequencyModelData',
]


@dataclass
class TrackFrequency(object):
    track: str
    frequency: float

    @classmethod
    def from_track_count(
        cls,
        track: str,
        count: int,
        total: int,
    ) -> 'TrackFrequency':
        return cls(
            track=track,
            frequency=count / total if total > 0 else 0.0,
        )


@dataclass
class WeightFrequency(object):
    weight: float
    frequency: float

    @property
    def weighted_frequency(self):
        return self.weight * self.frequency


@dataclass
class CombinedTrackFrequency(TrackFrequency):
    models_data: Dict[int, WeightFrequency] = field(default_factory=dict)

    @classmethod
    def from_track_frequency(
        cls,
        tf: TrackFrequency,
    ) -> 'CombinedTrackFrequency':
        return CombinedTrackFrequency(tf.track, tf.frequency)

    def add(self, skip: int, frequency: float, weight: float = 1.0):
        self.models_data[skip] = WeightFrequency(
            weight, frequency
        )

    def eval(self) -> 'CombinedTrackFrequency':
        self.frequency = sum(
            wf.weighted_frequency
            for wf in self.models_data.values()
        )
        return self


@dataclass
class FrequencyModelData(object):
    total: int = 0
    tracks: Counter = field(default_factory=Counter)

    def add(self, track: str):
        self.tracks[track] += 1
        self.total += 1

    def get(self, track: str, *, normalized: bool = True) -> float:
        if normalized:
            try:
                return self.tracks[track] / self.total
            except ZeroDivisionError:
                return 0.0
        else:
            return self.tracks[track]

    def get_top(
        self,
        top: Optional[int] = None,
        *, normalized: bool = True,
    ) -> Generator[TrackFrequency, None, None]:
        for track, count in self.tracks.most_common(top):
            if normalized:
                yield TrackFrequency.from_track_count(
                    track, count, self.total)
            else:
                yield TrackFrequency(track, count)

    def to_string(self):
        items = (f'{i[0]}-{i[1]}' for i in self.tracks.most_common())
        return f'{self.total}:{",".join(items)}'

    @classmethod
    def from_string(
        cls,
        string: str,
        cutoff: Optional[int] = None,
    ) -> 'FrequencyModelData':
        data = FrequencyModelData()
        total, items = string.split(':')
        for i, item in enumerate(items.split(',')):
            if cutoff is not None and i >= cutoff:
                break
            track, count = item.split('-')
            data.tracks[track] = int(count)

        data.total = int(total)
        return data
