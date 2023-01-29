from typing import Optional, Union, List, Dict, Any

from ya_cup_2022.df import dto
from ya_cup_2022.df.samplers import Sampler
from ya_cup_2022.frequency.pool import FrequencyModelPool

from ya_cup_2022.frequency import dto as f_dto


__all__ = [
    'FMPSampler',
]


class FMPSampler(Sampler):

    def __init__(
        self,
        work_dir: str, *,
        cutoff: Optional[int] = None,
        max_skip: Optional[int] = None,
        extended: bool = False,
        verbose: bool = True,
    ):
        self.work_dir = work_dir
        self.cutoff = cutoff
        self.max_skip = max_skip
        self.extended = extended
        self.verbose = verbose
        self.fmp: Optional[FrequencyModelPool] = None
        self.popular_tracks: Optional[List[dto.SampleItem]] = None

    @property
    def add_features(self) -> List[dto.Feature]:
        features = [dto.Feature('f', float)]
        if self.extended:
            for s in range(self.max_skip + 1):
                features.append(dto.Feature(f'f_{s}', float))
        return features

    def get_sample(
        self,
        tracks: List[str],
        count: int,
        **kwargs,
    ) -> List[dto.SampleItem]:
        self._lazy_init()
        ps = self.fmp.predict_rel_top(
            tracks,
            cutoff=count,
            max_skip=self.max_skip,
        )
        sample = [self._make_sample_item(p) for p in ps]
        return (sample + self.popular_tracks)[:count]

    def get_add_features(
        self,
        track: str,
        tracks: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        self._lazy_init()
        return self._make_sample_item(self.fmp.predict_frequency(
            track, tracks, max_skip=self.max_skip
        )).features

    def _make_sample_item(
        self,
        tf: Union[f_dto.TrackFrequency, f_dto.CombinedTrackFrequency],
    ) -> dto.SampleItem:
        item = dto.SampleItem(tf.track)
        item.features['f'] = tf.frequency
        if self.extended:
            for s in range(self.max_skip + 1):
                f = 0.0
                if isinstance(tf, f_dto.CombinedTrackFrequency):
                    md = tf.models_data.get(s)
                    if md is not None:
                        f = md.weighted_frequency
                item.features[f'f_{s}'] = f
        return item

    def _lazy_init(self):
        if self.fmp is None:
            self.fmp = FrequencyModelPool.load(
                self.work_dir,
                cutoff=self.cutoff,
                max_skip=self.max_skip,
                verbose=self.verbose,
            )
            self.popular_tracks = [
                self._make_sample_item(p)
                for p in self.fmp.predict_top(cutoff=self.cutoff)
            ]
