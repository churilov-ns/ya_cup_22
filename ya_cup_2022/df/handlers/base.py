import abc
from typing import List, Dict, Tuple, Any

from pandas.api.types import CategoricalDtype

from ya_cup_2022.df import dto
from ya_cup_2022.utils.log import LogUnit
from ya_cup_2022.utils.converter import (
    TrackArtistConverterType,
    TrackArtistConverterMixin,
)


__all__ = [
    'DataFrameHandler',
]


def _feature_ind(key: str) -> int:
    return int(key.split('_')[-1])


def _convert_pair(pair: str) -> Tuple[int]:
    return tuple(map(int, pair.split('-')))


class DataFrameHandler(abc.ABC, LogUnit, TrackArtistConverterMixin):

    def __init__(
        self, *,
        options: dto.GeneratorOptions,
        add_features: List[dto.Feature],
        converter: TrackArtistConverterType,
        verbose: bool = True,
    ):
        LogUnit.__init__(self, verbose=verbose)
        TrackArtistConverterMixin.__init__(
            self, converter=converter, verbose=verbose)
        self.options = options
        self.features: Dict[str, dto.Feature] = dict()
        self._generate_features(add_features)

    @abc.abstractmethod
    def init(self):
        raise NotImplemented

    @abc.abstractmethod
    def append(self, obj: dto.Object):
        raise NotImplemented

    @abc.abstractmethod
    def finalize(self):
        raise NotImplemented

    def _generate_features(self, add_features: List[dto.Feature]):
        self.features['user_id'] = dto.Feature('user_id', int)
        for i in range(self.options.n_prev_likes):
            key = f'prev_like_{i}'
            self.features[key] = dto.Feature(key, CategoricalDtype(
                categories=self.converter.all_tracks, ordered=False))
        for i in range(self.options.n_prev_likes):
            key = f'prev_artist_{i}'
            self.features[key] = dto.Feature(key, CategoricalDtype(
                categories=self.converter.all_artists, ordered=False))
        for i in range(self.options.n_fav_artists):
            key = f'fav_artist_{i}'
            self.features[key] = dto.Feature(key, CategoricalDtype(
                categories=self.converter.all_artists, ordered=False))
        for i in range(self.options.n_fav_artists):
            key = f'fav_artist_likes_{i}'
            self.features[key] = dto.Feature(key, int)
        self.features['cur_track_id'] = dto.Feature(
            'cur_track_id', CategoricalDtype(
                categories=self.converter.all_tracks, ordered=False))
        self.features['cur_artist_id'] = dto.Feature(
            'cur_artist_id', CategoricalDtype(
                categories=self.converter.all_artists, ordered=False))
        self.features['sampler_rank'] = dto.Feature('sampler_rank', float)
        for feature in add_features:
            self.features[feature.name] = feature
        if self.options.for_train:
            self.features['like'] = dto.Feature('like', int)
            self.features['rank'] = dto.Feature('like', float)

    @staticmethod
    def _extract_feature(key: str, obj: dto.Object) -> Any:
        if key == 'user_id':
            return obj.user_id
        elif key.startswith('prev_like_'):
            return obj.prev_likes[_feature_ind(key)]
        elif key.startswith('prev_artist_'):
            return obj.prev_artists[_feature_ind(key)]
        elif key.startswith('fav_artist_likes_'):
            return obj.fav_artists_likes[_feature_ind(key)]
        elif key.startswith('fav_artist_'):
            return obj.fav_artists[_feature_ind(key)]
        elif key == 'cur_track_id':
            return obj.cur_track_id
        elif key == 'cur_artist_id':
            return obj.cur_artist_id
        elif key == 'sampler_rank':
            return obj.sampler_rank
        elif key == 'like':
            return obj.like
        elif key == 'rank':
            return obj.rank
        elif key in obj.add_features:
            return obj.add_features[key]
        else:
            raise NotImplementedError(f'Cannot extract feature {key}')
