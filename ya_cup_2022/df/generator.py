from collections import Counter
from typing import Optional, List, Type, Dict, Any
from contextlib import contextmanager

from ya_cup_2022.df import dto
from ya_cup_2022.df.samplers import Sampler
from ya_cup_2022.df.handlers import DataFrameHandler
from ya_cup_2022.utils.log import LogUnit
from ya_cup_2022.utils.converter import (
    TrackArtistConverterMixin,
    TrackArtistConverterType,
)


class DataFrameGenerator(LogUnit, TrackArtistConverterMixin):

    def __init__(
        self, *,
        sampler: Sampler,
        options: dto.GeneratorOptions,
        converter: TrackArtistConverterType,
        handler_cls: Type[DataFrameHandler],
        handler_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        LogUnit.__init__(self, verbose=verbose)
        TrackArtistConverterMixin.__init__(
            self, converter=converter, verbose=verbose)
        self.sampler = sampler
        self.options = options
        self.handler_cls = handler_cls
        self.handler_params = handler_params
        if self.handler_params is None:
            self.handler_params = dict()

        self.handler: Optional[DataFrameHandler] = None
        self.appends_count: Optional[int] = None
        self.like_miss_count: Optional[int] = None

    def init(self):
        if self.handler is not None:
            raise RuntimeError('Already initialized')

        self.handler = self.handler_cls(
            options=self.options,
            add_features=self.sampler.add_features,
            converter=self.converter,
            verbose=self.verbose,
            **self.handler_params,
        )

        self.log('Starting DataFrame generation')
        self.handler.init()
        self.appends_count = 0
        self.like_miss_count = 0

    def append(self, user_id: int, tracks: List[str]):
        if self.handler is None:
            raise RuntimeError('Not initialized')
        self.appends_count += 1

        liked_track = None
        if self.options.for_train:
            liked_track = tracks.pop()

        sample = self.sampler.get_sample(tracks, self.options.sample_size)
        sample_size = len(sample)

        liked_track_ind = None
        if liked_track is not None:
            for i, item in enumerate(sample):
                if item.track == liked_track:
                    liked_track_ind = i
                    break
            if liked_track_ind is None:
                self.like_miss_count += 1
                return

        obj = self._create_object(user_id, tracks)
        for i, item in enumerate(sample):
            if item.track == liked_track:
                continue
            obj.cur_track_id = item.track
            obj.cur_artist_id = self.converter.get(item.track)
            obj.sampler_rank = (sample_size - i) / sample_size
            obj.add_features = item.features
            if self.options.for_train:
                obj.like = 0
                if i > liked_track_ind:
                    obj.rank = obj.sampler_rank
                else:
                    obj.rank = (sample_size - i - 1) / sample_size
            self.handler.append(obj)

        if liked_track is not None:
            obj.cur_track_id = liked_track
            obj.cur_artist_id = self.converter.get(liked_track)
            obj.sampler_rank = (sample_size - liked_track_ind) / sample_size
            obj.add_features = self.sampler.get_add_features(
                liked_track, tracks)
            obj.like = 1
            obj.rank = 1.0
            self.handler.append(obj)

    def finalize(self) -> Any:
        if self.handler is None:
            raise RuntimeError('Not initialized')
        self.log('Finalizing DataFrame generation')
        result = self.handler.finalize()

        self.log(f'Appends count: {self.appends_count}')
        if self.appends_count > 0:
            self.log(f'Like miss count: {self.like_miss_count} '
                     f'({self.like_miss_count * 100 / self.appends_count} %)')
        self.handler = None
        self.appends_count = None
        self.like_miss_count = None
        return result

    @contextmanager
    def begin(self):
        self.init()
        try:
            yield
        finally:
            self.finalize()

    def single(self, user_id: int, tracks: List[str]) -> Any:
        self.init()
        self.append(user_id, tracks)
        return self.finalize()

    def _create_object(self, user_id: int, tracks: List[str]) -> dto.Object:
        n_tracks = len(tracks)
        obj = dto.Object(
            user_id,
            prev_likes=['0'] * (self.options.n_prev_likes - n_tracks),
            prev_artists=['0'] * (self.options.n_prev_likes - n_tracks),
        )

        artists_likes = Counter()
        for i, track in enumerate(tracks):
            artist = self.converter.get(track)
            artists_likes[artist] += 1
            if i >= n_tracks - self.options.n_prev_likes:
                obj.prev_likes.append(track)
                obj.prev_artists.append(artist)

        artists_likes = artists_likes.most_common(
            self.options.n_fav_artists)
        for artist, likes in artists_likes:
            obj.fav_artists.append(artist)
            obj.fav_artists_likes.append(likes)
        while len(obj.fav_artists) < self.options.n_fav_artists:
            obj.fav_artists.append('0')
            obj.fav_artists_likes.append(0)

        return obj
