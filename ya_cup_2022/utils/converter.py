from typing import Dict, Any, Set, Union
from functools import cached_property

from ya_cup_2022.utils.log import LogUnit


__all__ = [
    'TrackArtistConverter',
    'read_track_artist_file',
    'TrackArtistConverterType',
    'TrackArtistConverterMixin',
]


class TrackArtistConverter(object):

    def __init__(self):
        self.data: Dict[str, str] = dict()

    def add(self, track: str, artist: str):
        self.data[track] = artist

    def get(self, track: str, fallback: Any = None) -> Any:
        return self.data.get(track, fallback)

    @cached_property
    def all_tracks(self) -> Set[str]:
        return set(self.data.keys())

    @cached_property
    def all_artists(self) -> Set[str]:
        return set(self.data.values())


def read_track_artist_file(
    file: str, *,
    verbose: bool = False,
) -> TrackArtistConverter:
    log = LogUnit(verbose=verbose)
    log.log(f'Loading tracks/artists from file {file}')

    converter = TrackArtistConverter()
    with open(file, 'rt') as f:
        for line in f:
            if line.startswith('trackId'):
                continue
            converter.add(*line.strip().split(','))

    log.log(f'Total # of tracks: {len(converter.all_tracks)}')
    log.log(f'Total # of artists: {len(converter.all_artists)}')
    return converter


TrackArtistConverterType = Union[TrackArtistConverter, str]


class TrackArtistConverterMixin(object):

    def __init__(
        self, *,
        converter: TrackArtistConverterType,
        verbose: bool = True,
    ):
        if isinstance(converter, TrackArtistConverter):
            self.converter = converter
        else:
            self.converter = read_track_artist_file(
                converter,
                verbose=verbose,
            )
