import os
from contextlib import suppress
from typing import Optional, List, Dict, Generator

from ya_cup_2022.utils.log import LogUnit
from ya_cup_2022.frequency.generator import FrequencyDataGenerator
from ya_cup_2022.frequency import models
from ya_cup_2022.frequency import dto


__all__ = [
    'FrequencyModelPool',
]


_WEIGHTS_FILENAME = '__weights__'


class FrequencyModelPool(object):

    def __init__(
        self, *,
        weights: Optional[Dict[int, float]] = None,
        default_weight: Optional[float] = None,
    ):
        self._abs_model = models.AbsFrequencyModel()
        self._rel_models: Dict[int, models.RelFrequencyModel] = dict()
        self.weights = weights
        if self.weights is None:
            self.weights = dict()
        self.default_weight = default_weight

    def add_model(self, model: models.FrequencyModel):
        if isinstance(model, models.AbsFrequencyModel):
            self._abs_model = model
        elif isinstance(model, models.RelFrequencyModel):
            self._rel_models[model.skip] = model

    def predict_top(
        self,
        track: Optional[str] = None, *,
        cutoff: Optional[int] = None,
        skip: int = 0,
    ) -> Generator[dto.TrackFrequency, None, None]:
        if track is None:
            return self._abs_model.predict_top(cutoff)
        elif skip in self._rel_models:
            return self._rel_models[skip].predict_top(track, cutoff)

    def get_weight(self, skip: int) -> float:
        weight = self.weights.get(skip, self.default_weight)
        return 1.0 / (skip + 1.0) ** 0.5 if weight is None else weight

    def predict_frequency(
        self,
        track: str,
        tracks: Optional[List[str]] = None,
        *, max_skip: Optional[int] = None,
    ) -> dto.CombinedTrackFrequency:
        if tracks is None:
            return dto.CombinedTrackFrequency(
                track, self._abs_model.predict(track))
        else:
            ctf = dto.CombinedTrackFrequency(track, 0.0)
            for i in range(len(tracks) - 1, -1, -1):
                skip = len(tracks) - 1 - i
                if max_skip is not None and skip > max_skip:
                    break
                model = self._rel_models.get(skip)
                if model is not None:
                    ctf.add(skip, model.predict(tracks[i], track),
                            self.get_weight(skip))
            return ctf.eval()

    def predict_rel_top(
        self,
        tracks: List[str], *,
        cutoff: Optional[int] = None,
        max_skip: Optional[int] = None,
    ) -> List[dto.CombinedTrackFrequency]:
        results = dict()
        tracks_set = set(tracks)
        for i in range(len(tracks) - 1, -1, -1):
            skip = len(tracks) - 1 - i
            if max_skip is not None and skip > max_skip:
                break

            model = self._rel_models.get(skip)
            if model is None:
                continue

            for tf in model.predict_top(tracks[i], cutoff):
                if tf.track not in tracks_set:
                    ctf = results.setdefault(
                        tf.track,
                        dto.CombinedTrackFrequency(tf.track, 0.0))
                    ctf.add(skip, tf.frequency, self.get_weight(skip))

        results = [ctf.eval() for ctf in results.values()]
        results.sort(key=lambda r: r.frequency, reverse=True)
        return results if cutoff is None else results[:cutoff]

    @classmethod
    def load(
        cls,
        work_dir: str, *,
        cutoff: Optional[int] = None,
        min_skip: Optional[int] = None,
        max_skip: Optional[int] = None,
        normalized: bool = True,
        verbose: bool = True,
    ) -> 'FrequencyModelPool':
        fms = FrequencyModelPool()
        log = LogUnit(verbose=verbose)
        log.log(f'Loading frequency model pool')

        for filename in os.listdir(work_dir):
            file_path = os.path.join(work_dir, filename)
            if not os.path.isfile(file_path):
                continue

            if filename == _WEIGHTS_FILENAME:
                log.log(f'Loading weights from file {file_path}')
                fms.weights = cls.load_weights(file_path)
                continue

            with suppress(ValueError):
                model_cls, model_params = (
                    FrequencyDataGenerator.filename_2_model(filename))
                model_params['verbose'] = verbose
                model_params['normalized'] = normalized

                model_skip = model_params.get('skip')
                if model_skip is not None:
                    if min_skip is not None and model_skip < min_skip:
                        continue
                    if max_skip is not None and model_skip > max_skip:
                        continue

                model_cutoff = cutoff
                if model_cls == models.AbsFrequencyModel:
                    model_cutoff = None

                log.log(f'Loading model from file {file_path}')
                fms.add_model(model_cls.load_model(
                    file_path, model_cutoff, **model_params
                ))

        log.log(f'Frequency model pool loaded')
        return fms

    @classmethod
    def load_weights(cls, weights_file: str) -> Dict[int, float]:
        weights = dict()
        with open(weights_file, 'rt') as f:
            for line in f:
                skip, weight = line.strip().split(' ')
                weights[int(skip)] = float(weight)
        return weights
