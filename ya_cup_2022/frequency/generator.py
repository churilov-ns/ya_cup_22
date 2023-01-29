import os
from typing import Generator, Tuple, Type
from pathlib import Path

from ya_cup_2022.utils.log import LogUnit
from ya_cup_2022.utils.wd import clear_dir
from ya_cup_2022.frequency import models


__all__ = [
    'FrequencyDataGenerator',
]


class FrequencyDataGenerator(LogUnit):

    def __init__(
        self,
        work_dir: str,
        max_skip: int = 0,
        *, verbose: bool = True,
    ):
        super().__init__(verbose=verbose)
        self.work_dir = work_dir
        self.max_skip = max_skip

    @staticmethod
    def model_2_filename(model: models.FrequencyModel) -> str:
        if isinstance(model, models.AbsFrequencyModel):
            return 'abs'
        elif isinstance(model, models.RelFrequencyModel):
            return f'rel_{model.skip}'
        else:
            raise ValueError('Unknown model cls')

    @staticmethod
    def filename_2_model(
        filename: str,
    ) -> Tuple[Type[models.FrequencyModel], dict]:
        if filename == 'abs':
            return models.AbsFrequencyModel, dict()
        elif filename.startswith('rel'):
            skip = int(filename.split('_')[-1])
            return models.RelFrequencyModel, {'skip': skip}
        else:
            raise ValueError('Unknown filename pattern')

    def generate(self, *files: str):
        self.log('Starting frequency data generation')

        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        clear_dir(self.work_dir)
        for model in self._models():
            model.fit(*files)
            model.save_model(self._get_model_file(model))

        self.log('End frequency data generation')

    def _models(self) -> Generator[models.FrequencyModel, None, None]:
        self.log('Fitting abs model')
        yield models.AbsFrequencyModel(verbose=self.verbose)

        for skip in range(self.max_skip + 1):
            self.log(f'Fitting rel model with skip={skip}')
            yield models.RelFrequencyModel(skip=skip, verbose=self.verbose)

    def _get_model_file(self, model: models.FrequencyModel) -> str:
        filename = self.model_2_filename(model)
        return os.path.join(self.work_dir, filename)
