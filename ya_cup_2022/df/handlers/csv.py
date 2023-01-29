from typing import Optional, Generator, TextIO

import pandas as pd

from ya_cup_2022.df import dto
from ya_cup_2022.df.handlers.base import DataFrameHandler


__all__ = [
    'CSVDataFrameHandler',
]


class CSVDataFrameHandler(DataFrameHandler):

    def __init__(self, *, filename: str, **parent_params):
        super().__init__(**parent_params)
        self.filename = filename
        self._csv: Optional[TextIO] = None

    def init(self):
        self.log(f'Opening file {self.filename}')
        self._csv = open(self.filename, 'wt')
        self._csv.write(f'{",".join(self.features.keys())}\n')

    def append(self, obj: dto.Object):
        to_write = [str(self._extract_feature(feature, obj))
                    for feature in self.features]
        self._csv.write(f'{",".join(to_write)}\n')

    def finalize(self):
        self.log(f'Closing file {self.filename}')
        self._csv.close()
        self._csv = None

    def load(self) -> pd.DataFrame:
        return self._process_df(pd.read_csv(self.filename, dtype=str))

    def load_chunks(
        self,
        chunk_size: int, *,
        max_chunks: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        params = {'dtype': str, 'chunksize': chunk_size}
        with pd.read_csv(self.filename, **params) as reader:
            for i, chunk in enumerate(reader):
                yield self._process_df(chunk)
                if max_chunks is not None and i + 1 >= max_chunks:
                    break

    def load_chunked(
        self,
        chunk_size: int, *,
        max_chunks: Optional[int] = None
    ) -> pd.DataFrame:
        return pd.concat(
            [chunk for chunk in
             self.load_chunks(chunk_size, max_chunks=max_chunks)],
            ignore_index=True,
        )

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            feature = self.features.get(col)
            if feature is not None:
                df[col] = df[col].astype(feature.type)
        return df
