from typing import Optional, Dict, List, Any

import pandas as pd

from ya_cup_2022.df import dto
from ya_cup_2022.df.handlers.base import DataFrameHandler


__all__ = [
    'PandasDataFrameHandler',
]


class PandasDataFrameHandler(DataFrameHandler):

    def __init__(self, **parent_params):
        super().__init__(**parent_params)
        self._data: Optional[Dict[str, List[Any]]] = None

    def init(self):
        self._data = dict()
        for feature in self.features:
            self._data[feature] = list()

    def append(self, obj: dto.Object):
        for feature in self.features:
            self._data[feature].append(self._extract_feature(feature, obj))

    def finalize(self) -> pd.DataFrame:
        df = pd.DataFrame(self._data)
        for col in df.columns:
            df[col] = df[col].astype(self.features[col].type)
        self._data = None
        return df
