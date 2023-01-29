from typing import Type, Optional, Union, List, Generator, Iterable

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)

import nmslib

from ya_cup_2022.knn import dto
from ya_cup_2022.utils.log import LogUnit
from ya_cup_2022.utils.converter import (
    TrackArtistConverter,
    read_track_artist_file,
)


__all__ = [
    'KNNModel',
]


VectorizerCls = Type[Union[CountVectorizer, TfidfVectorizer]]


class KNNModel(LogUnit):

    def __init__(
        self, *,
        vec_cls: VectorizerCls = CountVectorizer,
        min_df: Union[float, int] = 1,
        max_df: Union[float, int] = 1.0,
        max_features: Optional[int] = None,
        vec_add_params: Optional[dict] = None,
        n_neighbors: int = 10,
        space: str = 'cosinesimil_sparse',
        method: str = 'hnsw',
        num_threads: int = 0,
        index_params: Optional[dict] = None,
        query_params: Optional[dict] = None,
        track_artist_file: Optional[str] = None,
        distance_bias: float = 1e-6,
        verbose: bool = True,
    ):
        super().__init__(verbose=verbose)
        self.n_neighbors = n_neighbors
        self.num_threads = num_threads
        self.index_params = index_params
        self.query_params = query_params
        self.distance_bias = distance_bias
        self.likes: Optional[List[List[str]]] = None

        if vec_add_params is None:
            vec_add_params = dict()

        self.cv = vec_cls(
            lowercase=False,
            token_pattern=r'\b\d+\b',
            ngram_range=(1, 1),
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            **vec_add_params,
        )

        self.index = nmslib.init(
            space=space,
            method=method,
            data_type=nmslib.DataType.SPARSE_VECTOR,
        )

        self.converter: Optional[TrackArtistConverter] = None
        if track_artist_file is not None:
            self.converter = read_track_artist_file(
                track_artist_file,
                verbose=verbose,
            )

    def fit(self, *files: str, index_file: Optional[str] = None):
        self.log(' ---> Fitting KNN model')

        content = list()
        self.likes = list()
        for file in files:
            self.log(f'Reading file: {file}')
            with open(file, 'rt') as f:
                for line in f:
                    query = line.strip()
                    content.append(self._convert_query(query))
                    self.likes.append(query.split(' '))

        self.log('Fitting CV')
        X = self.cv.fit_transform(content)
        self.log(f'Space dimension: {X.shape[0]} x {X.shape[1]}')

        if index_file is None:
            self.log('Fitting NN')
            self.index.addDataPointBatch(X)
            self.index.createIndex(self.index_params, self.verbose)
        else:
            self.log('Loading NN')
            self.index.loadIndex(index_file, load_data=True)
        self.index.setQueryTimeParams(self.query_params)

        self.log(' ---> KNN model fit finished')

    def predict(
        self,
        tracks: List[str], *,
        cutoff: Optional[int] = None,
    ) -> List[dto.KNNResult]:
        tracks = self._convert_query(tracks)
        x = self.cv.transform([' '.join(tracks)])
        ids, distances = self.index.knnQueryBatch(x, k=self.n_neighbors)[0]
        return self._process_result(tracks, ids, distances, cutoff=cutoff)

    def batch_predict(
        self,
        queries: List[str], *,
        cutoff: Optional[int] = None,
    ) -> Generator[List[dto.KNNResult], None, None]:
        for i in range(len(queries)):
            queries[i] = self._convert_query(queries[i])
        X = self.cv.transform(queries)
        results = self.index.knnQueryBatch(
            X, k=self.n_neighbors, num_threads=self.num_threads)
        for query, result in zip(queries, results):
            yield self._process_result(
                query,
                result[0],
                result[1],
                cutoff=cutoff,
            )

    def _process_result(
        self,
        query: Union[List[str], str],
        ids: Iterable[int],
        distances: Iterable[float],
        *, cutoff: Optional[int] = None,
    ) -> List[dto.KNNResult]:
        if isinstance(query, str):
            query = query.split(' ')
        tracks_set = set(query)

        results = dict()
        for i, d in zip(ids, distances):
            for like in self.likes[i]:
                if like not in tracks_set:
                    res = results.setdefault(like, dto.KNNResult(like))
                    res.inv_dist += 1.0 / (d + self.distance_bias)

        results = [r for r in results.values()]
        results.sort(key=lambda r: r.inv_dist, reverse=True)
        return results if cutoff is None else results[:cutoff]

    def _convert_query(
        self,
        query: Union[List[str], str],
    ) -> Union[List[str], str]:
        if self.converter is None:
            return query
        else:
            return_str = False
            if isinstance(query, str):
                query = query.split(' ')
                return_str = True

            artists = [self.converter.get(t) for t in query]
            if return_str:
                artists = ' '.join(artists)
            return artists
