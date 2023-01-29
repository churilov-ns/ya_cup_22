from typing import Iterable
from dataclasses import dataclass
from collections import defaultdict


__all__ = [
    'InputFile',
    'combine_predictions',
]


@dataclass
class InputFile(object):
    filename: str
    weight: float = 1.0


@dataclass
class RecData(object):
    count: int = 0
    score: float = 0.0

    def update(self, pos: int, weight: float = 1.0):
        self.score += weight / (pos + 1.)
        self.count += 1

    @property
    def rank(self):
        return self.score


def combine_predictions(
    output_file: str,
    input_files: Iterable[InputFile],
    *, n: int = 100,
):
    handlers = list()
    weights = dict()

    try:
        for file in input_files:
            h = open(file.filename, 'rt')
            handlers.append(h)
            weights[h] = file.weight

        with open(output_file, 'wt') as f_out:
            user_id = 0
            while True:
                data = defaultdict(RecData)
                for handler in handlers:
                    line = handler.readline().strip()
                    if len(line) == 0:
                        continue
                    weight = weights[handler]
                    for i, track in enumerate(line.split(' ')):
                        if len(track) > 0:
                            data[track].update(i, weight)
                        else:
                            print(f'Blank object in {handler.name},'
                                  f' user_id {user_id} at pos {i}')
                if len(data) > 0:
                    recs = sorted(data.items(), key=lambda it: it[1].rank,
                                  reverse=True)
                    f_out.write(f'{" ".join([r[0] for r in recs[:n]])}\n')
                    user_id += 1
                else:
                    break

    finally:
        for handler in handlers:
            handler.close()
