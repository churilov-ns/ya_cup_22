from typing import Dict
from collections import defaultdict
from dataclasses import dataclass, field


__all__ = [
    'combine_df',
]


@dataclass
class Features(object):
    like: int = 0
    ranks: Dict[int, float] = field(default_factory=dict)

    def to_str(self, n_models: int) -> str:
        rank_str = ','.join(
            str(self.ranks.get(i, 0.)) for i in range(n_models))
        return f'{rank_str},{self.like}'


def combine_df(df_file: str, ans_file: str, *model_files: str):
    models = list()
    miss_count = 0
    users_count = 0

    try:
        for file in model_files:
            models.append(open(file, 'rt'))
        n_models = len(models)

        with open(df_file, 'wt') as f_out:
            header = ','.join(f'rank_{i}' for i in range(n_models))
            f_out.write(f'user_id,track_id,{header},like\n')

            with open(ans_file, 'rt') as f_ans:
                for user_id, line in enumerate(f_ans):
                    users_count += 1
                    liked_track = line.strip()
                    data = defaultdict(Features)

                    for i, model in enumerate(models):
                        tracks = model.readline().strip().split(' ')
                        for j, track in enumerate(tracks):
                            if len(track) == 0:
                                print(f'Blank object in {model.name},'
                                      f' line {user_id} at pos {j}')
                                continue
                            data[track].ranks[i] = 1. / (j + 1)

                    if liked_track not in data:
                        miss_count += 1
                        continue

                    data[liked_track].like = 1
                    for track, features in data.items():
                        f_out.write(f'{user_id},{track},'
                                    f'{features.to_str(n_models)}\n')

    finally:
        for model in models:
            model.close()
        miss_percent = miss_count * 100. / users_count
        print(f'# users: {users_count}')
        print(f'# misses: {miss_count} ({miss_percent} %)')
