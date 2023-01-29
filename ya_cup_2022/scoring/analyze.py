from typing import Tuple, Optional
from collections import Counter


__all__ = [
    'analyze_predictions',
    'miss_rank_error_decompose',
]


def analyze_predictions(
    target_path: str,
    predict_path: str,
    *, cutoff: Optional[int] = None,
) -> Tuple[Counter, int]:
    with open(target_path, 'rt') as f:
        targets = [line.strip() for line in f.readlines()]

    with open(predict_path, 'rt') as f:
        predictions = [line.strip() for line in f.readlines()]

    scores = Counter()
    for i, target in enumerate(targets):
        score = 0
        if i < len(predictions):
            for j, prediction in enumerate(predictions[i].split(' ')):
                if cutoff is not None and j >= cutoff:
                    break
                if prediction == target:
                    score = j + 1
                    break
        scores[score] += 1

    return scores, len(targets)


def miss_rank_error_decompose(
    scores: Counter,
    total: int,
) -> Tuple[float, float]:
    miss_error = scores[0] / total

    rank_error = 0.0
    for rank, count in scores.items():
        if rank == 0:
            rank_error += count
        else:
            rank_error += count / rank
    rank_error = 1 - rank_error / total

    return miss_error, rank_error
