from typing import List


__all__ = [
    'mrr_score',
]


def mrr_score(
    targets: List[str],
    predictions: List[List[str]],
    *, at: int = 100,
) -> float:
    score = 0.0
    for i in range(min(len(targets), len(predictions))):
        for j in range(min(at, len(predictions[i]))):
            if predictions[i][j] == targets[i]:
                score += 1. / (j + 1)
    return score / len(targets)
