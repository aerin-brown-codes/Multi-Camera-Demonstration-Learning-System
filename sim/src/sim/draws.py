import numpy as np


def inclusive_integer(rng: np.random.Generator, low_high) -> int:
    return int(rng.integers(low_high[0], low_high[1] + 1))


def chosen_by_share(rng: np.random.Generator, shares: dict[str, float]) -> str:
    return str(rng.choice(list(shares), p=list(shares.values())))
