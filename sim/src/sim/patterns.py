import numpy as np

from sim.draws import inclusive_integer


def checks(rng: np.random.Generator, a: np.ndarray, b: np.ndarray, px: int, cfg: dict) -> np.ndarray:
    squares_across = 2 * inclusive_integer(rng, cfg["check_pairs_across"])
    row, column = np.indices((px, px)) * squares_across // px
    return np.where(((row + column) % 2 == 0)[..., None], a, b)


def stripes(rng: np.random.Generator, a: np.ndarray, b: np.ndarray, px: int, cfg: dict) -> np.ndarray:
    bands_across = 2 * inclusive_integer(rng, cfg["stripe_pairs_across"])
    is_a = (np.arange(px) * bands_across // px) % 2 == 0
    image = np.broadcast_to(np.where(is_a[:, None], a, b), (px, px, 3))
    return image.transpose(1, 0, 2) if rng.random() < 0.5 else image


def speckle(rng: np.random.Generator, a: np.ndarray, b: np.ndarray, px: int, cfg: dict) -> np.ndarray:
    dots_across = int(rng.choice(cfg["speckle_dots_across"]))
    is_dot = rng.random((dots_across, dots_across)) < rng.uniform(*cfg["speckle_dot_share"])
    dot_px = px // dots_across
    return np.where(np.kron(is_dot, np.ones((dot_px, dot_px), bool))[..., None], b, a)


def fade(rng: np.random.Generator, a: np.ndarray, b: np.ndarray, px: int, cfg: dict) -> np.ndarray:
    there_and_back_so_repeats_join_up = np.abs(np.linspace(-1.0, 1.0, px, endpoint=False))
    image = a + (b - a) * np.broadcast_to(there_and_back_so_repeats_join_up[:, None, None], (px, px, 1))
    return image.transpose(1, 0, 2) if rng.random() < 0.5 else image


PATTERNS = {"checks": checks, "stripes": stripes, "speckle": speckle, "fade": fade}


def texture_of_two_colours(rng: np.random.Generator, pattern: str, a, b, cfg: dict) -> np.ndarray:
    image = PATTERNS[pattern](rng, np.asarray(a, float), np.asarray(b, float), cfg["texture_px"], cfg)
    return np.ascontiguousarray(np.round(image * 255).astype(np.uint8))
