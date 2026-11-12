import numpy as np
import pytest

from sim.patterns import PATTERNS, texture_of_two_colours

CFG = {"texture_px": 64, "check_pairs_across": [1, 4], "stripe_pairs_across": [1, 6],
       "speckle_dots_across": [16, 32], "speckle_dot_share": [0.05, 0.4]}
A, B = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)


@pytest.mark.parametrize("pattern", ["checks", "stripes", "speckle"])
def test_two_colour_pattern_holds_both_colours_and_nothing_else(pattern):
    for seed in range(20):
        image = texture_of_two_colours(np.random.default_rng(seed), pattern, A, B, CFG)
        assert image.shape == (64, 64, 3) and image.dtype == np.uint8
        colours = {tuple(c) for c in image.reshape(-1, 3)}
        assert colours == {(255, 0, 0), (0, 0, 255)}, f"{pattern} seed {seed} holds {colours}"


def test_checks_and_stripes_are_half_one_colour_half_the_other():
    for pattern in ("checks", "stripes"):
        image = texture_of_two_colours(np.random.default_rng(3), pattern, A, B, {**CFG, "texture_px": 240})
        assert (image[..., 0] == 255).mean() == 0.5, pattern


def test_fade_runs_from_one_colour_to_the_other_and_back_so_repeats_join_up():
    image = texture_of_two_colours(np.random.default_rng(0), "fade", A, B, CFG).astype(int)
    line = image[:, 0] if (image[0] == image[0, 0]).all() else image[0]
    assert tuple(line[0]) == (0, 0, 255) and tuple(line[len(line) // 2]) == (255, 0, 0), (
        "one colour at the edges, the other in the middle")
    assert np.abs(np.diff(np.concatenate([line, line[:1]]), axis=0)).max() <= 255 * 2 / 64 + 1, (
        "no jump inside the pattern or where one repeat meets the next")


def test_same_seed_gives_the_same_texture():
    for pattern in PATTERNS:
        first = texture_of_two_colours(np.random.default_rng(7), pattern, A, B, CFG)
        assert np.array_equal(first, texture_of_two_colours(np.random.default_rng(7), pattern, A, B, CFG))
