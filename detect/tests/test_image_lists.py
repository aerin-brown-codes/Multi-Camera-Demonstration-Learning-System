import copy

import pytest
import yaml

from detect.image_lists import check_images, training_images, write_lists_and_data_file
from detect.labels import label_file_of
from conftest import write_image_and_labels


@pytest.fixture
def two_sources(cfg, tmp_path):
    for source in ("sim", "real"):
        for n in range(10):
            write_image_and_labels(tmp_path / source, "train", f"{n:03d}", [(0, (10, 10, 50, 50))])
        for n in range(3):
            write_image_and_labels(tmp_path / source, "check", f"{n:03d}", [])
    return cfg


def images_from(images, source_name):
    return [image for image in images if image.parents[2].name == source_name]


def test_one_to_one_mix_holds_equal_numbers_from_each_source_and_no_image_twice(two_sources):
    images = training_images(two_sources)
    assert len(images_from(images, "sim")) == len(images_from(images, "real")) == 4
    assert len(set(images)) == 8


def test_same_seed_gives_the_same_list_and_another_seed_another(two_sources):
    assert training_images(two_sources) == training_images(copy.deepcopy(two_sources))
    assert training_images(two_sources) != training_images({**two_sources, "seed": 1})


def test_taking_more_of_one_source_leaves_the_pick_from_the_other_as_it_was(two_sources):
    more_real = copy.deepcopy(two_sources)
    more_real["sources"]["real"]["train_images"] = 7
    assert images_from(training_images(more_real), "sim") == images_from(training_images(two_sources), "sim")
    assert len(images_from(training_images(more_real), "real")) == 7


def test_asking_for_more_images_than_a_source_has_is_refused(two_sources):
    two_sources["sources"]["sim"]["train_images"] = 11
    with pytest.raises(ValueError, match="11 images asked for but only 10"):
        training_images(two_sources)


def test_every_check_image_of_every_source_is_watched(two_sources):
    assert len(check_images(two_sources)) == 6


def test_lists_and_data_file_are_what_the_training_tool_reads(two_sources, tmp_path):
    data_file = write_lists_and_data_file(tmp_path, two_sources)
    data = yaml.safe_load(data_file.read_text())
    assert data["names"] == {0: "hand", 1: "cube"}
    assert (tmp_path / "train.txt").read_text().split() == [str(image) for image in training_images(two_sources)]
    assert data["train"] == str(tmp_path / "train.txt") and data["val"] == str(tmp_path / "val.txt")
    assert len((tmp_path / "val.txt").read_text().split()) == 6


def test_image_without_a_label_file_stops_the_run_before_training(two_sources, tmp_path):
    label_file_of(check_images(two_sources)[0]).unlink()
    with pytest.raises(FileNotFoundError, match="1 images have no label file"):
        write_lists_and_data_file(tmp_path, two_sources)
