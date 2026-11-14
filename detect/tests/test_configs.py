import copy

from detect.config import CONFIGS, REPO, load_config

RUN_1 = load_config(CONFIGS / "train.yaml")


def test_run_2_is_run_1_with_twice_the_real_images_and_nothing_else():
    as_run_2 = copy.deepcopy(RUN_1)
    as_run_2["sources"]["real"]["train_images"] = 10000
    assert load_config(CONFIGS / "run2.yaml") == as_run_2


def test_trial_is_run_1_made_small():
    as_trial = copy.deepcopy(RUN_1)
    as_trial["sources"]["sim"]["train_images"] = as_trial["sources"]["real"]["train_images"] = 1000
    as_trial["epochs"] = 3
    assert load_config(CONFIGS / "trial.yaml") == as_trial


def test_run_1_scores_on_the_mac_and_on_colab_from_one_config():
    assert RUN_1["device"] == "auto", "a run trained with a named GPU cannot be scored on the Mac"


def test_the_live_detector_reads_frames_where_the_sim_sends_them_and_the_viewer_reads_boxes_where_it_sends_them():
    live, sim = load_config(CONFIGS / "live.yaml"), load_config(REPO / "sim" / "src" / "sim" / "configs" / "sim.yaml")
    assert live["feed_address"] == f"tcp://{sim['feed']['host']}:{sim['feed']['port']}"
    assert sim["view"]["detections_address"] == f"tcp://{live['publish']['host']}:{live['publish']['port']}"
