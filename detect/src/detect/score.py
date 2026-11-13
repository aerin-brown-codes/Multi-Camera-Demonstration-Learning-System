import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from detect.config import SET_SCORED, from_repo, load_config
from detect.scoring import ON_A_BOX_THAT_DOES_NOT_COUNT, RIGHT, Detection, Tally, judged
from detect.truth import WAYS_TO_READ_TRUTH, ImageWithTruth, group_between_edges

BY_CLASS, BY_VISIBLE_FRACTION, ALL_CLASSES = "class", "visible fraction", "all"
CONFIG_OF_A_RUN, WEIGHTS_OF_A_RUN = "train.yaml", Path("weights") / "best.pt"


def tallies_by_split(images: list[ImageWithTruth], detections_per_image: Iterable[list[Detection]],
                     cfg: dict) -> dict[str, dict[str, Tally]]:
    tallies = defaultdict(lambda: defaultdict(Tally))
    for image, detections in zip(images, detections_per_image, strict=True):
        outcome = judged(list(image.truths), detections, cfg["min_iou"])
        rows_of_the_image = [(BY_CLASS, ALL_CLASSES), *image.groups.items()]
        for truth, was_found in zip(image.truths, outcome.truth_was_found):
            if not truth.counts:
                continue
            rows = [*rows_of_the_image, (BY_CLASS, truth.class_name)]
            if truth.visible_fraction is not None:
                how_hidden = group_between_edges(truth.visible_fraction, cfg["visible_fraction"])
                rows.append((BY_VISIBLE_FRACTION, f"{truth.class_name}, {how_hidden}"))
            for split, group in rows:
                tallies[split][group].true_boxes += 1
                tallies[split][group].found += was_found
        for detection, was in zip(detections, outcome.detection_was):
            if was == ON_A_BOX_THAT_DOES_NOT_COUNT:
                continue
            for split, group in [*rows_of_the_image, (BY_CLASS, detection.class_name)]:
                tallies[split][group].detections += 1
                tallies[split][group].right += was == RIGHT
    return {split: {group: groups[group] for group in sorted(groups, key=lambda g: (g != ALL_CLASSES, g))}
            for split, groups in tallies.items()}


def meets_the_bar(tallies: dict[str, dict[str, Tally]], bar: dict) -> bool:
    with_something_to_find = [tally for tally in tallies.get(BY_CLASS, {}).values() if tally.true_boxes]
    return bool(with_something_to_find) and all(
        tally.found_share >= bar["found"] and tally.right_share is not None and tally.right_share >= bar["right"]
        for tally in with_something_to_find)


def scores_of(cfg: dict, detect) -> dict:
    sets = {}
    for name, source in cfg["sources"].items():
        images = WAYS_TO_READ_TRUTH[source["truth_from"]](from_repo(source["root"]), SET_SCORED, cfg["class_names"],
                                                          cfg["score"])
        tallies = tallies_by_split(images, detect([image.path for image in images]), cfg["score"])
        sets[name] = {
            "images": len(images), "meets_the_bar": meets_the_bar(tallies, cfg["score"]["pass_bar"]),
            "splits": {split: {group: {**asdict(tally), "found_share": tally.found_share,
                                       "right_share": tally.right_share} for group, tally in groups.items()}
                       for split, groups in tallies.items()}}
    return {"confidence": cfg["score"]["confidence"], "min_iou": cfg["score"]["min_iou"],
            "pass_bar": cfg["score"]["pass_bar"], "meets_the_bar": all(s["meets_the_bar"] for s in sets.values()),
            "sets": sets}


def as_percent(share: float | None) -> str:
    return "-" if share is None else f"{100 * share:.1f}%"


def as_markdown(scores: dict) -> str:
    bar = scores["pass_bar"]
    lines = [f"Pass bar (found at least {as_percent(bar['found'])}, right at least {as_percent(bar['right'])}, "
             f"every class of every set): {'met' if scores['meets_the_bar'] else 'not met'}. "
             f"Confidence cut-off {scores['confidence']}, overlap needed {scores['min_iou']}."]
    for name, scored in scores["sets"].items():
        lines += ["", f"## {name}: {scored['images']} images, bar {'met' if scored['meets_the_bar'] else 'not met'}", "",
                  "| Split | Group | True boxes | Found | Found share | Detections | Right | Right share |",
                  "|---|---|---|---|---|---|---|---|"]
        for split, groups in scored["splits"].items():
            for group, t in groups.items():
                lines.append(f"| {split} | {group} | {t['true_boxes']} | {t['found']} | {as_percent(t['found_share'])} "
                             f"| {t['detections']} | {t['right']} | {as_percent(t['right_share'])} |")
    return "\n".join(lines) + "\n"


def score_run(run_dir: Path, detect=None) -> dict:
    cfg = load_config(run_dir / CONFIG_OF_A_RUN)
    if detect is None:
        from detect.detector import detector_from_weights
        detect = detector_from_weights(run_dir / WEIGHTS_OF_A_RUN, cfg)
    scores = scores_of(cfg, detect)
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    (run_dir / "scores.md").write_text(as_markdown(scores))
    return scores


def main():
    parser = argparse.ArgumentParser(description="Score a run's weights on the test set of every source.")
    parser.add_argument("--run", type=Path, required=True, help="folder of the run, e.g. runs/first")
    run_dir = parser.parse_args().run
    print(as_markdown(score_run(run_dir)))
    print(f"written to {run_dir / 'scores.md'} and scores.json")


if __name__ == "__main__":
    main()
