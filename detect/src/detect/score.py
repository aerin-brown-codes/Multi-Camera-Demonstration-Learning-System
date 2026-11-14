import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from detect.config import SET_SCORED, SET_WATCHED_WHILE_TRAINING, from_repo, load_config
from detect.scoring import ON_A_BOX_THAT_DOES_NOT_COUNT, RIGHT, Detection, Tally, judged
from detect.truth import WAYS_TO_READ_TRUTH, ImageWithTruth, group_between_edges

BY_CLASS, BY_VISIBLE_FRACTION, ALL_CLASSES = "class", "visible fraction", "all"
CONFIG_OF_A_RUN, WEIGHTS_OF_A_RUN = "train.yaml", Path("weights") / "best.pt"
CUT_OFF_OF_A_RUN, PICKED = "cut_off.json", "picked"
TABLES_OF_A_RUN = {SET_WATCHED_WHILE_TRAINING: "check_scores", SET_SCORED: "scores"}
Seen = dict[str, tuple[list[ImageWithTruth], list[list[Detection]]]]


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


def seen_on(set_name: str, cfg: dict, detect) -> Seen:
    seen = {}
    for name, source in cfg["sources"].items():
        images = WAYS_TO_READ_TRUTH[source["truth_from"]](from_repo(source["root"]), set_name, cfg["class_names"],
                                                          cfg["score"])
        seen[name] = (images, list(detect([image.path for image in images])))
    return seen


def sure_enough(detections_per_image: list[list[Detection]], cut_off: float) -> list[list[Detection]]:
    return [[detection for detection in detections if detection.confidence >= cut_off]
            for detections in detections_per_image]


def scores_at(cut_off: float, set_name: str, seen: Seen, cfg: dict) -> dict:
    sets = {}
    for name, (images, detections_per_image) in seen.items():
        tallies = tallies_by_split(images, sure_enough(detections_per_image, cut_off), cfg["score"])
        sets[name] = {
            "images": len(images), "meets_the_bar": meets_the_bar(tallies, cfg["score"]["pass_bar"]),
            "splits": {split: {group: {**asdict(tally), "found_share": tally.found_share,
                                       "right_share": tally.right_share} for group, tally in groups.items()}
                       for split, groups in tallies.items()}}
    return {"set": set_name, "confidence": cut_off, "min_iou": cfg["score"]["min_iou"],
            "pass_bar": cfg["score"]["pass_bar"], "meets_the_bar": all(s["meets_the_bar"] for s in sets.values()),
            "sets": sets}


def worst_number(scores: dict) -> float:
    return min((share or 0.0 for scored in scores["sets"].values()
                for row in scored["splits"].get(BY_CLASS, {}).values() if row["true_boxes"]
                for share in (row["found_share"], row["right_share"])), default=0.0)


def worst_number_at_each_cut_off(seen_on_the_check_sets: Seen, cfg: dict) -> dict[float, float]:
    return {cut_off: worst_number(scores_at(cut_off, SET_WATCHED_WHILE_TRAINING, seen_on_the_check_sets, cfg))
            for cut_off in cfg["score"]["cut_offs"]}


def as_percent(share: float | None) -> str:
    return "-" if share is None else f"{100 * share:.1f}%"


def as_markdown(scores: dict) -> str:
    bar = scores["pass_bar"]
    lines = [f"The {scores['set']} sets. "
             f"Pass bar (found at least {as_percent(bar['found'])}, right at least {as_percent(bar['right'])}, "
             f"every class of every set): {'met' if scores['meets_the_bar'] else 'not met'}. "
             f"Confidence cut-off {scores['confidence']} (picked on the check sets), overlap needed {scores['min_iou']}."]
    for name, scored in scores["sets"].items():
        lines += ["", f"## {name}: {scored['images']} images, bar {'met' if scored['meets_the_bar'] else 'not met'}", "",
                  "| Split | Group | True boxes | Found | Found share | Detections | Right | Right share |",
                  "|---|---|---|---|---|---|---|---|"]
        for split, groups in scored["splits"].items():
            for group, t in groups.items():
                lines.append(f"| {split} | {group} | {t['true_boxes']} | {t['found']} | {as_percent(t['found_share'])} "
                             f"| {t['detections']} | {t['right']} | {as_percent(t['right_share'])} |")
    return "\n".join(lines) + "\n"


def picked_cut_off(run_dir: Path) -> float:
    return json.loads((run_dir / CUT_OFF_OF_A_RUN).read_text())[PICKED]


def score_run(run_dir: Path, detect=None) -> dict:
    cfg = load_config(run_dir / CONFIG_OF_A_RUN)
    if detect is None:
        from detect.detector import detector_from_weights
        detect = detector_from_weights(run_dir / WEIGHTS_OF_A_RUN, cfg)
    seen_on_the_check_sets = seen_on(SET_WATCHED_WHILE_TRAINING, cfg, detect)
    worst_at = worst_number_at_each_cut_off(seen_on_the_check_sets, cfg)
    cut_off = max(worst_at, key=worst_at.get)
    (run_dir / CUT_OFF_OF_A_RUN).write_text(json.dumps(
        {PICKED: cut_off, "worst_number_on_the_check_sets_at_each_cut_off": worst_at}, indent=2))
    seen = {SET_WATCHED_WHILE_TRAINING: seen_on_the_check_sets, SET_SCORED: seen_on(SET_SCORED, cfg, detect)}
    scores = {set_name: scores_at(cut_off, set_name, seen[set_name], cfg) for set_name in TABLES_OF_A_RUN}
    for set_name, file_name in TABLES_OF_A_RUN.items():
        (run_dir / f"{file_name}.json").write_text(json.dumps(scores[set_name], indent=2))
        (run_dir / f"{file_name}.md").write_text(as_markdown(scores[set_name]))
    return scores[SET_SCORED]


def main():
    parser = argparse.ArgumentParser(description="Pick a run's cut-off on the check sets, then score the check and "
                                                 "test set of every source at it.")
    parser.add_argument("--run", type=Path, required=True, help="folder of the run, e.g. runs/first")
    run_dir = parser.parse_args().run
    print(as_markdown(score_run(run_dir)))
    print(f"written to {run_dir}: scores.md (above), check_scores.md, cut_off.json")


if __name__ == "__main__":
    main()
