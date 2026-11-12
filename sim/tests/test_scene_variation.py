from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

import sim
from sim.rig import Rig, load_profiles
from sim.scene_variation import (PLAIN, apply_look, is_a_box_the_size_of_the_cube, is_red_like, look_in_words,
                                 look_with_extras_put_somewhere_else, name_of_extra, random_camera, random_extra,
                                 random_look)
from sim.world import TableTop, build_model, top_of_box_shaped_table

CONFIGS = Path(sim.__file__).parent / "configs"
CFG = yaml.safe_load((CONFIGS / "dataset.yaml").read_text())
LOOK = CFG["look"]
PROFILE = load_profiles(CONFIGS / "rig.yaml")[CFG["camera_profile"]]
TABLE = TableTop(np.zeros(2), np.array([0.6, 0.4]), 0.75)
CUBE_RGB = (0.8, 0.1, 0.1)
TARGET_PLACES = np.array([[0.1, 0.05], [-0.2, -0.1]])
ROUNDING = 1e-9


def looks(count, seed=0):
    rng = np.random.default_rng(seed)
    return [random_look(rng, TABLE, TARGET_PLACES, LOOK) for _ in range(count)]


def footprint_radius(extra):
    half = extra.half_sizes_as_mujoco_wants_them
    return {"box": lambda: np.hypot(half[0], half[1]), "ball": lambda: half[0], "cylinder": lambda: half[0],
            "capsule": lambda: half[0] + half[1]}[extra.shape]()


def test_table_top_is_read_from_the_scene():
    table = top_of_box_shaped_table(build_model("cube_hands", Rig(0.0, [])), CFG["table_geom"])
    assert np.allclose(table.centre_xy, TABLE.centre_xy) and np.allclose(table.half_size_xy, TABLE.half_size_xy)
    assert table.z == pytest.approx(TABLE.z)


def test_every_camera_is_in_its_distance_and_angle_range_above_the_table_and_looks_at_it():
    rng = np.random.default_rng(0)
    (near, far), (low, high) = CFG["cameras"]["distance_m"], CFG["cameras"]["height_deg"]
    seen_distances, seen_heights = [], []
    for _ in range(500):
        placed = random_camera(rng, "cam0", PROFILE, TABLE, CFG["cameras"])
        cam = placed.camera
        assert TABLE.is_under(placed.aim[:2]) and placed.aim[2] == TABLE.z, "a camera aims at a point on the table top"
        to_camera = cam.position - placed.aim
        assert np.linalg.norm(to_camera) == pytest.approx(placed.distance_m) and near <= placed.distance_m <= far
        assert np.rad2deg(np.arcsin(to_camera[2] / placed.distance_m)) == pytest.approx(placed.height_deg)
        assert low <= placed.height_deg <= high and cam.position[2] > TABLE.z, "never below the table top"
        aim_in_cam = cam.R_world_cam.T @ (placed.aim - cam.position)
        assert np.allclose(aim_in_cam[:2], 0.0, atol=1e-9) and aim_in_cam[2] > 0, "the aim is at the image centre"
        seen_distances.append(placed.distance_m)
        seen_heights.append(placed.height_deg)
    assert min(seen_distances) < near + 0.1 and max(seen_distances) > far - 0.1, "the whole distance range is used"
    assert min(seen_heights) < low + 8 and max(seen_heights) > high - 8, "the whole angle range is used"


def test_red_like_covers_the_cube_in_any_light_and_not_other_colours():
    red_like = partial(is_red_like, cfg=LOOK["extras"]["red_like"])
    for shade in (1.0, 0.6, 0.4):
        assert red_like(tuple(shade * c for c in CUBE_RGB)), f"the cube's red at {shade:.0%} brightness"
    assert red_like((1.0, 0.3, 0.3)) and red_like((0.9, 0.2, 0.05))
    for name, rgb in {"orange": (1.0, 0.6, 0.0), "blue": (0.1, 0.2, 0.9), "green": (0.1, 0.7, 0.2),
                      "grey": (0.5, 0.5, 0.5), "pale pink": (1.0, 0.8, 0.8), "purple": (0.5, 0.1, 0.8)}.items():
        assert not red_like(rgb), name


def test_no_extra_object_is_a_red_box_near_the_size_of_the_cube():
    only_cube_sized_boxes = {**LOOK["extras"], "shapes": ["box"], "size_m": [0.04, 0.06]}
    rng = np.random.default_rng(1)
    for _ in range(2000):
        extra = random_extra(rng, TABLE, TARGET_PLACES, only_cube_sized_boxes)
        assert is_a_box_the_size_of_the_cube(extra.shape, extra.half_sizes_as_mujoco_wants_them, LOOK["extras"])
        assert not is_red_like(extra.rgb, LOOK["extras"]["red_like"]), f"a {extra.rgb} box would need a cube label"
    reds_of_other_sizes_or_shapes = [
        e for look in looks(300) for e in look.extras if is_red_like(e.rgb, LOOK["extras"]["red_like"])]
    assert reds_of_other_sizes_or_shapes, "red is only kept off cube-sized boxes"
    assert not any(is_a_box_the_size_of_the_cube(e.shape, e.half_sizes_as_mujoco_wants_them, LOOK["extras"])
                   for e in reds_of_other_sizes_or_shapes)


def test_extra_objects_are_within_their_count_and_size_and_on_the_table():
    (fewest, most), (smallest, largest) = LOOK["extras"]["count"], LOOK["extras"]["size_m"]
    counts, shapes = set(), set()
    for look in looks(300):
        counts.add(len(look.extras))
        for extra in look.extras:
            shapes.add(extra.shape)
            half = np.array(extra.half_sizes_as_mujoco_wants_them)
            outer = 2 * (half if extra.shape != "capsule" else np.array([half[0], half[0] + half[1]]))
            assert (outer >= smallest - ROUNDING).all() and (outer <= largest + ROUNDING).all(), f"{extra.shape} {outer}"
            assert TABLE.is_under(extra.position[:2], footprint_radius(extra) - ROUNDING), "no extra hangs off the table"
    assert counts == set(range(fewest, most + 1)) and shapes == set(LOOK["extras"]["shapes"])


def test_about_half_the_extra_objects_are_put_beside_a_target():
    extras = [e for look in looks(400) for e in look.extras]
    beside = [e for e in extras if e.beside_a_target]
    assert abs(len(beside) / len(extras) - LOOK["extras"]["beside_chance"]) < 0.07
    for extra in beside:
        gap = min(np.linalg.norm(TARGET_PLACES - extra.position[:2], axis=1)) - footprint_radius(extra)
        assert -ROUNDING <= gap <= LOOK["extras"]["beside_within_m"] + ROUNDING
    rng = np.random.default_rng(0)
    assert not any(random_extra(rng, TABLE, np.empty((0, 2)), LOOK["extras"]).beside_a_target for _ in range(50)), (
        "with no target in the scene every extra goes anywhere on the table")


def test_only_the_named_extras_are_put_somewhere_else_and_each_stays_the_thing_it_was():
    rng = np.random.default_rng(0)
    moved = 0
    for look in looks(200):
        if len(look.extras) < 2:
            continue
        to_move = {name_of_extra(0), name_of_extra(len(look.extras) - 1)}
        after = look_with_extras_put_somewhere_else(rng, look, to_move, True, TABLE, TARGET_PLACES, LOOK["extras"])
        assert after.table is look.table and after.lights is look.lights and after.skin_rgb == look.skin_rgb
        for index, (was, now) in enumerate(zip(look.extras, after.extras, strict=True)):
            if name_of_extra(index) not in to_move:
                assert now is was, "an extra that sank into nothing stays where it is"
                continue
            moved += 1
            assert (now.shape, now.half_sizes_as_mujoco_wants_them, now.rgb) == (
                was.shape, was.half_sizes_as_mujoco_wants_them, was.rgb), "or large extras would be drawn less"
            assert not np.allclose(now.position[:2], was.position[:2])
            assert np.isclose(now.position[2], was.position[2]), "it still rests on the table"
            assert TABLE.is_under(now.position[:2], footprint_radius(now) - ROUNDING)
            assert not now.beside_a_target or was.beside_a_target, "one put anywhere is not moved beside a target"
            if now.beside_a_target:
                gap = min(np.linalg.norm(TARGET_PLACES - now.position[:2], axis=1)) - footprint_radius(now)
                assert -ROUNDING <= gap <= LOOK["extras"]["beside_within_m"] + ROUNDING
    assert moved > 100


def test_extra_that_found_no_room_beside_a_target_goes_anywhere_on_the_table():
    rng = np.random.default_rng(0)
    was_beside = 0
    for look in looks(100):
        everything = {name_of_extra(index) for index in range(len(look.extras))}
        after = look_with_extras_put_somewhere_else(rng, look, everything, False, TABLE, TARGET_PLACES, LOOK["extras"])
        was_beside += sum(extra.beside_a_target for extra in look.extras)
        assert not any(extra.beside_a_target for extra in after.extras), "or a crowded spot could stop the whole run"
    assert was_beside > 50


def test_lights_and_skin_stay_inside_their_ranges():
    lights, skin = LOOK["lights"], LOOK["skin"]
    counts, tones = set(), set()
    for look in looks(300):
        counts.add(len(look.lights))
        total = sum(light.brightness for light in look.lights)
        assert lights["total_brightness"][0] - ROUNDING <= total <= lights["total_brightness"][1] + ROUNDING
        assert lights["fill_brightness"][0] <= look.fill_brightness <= lights["fill_brightness"][1]
        for light in look.lights:
            assert light.position[2] > TABLE.z and TABLE.is_under(light.aim[:2])
        distance_to_each_tone = np.abs(np.array(skin["tones"]) - look.skin_rgb).max(axis=1)
        assert distance_to_each_tone.min() <= skin["shade_jitter"] + ROUNDING, "skin stays close to a listed tone"
        tones.add(int(distance_to_each_tone.argmin()))
    assert counts == set(range(lights["count"][0], lights["count"][1] + 1))
    assert tones == set(range(len(skin["tones"]))), "every skin tone turns up"


def test_look_is_built_into_the_scene():
    patterned = plain = None
    for look in looks(40, seed=5):
        if look.table.pattern != PLAIN and look.extras and patterned is None:
            patterned = look
        if look.table.pattern == PLAIN and plain is None:
            plain = look

    def built(look):
        model = build_model("cube_hands", Rig(0.0, []), partial(
            apply_look, look=look, table_geom=CFG["table_geom"], floor_geom=CFG["floor_geom"],
            skin_cfg=LOOK["skin"], backdrop_px=LOOK["backdrop_px"]))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    model, data = built(patterned)
    table = model.geom(CFG["table_geom"])
    assert table.matid[0] >= 0 and model.mat_texid[table.matid[0]].max() >= 0, "a patterned table gets a texture"
    assert model.nlight == len(patterned.lights), "the scene file's own lights are replaced"
    assert np.allclose(model.vis.headlight.diffuse, patterned.fill_brightness)
    for material in LOOK["skin"]["materials"]:
        assert np.allclose(model.material(material).rgba[:3], patterned.skin_rgb)
    for geom in LOOK["skin"]["geoms"]:
        assert np.allclose(model.geom(geom).rgba[:3], patterned.skin_rgb)
    for index, extra in enumerate(patterned.extras):
        geom = model.geom(name_of_extra(index))
        assert np.allclose(data.geom_xpos[geom.id], extra.position) and np.allclose(geom.rgba[:3], extra.rgb)
        gap_m = mujoco.mj_geomDistance(model, data, geom.id, table.id, 1.0, None)
        assert abs(gap_m) < 1e-6, f"{extra.shape} must rest on the table, not float or sink ({gap_m * 1000:.3f} mm)"

    model, _ = built(plain)
    table = model.geom(CFG["table_geom"])
    assert table.matid[0] < 0 and np.allclose(table.rgba[:3], plain.table.colours[0])
    assert look_in_words(plain)["table"] == {"pattern": PLAIN, "colours": [list(plain.table.colours[0])],
                                             "repeats": plain.table.repeats}
