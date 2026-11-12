import colorsys
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from sim.draws import chosen_by_share, inclusive_integer
from sim.patterns import texture_of_two_colours
from sim.poses import R_about_z, quat_wxyz
from sim.rig import Camera, Profile, look_at
from sim.world import TableTop

PLAIN = "plain"
FACES_OF_A_SKYBOX = 6
RGBA_THAT_LETS_THE_MATERIAL_SHOW = [0.5, 0.5, 0.5, 1.0]
R_LYING_ON_ITS_SIDE = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
MUJOCO_SHAPE = {"box": mujoco.mjtGeom.mjGEOM_BOX, "ball": mujoco.mjtGeom.mjGEOM_SPHERE,
                "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE}


@dataclass(frozen=True)
class PlacedCamera:
    camera: Camera
    aim: np.ndarray
    distance_m: float
    height_deg: float


@dataclass(frozen=True)
class Surface:
    pattern: str
    colours: tuple[tuple[float, ...], tuple[float, ...]]
    texture_or_none_when_plain: np.ndarray | None
    repeats: int


@dataclass(frozen=True)
class Light:
    position: np.ndarray
    aim: np.ndarray
    brightness: float
    parallel_rays: bool


@dataclass(frozen=True)
class ExtraObject:
    shape: str
    half_sizes_as_mujoco_wants_them: tuple[float, ...]
    position: np.ndarray
    R_world_object: np.ndarray
    rgb: tuple[float, ...]
    beside_a_target: bool


@dataclass(frozen=True)
class SceneLook:
    backdrop_colours: tuple[tuple[float, ...], tuple[float, ...]]
    table: Surface
    floor: Surface
    skin_rgb: tuple[float, ...]
    lights: list[Light]
    fill_brightness: float
    ambient: float
    extras: list[ExtraObject]


def point_seen_from(centre, distance_m: float, height_rad: float, side_rad: float) -> np.ndarray:
    return np.asarray(centre) + distance_m * np.array([np.cos(height_rad) * np.cos(side_rad),
                                                       np.cos(height_rad) * np.sin(side_rad), np.sin(height_rad)])


def random_camera(rng: np.random.Generator, name: str, profile: Profile, table: TableTop, cfg: dict) -> PlacedCamera:
    aim = np.array([*table.random_point_xy(rng), table.z])
    distance_m, height_deg = rng.uniform(*cfg["distance_m"]), rng.uniform(*cfg["height_deg"])
    position = point_seen_from(aim, distance_m, np.deg2rad(height_deg), rng.uniform(-np.pi, np.pi))
    return PlacedCamera(Camera(name, profile, position, look_at(position, aim)), aim, distance_m, height_deg)


def is_red_like(rgb, cfg: dict) -> bool:
    hue, saturation, value = colorsys.rgb_to_hsv(*rgb)
    degrees_from_red = min(hue, 1.0 - hue) * 360.0
    return (degrees_from_red <= cfg["hue_within_deg"] and saturation >= cfg["min_saturation"]
            and value >= cfg["min_value"])


def is_a_box_the_size_of_the_cube(extra_shape: str, half_sizes, cfg: dict) -> bool:
    low, high = cfg["cube_like_side_m"]
    return extra_shape == "box" and all(low <= 2 * half <= high for half in half_sizes)


def random_surface(rng: np.random.Generator, cfg: dict) -> Surface:
    pattern = chosen_by_share(rng, cfg["patterns"])
    a, b = tuple(rng.random(3)), tuple(rng.random(3))
    texture = None if pattern == PLAIN else texture_of_two_colours(rng, pattern, a, b, cfg)
    return Surface(pattern, (a, b), texture, inclusive_integer(rng, cfg["repeats"]))


def random_lights(rng: np.random.Generator, table: TableTop, cfg: dict) -> list[Light]:
    count = inclusive_integer(rng, cfg["count"])
    brightness_of_each = rng.dirichlet(np.ones(count)) * rng.uniform(*cfg["total_brightness"])
    table_centre = [*table.centre_xy, table.z]
    return [
        Light(
            position=point_seen_from(table_centre, rng.uniform(*cfg["distance_m"]),
                                     np.deg2rad(rng.uniform(*cfg["height_deg"])), rng.uniform(-np.pi, np.pi)),
            aim=np.array([*table.random_point_xy(rng), table.z]),
            brightness=float(brightness),
            parallel_rays=bool(rng.random() < cfg["parallel_rays_chance"]),
        )
        for brightness in brightness_of_each
    ]


def random_extra(rng: np.random.Generator, table: TableTop, places_of_targets_xy, cfg: dict) -> ExtraObject:
    shape = str(rng.choice(cfg["shapes"]))
    half = rng.uniform(*cfg["size_m"], size=3) / 2
    if shape == "box":
        half_sizes = tuple(half)
    elif shape == "ball":
        half_sizes = (half[0],)
    elif shape == "cylinder":
        half_sizes = (half[0], half[1])
    else:
        radius, half_length = min(half[0], half[1]), max(half[0], half[1])
        half_sizes = (radius, half_length - radius)

    rgb = tuple(rng.random(3))
    while is_a_box_the_size_of_the_cube(shape, half_sizes, cfg) and is_red_like(rgb, cfg["red_like"]):
        rgb = tuple(rng.random(3))

    beside = len(places_of_targets_xy) > 0 and bool(rng.random() < cfg["beside_chance"])
    return put_on_the_table(rng, shape, tuple(float(h) for h in half_sizes), rgb, beside, table, places_of_targets_xy,
                            cfg)


def put_on_the_table(rng: np.random.Generator, shape: str, half_sizes: tuple[float, ...], rgb, beside: bool,
                     table: TableTop, places_of_targets_xy, cfg: dict) -> ExtraObject:
    R = R_about_z(rng.uniform(-np.pi, np.pi))
    if shape == "box":
        footprint_radius, half_height = float(np.hypot(half_sizes[0], half_sizes[1])), half_sizes[2]
    elif shape == "ball":
        footprint_radius, half_height = half_sizes[0], half_sizes[0]
    elif shape == "cylinder":
        footprint_radius, half_height = half_sizes
    else:
        footprint_radius, half_height = half_sizes[0] + half_sizes[1], half_sizes[0]
        R = R @ R_LYING_ON_ITS_SIDE

    xy = None
    if beside:
        for _ in range(cfg["place_tries"]):
            target_xy = places_of_targets_xy[rng.integers(len(places_of_targets_xy))]
            away, side = footprint_radius + rng.uniform(0.0, cfg["beside_within_m"]), rng.uniform(-np.pi, np.pi)
            spot = target_xy + away * np.array([np.cos(side), np.sin(side)])
            if table.is_under(spot, footprint_radius):
                xy = spot
                break
    if xy is None:
        beside, xy = False, table.random_point_xy(rng, footprint_radius)
    return ExtraObject(shape, half_sizes, np.array([*xy, table.z + half_height]), R, rgb, beside)


def look_with_extras_put_somewhere_else(rng: np.random.Generator, look: SceneLook, names_of_those_to_move,
                                        still_trying_beside_a_target: bool, table: TableTop, places_of_targets_xy,
                                        cfg: dict) -> SceneLook:
    return replace(look, extras=[
        put_on_the_table(rng, e.shape, e.half_sizes_as_mujoco_wants_them, e.rgb,
                         e.beside_a_target and still_trying_beside_a_target, table, places_of_targets_xy, cfg)
        if name_of_extra(index) in names_of_those_to_move else e
        for index, e in enumerate(look.extras)])


def random_look(rng: np.random.Generator, table: TableTop, places_of_targets_xy, cfg: dict) -> SceneLook:
    skin, lights = cfg["skin"], cfg["lights"]
    tone = np.asarray(skin["tones"][rng.integers(len(skin["tones"]))])
    shade = np.clip(tone + rng.uniform(-skin["shade_jitter"], skin["shade_jitter"], 3), 0.0, 1.0)
    return SceneLook(
        backdrop_colours=(tuple(rng.random(3)), tuple(rng.random(3))),
        table=random_surface(rng, cfg["surfaces"]),
        floor=random_surface(rng, cfg["surfaces"]),
        skin_rgb=tuple(shade),
        lights=random_lights(rng, table, lights),
        fill_brightness=rng.uniform(*lights["fill_brightness"]),
        ambient=rng.uniform(*lights["ambient"]),
        extras=[random_extra(rng, table, places_of_targets_xy, cfg["extras"])
                for _ in range(inclusive_integer(rng, cfg["extras"]["count"]))],
    )


def name_of_extra(index: int) -> str:
    return f"extra_{index}"


def _cover(spec: mujoco.MjSpec, geom_name: str, surface: Surface) -> None:
    geom = spec.geom(geom_name)
    if surface.texture_or_none_when_plain is None:
        geom.rgba = [*surface.colours[0], 1.0]
        return
    name = f"{geom_name}_pattern"
    height_px, width_px, channels = surface.texture_or_none_when_plain.shape
    texture = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=width_px, height=height_px,
                               nchannel=channels)
    texture.data = surface.texture_or_none_when_plain.tobytes()
    material = spec.add_material(name=name, texrepeat=[surface.repeats, surface.repeats])
    textures = list(material.textures)
    textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    material.textures = textures
    geom.material = name
    geom.rgba = RGBA_THAT_LETS_THE_MATERIAL_SHOW


def apply_look(spec: mujoco.MjSpec, look: SceneLook, table_geom: str, floor_geom: str, skin_cfg: dict,
               backdrop_px: int) -> None:
    spec.add_texture(name="backdrop", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=look.backdrop_colours[0],
                     rgb2=look.backdrop_colours[1], width=backdrop_px, height=FACES_OF_A_SKYBOX * backdrop_px)
    _cover(spec, table_geom, look.table)
    _cover(spec, floor_geom, look.floor)
    for material in skin_cfg["materials"]:
        spec.material(material).rgba = [*look.skin_rgb, 1.0]
    for geom in skin_cfg["geoms"]:
        spec.geom(geom).rgba = [*look.skin_rgb, 1.0]

    for light in list(spec.lights):
        spec.delete(light)
    for light in look.lights:
        kind = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if light.parallel_rays else mujoco.mjtLightType.mjLIGHT_SPOT
        spec.worldbody.add_light(pos=light.position, dir=light.aim - light.position, type=kind,
                                 diffuse=[light.brightness] * 3)
    spec.visual.headlight.diffuse = [look.fill_brightness] * 3
    spec.visual.headlight.ambient = [look.ambient] * 3

    for index, extra in enumerate(look.extras):
        body = spec.worldbody.add_body(name=name_of_extra(index), pos=extra.position,
                                       quat=quat_wxyz(extra.R_world_object))
        size = [*extra.half_sizes_as_mujoco_wants_them, 0.0, 0.0][:3]
        body.add_geom(name=name_of_extra(index), type=MUJOCO_SHAPE[extra.shape], size=size, rgba=[*extra.rgb, 1.0])


def look_in_words(look: SceneLook) -> dict:
    def surface(s: Surface) -> dict:
        colours = s.colours[:1] if s.pattern == PLAIN else s.colours
        return {"pattern": s.pattern, "colours": [list(map(float, c)) for c in colours], "repeats": s.repeats}

    return {
        "backdrop_colours": [list(map(float, c)) for c in look.backdrop_colours],
        "table": surface(look.table),
        "floor": surface(look.floor),
        "skin_rgb": list(map(float, look.skin_rgb)),
        "lights": [{"position": light.position.tolist(), "brightness": light.brightness,
                    "parallel_rays": light.parallel_rays} for light in look.lights],
        "fill_brightness": float(look.fill_brightness),
        "ambient": float(look.ambient),
        "extras": [{"shape": e.shape, "half_sizes_m": list(e.half_sizes_as_mujoco_wants_them),
                    "position": e.position.tolist(), "rgb": list(map(float, e.rgb)),
                    "beside_a_target": e.beside_a_target} for e in look.extras],
    }
