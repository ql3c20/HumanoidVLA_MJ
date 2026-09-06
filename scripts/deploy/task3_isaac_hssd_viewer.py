#!/usr/bin/env python3
"""Optional Isaac Sim mirror for the working Task3 MuJoCo/DDS chain.

MuJoCo remains authoritative.  The script imports the exact Task3 MJCF into a
derived USD, adds either SIMPLE HSSD scene13 or a generic USD/USDZ background,
then mirrors qpos data received from the localhost DDS relay.  It never
publishes commands or edits the original MJCF/background assets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import socket
import struct
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCE = ROOT / "20260612_task3" / "20260612_154645_g1_sim"
DEFAULT_MJCF = DEFAULT_INSTANCE / "model_snapshot" / "mujoco" / "model" / "g1" / "scene_43dof.xml"
DEFAULT_HSSD = (
    Path("/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE")
    / "data/scenes/hssd/102344250/102344250_local.usd"
)
DEFAULT_OUTPUT_DIR = ROOT / "output/task3_isaac_hssd"
ISAAC_EGO_FRAME_MAGIC = b"HVEGO01\0"
SYNCED_RGB_RENDER_PASSES = 3
HSSD_SCENE_PROFILES = {
    "107734119_175999932": {
        # SIMPLE hssd:scene0: compact living room centered on a coffee table.
        # Hide that table and use its authored center for the MuJoCo task shelf.
        "scale": 1.0,
        "surface": "furniture/node_b914fb6bcc81386bfa1ff7a3eb8412b7ac581ff",
        "hide_prims": (),
    },
    "103997919_171031233": {
        # SIMPLE hssd:scene28: small one-bedroom apartment.
        "scale": 1.0,
        # The SIMPLE table anchor is in the living room.  Center this task on
        # the bedroom rug instead so the bed and bedroom furniture are visible.
        "surface": "furniture/fd57ac8d908762a5845a96d2ac41599775e17fbb",
        "hide_prims": (
            # Small brown rectangular bedside clock/decor in front of the task cabinet.
            "furniture/node_328222911172bbc4db868cde599bfd5c5c39b21",
        ),
    },
    "102344049": {
        # SIMPLE hssd:scene4 is authored in centimetres.
        "scale": 0.01,
        "surface": (
            "furniture/e8dd67ee07d375f40264caed68e8c5f76ed707cb/"
            "e8dd67ee07d375f40264caed68e8c5f76ed707cb___/"
            "e8dd67ee07d375f40264caed68e8c5f76ed707cb______root/"
            "e8dd67ee07d375f40264caed68e8c5f76ed707cb______root___"
        ),
        "hide_prims": (),
    },
    "102344250": {
        "scale": 1.0,
        "surface": "furniture/node_e7c674ea7231612862a5bb66960f0cfef5d8f0e",
        "hide_prims": (
            # Native round waste bin near the insertion area; hiding it keeps
            # the task pedal bin as the only garbage-can target.
            "furniture/node_a9c2d2f765a2399442f845b7da0ddaa61e82071",
            "furniture/ccf3f0ee76dd2263f77ad90a7cec59da84e047c8",
            "furniture/f71c22e2956fcc6e97ace5d6bc9334ddcd1842c1",
        ),
    },
    "102344280": {
        # SIMPLE hssd:scene3 is authored in centimetres.
        "scale": 0.01,
        "surface": "furniture/d68aaf2484eec4c754d3b6e07adc09293d8b36de",
        "hide_prims": (),
    },
}


def resolve_hssd_scene_profile(hssd_path: Path) -> dict:
    scene_name = hssd_path.stem.removesuffix("_local")
    try:
        return HSSD_SCENE_PROFILES[scene_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported HSSD scene {scene_name!r}; add its scale and insertion "
            "surface to HSSD_SCENE_PROFILES"
        ) from exc


def format_progress_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_progress_line(current: int, total: int, started_at: float) -> str:
    elapsed = max(time.monotonic() - started_at, 1e-9)
    rate = current / elapsed
    remaining = max(total - current, 0)
    eta = remaining / rate if rate > 0.0 else 0.0
    bar_width = 30
    filled = min(bar_width, int(bar_width * current / max(total, 1)))
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = 100.0 * current / max(total, 1)
    return (
        f"[Task3IsaacReplay] [{bar}] {current}/{total} {percent:6.2f}% "
        f"{rate:5.1f} frame/s elapsed={format_progress_duration(elapsed)} "
        f"ETA={format_progress_duration(eta)}"
    )


def publish_live_ego_frame(
    path: Path,
    rgb,
    *,
    sequence: int,
    sim_time: float,
    jpeg_quality: int,
    session_id: str | None = None,
    request_id: int | None = None,
    state_sequence: int | None = None,
) -> None:
    """Atomically publish one JPEG plus synchronization metadata in shared memory."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    jpeg = io.BytesIO()
    Image.fromarray(rgb).save(jpeg, format="JPEG", quality=jpeg_quality)
    metadata_dict = {
        "schema_version": (
            "humanoidvla.isaac_ego.v2"
            if request_id is not None
            else "humanoidvla.isaac_ego.v1"
        ),
        "sequence": int(sequence),
        "sim_time": float(sim_time),
        "published_monotonic_s": time.monotonic(),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "encoding": "jpeg",
    }
    if session_id is not None:
        metadata_dict["session_id"] = str(session_id)
    if request_id is not None:
        metadata_dict["request_id"] = int(request_id)
    if state_sequence is not None:
        metadata_dict["state_sequence"] = int(state_sequence)
    metadata = json.dumps(metadata_dict, separators=(",", ":")).encode("utf-8")
    payload = ISAAC_EGO_FRAME_MAGIC + struct.pack("<I", len(metadata)) + metadata + jpeg.getvalue()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)

ROBOT_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]
TASK_OBJECT_PROFILES = (
    {
        "name": "three_drawer_cabinet",
        "prim": "/World/Task3/Geometry/drawer_mount",
        # Task5 SCQ: the cabinet is fixed and the three drawers are scalar
        # slide joints at qpos[50:53].  Keep the order authored by the MJCF.
        "joints": [
            "drawer_lower_drawer_slide",
            "drawer_middle_drawer_slide",
            "drawer_upper_drawer_slide",
        ],
    },
    {
        "name": "step_trash_can",
        "prim": "/World/Task3/Geometry/step_trash_can_mount",
        # MuJoCo qpos order in the original 20260612 Task3 scene.
        "joints": [
            "step_trash_can_pedal_hinge_joint",
            "step_trash_can_lid_hinge_joint",
        ],
    },
    {
        "name": "custom_step_trash_can",
        "prim": "/World/Task3/Geometry/custom_step_trash_can_mount",
        # MuJoCo qpos order in 20260804_task3_new is lid, then pedal.
        "joints": [
            "custom_step_trash_can_lid_hinge_joint",
            "custom_step_trash_can_pedal_hinge_joint",
        ],
    },
)

TORSO_PRIM_PATH = (
    "/World/Task3/Geometry/pelvis/waist_yaw_link/waist_roll_link/torso_link"
)
QPOS_COLUMN_RE = re.compile(r"\[qpos(\d+)\]$")


def resolve_mjcf_qpos_layout(mjcf_path: Path) -> tuple[dict[str, dict], int]:
    """Expand MJCF includes and derive named joint qpos addresses."""
    joints: dict[str, dict] = {}
    qpos_address = 0
    include_stack: list[Path] = []

    def visit(element, source_dir: Path, body_path: tuple[str, ...], in_worldbody: bool):
        nonlocal qpos_address
        if element.tag == "include":
            include_path = (source_dir / element.attrib["file"]).resolve()
            if include_path in include_stack:
                raise ValueError(f"recursive MJCF include: {include_path}")
            include_stack.append(include_path)
            included_root = ET.parse(include_path).getroot()
            for child in included_root:
                visit(child, include_path.parent, body_path, in_worldbody)
            include_stack.pop()
            return
        if element.tag == "worldbody":
            for child in element:
                visit(child, source_dir, body_path, True)
            return
        if not in_worldbody:
            return
        if element.tag == "body":
            body_name = element.get("name")
            next_body_path = body_path + ((body_name,) if body_name else ())
            for child in element:
                visit(child, source_dir, next_body_path, True)
            return
        if element.tag == "frame":
            for child in element:
                visit(child, source_dir, body_path, True)
            return
        if element.tag not in ("joint", "freejoint"):
            return

        joint_name = element.get("name")
        if not joint_name:
            raise ValueError(f"unnamed MJCF {element.tag} in {mjcf_path}")
        if joint_name in joints:
            raise ValueError(f"duplicate MJCF joint name: {joint_name}")
        joint_type = "free" if element.tag == "freejoint" else element.get("type", "hinge")
        width = {"free": 7, "ball": 4, "hinge": 1, "slide": 1}.get(joint_type)
        if width is None:
            raise ValueError(f"unsupported MJCF joint type {joint_type}: {joint_name}")
        joints[joint_name] = {
            "qpos_address": qpos_address,
            "width": width,
            "type": joint_type,
            "body_path": body_path,
        }
        qpos_address += width

    resolved_mjcf = mjcf_path.expanduser().resolve()
    include_stack.append(resolved_mjcf)
    root = ET.parse(resolved_mjcf).getroot()
    for child in root:
        visit(child, resolved_mjcf.parent, (), False)
    include_stack.pop()
    return joints, qpos_address


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--hssd-usd", type=Path, default=DEFAULT_HSSD)
    parser.add_argument(
        "--background-mode", choices=("hssd", "generic-3dgs"), default="hssd"
    )
    parser.add_argument("--background-scale", type=float, default=1.0)
    parser.add_argument(
        "--background-rotate-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0)
    )
    parser.add_argument(
        "--background-translate", type=float, nargs=3, default=(0.0, 0.0, 0.0)
    )
    parser.add_argument(
        "--hide-background-prim",
        action="append",
        default=[],
        metavar="RELATIVE_PRIM_PATH",
        help=(
            "Hide an additional prim below the referenced background root in "
            "the Isaac session layer. Repeat for multiple task-specific props."
        ),
    )
    parser.add_argument(
        "--background-align-min-z",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--robot-colors-srgb-to-linear",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Treat direct robot geom RGB values as sRGB and convert them to "
            "linear RGB in the derived Isaac-only MJCF bundle."
        ),
    )
    parser.add_argument(
        "--smooth-task-table-cylinder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace the Task1 primitive round-table top with an equivalent "
            "128-segment visual mesh in the derived Isaac-only MJCF bundle."
        ),
    )
    parser.add_argument(
        "--remove-task6-top-shelf-visual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Remove the top tier from the known Task6 shelf visual meshes, "
            "while retaining two front-back rails connecting the four full "
            "posts, in the derived Isaac MJCF bundle. MuJoCo source assets "
            "and collision geometry remain unchanged."
        ),
    )
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=23331)
    parser.add_argument("--live-ego-frame-path", type=Path)
    parser.add_argument("--live-ego-rate-hz", type=float, default=15.0)
    parser.add_argument(
        "--live-ego-throttle-render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Render only when the next live Ego frame is due. This keeps the "
            "UDP mirror latest-only and avoids an unbounded Isaac render loop "
            "during policy evaluation."
        ),
    )
    parser.add_argument(
        "--live-ego-strict-request-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Render exactly once for each new session_id/request_id state and "
            "echo those identifiers in the HVEGO metadata."
        ),
    )
    parser.add_argument("--live-ego-jpeg-quality", type=int, default=95)
    parser.add_argument("--live-ego-width", type=int, default=640)
    parser.add_argument("--live-ego-height", type=int, default=480)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=(-2.5, -1.65, 1.8))
    parser.add_argument("--camera-target", type=float, nargs=3, default=(0.55, -0.05, 0.72))
    parser.add_argument("--camera-focal-length", type=float, default=23.0)
    parser.add_argument(
        "--camera-fovy",
        type=float,
        default=0.0,
        help="Exact vertical FOV in degrees; 0 keeps the focal-length-only behavior",
    )
    parser.add_argument(
        "--camera-mode", choices=("world", "head"), default="world"
    )
    parser.add_argument(
        "--no-head-camera",
        action="store_true",
        help=(
            "Keep the main viewport on the world camera initialized by "
            "--camera-eye/--camera-target. This does not disable an explicitly "
            "enabled --ego-inset."
        ),
    )
    parser.add_argument(
        "--ego-inset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open a separate torso-following Ego viewport at the top right.",
    )
    parser.add_argument("--ego-width", type=int, default=480)
    parser.add_argument("--ego-height", type=int, default=360)
    parser.add_argument(
        "--ego-window-width",
        type=int,
        default=0,
        help="Ego GUI window width; 0 uses --ego-width.",
    )
    parser.add_argument(
        "--ego-window-height",
        type=int,
        default=0,
        help="Ego GUI window height; 0 uses --ego-height.",
    )
    parser.add_argument("--ego-window-margin", type=int, default=18)
    parser.add_argument("--ego-camera-fovy", type=float, default=70.0)
    parser.add_argument(
        "--ego-camera-pitch-deg",
        type=float,
        default=15.0,
        help="Additional downward pitch for the Ego inset (degrees).",
    )
    parser.add_argument("--head-camera-parent", default=TORSO_PRIM_PATH)
    parser.add_argument(
        "--head-camera-eye", type=float, nargs=3, default=(0.06, 0.0, 0.45)
    )
    parser.add_argument(
        "--head-camera-pitch-deg",
        type=float,
        default=0.0,
        help="Additional downward pitch for the main head camera (degrees).",
    )
    parser.add_argument(
        "--head-camera-forward",
        type=float,
        nargs=3,
        default=(0.71735609, 0.0, -0.69670671),
    )
    parser.add_argument(
        "--head-camera-up",
        type=float,
        nargs=3,
        default=(0.69670671, 0.0, 0.71735609),
    )
    parser.add_argument("--head-camera-near-clip", type=float, default=0.2)
    parser.add_argument(
        "--task-translate", type=float, nargs=3, default=(0.0, 0.0, 0.0)
    )
    parser.add_argument(
        "--transform-task-root",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply --task-translate/--task-yaw-deg to /World/Task3 so fixed "
            "task assets and their joints move with the driven robot and objects."
        ),
    )
    parser.add_argument(
        "--task-yaw-deg",
        type=float,
        default=0.0,
        help="Rotate the complete MuJoCo task about world Z; positive is counter-clockwise.",
    )
    parser.add_argument(
        "--trash-translate", type=float, nargs=3, default=(0.0, 0.0, 0.0)
    )
    parser.add_argument("--robot-z-offset", type=float, default=0.012)
    parser.add_argument("--trash-z-offset", type=float, default=0.006)
    parser.add_argument("--free-object-z-offset", type=float, default=0.0)
    parser.add_argument(
        "--floor-material-dir",
        type=Path,
        help=(
            "Optional visual-only PBR floor directory containing "
            "*_BaseColor.png, *_N.png, and *_ORM.png."
        ),
    )
    parser.add_argument("--floor-size", type=float, default=8.0)
    parser.add_argument("--floor-center", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--floor-z", type=float, default=0.012)
    parser.add_argument("--floor-uv-scale", type=float, default=4.0)
    parser.add_argument(
        "--light-rig",
        choices=("scripted", "default"),
        default="scripted",
        help=(
            "Use the existing scripted Dome/Distant lights or Isaac Sim's "
            "built-in Light Rigs > Default asset."
        ),
    )
    parser.add_argument(
        "--randomize-lighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample one deterministic lighting setup for the whole episode.",
    )
    parser.add_argument("--lighting-seed", type=int, default=0)
    parser.add_argument(
        "--live-lighting-manifest",
        type=Path,
        help=(
            "Restore exact per-recording scripted lighting from the training "
            "dataset meta/lighting.jsonl during live Ego evaluation."
        ),
    )
    parser.add_argument(
        "--dome-intensity-range", type=float, nargs=2, default=(250.0, 700.0)
    )
    parser.add_argument(
        "--distant-intensity-range", type=float, nargs=2, default=(800.0, 2200.0)
    )
    parser.add_argument(
        "--light-temperature-range", type=float, nargs=2, default=(3500.0, 7500.0)
    )
    parser.add_argument(
        "--distant-rotate-x-range", type=float, nargs=2, default=(285.0, 345.0)
    )
    parser.add_argument(
        "--distant-rotate-z-range", type=float, nargs=2, default=(0.0, 360.0)
    )
    parser.add_argument(
        "--replay-csv",
        type=Path,
        help="Offline Task3 data.csv to replay instead of subscribing to DDS.",
    )
    parser.add_argument(
        "--replay-output-dir",
        type=Path,
        help="Directory for synchronized Isaac RGB frames and manifest.json.",
    )
    parser.add_argument(
        "--replay-batch-plan",
        type=Path,
        help=(
            "JSON plan for rendering multiple compatible recordings in one "
            "Isaac process. Mutually exclusive with --replay-csv/output-dir."
        ),
    )
    parser.add_argument("--replay-downsample", type=int, default=8)
    parser.add_argument(
        "--replay-png-compress-level",
        type=int,
        choices=range(10),
        default=1,
        help="Lossless PNG compression level for temporary replay frames.",
    )
    parser.add_argument(
        "--replay-policy-valid-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--replay-overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def resolve_floor_textures(material_dir: Path) -> dict[str, Path]:
    """Resolve the Synthesis PBR maps without depending on Isaac imports."""
    material_dir = material_dir.expanduser().resolve()
    if not material_dir.is_dir():
        raise NotADirectoryError(material_dir)
    suffixes = {
        "base_color": "_BaseColor.png",
        "normal": "_N.png",
        "orm": "_ORM.png",
    }
    textures = {}
    for label, suffix in suffixes.items():
        candidates = sorted(
            path
            for path in material_dir.rglob(f"*{suffix}")
            if not any(part.startswith(".") or part == "__MACOSX" for part in path.parts)
        )
        if len(candidates) != 1:
            raise ValueError(
                f"expected exactly one *{suffix} below {material_dir}, "
                f"found {len(candidates)}"
            )
        textures[label] = candidates[0]
    return textures


def add_visual_floor(stage, args, textures, *, Gf, Sdf, UsdGeom, UsdShade) -> None:
    """Add a render-only PBR quad; it has no collision or rigid-body schemas."""
    if args.floor_size <= 0 or args.floor_uv_scale <= 0:
        raise ValueError("floor size and UV scale must be positive")

    half = args.floor_size * 0.5
    center_x, center_y = args.floor_center
    mesh = UsdGeom.Mesh.Define(stage, "/World/Task3VisualFloor")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(center_x - half, center_y - half, args.floor_z),
            Gf.Vec3f(center_x + half, center_y - half, args.floor_z),
            Gf.Vec3f(center_x + half, center_y + half, args.floor_z),
            Gf.Vec3f(center_x - half, center_y + half, args.floor_z),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)

    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    uv = float(args.floor_uv_scale)
    st.Set(
        [
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(uv, 0.0),
            Gf.Vec2f(uv, uv),
            Gf.Vec2f(0.0, uv),
        ]
    )

    material = UsdShade.Material.Define(stage, "/World/Looks/Task3FloorMaterial")
    surface = UsdShade.Shader.Define(stage, "/World/Looks/Task3FloorMaterial/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    reader = UsdShade.Shader.Define(stage, "/World/Looks/Task3FloorMaterial/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def texture_shader(name, path, *, color_space):
        shader = UsdShade.Shader.Define(
            stage, f"/World/Looks/Task3FloorMaterial/{name}"
        )
        shader.CreateIdAttr("UsdUVTexture")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
        shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader_output)
        return shader

    base_color = texture_shader("BaseColor", textures["base_color"], color_space="sRGB")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        base_color.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )

    normal = texture_shader("Normal", textures["normal"], color_space="raw")
    normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 2))
    normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, -1))
    surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )

    orm = texture_shader("ORM", textures["orm"], color_space="raw")
    surface.GetInput("roughness").ConnectToSource(
        orm.CreateOutput("g", Sdf.ValueTypeNames.Float)
    )
    surface.GetInput("metallic").ConnectToSource(
        orm.CreateOutput("b", Sdf.ValueTypeNames.Float)
    )
    material.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    print(
        f"[Task3Isaac] visual_floor={textures['base_color'].parent} "
        f"size={args.floor_size} uv_scale={args.floor_uv_scale} z={args.floor_z}",
        flush=True,
    )


def load_replay_frames(
    csv_path: Path, *, downsample: int, policy_valid_only: bool
) -> list[dict]:
    """Load exactly the rows selected by the existing LeRobot converter."""
    if downsample <= 0:
        raise ValueError("replay downsample must be positive")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"replay CSV has no header: {csv_path}")
        indexed_qpos = []
        for column in reader.fieldnames:
            match = QPOS_COLUMN_RE.search(column)
            if match:
                indexed_qpos.append((int(match.group(1)), column))
        indexed_qpos.sort()
        if not indexed_qpos or indexed_qpos[0][0] != 0:
            raise ValueError(f"replay CSV has no indexed qpos columns: {csv_path}")
        expected = list(range(indexed_qpos[-1][0] + 1))
        actual = [index for index, _ in indexed_qpos]
        if actual != expected:
            raise ValueError("replay qpos columns are not contiguous from qpos0")

        selected = []
        valid_index = 0
        for source_row_index, row in enumerate(reader):
            if policy_valid_only and "policy_valid" in row:
                if int(float(row["policy_valid"])) != 1:
                    continue
            keep = valid_index % downsample == 0
            valid_index += 1
            if not keep:
                continue
            selected.append(
                {
                    "source_row_index": source_row_index,
                    "sample_index": int(float(row.get("sample_index", source_row_index))),
                    "control_time_s": float(row.get("control_time_s", "nan")),
                    "mujoco_time_s": float(row.get("mujoco_time_s", "nan")),
                    "qpos": [float(row[column]) for _, column in indexed_qpos],
                }
            )
    if len(selected) < 2:
        raise ValueError(
            f"replay needs at least two selected rows, found {len(selected)}"
        )
    return selected


def prepare_replay_output(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("frame_*.png"))
    existing += [
        path
        for path in (output_dir / "manifest.json", output_dir / "manifest.partial.json")
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"replay output is not empty: {output_dir}; pass --replay-overwrite"
        )
    if overwrite:
        for path in existing:
            path.unlink()


def resolve_task_object_profile(
    mjcf_path: Path, qpos_layout: dict[str, dict]
) -> dict | None:
    """Select an optional imported task-object articulation by joint names."""
    scene_joint_names = set(qpos_layout)
    matches = [
        profile
        for profile in TASK_OBJECT_PROFILES
        if set(profile["joints"]).issubset(scene_joint_names)
    ]
    if len(matches) > 1:
        raise ValueError(
            "expected at most one supported task-object profile in "
            f"{mjcf_path}, found {[profile['name'] for profile in matches]}; "
            f"scene joints={sorted(scene_joint_names)}"
        )
    return matches[0] if matches else None


def resolve_free_task_objects(qpos_layout: dict[str, dict]) -> list[dict]:
    """Map non-robot MJCF free joints to their imported rigid-body prims."""
    objects = []
    for joint_name, joint in qpos_layout.items():
        if joint["type"] != "free" or joint_name == "floating_base_joint":
            continue
        if not joint["body_path"]:
            raise ValueError(f"free joint has no parent body: {joint_name}")
        objects.append(
            {
                "joint": joint_name,
                "qpos_address": joint["qpos_address"],
                "prim": "/World/Task3/Geometry/" + "/".join(joint["body_path"]),
            }
        )
    return objects


def mjcf_xml_bundle_sha256(mjcf_path: Path) -> str:
    """Hash every XML imported from a recording's g1 snapshot directory."""
    digest = hashlib.sha256()
    digest.update(b"task3-render-source-v4-always-derived-unique-textures")
    for xml_path in sorted(mjcf_path.parent.glob("*.xml")):
        digest.update(xml_path.name.encode("utf-8"))
        digest.update(xml_path.read_bytes())
    return digest.hexdigest()


def load_replay_batch_plan(plan_path: Path, mjcf_path: Path) -> list[dict]:
    plan_path = plan_path.expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "task3_isaac_replay_batch.v1":
        raise ValueError(f"unsupported replay batch plan: {plan_path}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"replay batch plan has no episodes: {plan_path}")

    expected_bundle_hash = mjcf_xml_bundle_sha256(mjcf_path)
    jobs = []
    seen_outputs = set()
    for index, entry in enumerate(episodes):
        if not isinstance(entry, dict):
            raise ValueError(f"batch episode {index} must be an object")
        csv_path = Path(entry["csv"]).expanduser().resolve()
        episode_mjcf = Path(entry["mjcf"]).expanduser().resolve()
        output_dir = Path(entry["output_dir"]).expanduser().resolve()
        for required in (csv_path, episode_mjcf):
            if not required.is_file():
                raise FileNotFoundError(required)
        bundle_hash = mjcf_xml_bundle_sha256(episode_mjcf)
        if bundle_hash != expected_bundle_hash:
            raise ValueError(
                "batch recording MJCF bundle differs from the loaded stage: "
                f"episode={csv_path.parent.name} expected={expected_bundle_hash} "
                f"actual={bundle_hash}"
            )
        output_key = str(output_dir)
        if output_key in seen_outputs:
            raise ValueError(f"duplicate replay output directory: {output_dir}")
        seen_outputs.add(output_key)
        jobs.append(
            {
                "csv": csv_path,
                "output_dir": output_dir,
                "lighting_seed": int(entry.get("lighting_seed", 0)),
            }
        )
    return jobs


def load_live_lighting_manifest(
    manifest_path: Path,
) -> tuple[dict[int, dict], dict[str, dict]]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    by_index: dict[int, dict] = {}
    by_name: dict[str, dict] = {}
    required = {
        "dome_intensity",
        "distant_intensity",
        "color_temperature_k",
        "distant_rotate_xyz_deg",
        "seed",
    }
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        entry = json.loads(line)
        episode_index = int(entry["episode_index"])
        recording_name = str(entry["recording_name"])
        lighting = dict(entry["lighting"])
        missing = sorted(required - lighting.keys())
        if missing:
            raise ValueError(
                f"training lighting entry {line_number} is missing {missing}: "
                f"{manifest_path}"
            )
        if episode_index in by_index:
            raise ValueError(f"duplicate lighting episode_index={episode_index}")
        if recording_name in by_name:
            raise ValueError(f"duplicate lighting recording_name={recording_name}")
        normalized = {
            "episode_index": episode_index,
            "recording_name": recording_name,
            "lighting": lighting,
        }
        by_index[episode_index] = normalized
        by_name[recording_name] = normalized
    if not by_index:
        raise ValueError(f"training lighting manifest is empty: {manifest_path}")
    return by_index, by_name


def srgb_channel_to_linear(value: float) -> float:
    """Convert one normalized sRGB channel to linear RGB."""
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def linearize_robot_geom_rgba(tree: ET.ElementTree) -> int:
    """Linearize direct robot geom colors while preserving alpha."""
    converted = 0
    for geom in tree.iter("geom"):
        rgba = geom.get("rgba")
        if not rgba:
            continue
        channels = [float(value) for value in rgba.split()]
        if len(channels) != 4:
            raise ValueError(f"expected four geom rgba channels, got: {rgba!r}")
        channels[:3] = [srgb_channel_to_linear(value) for value in channels[:3]]
        geom.set("rgba", " ".join(f"{value:.9g}" for value in channels))
        converted += 1
    return converted


def write_cylinder_visual_obj(
    path: Path,
    *,
    radius: float,
    half_height: float,
    segments: int = 128,
) -> None:
    """Write a smooth, closed Z-axis cylinder used only by the Isaac import."""
    if radius <= 0.0 or half_height <= 0.0:
        raise ValueError(
            f"cylinder dimensions must be positive: radius={radius} "
            f"half_height={half_height}"
        )
    if segments < 3:
        raise ValueError(f"cylinder must have at least three segments: {segments}")

    lines = [
        "# Generated visual-only cylinder for the derived Isaac MJCF bundle",
        "o smooth_task_table_cylinder",
        f"v 0 0 {half_height:.9g}",
        f"v 0 0 {-half_height:.9g}",
    ]
    for z in (half_height, -half_height):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            lines.append(
                f"v {radius * math.cos(angle):.9g} "
                f"{radius * math.sin(angle):.9g} {z:.9g}"
            )

    lines.extend(("vn 0 0 1", "vn 0 0 -1"))
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        lines.append(f"vn {math.cos(angle):.9g} {math.sin(angle):.9g} 0")

    top_start = 3
    bottom_start = top_start + segments
    radial_normal_start = 3
    for index in range(segments):
        next_index = (index + 1) % segments
        top = top_start + index
        top_next = top_start + next_index
        bottom = bottom_start + index
        bottom_next = bottom_start + next_index
        normal = radial_normal_start + index
        normal_next = radial_normal_start + next_index
        lines.append(f"f 1//1 {top}//1 {top_next}//1")
        lines.append(f"f 2//2 {bottom_next}//2 {bottom}//2")
        lines.append(
            f"f {top}//{normal} {bottom}//{normal} "
            f"{bottom_next}//{normal_next}"
        )
        lines.append(
            f"f {top}//{normal} {bottom_next}//{normal_next} "
            f"{top_next}//{normal_next}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replace_task_table_cylinder_visual(
    tree: ET.ElementTree,
    generated_visual_dir: Path,
    *,
    segments: int = 128,
) -> int:
    """Replace Task1's analytic table-top visual with an equivalent OBJ mesh."""
    target_name = "primitive_round_table_top_visual"
    mesh_name = f"{target_name}_{segments}seg_mesh"
    matches = [geom for geom in tree.iter("geom") if geom.get("name") == target_name]
    if not matches:
        return 0
    if len(matches) != 1:
        raise ValueError(f"expected one {target_name!r} geom, found {len(matches)}")

    geom = matches[0]
    if geom.get("type") != "cylinder":
        raise ValueError(f"{target_name!r} is not a cylinder: {geom.get('type')!r}")
    size = [float(value) for value in geom.get("size", "").split()]
    if len(size) != 2:
        raise ValueError(f"expected cylinder radius/half-height, got: {size}")

    generated_obj = generated_visual_dir / f"{mesh_name}.obj"
    write_cylinder_visual_obj(
        generated_obj,
        radius=size[0],
        half_height=size[1],
        segments=segments,
    )
    asset = tree.getroot().find("asset")
    if asset is None:
        raise ValueError("MJCF has no <asset> section for the generated table mesh")
    ET.SubElement(
        asset,
        "mesh",
        name=mesh_name,
        file=str(generated_obj.resolve()),
    )
    geom.set("type", "mesh")
    geom.set("mesh", mesh_name)
    geom.attrib.pop("size", None)
    return 1


def write_obj_clipped_below_y(
    source: Path,
    destination: Path,
    *,
    cutoff_y: float,
    vertex_transform=None,
) -> tuple[int, int]:
    """Clip a triangle-only OBJ against y <= cutoff_y without touching source."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    object_name = destination.stem
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if not fields or fields[0] == "#":
            continue
        if fields[0] == "o" and len(fields) >= 2:
            object_name = fields[1]
        elif fields[0] == "v":
            if len(fields) != 4:
                raise ValueError(
                    f"expected xyz vertex in {source}:{line_number}: {line!r}"
                )
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f":
            if len(fields) != 4 or any("/" in value for value in fields[1:]):
                raise ValueError(
                    f"expected vertex-only triangle in {source}:{line_number}: "
                    f"{line!r}"
                )
            indices = tuple(int(value) for value in fields[1:4])
            if any(index <= 0 for index in indices):
                raise ValueError(
                    f"expected positive OBJ indices in {source}:{line_number}"
                )
            faces.append(indices)

    if not vertices or not faces:
        raise ValueError(f"OBJ has no vertices/faces: {source}")

    clipped_vertices: list[tuple[float, float, float]] = []
    clipped_faces: list[tuple[int, int, int]] = []

    def intersection(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        denominator = end[1] - start[1]
        if abs(denominator) < 1e-12:
            return (start[0], cutoff_y, start[2])
        ratio = (cutoff_y - start[1]) / denominator
        return tuple(
            start[axis] + ratio * (end[axis] - start[axis])
            for axis in range(3)
        )

    for face in faces:
        polygon = [vertices[index - 1] for index in face]
        output: list[tuple[float, float, float]] = []
        previous = polygon[-1]
        previous_inside = previous[1] <= cutoff_y
        for current in polygon:
            current_inside = current[1] <= cutoff_y
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous = current
            previous_inside = current_inside
        if len(output) < 3:
            continue
        first = len(clipped_vertices) + 1
        clipped_vertices.extend(output)
        for index in range(1, len(output) - 1):
            clipped_faces.append((first, first + index, first + index + 1))

    if not clipped_faces:
        raise ValueError(f"clipping removed the complete OBJ: {source}")
    lines = [
        "# Derived Isaac-only mesh; authoritative source is unchanged",
        f"# source={source}",
        f"# kept y <= {cutoff_y:.9g}",
        f"o {object_name}_without_top_tier",
    ]
    if vertex_transform is not None:
        clipped_vertices = [vertex_transform(vertex) for vertex in clipped_vertices]
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in clipped_vertices)
    lines.extend(f"f {a} {b} {c}" for a, b, c in clipped_faces)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(faces), len(clipped_faces)


def close_task6_shelf_layer_gaps(
    vertex: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Expand each retained shelf tier to overlap the four posts by 2 mm."""
    x, y, z = vertex
    tier_planes = (0.18, 0.592, 1.0, 1.4)
    tier_x_bounds = (
        (-0.359, 0.371),
        (-0.370, 0.350),
        (-0.375, 0.335),
        (-0.378, 0.322),
    )
    tier_index = min(
        range(len(tier_planes)), key=lambda index: abs(y - tier_planes[index])
    )
    source_min_x, source_max_x = tier_x_bounds[tier_index]
    target_min_x = -0.379
    target_max_x = 0.349
    x = target_min_x + (x - source_min_x) * (
        (target_max_x - target_min_x) / (source_max_x - source_min_x)
    )
    # Original rim depth is [-0.205, 0.205]; the post inner faces are at
    # +/-0.207.  Two millimetres of visual overlap avoids a raster seam.
    z *= 0.209 / 0.205
    return x, y, z


def write_task6_frame_with_top_depth_rails(
    source: Path,
    destination: Path,
    *,
    top_min_y: float,
) -> tuple[int, int]:
    """Keep the four full posts and two front-back rails, not the top crossbars."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if not fields or fields[0] == "#":
            continue
        if fields[0] == "v":
            if len(fields) != 4:
                raise ValueError(
                    f"expected xyz vertex in {source}:{line_number}: {line!r}"
                )
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f":
            if len(fields) != 4 or any("/" in value for value in fields[1:]):
                raise ValueError(
                    f"expected vertex-only triangle in {source}:{line_number}: "
                    f"{line!r}"
                )
            faces.append(tuple(int(value) - 1 for value in fields[1:4]))

    if not vertices or not faces:
        raise ValueError(f"OBJ has no vertices/faces: {source}")

    parent = list(range(len(vertices)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for face in faces:
        union(face[0], face[1])
        union(face[0], face[2])

    component_vertices: dict[int, list[int]] = {}
    for index in range(len(vertices)):
        component_vertices.setdefault(find(index), []).append(index)

    removed_components: set[int] = set()
    for root, indices in component_vertices.items():
        xs = [vertices[index][0] for index in indices]
        ys = [vertices[index][1] for index in indices]
        zs = [vertices[index][2] for index in indices]
        x_span = max(xs) - min(xs)
        z_span = max(zs) - min(zs)
        # The known Task6 frame has two top beams spanning left-right and two
        # spanning front-back.  Remove only the former; retain the latter and
        # the complete four posts so the side beams are physically connected.
        if min(ys) > top_min_y and x_span > z_span:
            removed_components.add(root)

    if len(removed_components) != 2:
        raise ValueError(
            "expected exactly two Task6 top left-right frame beams, found "
            f"{len(removed_components)} in {source}"
        )
    kept_faces = [
        face for face in faces if find(face[0]) not in removed_components
    ]
    used_indices = sorted({index for face in kept_faces for index in face})
    remap = {old: new for new, old in enumerate(used_indices, start=1)}
    lines = [
        "# Derived Isaac-only Task6 frame; authoritative source is unchanged",
        "# Four full posts and the two front-back top rails are retained",
        "o supplied_shelf_frame_without_top_tier",
    ]
    lines.extend(
        f"v {vertices[index][0]:.9g} {vertices[index][1]:.9g} "
        f"{vertices[index][2]:.9g}"
        for index in used_indices
    )
    lines.extend(
        f"f {remap[a]} {remap[b]} {remap[c]}" for a, b, c in kept_faces
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(faces), len(kept_faces)


def replace_task6_top_shelf_visuals(
    tree: ET.ElementTree,
    bundle_g1: Path,
    generated_visual_dir: Path,
    *,
    cutoff_y: float = 1.52,
) -> tuple[int, int, int]:
    """Use clipped derived meshes for the known five-tier Task6 shelf."""
    target_meshes = (
        "supplied_shelf_object_visual",
        "supplied_shelf_object_visual_grid",
        "supplied_shelf_object_visual_rims",
    )
    mesh_by_name = {
        mesh.get("name"): mesh
        for mesh in tree.getroot().findall("./asset/mesh")
        if mesh.get("name") in target_meshes
    }
    if set(mesh_by_name) != set(target_meshes):
        raise ValueError(
            "--remove-task6-top-shelf-visual requires the known Task6 shelf "
            f"meshes; found={sorted(mesh_by_name)}"
        )
    source_faces = 0
    derived_faces = 0
    for mesh_name in target_meshes:
        mesh = mesh_by_name[mesh_name]
        file_attr = mesh.get("file")
        if not file_attr:
            raise ValueError(f"Task6 shelf mesh has no file: {mesh_name}")
        source = (bundle_g1 / file_attr).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = generated_visual_dir / f"{mesh_name}_without_top_tier.obj"
        if mesh_name == "supplied_shelf_object_visual":
            before, after = write_task6_frame_with_top_depth_rails(
                source,
                destination,
                top_min_y=cutoff_y,
            )
        else:
            before, after = write_obj_clipped_below_y(
                source,
                destination,
                cutoff_y=cutoff_y,
                vertex_transform=close_task6_shelf_layer_gaps,
            )
        source_faces += before
        derived_faces += after
        # Isaac Sim 6 resolves a relative mesh path below model/g1/meshes,
        # regardless of the derived MJCF's directory.  Keep the generated
        # asset in the visual-only bundle and point to it explicitly.
        mesh.set("file", str(destination.resolve()))
    return len(target_meshes), source_faces, derived_faces


def prepare_mjcf_source(
    mjcf_path: Path,
    output_dir: Path,
    *,
    robot_colors_srgb_to_linear: bool = False,
    smooth_task_table_cylinder: bool = False,
    remove_task6_top_shelf_visual: bool = False,
) -> tuple[Path, str]:
    """Create a visual-only import bundle with collision and texture fixes."""
    source_dir = mjcf_path.parent
    xml_paths = sorted(source_dir.glob("*.xml"))
    source_hash = mjcf_xml_bundle_sha256(mjcf_path)
    variant_tags = []
    if robot_colors_srgb_to_linear:
        variant_tags.append("robot-srgb-linear-v1")
    if smooth_task_table_cylinder:
        variant_tags.append("task-table-cylinder-mesh-128-v1")
    if remove_task6_top_shelf_visual:
        variant_tags.append(
            "task6-shelf-without-top-tier-depth-rails-closed-gaps-y1.52-v3"
        )
    if variant_tags:
        source_id = hashlib.sha256(
            f"{source_hash}:{':'.join(variant_tags)}".encode("utf-8")
        ).hexdigest()[:12]
    else:
        source_id = source_hash[:12]

    bundle_g1 = output_dir / "task3_mjcf_sources" / source_id / "mujoco/model/g1"
    bundle_g1.mkdir(parents=True, exist_ok=True)
    links = {
        bundle_g1 / "meshes": ROOT / "mujoco/model/g1/meshes",
        bundle_g1 / "objects": ROOT / "mujoco/model/g1/objects",
        bundle_g1 / "textures": ROOT / "mujoco/model/g1/textures",
        # Recording XMLs contain both ../task_assets and ../../task_assets
        # spellings.  Provide both derived-bundle locations without changing
        # the authoritative snapshot XML or repository assets.
        bundle_g1.parent / "task_assets": ROOT / "mujoco/model/task_assets",
        bundle_g1.parent / "robotwin_assets": ROOT / "mujoco/model/robotwin_assets",
        bundle_g1.parent.parent / "task_assets": ROOT / "mujoco/model/task_assets",
        bundle_g1.parent.parent / "robotwin_assets": ROOT / "mujoco/model/robotwin_assets",
    }
    for link, target in links.items():
        if not target.exists():
            raise FileNotFoundError(target)
        if link.is_symlink():
            if link.resolve() != target.resolve():
                link.unlink()
            else:
                continue
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)

    # The Isaac MJCF importer flattens source textures into one Textures/
    # directory.  Task4's round table and green box both use a file named
    # texture_diffuse.png from different asset directories, so the later copy
    # overwrites the former and turns the table green.  Give every file-backed
    # MJCF texture a stable unique basename in this derived bundle.
    unique_texture_dir = bundle_g1 / "isaac_unique_textures"
    unique_texture_dir.mkdir(parents=True, exist_ok=True)
    generated_visual_dir = bundle_g1 / "isaac_generated_visuals"
    removed_collision_geoms = 0
    remapped_textures = 0
    linearized_robot_geoms = 0
    smoothed_table_visuals = 0
    trimmed_shelf_visuals = 0
    shelf_source_faces = 0
    shelf_derived_faces = 0
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        for parent in tree.iter():
            for child in list(parent):
                if child.tag == "geom" and child.get("group") == "3":
                    parent.remove(child)
                    removed_collision_geoms += 1
        if robot_colors_srgb_to_linear and xml_path.name.startswith("g1_"):
            linearized_robot_geoms += linearize_robot_geom_rgba(tree)
        if smooth_task_table_cylinder and xml_path.resolve() == mjcf_path.resolve():
            smoothed_table_visuals += replace_task_table_cylinder_visual(
                tree,
                generated_visual_dir,
            )
        if (
            remove_task6_top_shelf_visual
            and xml_path.resolve() == mjcf_path.resolve()
        ):
            (
                trimmed_shelf_visuals,
                shelf_source_faces,
                shelf_derived_faces,
            ) = replace_task6_top_shelf_visuals(
                tree,
                bundle_g1,
                generated_visual_dir,
            )
        for texture in tree.iter("texture"):
            file_attr = texture.get("file")
            if not file_attr:
                continue
            source_texture = (bundle_g1 / file_attr).resolve()
            if not source_texture.is_file():
                raise FileNotFoundError(
                    f"MJCF texture not found: {file_attr} resolved to {source_texture}"
                )
            texture_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", texture.get("name", "texture"))
            path_tag = hashlib.sha256(str(source_texture).encode("utf-8")).hexdigest()[:8]
            unique_name = f"{texture_name}_{path_tag}_{source_texture.name}"
            unique_texture = unique_texture_dir / unique_name
            if unique_texture.is_symlink():
                if unique_texture.resolve() != source_texture:
                    unique_texture.unlink()
                else:
                    texture.set("file", f"isaac_unique_textures/{unique_name}")
                    remapped_textures += 1
                    continue
            if unique_texture.exists():
                raise FileExistsError(unique_texture)
            unique_texture.symlink_to(source_texture)
            texture.set("file", f"isaac_unique_textures/{unique_name}")
            remapped_textures += 1
        tree.write(
            bundle_g1 / xml_path.name,
            encoding="utf-8",
            xml_declaration=True,
        )

    bundled_mjcf = bundle_g1 / mjcf_path.name
    print(
        f"[Task3Isaac] mjcf_source_bundle={bundled_mjcf} "
        f"original={mjcf_path} removed_collision_geoms={removed_collision_geoms} "
        f"remapped_textures={remapped_textures} "
        f"linearized_robot_geoms={linearized_robot_geoms} "
        f"smoothed_table_visuals={smoothed_table_visuals} "
        f"trimmed_shelf_visuals={trimmed_shelf_visuals} "
        f"shelf_faces={shelf_source_faces}->{shelf_derived_faces}",
        flush=True,
    )
    return bundled_mjcf, source_id


def import_task3_mjcf(
    mjcf_path: Path,
    output_dir: Path,
    *,
    robot_colors_srgb_to_linear: bool = False,
    smooth_task_table_cylinder: bool = False,
    remove_task6_top_shelf_visual: bool = False,
) -> Path:
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

    import_mjcf_path, source_id = prepare_mjcf_source(
        mjcf_path,
        output_dir,
        robot_colors_srgb_to_linear=robot_colors_srgb_to_linear,
        smooth_task_table_cylinder=smooth_task_table_cylinder,
        remove_task6_top_shelf_visual=remove_task6_top_shelf_visual,
    )
    import_name = f"{mjcf_path.stem}_{source_id}"
    expected = (
        output_dir
        / "task3_mjcf"
        / import_name
        / import_mjcf_path.stem
        / f"{import_mjcf_path.stem}.usda"
    )

    def import_is_complete(root_usd: Path) -> bool:
        """Reject truncated importer output even when its root USD exists."""
        geometry_payload = root_usd.parent / "payloads/geometries.usd"
        return (
            root_usd.is_file()
            and root_usd.stat().st_size > 0
            and geometry_payload.is_file()
            and geometry_payload.stat().st_size > 0
        )

    if (
        import_is_complete(expected)
        and expected.stat().st_mtime >= mjcf_path.stat().st_mtime
    ):
        return expected
    config = MJCFImporterConfig(
        mjcf_path=str(import_mjcf_path),
        usd_path=str(output_dir / "task3_mjcf" / import_name),
        import_scene=True,
        merge_mesh=False,
        collision_from_visuals=False,
        allow_self_collision=False,
        fix_base=None,
    )
    imported = Path(MJCFImporter(config).import_mjcf()).resolve()
    if not import_is_complete(imported):
        geometry_payload = imported.parent / "payloads/geometries.usd"
        geometry_size = (
            geometry_payload.stat().st_size if geometry_payload.is_file() else -1
        )
        raise RuntimeError(
            "MJCF importer produced an incomplete USD cache: "
            f"root={imported} geometry_payload={geometry_payload} "
            f"geometry_size={geometry_size}"
        )
    return imported


def hide_collision_render_prims(stage, root_path: str) -> list[str]:
    """Hide collision-only shapes while retaining their physics schemas.

    The MJCF importer authors transparent collision geoms with display opacity
    zero.  RTX can still draw those analytic shapes, which makes a detailed
    task-object mesh (for example the Task2 Coke bottle) look like its cylinder
    collider.  Visibility is overridden only in this Isaac stage; the imported
    USD and the authoritative MuJoCo model remain unchanged.
    """
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"cannot hide collision render prims: missing {root_path}")

    hidden = []
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if name.endswith("_collision") or "_collision_" in name:
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
                hidden.append(prim.GetPath().pathString)
    return hidden


def set_camera(
    stage,
    viewport,
    eye,
    target,
    focal_length: float,
    *,
    path: str = "/World/Task3PreviewCamera",
    up=(0.0, 0.0, 1.0),
    near_clip: float = 0.01,
    fovy: float = 0.0,
    aspect: float = 16.0 / 9.0,
) -> str:
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length)
    if fovy > 0.0:
        if fovy >= 180.0 or aspect <= 0.0:
            raise ValueError("camera fovy must be below 180 degrees and aspect positive")
        # USD camera aperture and focal length use the same arbitrary unit.
        # Author both apertures so the requested vertical FOV remains exact at
        # the selected render-product aspect ratio.
        import math

        vertical_aperture = 2.0 * focal_length * math.tan(math.radians(fovy) * 0.5)
        camera.CreateVerticalApertureAttr(vertical_aperture)
        camera.CreateHorizontalApertureAttr(vertical_aperture * aspect)
    camera.CreateClippingRangeAttr(Gf.Vec2f(near_clip, 1000.0))
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)
    )
    xformable = UsdGeom.Xformable(camera)
    transform_ops = [
        op for op in xformable.GetOrderedXformOps() if op.GetOpType() == op.TypeTransform
    ]
    transform_op = transform_ops[0] if transform_ops else xformable.AddTransformOp()
    transform_op.Set(view.GetInverse())
    if viewport is not None:
        viewport.camera_path = path
    return path


def quaternion_wxyz_to_rotation(quaternion):
    import numpy as np

    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm((w, x, y, z)))
    if norm < 1e-8:
        raise ValueError("head camera parent quaternion must be non-zero")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def resolve_head_camera_vectors(args, pitch_down_deg: float = 0.0):
    import numpy as np

    local_forward = np.asarray(args.head_camera_forward, dtype=np.float64).copy()
    local_up = np.asarray(args.head_camera_up, dtype=np.float64).copy()
    if np.linalg.norm(local_forward) < 1e-8 or np.linalg.norm(local_up) < 1e-8:
        raise ValueError("head camera forward/up vectors must be non-zero")
    local_forward /= np.linalg.norm(local_forward)
    local_up -= np.dot(local_up, local_forward) * local_forward
    if np.linalg.norm(local_up) < 1e-8:
        raise ValueError("head camera forward and up vectors must not be parallel")
    local_up /= np.linalg.norm(local_up)
    if pitch_down_deg:
        import math

        angle = math.radians(pitch_down_deg)
        forward = math.cos(angle) * local_forward - math.sin(angle) * local_up
        up = math.sin(angle) * local_forward + math.cos(angle) * local_up
        local_forward, local_up = forward, up
    return local_forward, local_up


def update_head_camera(
    stage,
    camera_path: str,
    parent_position,
    parent_orientation,
    args,
    *,
    fovy: float | None = None,
    aspect: float | None = None,
    pitch_down_deg: float = 0.0,
):
    import numpy as np

    rotation = quaternion_wxyz_to_rotation(parent_orientation)
    position = np.asarray(parent_position, dtype=np.float64)
    local_eye = np.asarray(args.head_camera_eye, dtype=np.float64)
    local_forward, local_up = resolve_head_camera_vectors(args, pitch_down_deg)
    eye = position + rotation @ local_eye
    forward = rotation @ local_forward
    up = rotation @ local_up
    set_camera(
        stage,
        viewport=None,
        eye=eye,
        target=eye + forward,
        focal_length=args.camera_focal_length,
        path=camera_path,
        up=up,
        near_clip=args.head_camera_near_clip,
        fovy=args.camera_fovy if fovy is None else fovy,
        aspect=args.width / args.height if aspect is None else aspect,
    )
    return eye


def set_head_camera(
    stage,
    viewport,
    args,
    *,
    path: str = "/World/Task3HeadCamera",
    fovy: float | None = None,
    aspect: float | None = None,
    pitch_down_deg: float = 0.0,
    label: str = "head",
) -> str:
    import numpy as np
    from pxr import Gf, UsdGeom

    parent = stage.GetPrimAtPath(args.head_camera_parent)
    if not parent.IsValid():
        raise RuntimeError(
            f"head camera parent prim is missing: {args.head_camera_parent}"
        )
    eye = np.asarray(args.head_camera_eye, dtype=np.float64)
    forward, up = resolve_head_camera_vectors(args, pitch_down_deg)
    parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(parent)
    world_eye = parent_world.Transform(Gf.Vec3d(*eye))
    world_forward = parent_world.TransformDir(Gf.Vec3d(*forward))
    world_up = parent_world.TransformDir(Gf.Vec3d(*up))
    camera_path = set_camera(
        stage,
        viewport,
        world_eye,
        world_eye + world_forward,
        args.camera_focal_length,
        path=path,
        up=world_up,
        near_clip=args.head_camera_near_clip,
        fovy=args.camera_fovy if fovy is None else fovy,
        aspect=args.width / args.height if aspect is None else aspect,
    )
    print(
        f"[Task3Isaac] camera_mode={label} parent={args.head_camera_parent} "
        f"local_eye={tuple(np.round(eye, 4))} "
        f"local_forward={tuple(np.round(forward, 4))} "
        f"pitch_down={pitch_down_deg:.1f} "
        f"fovy={args.camera_fovy if fovy is None else fovy}",
        flush=True,
    )
    return camera_path


def main() -> int:
    args = parse_args()
    for range_name in (
        "dome_intensity_range",
        "distant_intensity_range",
        "light_temperature_range",
        "distant_rotate_x_range",
        "distant_rotate_z_range",
    ):
        lower, upper = getattr(args, range_name)
        if not (float("-inf") < lower <= upper < float("inf")):
            raise ValueError(f"invalid --{range_name.replace('_', '-')}: {(lower, upper)}")
    if args.dome_intensity_range[0] < 0 or args.distant_intensity_range[0] < 0:
        raise ValueError("lighting intensity ranges must be non-negative")
    if args.light_temperature_range[0] <= 0:
        raise ValueError("light temperature range must be positive")
    if args.light_rig == "default" and args.randomize_lighting:
        raise ValueError(
            "--randomize-lighting cannot be combined with --light-rig default"
        )
    if args.live_lighting_manifest is not None and args.randomize_lighting:
        raise ValueError(
            "--live-lighting-manifest restores exact training values and cannot "
            "be combined with --randomize-lighting"
        )
    if args.live_lighting_manifest is not None and args.light_rig != "scripted":
        raise ValueError(
            "--live-lighting-manifest currently requires --light-rig scripted"
        )
    if args.ego_width <= 0 or args.ego_height <= 0:
        raise ValueError("Ego inset width and height must be positive")
    if args.ego_window_width < 0 or args.ego_window_height < 0:
        raise ValueError("Ego GUI window width and height must be non-negative")
    if args.ego_window_margin < 0:
        raise ValueError("Ego inset margin must be non-negative")
    if not 0.0 < args.ego_camera_fovy < 180.0:
        raise ValueError("Ego camera FOV must be between 0 and 180 degrees")
    if not -89.0 < args.ego_camera_pitch_deg < 89.0:
        raise ValueError("Ego camera pitch must be between -89 and 89 degrees")
    if not -89.0 < args.head_camera_pitch_deg < 89.0:
        raise ValueError("Head camera pitch must be between -89 and 89 degrees")
    if args.ego_inset and args.headless:
        raise ValueError("--ego-inset requires the Isaac GUI")
    if args.no_head_camera:
        args.camera_mode = "world"
    mjcf_path = args.mjcf.expanduser().resolve()
    hssd_path = args.hssd_usd.expanduser().resolve()
    hssd_profile = (
        resolve_hssd_scene_profile(hssd_path)
        if args.background_mode == "hssd"
        else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    qpos_layout, expected_nq = resolve_mjcf_qpos_layout(mjcf_path)
    root_joint = qpos_layout.get("floating_base_joint")
    if root_joint is None or root_joint["qpos_address"] != 0 or root_joint["width"] != 7:
        raise ValueError("expected floating_base_joint at qpos[0:7]")
    task_object_profile = resolve_task_object_profile(mjcf_path, qpos_layout)
    free_task_objects = resolve_free_task_objects(qpos_layout)
    task_label = "Task4" if task_object_profile is None else (
        "Task2" if free_task_objects else "Task3"
    )
    if task_object_profile is None:
        print(
            f"[Task3Isaac] task_object_profile=none expected_nq={expected_nq}",
            flush=True,
        )
    else:
        print(
            f"[Task3Isaac] task_object_profile={task_object_profile['name']} "
            f"prim={task_object_profile['prim']} "
            f"joints={task_object_profile['joints']} expected_nq={expected_nq}",
            flush=True,
        )
    for free_object in free_task_objects:
        address = free_object["qpos_address"]
        print(
            f"[Task3Isaac] free_object joint={free_object['joint']} "
            f"qpos={address}:{address + 7} prim={free_object['prim']}",
            flush=True,
        )
    floor_textures = None
    if args.floor_material_dir is not None:
        floor_textures = resolve_floor_textures(args.floor_material_dir)
    replay_jobs = []
    if (args.replay_csv is None) != (args.replay_output_dir is None):
        raise ValueError("--replay-csv and --replay-output-dir must be used together")
    if args.replay_batch_plan is not None and args.replay_csv is not None:
        raise ValueError(
            "--replay-batch-plan is mutually exclusive with --replay-csv/output-dir"
        )
    if args.replay_csv is not None:
        if args.dds:
            raise ValueError("offline replay requires --no-dds")
        replay_csv = args.replay_csv.expanduser().resolve()
        if not replay_csv.is_file():
            raise FileNotFoundError(replay_csv)
        replay_output_dir = args.replay_output_dir.expanduser().resolve()
        replay_jobs.append(
            {
                "csv": replay_csv,
                "output_dir": replay_output_dir,
                "lighting_seed": int(args.lighting_seed),
            }
        )
    elif args.replay_batch_plan is not None:
        if args.dds:
            raise ValueError("offline replay batch requires --no-dds")
        replay_jobs = load_replay_batch_plan(args.replay_batch_plan, mjcf_path)

    live_lighting_by_index: dict[int, dict] = {}
    live_lighting_by_name: dict[str, dict] = {}
    if args.live_lighting_manifest is not None:
        live_lighting_by_index, live_lighting_by_name = load_live_lighting_manifest(
            args.live_lighting_manifest
        )
        print(
            f"[Task3Isaac] loaded_training_lighting_manifest="
            f"{args.live_lighting_manifest.expanduser().resolve()} "
            f"recordings={len(live_lighting_by_index)}",
            flush=True,
        )

    if replay_jobs:
        print(
            f"[Task3IsaacBatch] recordings={len(replay_jobs)} "
            f"single_isaac_process=1",
            flush=True,
        )
    for path in (mjcf_path, hssd_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    launch_config = {
        "headless": args.headless,
        "width": args.width,
        "height": args.height,
        "renderer": "RayTracedLighting",
        "multi_gpu": False,
    }
    extra_args = []
    if args.light_rig == "default":
        extra_args.extend(["--enable", "omni.kit.viewport.menubar.lighting"])
    if args.background_mode == "generic-3dgs":
        extra_args.extend([
            "--enable",
            "omni.rtx.spg",
            "--enable",
            "isaacsim.replicator.nurec_utils",
            "--/renderer/multiGpu/enabled=false",
        ])
    if extra_args:
        launch_config["extra_args"] = extra_args
    app = SimulationApp(launch_config)
    udp = None
    nurec_render_product = None
    nurec_render_product_path = None
    replay_render_product = None
    replay_rgb_annotator = None
    ego_viewport_window = None
    ego_camera_path = None
    try:
        import numpy as np
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim, SingleArticulation
        from omni.kit.viewport.utility import (
            capture_viewport_to_file,
            create_viewport_window,
            get_active_viewport,
        )
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        task_usd = import_task3_mjcf(
            mjcf_path,
            output_dir,
            robot_colors_srgb_to_linear=args.robot_colors_srgb_to_linear,
            smooth_task_table_cylinder=args.smooth_task_table_cylinder,
            remove_task6_top_shelf_visual=args.remove_task6_top_shelf_visual,
        )
        print(f"[Task3Isaac] task_usd={task_usd}", flush=True)

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        # Isaac 6 only propagates articulation tensor state into the
        # kinematic/Fabric render scene from SimulationContext.render() when
        # the simulation view is CUDA-backed.
        world = World(stage_units_in_meters=1.0, backend="torch", device="cuda:0")

        def as_numpy(value):
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        task_prim = stage.DefinePrim("/World/Task3", "Xform")
        task_prim.GetReferences().AddReference(str(task_usd))
        background_prim_path = (
            "/World/HSSDScene13"
            if args.background_mode == "hssd"
            else "/World/Background3DGS"
        )
        room_prim = stage.DefinePrim(background_prim_path, "Xform")
        room_prim.GetReferences().AddReference(str(hssd_path))
        is_nurec = False
        if args.background_mode == "generic-3dgs":
            from isaacsim.replicator.nurec_utils.rendering_setup import (
                setup_for_rendering,
            )

            # Large USDZ files can defer their nested NuRec payload.  Force
            # composition before classifying so a cold launch does not race
            # the volume load and incorrectly select the ordinary USD path.
            stage.Load(room_prim.GetPath())
            nurec_ok, is_nurec, has_spg, nurec_problems = setup_for_rendering(stage)
            print(
                f"[Task3Isaac] nurec={is_nurec} spg={has_spg} "
                f"setup_ok={nurec_ok}",
                flush=True,
            )
            if not nurec_ok:
                raise RuntimeError("NuRec setup failed: " + "; ".join(nurec_problems))
        while context.get_stage_loading_status()[2] > 0:
            app.update()

        task_offset = np.asarray(args.task_translate, dtype=np.float64)
        task_yaw_rad = np.deg2rad(args.task_yaw_deg)
        task_yaw_rotation = np.asarray(
            [
                [np.cos(task_yaw_rad), -np.sin(task_yaw_rad), 0.0],
                [np.sin(task_yaw_rad), np.cos(task_yaw_rad), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        task_yaw_orientation = np.asarray(
            [np.cos(task_yaw_rad / 2.0), 0.0, 0.0, np.sin(task_yaw_rad / 2.0)],
            dtype=np.float64,
        )

        if args.transform_task_root:
            task_xform = UsdGeom.Xformable(task_prim)
            task_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(*task_offset)
            )
            task_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(
                    float(task_yaw_orientation[0]),
                    Gf.Vec3d(*(float(value) for value in task_yaw_orientation[1:])),
                )
            )
            print(
                f"[Task3Isaac] transformed_task_root translate={tuple(task_offset)} "
                f"yaw_deg={args.task_yaw_deg:.3f}",
                flush=True,
            )

        hidden_collision_prims = hide_collision_render_prims(stage, "/World/Task3")
        print(
            f"[Task3Isaac] hidden_collision_render_prims="
            f"{len(hidden_collision_prims)}",
            flush=True,
        )

        if args.background_mode == "generic-3dgs" and not is_nurec:
            nurec_ok, is_nurec, has_spg, nurec_problems = setup_for_rendering(stage)
            print(
                f"[Task3Isaac] nurec_after_load={is_nurec} spg={has_spg} "
                f"setup_ok={nurec_ok}",
                flush=True,
            )
            if not nurec_ok:
                raise RuntimeError(
                    "NuRec setup after payload load failed: "
                    + "; ".join(nurec_problems)
                )
            if not is_nurec:
                raise RuntimeError(
                    "generic-3dgs asset finished loading but contains no NuRec volume: "
                    f"{hssd_path}"
                )

        if args.background_mode == "generic-3dgs":
            for prim in stage.Traverse():
                if prim.GetTypeName() not in ("Camera", "RenderProduct"):
                    continue
                details = ""
                if prim.GetTypeName() == "RenderProduct":
                    details = (
                        f" camera_targets="
                        f"{prim.GetRelationship('camera').GetTargets()}"
                    )
                print(
                    f"[Task3Isaac] authored_{prim.GetTypeName().lower()}="
                    f"{prim.GetPath()}{details}",
                    flush=True,
                )

        # The imported MJCF plane and the HSSD floor would be coincident.
        # Keep HSSD as the only visible floor; MuJoCo remains authoritative for
        # the actual contact and collision response.
        task_floor = stage.GetPrimAtPath("/World/Task3/Geometry/floor")
        if task_floor.IsValid():
            UsdGeom.Imageable(task_floor).MakeInvisible()

        # SIMPLE/HSSD and generic 3DGS backgrounds need different alignment.
        # Keep both as authored references and apply only session-layer xforms.
        room_xform = UsdGeom.Xformable(room_prim)
        room_ops = {op.GetOpName(): op for op in room_xform.GetOrderedXformOps()}
        translate = room_ops.get("xformOp:translate") or room_xform.AddTranslateOp()
        rotate = room_ops.get("xformOp:rotateXYZ") or room_xform.AddRotateXYZOp()
        scale = room_ops.get("xformOp:scale") or room_xform.AddScaleOp()
        if args.background_mode == "hssd":
            rotate.Set(Gf.Vec3f(90, 0, 0))
            hssd_scale = float(hssd_profile["scale"])
            scale.Set(Gf.Vec3f(hssd_scale, hssd_scale, hssd_scale))
            hssd_surface = str(hssd_profile["surface"])
            surface = stage.GetPrimAtPath(f"{background_prim_path}/{hssd_surface}")
            if not surface.IsValid():
                raise RuntimeError(
                    f"HSSD insertion surface is missing: {hssd_surface} in {hssd_path}"
                )
            bbox = UsdGeom.BBoxCache(
                0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
            ).ComputeWorldBound(surface).ComputeAlignedRange()
            surface_center = np.asarray(bbox.GetMin(), dtype=np.float64)
            surface_center += np.asarray(bbox.GetMax(), dtype=np.float64)
            surface_center *= 0.5
            room_offset = np.asarray([0.0, 0.25, 0.0]) - surface_center
            # Keep the profile anchor as the default, while allowing an
            # explicit session-layer offset for task-specific room placement.
            # This moves only the visual HSSD reference; MuJoCo task geometry
            # and all qpos values remain in their original coordinates.
            room_offset += np.asarray(args.background_translate, dtype=np.float64)
            translate.Set(Gf.Vec3d(*room_offset))
            UsdGeom.Imageable(surface).MakeInvisible()
            for relative_path in hssd_profile["hide_prims"]:
                background_prim = stage.GetPrimAtPath(
                    f"{background_prim_path}/{relative_path}"
                )
                if background_prim.IsValid():
                    UsdGeom.Imageable(background_prim).MakeInvisible()
            for relative_path in args.hide_background_prim:
                relative_path = str(relative_path).strip().strip("/")
                if not relative_path or ".." in relative_path.split("/"):
                    raise ValueError(
                        "--hide-background-prim must be a safe path relative to "
                        f"the background root, got: {relative_path!r}"
                    )
                background_prim = stage.GetPrimAtPath(
                    f"{background_prim_path}/{relative_path}"
                )
                if not background_prim.IsValid():
                    raise RuntimeError(
                        "Background prim requested for hiding is missing: "
                        f"{relative_path} in {hssd_path}"
                    )
                UsdGeom.Imageable(background_prim).MakeInvisible()
                print(f"[Task3Isaac] hidden_background_prim={relative_path}")
            ceilings = stage.GetPrimAtPath(f"{background_prim_path}/ceilings")
            if ceilings.IsValid():
                UsdGeom.Imageable(ceilings).MakeInvisible()
        else:
            rotate.Set(Gf.Vec3f(*args.background_rotate_xyz))
            scale.Set(Gf.Vec3f(*([args.background_scale] * 3)))
            room_offset = np.asarray(args.background_translate, dtype=np.float64)
            translate.Set(Gf.Vec3d(*room_offset))

        bounds_cache = UsdGeom.BBoxCache(
            0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        # Align the visible background floor with MuJoCo z=0 when the asset
        # provides a finite authored extent.  The floor remains visual-only.
        bounds_cache.Clear()
        room_bounds = bounds_cache.ComputeWorldBound(room_prim).ComputeAlignedRange()
        room_min = np.asarray(room_bounds.GetMin(), dtype=np.float64)
        room_max = np.asarray(room_bounds.GetMax(), dtype=np.float64)
        finite_bounds = bool(
            np.all(np.isfinite(room_min))
            and np.all(np.isfinite(room_max))
            and np.all(room_max > room_min)
        )
        if args.background_align_min_z and finite_bounds:
            room_offset[2] -= room_min[2]
            translate.Set(Gf.Vec3d(*room_offset))
        elif args.background_align_min_z:
            print(
                "[Task3Isaac] background has no finite bounds; "
                "skipping automatic min-z alignment",
                flush=True,
            )
        bounds_cache.Clear()
        for label, prim in (("task", task_prim), ("room", room_prim)):
            bounds = bounds_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            print(
                f"[Task3Isaac] {label}_bounds min={tuple(bounds.GetMin())} "
                f"max={tuple(bounds.GetMax())}",
                flush=True,
            )

        if floor_textures is not None:
            add_visual_floor(
                stage,
                args,
                floor_textures,
                Gf=Gf,
                Sdf=Sdf,
                UsdGeom=UsdGeom,
                UsdShade=UsdShade,
            )

        dome = None
        distant = None
        distant_rotate_op = None
        if args.light_rig == "default":
            from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode

            rig_applied, active_rig, previous_rig = _set_lighting_mode(
                "Default",
                usd_context=context,
            )
            if not rig_applied:
                raise RuntimeError(
                    "failed to apply Isaac Sim Light Rigs > Default: "
                    f"active={active_rig!r} previous={previous_rig!r}"
                )
        else:
            dome = UsdLux.DomeLight.Define(stage, "/World/Task3DomeLight")
            distant = UsdLux.DistantLight.Define(stage, "/World/Task3DistantLight")
            distant_xform = UsdGeom.Xformable(distant)
            distant_rotate_ops = [
                op
                for op in distant_xform.GetOrderedXformOps()
                if op.GetOpType() == op.TypeRotateXYZ
            ]
            distant_rotate_op = (
                distant_rotate_ops[0]
                if distant_rotate_ops
                else distant_xform.AddRotateXYZOp()
            )

        def apply_episode_lighting(
            seed: int, *, exact_training_lighting: dict | None = None
        ) -> dict:
            if args.light_rig == "default":
                lighting = {
                    "rig": "Default",
                    "randomized": False,
                    "seed": int(seed),
                    "dome_intensity": 1.0,
                    "dome_exposure": 9.0,
                    "dome_color_temperature_k": 6150.0,
                    "distant_intensity": 1.0,
                    "distant_exposure": 10.0,
                    "distant_color_temperature_k": 7250.0,
                }
                print(
                    f"[Task3Isaac] lighting={json.dumps(lighting, sort_keys=True)}",
                    flush=True,
                )
                return lighting
            if exact_training_lighting is not None:
                source = dict(exact_training_lighting)
                if source.get("rig", "scripted") != "scripted":
                    raise ValueError(
                        "training lighting manifest entry must use rig=scripted"
                    )
                rotation = [float(value) for value in source["distant_rotate_xyz_deg"]]
                if len(rotation) != 3:
                    raise ValueError(
                        "training distant_rotate_xyz_deg must contain three values"
                    )
                temperature = source.get("color_temperature_k")
                lighting = {
                    "rig": "scripted",
                    "randomized": False,
                    "source": "training_manifest",
                    "seed": int(source["seed"]),
                    "dome_intensity": float(source["dome_intensity"]),
                    "distant_intensity": float(source["distant_intensity"]),
                    "color_temperature_k": (
                        None if temperature is None else float(temperature)
                    ),
                    "distant_rotate_xyz_deg": rotation,
                }
            else:
                lighting = {
                    "rig": "scripted",
                    "randomized": bool(args.randomize_lighting),
                    "source": "runtime_seed" if args.randomize_lighting else "fixed",
                    "seed": int(seed),
                    "dome_intensity": 450.0,
                    "distant_intensity": 1400.0,
                    "color_temperature_k": None,
                    "distant_rotate_xyz_deg": [315.0, 0.0, 35.0],
                }
            if exact_training_lighting is None and args.randomize_lighting:
                light_rng = np.random.default_rng(seed)
                lighting.update(
                    {
                        "dome_intensity": float(
                            light_rng.uniform(*args.dome_intensity_range)
                        ),
                        "distant_intensity": float(
                            light_rng.uniform(*args.distant_intensity_range)
                        ),
                        "color_temperature_k": float(
                            light_rng.uniform(*args.light_temperature_range)
                        ),
                        "distant_rotate_xyz_deg": [
                            float(light_rng.uniform(*args.distant_rotate_x_range)),
                            0.0,
                            float(light_rng.uniform(*args.distant_rotate_z_range)),
                        ],
                    }
                )
            dome.CreateIntensityAttr().Set(lighting["dome_intensity"])
            distant.CreateIntensityAttr().Set(lighting["distant_intensity"])
            distant_rotate_op.Set(Gf.Vec3f(*lighting["distant_rotate_xyz_deg"]))
            for light in (dome, distant):
                light.GetPrim().CreateAttribute(
                    "inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool
                ).Set(lighting["color_temperature_k"] is not None)
                temperature_attr = light.GetPrim().CreateAttribute(
                    "inputs:colorTemperature", Sdf.ValueTypeNames.Float
                )
                if lighting["color_temperature_k"] is not None:
                    temperature_attr.Set(lighting["color_temperature_k"])
            print(
                f"[Task3Isaac] lighting={json.dumps(lighting, sort_keys=True)}",
                flush=True,
            )
            return lighting

        initial_lighting_seed = (
            replay_jobs[0]["lighting_seed"] if replay_jobs else args.lighting_seed
        )
        lighting = apply_episode_lighting(initial_lighting_seed)

        # NuRec volume render state is created on the first Hydra sync.  Bind
        # the intended camera and RenderProduct before that sync; creating it
        # after world.reset() can leave the volume using its initial viewport
        # camera even though ordinary USD meshes follow the new camera.
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("Isaac Sim has no active viewport")
        viewport.set_texture_resolution((args.width, args.height))
        if args.camera_mode == "head":
            camera_path = set_head_camera(
                stage,
                viewport,
                args,
                pitch_down_deg=args.head_camera_pitch_deg,
            )
        else:
            camera_path = set_camera(
                stage,
                viewport,
                args.camera_eye,
                args.camera_target,
                args.camera_focal_length,
                fovy=args.camera_fovy,
                aspect=args.width / args.height,
            )
            print(
                f"[Task3Isaac] camera_mode=world eye={tuple(args.camera_eye)} "
                f"target={tuple(args.camera_target)}",
                flush=True,
            )
        if args.ego_inset:
            ego_window_width = args.ego_window_width or args.ego_width
            ego_window_height = args.ego_window_height or args.ego_height
            ego_camera_path = set_head_camera(
                stage,
                viewport=None,
                args=args,
                path="/World/Task3EgoInsetCamera",
                fovy=args.ego_camera_fovy,
                aspect=args.ego_width / args.ego_height,
                pitch_down_deg=args.ego_camera_pitch_deg,
                label="ego_inset",
            )
            ego_x = max(args.width - ego_window_width - args.ego_window_margin, 0)
            ego_y = args.ego_window_margin
            ego_viewport_window = create_viewport_window(
                name=f"{task_label} Ego",
                width=ego_window_width,
                height=ego_window_height,
                position_x=ego_x,
                position_y=ego_y,
                camera_path=Sdf.Path(ego_camera_path),
            )
            if ego_viewport_window is None:
                raise RuntimeError(
                    f"failed to create the {task_label} Ego viewport window"
                )
            ego_viewport_window.viewport_api.set_texture_resolution(
                (args.ego_width, args.ego_height)
            )
            print(
                f"[Task3Isaac] ego_inset_texture={args.ego_width}x{args.ego_height} "
                f"window={ego_window_width}x{ego_window_height} "
                f"position=({ego_x},{ego_y}) camera={ego_camera_path}",
                flush=True,
            )
        if args.background_mode == "generic-3dgs":
            authored_render_products = [
                prim
                for prim in stage.Traverse()
                if prim.GetTypeName() == "RenderProduct"
                and str(prim.GetPath()).startswith("/Render/")
            ]
            if authored_render_products:
                # NuRec USDZ exports carry a renderer-specific RenderProduct.
                # Reuse it and repoint its camera; a newly created generic RP
                # can render the volume from its baked export camera only.
                render_product_prim = authored_render_products[0]
                render_product_prim.GetRelationship("camera").SetTargets([camera_path])
                nurec_render_product_path = str(render_product_prim.GetPath())
            else:
                import omni.replicator.core as rep

                nurec_render_product = rep.create.render_product(
                    camera_path, (args.width, args.height), force_new=True
                )
                nurec_render_product_path = nurec_render_product.path
                render_product_prim = stage.GetPrimAtPath(nurec_render_product_path)
            camera_targets = render_product_prim.GetRelationship("camera").GetTargets()
            viewport.render_product_path = nurec_render_product_path
            camera_world = UsdGeom.XformCache().GetLocalToWorldTransform(
                stage.GetPrimAtPath(camera_path)
            )
            print(
                f"[Task3Isaac] NuRec viewport render_product="
                f"{nurec_render_product_path} camera_targets={camera_targets} "
                f"camera_xyz={tuple(camera_world.ExtractTranslation())}",
                flush=True,
            )

        if replay_jobs or args.live_ego_frame_path is not None:
            import omni.replicator.core as rep

            rgb_camera_path = (
                ego_camera_path
                if args.live_ego_frame_path is not None and ego_camera_path is not None
                else camera_path
            )
            rgb_resolution = (
                (args.live_ego_width, args.live_ego_height)
                if args.live_ego_frame_path is not None
                else (args.width, args.height)
            )
            if nurec_render_product_path is None:
                replay_render_product = rep.create.render_product(
                    rgb_camera_path, rgb_resolution, force_new=True
                )
                replay_render_product_path = replay_render_product.path
            else:
                replay_render_product_path = nurec_render_product_path
            replay_rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            replay_rgb_annotator.attach([replay_render_product_path])
            print(
                f"[Task3IsaacRGB] render_product={replay_render_product_path} "
                f"camera={rgb_camera_path} resolution={rgb_resolution[0]}x{rgb_resolution[1]} "
                f"render_passes={SYNCED_RGB_RENDER_PASSES}",
                flush=True,
            )

        for _ in range(8):
            app.update()
        world.reset()
        robot = SingleArticulation("/World/Task3/Geometry/pelvis", name="task3_robot")
        trash = (
            SingleArticulation(task_object_profile["prim"], name="task3_trash_can")
            if task_object_profile is not None
            else None
        )
        robot.initialize()
        if trash is not None:
            trash.initialize()
        free_object_drivers = []
        for object_index, free_object in enumerate(free_task_objects):
            root_prim = stage.GetPrimAtPath(free_object["prim"])
            if not root_prim.IsValid():
                raise RuntimeError(
                    f"Imported free-object prim is missing: {free_object['prim']}"
                )
            rigid_paths = [
                str(prim.GetPath())
                for prim in Usd.PrimRange(root_prim)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            if free_object["prim"] not in rigid_paths:
                raise RuntimeError(
                    "Imported free-object root is not a rigid body: "
                    f"{free_object['prim']} descendants={rigid_paths}"
                )
            rigid_paths = [free_object["prim"]] + sorted(
                path for path in rigid_paths if path != free_object["prim"]
            )
            driven_prims = []
            for rigid_index, rigid_path in enumerate(rigid_paths):
                object_view = RigidPrim(
                    prim_paths_expr=rigid_path,
                    name=f"task_free_object_{object_index}_{rigid_index}",
                    reset_xform_properties=False,
                )
                object_view.initialize()
                driven_prims.append({"path": rigid_path, "view": object_view})
            free_object_drivers.append(
                {"mapping": free_object, "driven_prims": driven_prims}
            )
            print(
                f"[Task3Isaac] free_object_rigid_prims joint={free_object['joint']} "
                f"prims={rigid_paths}",
                flush=True,
            )
        head_tracker = None
        if args.camera_mode == "head" or args.ego_inset:
            head_tracker = RigidPrim(
                prim_paths_expr=args.head_camera_parent,
                name="task3_head_camera_parent",
                reset_xform_properties=False,
            )
            head_tracker.initialize()

        def update_tracked_head_cameras(parent_position, parent_orientation):
            main_eye = None
            inset_eye = None
            if args.camera_mode == "head":
                main_eye = update_head_camera(
                    stage,
                    camera_path,
                    parent_position,
                    parent_orientation,
                    args,
                    pitch_down_deg=args.head_camera_pitch_deg,
                )
            if ego_camera_path is not None:
                inset_eye = update_head_camera(
                    stage,
                    ego_camera_path,
                    parent_position,
                    parent_orientation,
                    args,
                    fovy=args.ego_camera_fovy,
                    aspect=args.ego_width / args.ego_height,
                    pitch_down_deg=args.ego_camera_pitch_deg,
                )
            return main_eye if main_eye is not None else inset_eye
        # Keep the timeline playing so SimulationContext.render() flushes
        # articulation tensor writes into the kinematic/render state.  render()
        # temporarily disables physics simulation during app.update(), so Isaac
        # remains a read-only visualizer driven by MuJoCo rather than a second
        # dynamics simulator.
        world.play()

        def normalized_quaternion(quaternion):
            quaternion = np.asarray(quaternion, dtype=np.float64)
            norm = float(np.linalg.norm(quaternion))
            if norm < 1e-8:
                raise ValueError("free-object quaternion must be non-zero")
            return quaternion / norm

        def multiply_quaternions(left, right):
            lw, lx, ly, lz = normalized_quaternion(left)
            rw, rx, ry, rz = normalized_quaternion(right)
            return normalized_quaternion(
                np.asarray(
                    [
                        lw * rw - lx * rx - ly * ry - lz * rz,
                        lw * rx + lx * rw + ly * rz - lz * ry,
                        lw * ry - lx * rz + ly * rw + lz * rx,
                        lw * rz + lx * ry - ly * rx + lz * rw,
                    ],
                    dtype=np.float64,
                )
            )

        robot_position, robot_orientation = robot.get_world_pose()
        robot_position = np.asarray(as_numpy(robot_position), dtype=np.float64)
        robot_orientation = np.asarray(as_numpy(robot_orientation), dtype=np.float64)
        trash_offset = np.asarray(args.trash_translate, dtype=np.float64)

        def transform_task_position(position):
            return task_yaw_rotation @ np.asarray(position, dtype=np.float64) + task_offset

        def transform_task_orientation(orientation):
            return multiply_quaternions(task_yaw_orientation, orientation)

        if not args.transform_task_root:
            robot_position = transform_task_position(robot_position)
            robot_orientation = transform_task_orientation(robot_orientation)
        robot_position[2] += args.robot_z_offset
        robot.set_world_pose(robot_position, robot_orientation)
        if trash is not None:
            trash_position, trash_orientation = trash.get_world_pose()
            trash_position = np.asarray(as_numpy(trash_position), dtype=np.float64)
            trash_orientation = np.asarray(
                as_numpy(trash_orientation), dtype=np.float64
            )
            if not args.transform_task_root:
                trash_position = transform_task_position(trash_position)
                trash_orientation = transform_task_orientation(trash_orientation)
            trash_position += trash_offset
            trash_position[2] += args.trash_z_offset
            trash.set_world_pose(trash_position, trash_orientation)

        for driver in free_object_drivers:
            initial_poses = []
            for driven_prim in driver["driven_prims"]:
                positions, orientations = driven_prim["view"].get_world_poses(usd=False)
                initial_poses.append(
                    (
                        np.asarray(as_numpy(positions[0]), dtype=np.float64),
                        normalized_quaternion(as_numpy(orientations[0])),
                    )
                )
            root_position, root_orientation = initial_poses[0]
            root_rotation = quaternion_wxyz_to_rotation(root_orientation)
            root_inverse = root_orientation.copy()
            root_inverse[1:] *= -1.0
            for driven_prim, (position, orientation) in zip(
                driver["driven_prims"], initial_poses
            ):
                driven_prim["relative_position"] = (
                    root_rotation.T @ (position - root_position)
                )
                driven_prim["relative_orientation"] = multiply_quaternions(
                    root_inverse, orientation
                )
                translated_position = (
                    position
                    if args.transform_task_root
                    else transform_task_position(position)
                )
                translated_position[2] += args.free_object_z_offset
                driven_prim["view"].set_world_poses(
                    positions=torch.as_tensor(
                        translated_position[np.newaxis, :],
                        dtype=torch.float32,
                        device="cuda:0",
                    ),
                    orientations=torch.as_tensor(
                        transform_task_orientation(orientation)[np.newaxis, :],
                        dtype=torch.float32,
                        device="cuda:0",
                    ),
                )
        print(
            f"[Task3Isaac] task_translate={tuple(task_offset)} "
            f"task_yaw_deg={args.task_yaw_deg:.3f} "
            f"trash_translate={tuple(trash_offset)}",
            flush=True,
        )

        robot_indices = [robot.get_dof_index(name) for name in ROBOT_JOINTS]
        trash_indices = (
            [trash.get_dof_index(name) for name in task_object_profile["joints"]]
            if trash is not None
            else []
        )
        if any(index < 0 for index in robot_indices + trash_indices):
            raise RuntimeError("Imported Task3 USD is missing one or more MuJoCo joints")
        robot_qpos_indices = [qpos_layout[name]["qpos_address"] for name in ROBOT_JOINTS]
        trash_qpos_indices = (
            [
                qpos_layout[name]["qpos_address"]
                for name in task_object_profile["joints"]
            ]
            if task_object_profile is not None
            else []
        )
        if any(qpos_layout[name]["width"] != 1 for name in ROBOT_JOINTS):
            raise ValueError("robot articulation mapping requires scalar MJCF joints")
        if task_object_profile is not None and any(
            qpos_layout[name]["width"] != 1
            for name in task_object_profile["joints"]
        ):
            raise ValueError("trash articulation mapping requires scalar MJCF joints")
        robot_indices_cuda = torch.as_tensor(robot_indices, dtype=torch.int64, device="cuda:0")
        trash_indices_cuda = torch.as_tensor(trash_indices, dtype=torch.int64, device="cuda:0")

        if args.dds:
            udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp.setblocking(False)
            udp.bind((args.udp_host, args.udp_port))
            print(
                f"[Task3Isaac] waiting for MuJoCo state on udp://{args.udp_host}:{args.udp_port}",
                flush=True,
            )

        latest = None
        frames = 0
        live_ego_sequence = 0
        last_rendered_request_key = None
        live_ego_deadline = time.monotonic()
        live_ego_frame_path = (
            args.live_ego_frame_path.expanduser().resolve()
            if args.live_ego_frame_path is not None else None
        )
        if live_ego_frame_path is not None:
            if replay_jobs:
                raise ValueError("--live-ego-frame-path cannot be combined with offline replay")
            if args.live_ego_rate_hz <= 0 or args.live_ego_width <= 0 or args.live_ego_height <= 0:
                raise ValueError("live ego rate and resolution must be positive")
            if not 1 <= args.live_ego_jpeg_quality <= 100:
                raise ValueError("--live-ego-jpeg-quality must be in [1, 100]")
            live_ego_frame_path.unlink(missing_ok=True)
            print(
                f"[Task3IsaacRGB] live ego -> {live_ego_frame_path} "
                f"rate={args.live_ego_rate_hz:.1f}Hz jpeg_quality={args.live_ego_jpeg_quality} "
                f"throttle_render={int(args.live_ego_throttle_render)} "
                f"strict_request_sync={int(args.live_ego_strict_request_sync)}",
                flush=True,
            )
        elif args.live_ego_throttle_render:
            raise ValueError(
                "--live-ego-throttle-render requires --live-ego-frame-path"
            )
        if args.live_ego_strict_request_sync and (live_ego_frame_path is None or udp is None):
            raise ValueError(
                "--live-ego-strict-request-sync requires DDS UDP and --live-ego-frame-path"
            )
        first_sync = False
        live_lighting_recording_key = None
        last_rendered_sim_time = None
        sync_frames = 0
        last_camera_eye = None
        def apply_qpos(qpos):
            nonlocal sync_frames
            qpos = np.asarray(qpos, dtype=np.float64)
            if qpos.size != expected_nq:
                raise ValueError(
                    f"MJCF expects nq={expected_nq}, received {qpos.size} qpos values"
                )
            target_robot_position = transform_task_position(qpos[:3])
            target_robot_position[2] += args.robot_z_offset
            robot.set_world_pose(
                position=target_robot_position,
                orientation=transform_task_orientation(qpos[3:7]),
            )
            robot.set_joint_positions(
                torch.as_tensor(
                    qpos[robot_qpos_indices], dtype=torch.float32, device="cuda:0"
                ),
                joint_indices=robot_indices_cuda,
            )
            if trash is not None:
                trash.set_joint_positions(
                    torch.as_tensor(
                        qpos[trash_qpos_indices], dtype=torch.float32, device="cuda:0"
                    ),
                    joint_indices=trash_indices_cuda,
                )
            for driver in free_object_drivers:
                free_object = driver["mapping"]
                address = free_object["qpos_address"]
                object_position = transform_task_position(qpos[address : address + 3])
                object_position[2] += args.free_object_z_offset
                object_orientation = transform_task_orientation(
                    qpos[address + 3 : address + 7]
                )
                object_rotation = quaternion_wxyz_to_rotation(object_orientation)
                for driven_prim in driver["driven_prims"]:
                    driven_position = (
                        object_position
                        + object_rotation @ driven_prim["relative_position"]
                    )
                    driven_orientation = multiply_quaternions(
                        object_orientation, driven_prim["relative_orientation"]
                    )
                    driven_prim["view"].set_world_poses(
                        positions=torch.as_tensor(
                            driven_position[np.newaxis, :],
                            dtype=torch.float32,
                            device="cuda:0",
                        ),
                        orientations=torch.as_tensor(
                            driven_orientation[np.newaxis, :],
                            dtype=torch.float32,
                            device="cuda:0",
                        ),
                    )
                    driven_prim["target_position"] = driven_position
                    driven_prim["target_orientation"] = driven_orientation
            sync_frames += 1
            return target_robot_position

        def verify_free_object_pose_readback():
            for driver in free_object_drivers:
                for driven_prim in driver["driven_prims"]:
                    positions, orientations = driven_prim["view"].get_world_poses(
                        usd=False
                    )
                    position = np.asarray(as_numpy(positions[0]), dtype=np.float64)
                    orientation = normalized_quaternion(as_numpy(orientations[0]))
                    position_error = float(
                        np.linalg.norm(position - driven_prim["target_position"])
                    )
                    target_orientation = driven_prim["target_orientation"]
                    orientation_error = float(
                        min(
                            np.linalg.norm(orientation - target_orientation),
                            np.linalg.norm(orientation + target_orientation),
                        )
                    )
                    if position_error > 1e-3 or orientation_error > 1e-3:
                        raise RuntimeError(
                            "Free-object render pose did not reach its target: "
                            f"prim={driven_prim['path']} "
                            f"position_error={position_error:.6g} "
                            f"orientation_error={orientation_error:.6g}"
                        )

        if replay_jobs:
            from PIL import Image

            batch_started_at = time.monotonic()
            progress_is_tty = sys.stdout.isatty()
            for job_index, replay_job in enumerate(replay_jobs):
                replay_csv = replay_job["csv"]
                replay_output_dir = replay_job["output_dir"]
                prepare_replay_output(
                    replay_output_dir,
                    overwrite=args.replay_overwrite,
                )
                replay_rows = load_replay_frames(
                    replay_csv,
                    downsample=args.replay_downsample,
                    policy_valid_only=args.replay_policy_valid_only,
                )
                transition_count = len(replay_rows)
                if args.max_frames > 0:
                    transition_count = min(transition_count, args.max_frames)
                print(
                    f"[Task3IsaacBatch] episode={job_index + 1}/{len(replay_jobs)} "
                    f"recording={replay_csv.parent.name} frames={transition_count}",
                    flush=True,
                )
                lighting = apply_episode_lighting(replay_job["lighting_seed"])
                manifest_frames = []
                partial_manifest = replay_output_dir / "manifest.partial.json"
                progress_started_at = time.monotonic()

                # Prime Fabric, the moving head camera, lighting, and RGB
                # readback for this episode without rebuilding the Isaac stage.
                apply_qpos(replay_rows[0]["qpos"])
                for _ in range(2):
                    world.render()
                    if head_tracker is not None:
                        warm_positions, warm_orientations = head_tracker.get_world_poses(
                            usd=False
                        )
                        update_tracked_head_cameras(
                            as_numpy(warm_positions[0]),
                            as_numpy(warm_orientations[0]),
                        )
                    world.render()
                    world.render()
                    replay_rgb_annotator.get_data()

                for frame_index, replay_row in enumerate(
                    replay_rows[:transition_count]
                ):
                    apply_qpos(replay_row["qpos"])
                    world.render()
                    if head_tracker is not None:
                        head_positions, head_orientations = head_tracker.get_world_poses(
                            usd=False
                        )
                        head_position = as_numpy(head_positions[0])
                        head_orientation = as_numpy(head_orientations[0])
                        last_camera_eye = update_tracked_head_cameras(
                            head_position, head_orientation
                        )
                    for _ in range(SYNCED_RGB_RENDER_PASSES - 1):
                        world.render()
                    if frame_index == 0 or frame_index == transition_count - 1:
                        verify_free_object_pose_readback()
                        if trash is not None:
                            task_joint_target = np.asarray(replay_row["qpos"])[
                                trash_qpos_indices
                            ]
                            task_joint_tensor = as_numpy(trash.get_joint_positions())[
                                trash_indices
                            ]
                            print(
                                f"[Task3IsaacReplay] task_object_joint_target="
                                f"{tuple(np.round(task_joint_target, 4))} "
                                f"tensor={tuple(np.round(task_joint_tensor, 4))}",
                                flush=True,
                            )
                    rgb = replay_rgb_annotator.get_data()
                    if isinstance(rgb, dict):
                        rgb = rgb.get("data", rgb)
                    rgb = np.asarray(rgb)
                    if (
                        rgb.ndim != 3
                        or rgb.shape[0] != args.height
                        or rgb.shape[1] != args.width
                    ):
                        raise RuntimeError(f"unexpected Isaac RGB shape: {rgb.shape}")
                    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
                    image_name = f"frame_{frame_index:06d}.png"
                    Image.fromarray(rgb, mode="RGB").save(
                        replay_output_dir / image_name,
                        compress_level=args.replay_png_compress_level,
                    )
                    frame_meta = {
                        key: replay_row[key]
                        for key in (
                            "source_row_index",
                            "sample_index",
                            "control_time_s",
                            "mujoco_time_s",
                        )
                    }
                    frame_meta.update(
                        {"frame_index": frame_index, "image": image_name}
                    )
                    if head_tracker is not None:
                        frame_meta["head_position_w"] = [
                            float(v) for v in head_position
                        ]
                        frame_meta["head_orientation_wxyz"] = [
                            float(v) for v in head_orientation
                        ]
                        frame_meta["camera_eye_w"] = [
                            float(v) for v in last_camera_eye
                        ]
                    manifest_frames.append(frame_meta)
                    completed_frames = frame_index + 1
                    progress_line = render_progress_line(
                        completed_frames,
                        transition_count,
                        progress_started_at,
                    )
                    if progress_is_tty:
                        print(f"\r{progress_line}\033[K", end="", flush=True)
                    if (
                        completed_frames % 25 == 0
                        or completed_frames == transition_count
                    ):
                        partial_manifest.write_text(
                            json.dumps(
                                {"complete": False, "frames": manifest_frames},
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        if not progress_is_tty:
                            print(progress_line, flush=True)

                if progress_is_tty:
                    print(flush=True)

                manifest = {
                    "schema_version": "task3_isaac_replay_rgb.v1",
                    "complete": True,
                    "source_csv": str(replay_csv),
                    "recording_name": replay_csv.parent.name,
                    "downsample": args.replay_downsample,
                    "png_compress_level": args.replay_png_compress_level,
                    "policy_valid_only": args.replay_policy_valid_only,
                    "fps": 50,
                    "width": args.width,
                    "height": args.height,
                    "camera_mode": args.camera_mode,
                    "camera_path": camera_path,
                    "camera_world_eye": list(args.camera_eye),
                    "camera_world_target": list(args.camera_target),
                    "camera_local_eye": list(args.head_camera_eye),
                    "camera_local_forward": list(args.head_camera_forward),
                    "camera_local_up": list(args.head_camera_up),
                    "camera_pitch_down_deg": args.head_camera_pitch_deg,
                    "camera_fovy_deg": args.camera_fovy,
                    "camera_focal_length": args.camera_focal_length,
                    "camera_near_clip": args.head_camera_near_clip,
                    "remove_task6_top_shelf_visual": (
                        args.remove_task6_top_shelf_visual
                    ),
                    "background_mode": args.background_mode,
                    "background_asset": str(hssd_path),
                    "background_translate": list(args.background_translate),
                    "hidden_background_prims": list(args.hide_background_prim),
                    "task_translate": list(args.task_translate),
                    "transform_task_root": args.transform_task_root,
                    "task_yaw_deg": args.task_yaw_deg,
                    "trash_translate": list(args.trash_translate),
                    "robot_z_offset": args.robot_z_offset,
                    "trash_z_offset": args.trash_z_offset,
                    "lighting": lighting,
                    "source_sampled_rows": len(replay_rows),
                    "full_episode": transition_count == len(replay_rows),
                    "frames": manifest_frames,
                }
                (replay_output_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                partial_manifest.unlink(missing_ok=True)
                batch_elapsed = time.monotonic() - batch_started_at
                completed_jobs = job_index + 1
                batch_eta = (
                    batch_elapsed
                    * (len(replay_jobs) - completed_jobs)
                    / completed_jobs
                )
                print(
                    f"[Task3IsaacBatch] complete={completed_jobs}/{len(replay_jobs)} "
                    f"frames={len(manifest_frames)} "
                    f"elapsed={format_progress_duration(batch_elapsed)} "
                    f"ETA={format_progress_duration(batch_eta)} "
                    f"output={replay_output_dir}",
                    flush=True,
                )

        while not replay_jobs and app.is_running():
            live_state_reset = False
            if udp is not None:
                while True:
                    try:
                        payload, _ = udp.recvfrom(65535)
                        latest = json.loads(payload.decode("utf-8"))
                    except BlockingIOError:
                        break
            now = time.monotonic()
            request_key = None
            if args.live_ego_strict_request_sync and latest is None:
                time.sleep(0.002)
                continue
            if args.live_ego_strict_request_sync:
                session_id = str(latest.get("session_id", ""))
                request_id = int(latest.get("request_id", -1))
                state_sequence = int(latest.get("state_sequence", -1))
                if not session_id or request_id < 0 or state_sequence != request_id:
                    raise ValueError(
                        "strict live Ego state requires session_id and equal nonnegative "
                        "request_id/state_sequence"
                    )
                request_key = (session_id, request_id)
                if request_key == last_rendered_request_key:
                    time.sleep(0.002)
                    continue
            render_due = (
                args.live_ego_strict_request_sync
                or not args.live_ego_throttle_render
                or live_ego_frame_path is None
                or now >= live_ego_deadline
            )
            if not render_due:
                # Continue draining UDP at a low cost so the next render uses
                # only the newest authoritative MuJoCo state.  In particular,
                # do not let a 2K GUI viewport render as fast as possible while
                # the policy consumes only ``live_ego_rate_hz`` images.
                time.sleep(min(max(live_ego_deadline - now, 0.0), 0.005))
                continue
            if latest is not None:
                latest_sim_time = float(latest["sim_time"])
                live_state_reset = (
                    last_rendered_sim_time is None
                    or latest_sim_time <= last_rendered_sim_time
                )
                if live_lighting_by_index:
                    recording_index = latest.get("recording_index")
                    recording_name = str(latest.get("recording_name", ""))
                    if recording_index is None:
                        raise ValueError(
                            "live training lighting requires recording_index metadata"
                        )
                    recording_index = int(recording_index)
                    entry = live_lighting_by_index.get(recording_index)
                    if entry is None:
                        raise KeyError(
                            f"recording_index={recording_index} is absent from the "
                            "training lighting manifest"
                        )
                    if recording_name and entry["recording_name"] != recording_name:
                        raise ValueError(
                            "training lighting recording mismatch: "
                            f"index={recording_index} manifest={entry['recording_name']} "
                            f"request={recording_name}"
                        )
                    recording_key = (recording_index, entry["recording_name"])
                    if recording_key != live_lighting_recording_key:
                        lighting = apply_episode_lighting(
                            int(entry["lighting"]["seed"]),
                            exact_training_lighting=entry["lighting"],
                        )
                        live_lighting_recording_key = recording_key
                        print(
                            f"[Task3Isaac] restored_training_lighting "
                            f"recording_index={recording_index} "
                            f"recording={entry['recording_name']} "
                            f"seed={entry['lighting']['seed']}",
                            flush=True,
                        )
                elif args.randomize_lighting:
                    recording_index = latest.get("recording_index")
                    if recording_index is not None:
                        recording_index = int(recording_index)
                        recording_key = (recording_index, "")
                        if recording_key != live_lighting_recording_key:
                            lighting = apply_episode_lighting(
                                args.lighting_seed + recording_index
                            )
                            live_lighting_recording_key = recording_key
                            print(
                                f"[Task3Isaac] recording_index={recording_index} "
                                f"matched_training_lighting_seed="
                                f"{args.lighting_seed + recording_index}",
                                flush=True,
                            )
                qpos = np.asarray(latest["qpos"], dtype=np.float64)
                robot_position = apply_qpos(qpos)
                if not first_sync:
                    print(
                        f"[Task3Isaac] first_sync sim_time={latest['sim_time']:.4f} "
                        f"nq={qpos.size}",
                        flush=True,
                    )
                    first_sync = True
                elif sync_frames % 120 == 0:
                    rendered_position, _ = robot.get_world_pose()
                    rendered_position = as_numpy(rendered_position)
                    object_status = ""
                    if free_object_drivers:
                        root_view = free_object_drivers[0]["driven_prims"][0]["view"]
                        object_positions, _ = root_view.get_world_poses(usd=False)
                        object_status = (
                            " free_object_xyz="
                            f"{tuple(np.round(as_numpy(object_positions)[0], 4))}"
                        )
                    print(
                        f"[Task3Isaac] sync sim_time={latest['sim_time']:.4f} "
                        f"target_xyz={tuple(np.round(robot_position, 4))} "
                        f"tensor_xyz={tuple(np.round(rendered_position, 4))}"
                        f"{object_status}",
                        flush=True,
                    )
            # Match the offline training-data renderer exactly: first submit the
            # new articulation/free-object state to Fabric, then read the new
            # head pose and update the Ego camera, and finally let the same
            # temporal renderer settle before RGB readback.  Reading the head
            # pose before this first render made the live camera one state late;
            # a single DLSS render also left visible temporal ghosting.
            render_passes = SYNCED_RGB_RENDER_PASSES * (2 if live_state_reset else 1)
            if live_state_reset:
                print(
                    f"[Task3IsaacRGB] reset_settle sim_time="
                    f"{float(latest['sim_time']):.4f} render_passes={render_passes}",
                    flush=True,
                )
            world.render()
            if head_tracker is not None:
                head_positions, head_orientations = head_tracker.get_world_poses(
                    usd=False
                )
                head_position = as_numpy(head_positions[0])
                head_orientation = as_numpy(head_orientations[0])
                last_camera_eye = update_tracked_head_cameras(
                    head_position, head_orientation
                )
            for _ in range(render_passes - 1):
                world.render()
            now = time.monotonic()
            if (
                live_ego_frame_path is not None
                and latest is not None
                and (args.live_ego_strict_request_sync or now >= live_ego_deadline)
            ):
                rgb = replay_rgb_annotator.get_data()
                if isinstance(rgb, dict):
                    rgb = rgb.get("data", rgb)
                rgb = np.asarray(rgb)
                if rgb.ndim == 3 and rgb.shape[:2] == (args.live_ego_height, args.live_ego_width):
                    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
                    publish_live_ego_frame(
                        live_ego_frame_path, rgb, sequence=live_ego_sequence,
                        sim_time=float(latest["sim_time"]), jpeg_quality=args.live_ego_jpeg_quality,
                        session_id=latest.get("session_id"),
                        request_id=latest.get("request_id"),
                        state_sequence=latest.get("state_sequence"),
                    )
                    live_ego_sequence += 1
                    if args.live_ego_strict_request_sync:
                        last_rendered_request_key = request_key
                    last_rendered_sim_time = float(latest["sim_time"])
                if not args.live_ego_strict_request_sync:
                    live_ego_deadline = now + 1.0 / args.live_ego_rate_hz
            frames += 1
            if args.max_frames > 0 and frames >= args.max_frames:
                break

        if args.capture:
            import asyncio

            output = args.capture.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            if last_camera_eye is not None:
                print(
                    f"[Task3Isaac] capture head_camera_xyz="
                    f"{tuple(np.round(last_camera_eye, 4))}",
                    flush=True,
                )
            if nurec_render_product_path is not None:
                render_product_prim = stage.GetPrimAtPath(nurec_render_product_path)
                camera_targets = render_product_prim.GetRelationship("camera").GetTargets()
                camera_world = UsdGeom.XformCache().GetLocalToWorldTransform(
                    stage.GetPrimAtPath(camera_path)
                )
                print(
                    f"[Task3Isaac] capture viewport_render_product="
                    f"{viewport.render_product_path} camera_targets={camera_targets} "
                    f"camera_xyz={tuple(camera_world.ExtractTranslation())}",
                    flush=True,
                )
            capture = capture_viewport_to_file(viewport, file_path=str(output))
            task = asyncio.ensure_future(capture.wait_for_result(completion_frames=30))
            while not task.done():
                app.update()
            task.result()
            for _ in range(30):
                if output.is_file() and output.stat().st_size > 0:
                    break
                app.update()
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("Isaac viewport capture did not produce an image")
            print(f"[Task3Isaac] preview={output}", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if udp is not None:
            udp.close()
        if "live_ego_frame_path" in locals() and live_ego_frame_path is not None:
            print(f"[Task3IsaacRGB] published_frames={live_ego_sequence}", flush=True)
            live_ego_frame_path.unlink(missing_ok=True)
        if replay_rgb_annotator is not None:
            replay_rgb_annotator.detach()
        if replay_render_product is not None:
            replay_render_product.destroy()
        if nurec_render_product is not None:
            nurec_render_product.destroy()
        if ego_viewport_window is not None:
            try:
                ego_viewport_window.viewport_widget.destroy()
                ego_viewport_window.destroy()
            except Exception:
                pass
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
