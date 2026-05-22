"""Polyline-track MDP terms for WheeledLab racing on real tracks.

The drift/racing tasks use an analytic oval. This module lets the racing task
run on REAL track polylines (goat_racer `tracks/data/<name>/`: centerline.csv,
wall_inner.csv, wall_outer.csv, raceline_waypoints.csv — `x_m,y_m`).

The arc-length progress + cross-track + nearest-point math is ported from
isaaclab/.../leatherback_track_v4_env.py. Functions follow the Isaac Lab MDP
convention (take `env`, return per-env tensors); the reset is a ManagerTermBase
mirroring WheeledLab's reset_root_state_along_track.

All envs share ONE track at its world coordinates (env_spacing=0, like drift),
so `root_pos_w` is directly comparable to the track tensors.

NOTE (fork portability): tracks are referenced from the goat_racer repo at
`<repo>/tracks/data`. Override with env var WHEELEDLAB_TRACKS_DIR. To make the
fork standalone, bundle the chosen track CSVs into the package.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Track loading (cached per name+device)
# ---------------------------------------------------------------------------

_TRACK_CACHE: dict = {}


def _tracks_dir() -> Path:
    env_dir = os.environ.get("WHEELEDLAB_TRACKS_DIR")
    if env_dir:
        return Path(env_dir)
    # external/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/racing/polytrack.py
    # parents[6] -> goat_racer repo root
    return Path(__file__).resolve().parents[6] / "tracks" / "data"


def _load_xy(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{path}: expected (N,2), got {arr.shape}")
    return arr


def load_track(name: str, device) -> dict:
    key = (name, str(device))
    if key in _TRACK_CACHE:
        return _TRACK_CACHE[key]
    d = _tracks_dir() / name
    if not d.is_dir():
        raise FileNotFoundError(f"track dir not found: {d} (set WHEELEDLAB_TRACKS_DIR)")
    center = _load_xy(d / "centerline.csv")
    inner = _load_xy(d / "wall_inner.csv")
    outer = _load_xy(d / "wall_outer.csv")

    # unit tangents (closed loop), segment lengths to next point, cumulative arc
    fwd = np.roll(center, -1, axis=0) - center
    back = center - np.roll(center, 1, axis=0)
    t = 0.5 * (fwd + back)
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    tn = np.where(tn < 1e-12, 1.0, tn)
    tang = (t / tn).astype(np.float32)
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1).astype(np.float32)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)[:-1]]).astype(np.float32)
    total = float(seg.sum())

    # half width: min distance from each centerline point to either wall
    def _mind(pts: np.ndarray, pl: np.ndarray) -> np.ndarray:
        d2 = ((pts[:, None, :] - pl[None, :, :]) ** 2).sum(axis=-1)
        return np.sqrt(d2.min(axis=1))
    half_width = np.minimum(_mind(center, inner), _mind(center, outer)).astype(np.float32)

    track = {
        "center": torch.tensor(center, device=device),
        "tang": torch.tensor(tang, device=device),
        "seg": torch.tensor(seg, device=device),
        "cumlen": torch.tensor(cumlen, device=device),
        "half_width": torch.tensor(half_width, device=device),
        "total": total,
    }
    _TRACK_CACHE[key] = track
    return track


def _project(env, track: dict):
    """Project each env's world xy onto the centerline.

    Returns (idx, arc, lateral): nearest centerline index, arc length along the
    track, and signed lateral offset (left positive).
    """
    pos = mdp.root_pos_w(env)[:, :2]  # (E,2)
    center = track["center"]          # (N,2)
    d2 = (center.unsqueeze(0) - pos.unsqueeze(1)).pow(2).sum(dim=-1)  # (E,N)
    idx = d2.argmin(dim=-1)           # (E,)
    seg_start = center[idx]
    tang = track["tang"][idx]
    rel = pos - seg_start
    along = (rel[:, 0] * tang[:, 0] + rel[:, 1] * tang[:, 1]).clamp(min=0.0)
    along = torch.minimum(along, track["seg"][idx])
    arc = track["cumlen"][idx] + along
    nx = -tang[:, 1]
    ny = tang[:, 0]
    lateral = rel[:, 0] * nx + rel[:, 1] * ny
    return idx, arc, lateral


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------


def arc_progress(env, track_name: str):
    """Forward progress (Δ arc length along the centerline) per step."""
    track = load_track(track_name, env.device)
    _, arc, _ = _project(env, track)
    prev = getattr(env, "_poly_prev_arc", None)
    if prev is None or prev.shape[0] != env.num_envs:
        prev = arc.clone()
        env._poly_prev_arc = prev
    delta = arc - prev
    L = track["total"]
    # wrap (lap completion / projection jitter shouldn't read as a huge jump)
    delta = torch.where(delta < -0.5 * L, delta + L, delta)
    delta = torch.where(delta > 0.5 * L, delta - L, delta)
    env._poly_prev_arc = arc.clone()
    return delta


def cross_track(env, track_name: str):
    """Absolute lateral distance from the centerline (penalty; use weight<0)."""
    track = load_track(track_name, env.device)
    _, _, lateral = _project(env, track)
    return lateral.abs()


def forward_speed(env, target: float = 5.0):
    """Ground speed, capped at `target` (reward; use weight>0)."""
    v = mdp.base_lin_vel(env)
    speed = torch.norm(v[..., :2], dim=-1)
    return speed.clamp(max=target)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


def off_track(env, track_name: str, margin: float = 0.3):
    """True when the car is beyond the track half-width (+margin)."""
    track = load_track(track_name, env.device)
    idx, _, lateral = _project(env, track)
    hw = track["half_width"][idx]
    return lateral.abs() > (hw + margin)


# ---------------------------------------------------------------------------
# Reset along the polyline
# ---------------------------------------------------------------------------


class reset_root_state_on_polytrack(ManagerTermBase):
    """Reset the robot to a random point along the track centerline, facing along
    the local tangent, with small lateral/yaw noise. Also seeds the per-env
    `_poly_prev_arc` so the first post-reset progress delta is ~0.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.track_name = cfg.params.get("track_name", "Oval")
        self.track = load_track(self.track_name, self.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        track_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        pos_noise: float = 0.2,
        yaw_noise: float = 0.3,
        z_offset: float = 0.05,
    ):
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        track = self.track
        n = len(env_ids)
        N = track["center"].shape[0]
        j = torch.randint(N, (n,), device=env.device)

        center = track["center"][j]   # (n,2)
        tang = track["tang"][j]
        nx = -tang[:, 1]
        ny = tang[:, 0]
        lat = (2 * torch.rand(n, device=env.device) - 1.0) * pos_noise
        xy = center + torch.stack([nx * lat, ny * lat], dim=-1)

        z = torch.full((n, 1), z_offset, device=env.device)
        posns = torch.cat([xy, z], dim=-1)

        yaw = torch.atan2(tang[:, 1], tang[:, 0])
        yaw = yaw + (2 * torch.rand(n, device=env.device) - 1.0) * yaw_noise
        zeros = torch.zeros(n, device=env.device)
        oris = math_utils.quat_from_euler_xyz(roll=zeros, pitch=zeros, yaw=yaw)

        asset.write_root_pose_to_sim(torch.cat([posns, oris], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)

        # seed previous arc so the first progress delta after reset is ~0
        prev = getattr(env, "_poly_prev_arc", None)
        if prev is None or prev.shape[0] != env.num_envs:
            prev = torch.zeros(env.num_envs, device=env.device)
            env._poly_prev_arc = prev
        env._poly_prev_arc[env_ids] = track["cumlen"][j]
