"""F1Tenth racing on a REAL polyline track (not the analytic oval).

Builds on F1TenthRaceRLEnvCfg (same F1Tenth asset, plane scene, domain
randomization, actions, PPO agent) but replaces the oval-based reward /
termination / reset with polyline-track versions from `polytrack.py` that
operate on goat_racer track data (tracks/data/<TRACK>/). Default TRACK = Oval
(32x12 m, F1Tenth-scale). Switch TRACK to any track in tracks/data (Austin,
Spielberg, Spa, ...).

All envs share one track at its world coordinates (env_spacing=0), so
root_pos_w is directly comparable to the track tensors.
"""

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)

from wheeledlab_tasks.drifting.f1tenth_drift_env_cfg import F1TenthDriftEventsRandomCfg
from wheeledlab_tasks.drifting.mushr_drift_env_cfg import DriftCurriculumCfg
from .f1tenth_race_env_cfg import F1TenthRaceRLEnvCfg, RACE_MAX_SPEED
from . import polytrack

# Which track to race. Any dir under goat_racer tracks/data/ works.
TRACK = "Oval"

# Top speed for polyline racing. Lower than the analytic-oval racing (5.0):
# real tracks (Oval ~6 m-radius corners, ~2 m wide) can't be held at 5 m/s
# (lateral accel exceeds tire grip -> car always leaves the track and the
# policy collapses, out_of_bounds ~1.0). 3 m/s is feasible. [autopilot tuned]
POLY_MAX_SPEED = 3.0


######################
###### REWARDS #######
######################


@configclass
class PolyTrackRewardsCfg:
    """Racing rewards on a polyline track."""

    # Forward progress along the centerline (Δ arc length) — the core reward.
    progress = RewTerm(func=polytrack.arc_progress, weight=50.0, params={"track_name": TRACK})

    # Reward speed up to the (feasible) racing target.
    speed = RewTerm(func=polytrack.forward_speed, weight=2.0, params={"target": POLY_MAX_SPEED})

    # Penalty: stay near the centerline / racing line (stiffened).
    cross_track = RewTerm(func=polytrack.cross_track, weight=-12.0, params={"track_name": TRACK})

    # Heavy penalty for leaving the track.
    term_pens = RewTerm(
        func=mdp.rewards.is_terminated_term,
        params={"term_keys": ["out_of_bounds"]},
        weight=-500.0,
    )


##########################
###### TERMINATION #######
##########################


@configclass
class PolyTrackTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(
        func=polytrack.off_track,
        params={"track_name": TRACK, "margin": 0.6},
    )


#####################
###### EVENTS #######
#####################


@configclass
class PolyTrackEventsCfg(F1TenthDriftEventsRandomCfg):
    """Reuse F1Tenth drift domain randomization, but reset along the polyline."""

    reset_root_state = EventTerm(
        func=polytrack.reset_root_state_on_polytrack,
        params={
            "track_name": TRACK,
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_noise": 0.2,
            "yaw_noise": 0.3,
        },
        mode="reset",
    )


######################
###### RL ENV ########
######################


@configclass
class F1TenthPolyRaceRLEnvCfg(F1TenthRaceRLEnvCfg):
    """F1Tenth racing on a real polyline track."""

    rewards: PolyTrackRewardsCfg = PolyTrackRewardsCfg()
    terminations: PolyTrackTerminationsCfg = PolyTrackTerminationsCfg()
    events: PolyTrackEventsCfg = PolyTrackEventsCfg()
    curriculum: DriftCurriculumCfg = None

    def __post_init__(self):
        super().__post_init__()
        # Longer episodes — a real lap is longer than the small oval.
        self.episode_length_s = 20
        self.actions.throttle_steer.scale = (POLY_MAX_SPEED, 0.488)


@configclass
class F1TenthPolyRacePlayEnvCfg(F1TenthPolyRaceRLEnvCfg):
    """Playback config — keep terminations off so it drives continuously."""

    rewards: PolyTrackRewardsCfg = None
    terminations: PolyTrackTerminationsCfg = None
    curriculum: DriftCurriculumCfg = None

    def __post_init__(self):
        super().__post_init__()
