"""F1Tenth DRIFTING on a real polyline track.

Combines the two ideas: drive around a REAL track (polyline geometry from
polytrack.py — arc-length progress, cross-track, off-track, reset-along-track)
while DRIFTING (reward side-slip with a curriculum, like the WheeledLab drift
task). Builds on F1TenthPolyRaceRLEnvCfg (F1Tenth asset, plane scene, domain
randomization, poly reset) and swaps in a drift-flavored reward + curriculum,
with a wider off-track margin so the car has room to slide.

User ask: "try and get it drifting around the new tracks." Default TRACK = Oval.
"""

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    CurriculumTermCfg as CurrTerm,
)

from wheeledlab.envs.mdp import increase_reward_weight_over_time
from wheeledlab_tasks.drifting.mushr_drift_env_cfg import side_slip, SLIP_THRESHOLD
from .f1tenth_race_env_cfg import RACE_MAX_SPEED
from .f1tenth_polytrack_env_cfg import F1TenthPolyRaceRLEnvCfg, TRACK
from . import polytrack


######################
###### REWARDS #######
######################


@configclass
class PolyDriftRewardsCfg:
    """Drift-around-the-track rewards: progress + side-slip, gentle line keeping."""

    # Go around the real track (Δ arc length).
    progress = RewTerm(func=polytrack.arc_progress, weight=30.0, params={"track_name": TRACK})

    # Reward sliding (drift). Ramped up by the curriculum below.
    side_slip = RewTerm(
        func=side_slip,
        weight=10.0,
        params={"min_thresh": 0.25, "max_thresh": SLIP_THRESHOLD, "min_vel_x": 1.0},
    )

    # Keep some speed up.
    speed = RewTerm(func=polytrack.forward_speed, weight=1.0, params={"target": RACE_MAX_SPEED})

    # Gentle line keeping — small so it doesn't fight the drift angle.
    cross_track = RewTerm(func=polytrack.cross_track, weight=-3.0, params={"track_name": TRACK})

    # Penalty for leaving the (wider) track bounds.
    term_pens = RewTerm(
        func=mdp.rewards.is_terminated_term,
        params={"term_keys": ["out_of_bounds"]},
        weight=-500.0,
    )


########################
###### CURRICULUM ######
########################


@configclass
class PolyDriftCurriculumCfg:
    """Ramp up the drift (side-slip) reward over training, like WheeledLab drift."""

    more_slip = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "side_slip",
            "increase": 10.0,
            "episodes_per_increase": 20,
            "max_increases": 10,
        },
    )


##########################
###### TERMINATION #######
##########################


@configclass
class PolyDriftTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Wider margin than racing — drifting needs room to slide. [Stage E: 0.6->0.9
    # to give the higher-speed (4.5 m/s) drift more room before terminating.]
    out_of_bounds = DoneTerm(
        func=polytrack.off_track,
        params={"track_name": TRACK, "margin": 0.9},
    )


######################
###### RL ENV ########
######################


@configclass
class F1TenthPolyDriftRLEnvCfg(F1TenthPolyRaceRLEnvCfg):
    """F1Tenth drifting around a real polyline track."""

    rewards: PolyDriftRewardsCfg = PolyDriftRewardsCfg()
    terminations: PolyDriftTerminationsCfg = PolyDriftTerminationsCfg()
    curriculum: PolyDriftCurriculumCfg = PolyDriftCurriculumCfg()
    # events (poly reset + domain randomization) inherited.

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20
        # [Stage E] Drift needs more speed than clean racing: at 3 m/s side-slip
        # stayed ~0.02 (barely sliding). Push to 4.5 m/s so the rear can break
        # traction and actually drift the real track.
        self.actions.throttle_steer.scale = (4.5, 0.488)


@configclass
class F1TenthPolyDriftPlayEnvCfg(F1TenthPolyDriftRLEnvCfg):
    """Playback config — no rewards/terminations/curriculum."""

    rewards: PolyDriftRewardsCfg = None
    terminations: PolyDriftTerminationsCfg = None
    curriculum: PolyDriftCurriculumCfg = None

    def __post_init__(self):
        super().__post_init__()
