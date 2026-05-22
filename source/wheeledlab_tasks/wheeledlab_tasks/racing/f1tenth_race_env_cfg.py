"""F1Tenth racing task — clean fast laps on the WheeledLab oval.

Built on the WheeledLab F1Tenth *drift* task: it reuses the same F1Tenth
asset, scene, domain-randomization events, observations, actions, terminations
and ManagerBased RL framework. The only real difference is the REWARD.

Drift maximizes side-slip; racing wants fast, clean laps. So we:
  - reward lap progress (yaw rate ≈ orbital rate on the centered oval) and
    carrying speed through corners,
  - reward a high target speed,
  - penalize leaving the racing line (cross-track) and sliding (side-slip),
  - drop the drift curriculum (slip / turn-left-go-right ramps),
  - raise the top speed and lengthen the episode for multi-lap rollouts.

This is the first "our racing task" on the proven WheeledLab foundation,
replacing the hand-built FBD twin approach. A follow-up will swap the analytic
oval for our real track polylines (Oval/Austin/Spielberg/Spa) with arc-length
progress + cross-track from the goat_racer track data.
"""

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm

from wheeledlab_tasks.drifting.f1tenth_drift_env_cfg import F1TenthDriftRLEnvCfg
from wheeledlab_tasks.drifting.mushr_drift_env_cfg import (
    track_progress_rate,
    vel_dist,
    cross_track_dist,
    energy_through_turn,
    side_slip,
    DriftCurriculumCfg,
    DriftTerminationsCfg,
    STRAIGHT,
    LINE_RADIUS,
    SLIP_THRESHOLD,
)

# Racing top speed — faster than the drift target (3.0 m/s).
RACE_MAX_SPEED = 5.0

######################
###### REWARDS #######
######################


@configclass
class F1TenthRaceRewardsCfg:
    """Racing rewards: lap progress + speed + clean line. No drift reward."""

    # Core: rate of progress around the oval (car yaw rate ≈ orbital rate on a
    # track centered at the origin). Rewards lapping faster.
    progress = RewTerm(func=track_progress_rate, weight=60.0)

    # Soft speed target: vel_dist = (speed - target)^2 + offset; with a negative
    # weight this rewards being near RACE_MAX_SPEED (peaks at the target).
    vel = RewTerm(
        func=vel_dist,
        weight=-3.0,
        params={"speed_target": RACE_MAX_SPEED, "offset": -(RACE_MAX_SPEED**2)},
    )

    # Carry speed through the corners.
    turn_energy = RewTerm(
        func=energy_through_turn,
        weight=15.0,
        params={"straight": STRAIGHT},
    )

    # Penalty: stay on the racing line (distance from the centerline radius).
    cross_track = RewTerm(
        func=cross_track_dist,
        weight=-40.0,
        params={"straight": STRAIGHT, "track_radius": LINE_RADIUS, "p": 1, "offset": -1.0},
    )

    # Penalty: discourage sliding (clean racing, unlike drift).
    side_slip = RewTerm(
        func=side_slip,
        weight=-2.0,
        params={"min_thresh": 0.25, "max_thresh": SLIP_THRESHOLD, "min_vel_x": 1.0},
    )

    # Heavy penalty for leaving the track.
    term_pens = RewTerm(
        func=mdp.rewards.is_terminated_term,
        params={"term_keys": ["out_of_bounds"]},
        weight=-5000.0,
    )


######################
###### RL ENV ########
######################


@configclass
class F1TenthRaceRLEnvCfg(F1TenthDriftRLEnvCfg):
    """F1Tenth racing on the WheeledLab oval — fast, clean laps."""

    rewards: F1TenthRaceRewardsCfg = F1TenthRaceRewardsCfg()
    # Drop the drift-specific curriculum (side-slip / tlgr ramps).
    curriculum: DriftCurriculumCfg = None

    def __post_init__(self):
        super().__post_init__()
        # Longer episodes so the policy runs multiple laps.
        self.episode_length_s = 15
        # Scale actions to the racing top speed.
        self.actions.throttle_steer.scale = (RACE_MAX_SPEED, 0.488)


######################
###### PLAY ENV ######
######################


@configclass
class F1TenthRacePlayEnvCfg(F1TenthRaceRLEnvCfg):
    """Playback config — no rewards/terminations/curriculum (drive forever)."""

    rewards: F1TenthRaceRewardsCfg = None
    terminations: DriftTerminationsCfg = None
    curriculum: DriftCurriculumCfg = None

    def __post_init__(self):
        super().__post_init__()
