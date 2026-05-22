# Porting WheeledLab to Isaac Lab 2.3.x / Isaac Sim 5.1 / rsl-rl 3.x

WheeledLab upstream targets **Isaac Sim 4.5.0 + Isaac Lab 2.0.2 + rsl-rl 2.3**.
This fork makes it run on the newer stack we have:

| Component | Upstream target | This environment |
|-----------|-----------------|------------------|
| Isaac Sim | 4.5.0           | **5.1.0**        |
| Isaac Lab | 2.0.2           | **2.3.2**        |
| rsl-rl-lib| ~2.3            | **3.1.2**        |
| gymnasium | 1.0.0           | **1.2.1**        |
| numpy     | <2              | 1.26.0 (ok)      |
| Python    | 3.10            | 3.10             |

**Status:** `Isaac-F1TenthDriftRL-v0` and `Isaac-MushrDriftRL-v0` train cleanly
(~41k steps/s headless, drift reward terms active, checkpoints + tensorboard
written). Elevation / Visual not yet exercised.

## 1. Installation change — install with `--no-deps`

The four packages pin `gymnasium==1.0.0`, but Isaac Lab 2.3.2 needs 1.2.1.
Installing deps would downgrade gymnasium and break Isaac Lab. Install editable
**without dependencies** (rsl-rl 3.1.2 already satisfies `rsl-rl-lib>=2.3.0`):

```bash
cd source
for p in wheeledlab wheeledlab_assets wheeledlab_tasks wheeledlab_rl; do
    python -m pip install -e "$p" --no-deps
done
```

## 2. Runtime dependency — PyAV

`wheeledlab_rl/utils/custom_video_recorder.py` hard-imports `av` (PyAV) at
startup. It is not declared anywhere. Install it (numpy stays <2):

```bash
python -m pip install av   # tested with av 17.0.1
```

## 3. Code patch — `isaaclab.utils.io` dropped pickle helpers

Isaac Lab removed `dump_pickle` / `load_pickle` from `isaaclab.utils.io` after
2.0.2 (only `dump_yaml`/`load_yaml`/`load_torchscript_model` remain). WheeledLab
needs pickle for `RunConfig` (contains slices that YAML can't round-trip).

- **Added** `source/wheeledlab_rl/wheeledlab_rl/utils/pickle_io.py` re-providing
  the two functions (behaviour identical to old isaaclab).
- **Repointed imports:**
  - `scripts/train_rl.py`: `from isaaclab.utils.io import dump_yaml, dump_pickle`
    → import `dump_yaml` from isaaclab, `dump_pickle` from the shim.
  - `scripts/play_policy.py`: `load_pickle` now from the shim.

## 4. Code patch — rsl-rl 2.x → 3.x runner incompatibility

`wheeledlab_rl/utils/modified_rsl_rl_runner.py` subclasses
`rsl_rl.runners.OnPolicyRunner` and overrides `learn()` with a hand-rolled
rollout loop written against the **rsl-rl 2.x** API
(`get_observations() -> (obs, extras)`, old `step`/`act`/normalizer
signatures). On rsl-rl 3.1.2 this fails immediately:
`ValueError: too many values to unpack (expected 2)`.

The runner's `__init__` already works with 3.x (the alg/policy build fine), so
`learn()` now **delegates to the stock 3.x `super().learn()`**, which speaks the
3.x env/obs/alg API. WheeledLab's custom per-step logging is dropped (it was the
only reason for the override); stock tensorboard logging is used instead
(`logger_type` is forced to `"tensorboard"` because stock `learn()` asserts on a
`None` logger when wandb is off). The original 2.x loop is left in place but
unreachable, for reference.

## 5. Known cosmetic issue (not fixed) — wheel visual meshes

In the GUI under Isaac Sim 5.1 the wheel **visual** meshes don't render (the cars
look like they "lost their wheels"). The wheel **colliders** are fine — PhysX
parses each `*_wheel_link` and builds convex-hull colliders ("triangle mesh
collision … falling back to convexHull"), so physics/training are unaffected.
Likely a mesh/material asset that doesn't resolve the same way in 5.1 as 4.5.
Headless training is unaffected.

## How to run

```bash
# headless training (real run)
isaaclab.sh -p source/wheeledlab_rl/scripts/train_rl.py --headless \
    -r F1TENTH_DRIFT_CONFIG train.log.no_wandb=True
# available configs: F1TENTH_DRIFT_CONFIG, RSS_DRIFT_CONFIG (mushr),
#                    RSS_ELEV_CONFIG (mushr), RSS_VISUAL_CONFIG (mushr)

# GUI (watch; slow — renders every env): drop --headless, set DISPLAY
```
