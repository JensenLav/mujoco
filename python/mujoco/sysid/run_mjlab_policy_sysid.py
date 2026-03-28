"""Run an MjLab H1_2 velocity policy via ONNX, collect walking data, and run sysID.

Loads the trained policy checkpoint (ONNX export), runs it on the freestanding
H1_2 model, collects trajectory data, then runs Active-SPI Stage 1 system
identification to recover the robot's physical parameters.

Usage:
    cd python/mujoco/sysid
    python run_mjlab_policy_sysid.py
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mujoco
import mujoco.rollout as rollout
import numpy as np
import onnxruntime as ort
from absl import logging
from mujoco import sysid
from mujoco.sysid._src import timeseries

try:
    from mujoco.sysid._src.actuator_config import (
        PositionActuatorCfg,
        apply_actuator_config,
        create_position_actuator,
    )
except ModuleNotFoundError:
    from _src.actuator_config import (
        PositionActuatorCfg,
        apply_actuator_config,
        create_position_actuator,
    )

from h1_2_sysid_active_spi import (
    H1_2_FREESTANDING_XML,
    segment_trajectory,
)

logging.set_verbosity("INFO")

# ============================================================================
# Policy / env configuration (from deploy.yaml and env.yaml)
# ============================================================================

POLICY_ONNX = (
    "/home/jensen/unitree_rl_mjlab/logs/rsl_rl/"
    "h1_2_velocity/2026-03-23_19-50-41/policy.onnx"
)

CONTROL_DT = 0.02  # policy runs at 50 Hz (decimation=4, sim_dt=0.005)
SIM_DT = 0.002     # sysid model timestep
DECIMATION = int(CONTROL_DT / SIM_DT)

# Joint order in MjLab (matches deploy.yaml / actuator ordering)
MJLAB_JOINT_ORDER = [
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

DEFAULT_JOINT_POS = np.array([
    0, -0.2, 0, 0.5, -0.3, 0,   # left leg
    0, -0.2, 0, 0.5, -0.3, 0,   # right leg
    0,                            # torso
    0.28, 0, 0, 0.52, 0, 0, 0,  # left arm
    0.28, 0, 0, 0.52, 0, 0, 0,  # right arm
], dtype=np.float32)

ACTION_SCALE = np.array([
    0.51, 0.51, 0.51, 0.48, 0.51, 0.51,  # left leg
    0.51, 0.51, 0.51, 0.48, 0.51, 0.51,  # right leg
    0.51,                                  # torso
    0.51, 0.51, 0.57, 0.57, 0.57, 0.57, 0.57,  # left arm
    0.51, 0.51, 0.57, 0.57, 0.57, 0.57, 0.57,  # right arm
], dtype=np.float32)

# Actuator PD gains from training env (env.yaml) -- these are the "true"
# parameters that sysID should recover
TRAINING_ACTUATORS: list[tuple[str, PositionActuatorCfg]] = [
    ("left_hip_yaw_joint",        PositionActuatorCfg(98.7,  6.3, 200, armature=0.025, frictionloss=0.0)),
    ("left_hip_pitch_joint",      PositionActuatorCfg(98.7,  6.3, 200, armature=0.025, frictionloss=0.0)),
    ("left_hip_roll_joint",       PositionActuatorCfg(98.7,  6.3, 200, armature=0.025, frictionloss=0.0)),
    ("left_knee_joint",           PositionActuatorCfg(157.7, 10.1, 300, armature=0.04, frictionloss=0.0)),
    ("left_ankle_pitch_joint",    PositionActuatorCfg(19.7,  1.3, 40, armature=0.005, frictionloss=0.0)),
    ("left_ankle_roll_joint",     PositionActuatorCfg(19.7,  1.3, 40, armature=0.005, frictionloss=0.0)),
    ("right_hip_yaw_joint",       PositionActuatorCfg(98.7,  6.3, 200, armature=0.025, frictionloss=0.0)),
    ("right_hip_pitch_joint",     PositionActuatorCfg(98.7,  6.3, 200, armature=0.025, frictionloss=0.0)),
    ("right_hip_roll_joint",      PositionActuatorCfg(98.7,  6.3, 200, armature=0.025, frictionloss=0.0)),
    ("right_knee_joint",          PositionActuatorCfg(157.7, 10.1, 300, armature=0.04, frictionloss=0.0)),
    ("right_ankle_pitch_joint",   PositionActuatorCfg(19.7,  1.3, 40, armature=0.005, frictionloss=0.0)),
    ("right_ankle_roll_joint",    PositionActuatorCfg(19.7,  1.3, 40, armature=0.005, frictionloss=0.0)),
    ("torso_joint",               PositionActuatorCfg(98.7,  6.3, 200, armature=0.025, frictionloss=0.0)),
    ("left_shoulder_pitch_joint", PositionActuatorCfg(19.7,  1.3, 40, armature=0.005, frictionloss=0.0)),
    ("left_shoulder_roll_joint",  PositionActuatorCfg(19.7,  1.3, 40, armature=0.005, frictionloss=0.0)),
    ("left_shoulder_yaw_joint",   PositionActuatorCfg(7.9,   0.5, 18, armature=0.002, frictionloss=0.0)),
    ("left_elbow_joint",          PositionActuatorCfg(7.9,   0.5, 18, armature=0.002, frictionloss=0.0)),
    ("left_wrist_roll_joint",     PositionActuatorCfg(7.9,   0.5, 18, armature=0.002, frictionloss=0.0)),
    ("left_wrist_pitch_joint",    PositionActuatorCfg(7.9,   0.5, 18, armature=0.002, frictionloss=0.0)),
    ("left_wrist_yaw_joint",      PositionActuatorCfg(7.9,   0.5, 18, armature=0.002, frictionloss=0.0)),
    ("right_shoulder_pitch_joint", PositionActuatorCfg(19.7, 1.3, 40, armature=0.005, frictionloss=0.0)),
    ("right_shoulder_roll_joint", PositionActuatorCfg(19.7,  1.3, 40, armature=0.005, frictionloss=0.0)),
    ("right_shoulder_yaw_joint",  PositionActuatorCfg(7.9,   0.5, 18, armature=0.002, frictionloss=0.0)),
    ("right_elbow_joint",         PositionActuatorCfg(7.9,   0.5, 18, armature=0.002, frictionloss=0.0)),
    ("right_wrist_roll_joint",    PositionActuatorCfg(7.9,   0.5, 18, armature=0.002, frictionloss=0.0)),
    ("right_wrist_pitch_joint",   PositionActuatorCfg(7.9,   0.5, 18, armature=0.002, frictionloss=0.0)),
    ("right_wrist_yaw_joint",     PositionActuatorCfg(7.9,   0.5, 18, armature=0.002, frictionloss=0.0)),
]

LEG_JOINT_NAMES = [
    "left_hip_yaw_joint",   "left_hip_pitch_joint",  "left_hip_roll_joint",
    "left_knee_joint",      "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint",  "right_hip_pitch_joint",  "right_hip_roll_joint",
    "right_knee_joint",     "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

DURATION_PER_CMD = 5.0

# Multiple velocity commands for diverse motion excitation
# Each is [vx, vy, wz] -- more diversity = better parameter identifiability
VELOCITY_CMDS = [
    np.array([0.5,  0.0,  0.0], dtype=np.float32),   # walk forward
    np.array([0.0,  0.3,  0.0], dtype=np.float32),   # sidestep left
    np.array([0.0, -0.3,  0.0], dtype=np.float32),   # sidestep right
    np.array([0.3,  0.0,  0.5], dtype=np.float32),   # walk + turn left
    np.array([0.8,  0.0,  0.0], dtype=np.float32),   # walk fast forward
    np.array([0.0,  0.0,  0.0], dtype=np.float32),   # stand still
]
VELOCITY_CMD_NAMES = [
    "fwd_0.5", "side_L_0.3", "side_R_0.3",
    "fwd_turn_L", "fwd_fast_0.8", "stand",
]

# Real robot data: set to a CSV path to use real data, or None for sim
# CSV must have columns: time, {Joint}_q_cmd, {Joint}_q_meas, {Joint}_dq_meas,
#   {Joint}_tau_meas, imu_quat_w/x/y/z, imu_gyro_x/y/z, imu_acc_x/y/z
# Plus a {stem}_meta.txt with joint_names: [...]
REAL_DATA_CSV = None  # e.g. "data/real_walking.csv"

# SysID settings
N_CLIPS = 3
CLIP_H_MIN = 200
CLIP_H_MAX = 400
NOISE_STD = 0.01

# Optimizer: "mujoco" (fast, gradient-based NLS) or "cma" (sampling-based)
OPTIMIZER = "mujoco"

# CMA-ES settings (only used when OPTIMIZER="cma")
import cma
CMAES_POPSIZE = 16
CMAES_MAXITER = 60
CMAES_SIGMA0 = 0.3

# Set to True to compute per-joint kp-vs-kv cost landscape plots (slow)
PLOT_COST_LANDSCAPES = True


# ============================================================================
# Observation builder
# ============================================================================


def build_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cmd: np.ndarray,
    last_action: np.ndarray,
    actuator_names: list[str],
) -> np.ndarray:
    """Build the 90-D actor observation vector matching MjLab training env.

    Order: base_ang_vel(3) + projected_gravity(3) + command(3)
           + joint_pos_rel(27) + joint_vel(27) + last_action(27)
    """
    # Base angular velocity from IMU gyro sensor (body-frame, on torso)
    gyro_id = model.sensor("imu_gyro").id
    base_ang_vel = data.sensordata[
        model.sensor_adr[gyro_id]:model.sensor_adr[gyro_id] + 3
    ].copy()

    # Projected gravity: use IMU quaternion to rotate world gravity into body frame
    quat_id = model.sensor("imu_quat").id
    quat = data.sensordata[
        model.sensor_adr[quat_id]:model.sensor_adr[quat_id] + 4
    ].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    projected_gravity = R.T @ np.array([0.0, 0.0, -1.0])

    # Joint positions relative to default (in MjLab joint order)
    joint_pos_rel = np.zeros(27, dtype=np.float32)
    joint_vel = np.zeros(27, dtype=np.float32)
    for i, jn in enumerate(MJLAB_JOINT_ORDER):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jnt_id >= 0:
            joint_pos_rel[i] = data.qpos[model.jnt_qposadr[jnt_id]] - DEFAULT_JOINT_POS[i]
            joint_vel[i] = data.qvel[model.jnt_dofadr[jnt_id]]

    obs = np.concatenate([
        base_ang_vel.astype(np.float32),
        projected_gravity.astype(np.float32),
        cmd.astype(np.float32),
        joint_pos_rel,
        joint_vel,
        last_action.astype(np.float32),
    ])
    return obs.reshape(1, -1)


# ============================================================================
# Data collection with ONNX policy
# ============================================================================


def collect_walking_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    session: ort.InferenceSession,
    duration: float,
    cmd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the ONNX policy and collect (times, ctrl, sensor, state) arrays.

    The policy runs at CONTROL_DT (50 Hz) while sim runs at SIM_DT (500 Hz).
    We log at SIM_DT for maximum sysID resolution.
    """
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    # Map from MjLab order -> model actuator index
    mjlab_to_act = []
    for jn in MJLAB_JOINT_ORDER:
        mjlab_to_act.append(actuator_names.index(jn))

    n_steps = int(duration / SIM_DT)
    ctrl_log = np.zeros((n_steps, model.nu))
    sensor_log = np.zeros((n_steps, model.nsensordata))
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
    state_log = np.zeros((n_steps, nstate))
    times_log = np.zeros(n_steps)

    last_action = np.zeros(27, dtype=np.float32)
    current_ctrl = np.zeros(model.nu)

    # Set initial default pose (matching training env init z=1.02)
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.02
    for i, jn in enumerate(MJLAB_JOINT_ORDER):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jnt_id >= 0:
            data.qpos[model.jnt_qposadr[jnt_id]] = DEFAULT_JOINT_POS[i]
            current_ctrl[mjlab_to_act[i]] = DEFAULT_JOINT_POS[i]

    data.ctrl[:] = current_ctrl
    for _ in range(200):
        mujoco.mj_step(model, data)

    input_name = session.get_inputs()[0].name
    print(f"  Collecting {duration}s at {1/SIM_DT:.0f}Hz "
          f"(policy at {1/CONTROL_DT:.0f}Hz, decimation={DECIMATION})...")

    for step in range(n_steps):
        if step % DECIMATION == 0:
            obs = build_obs(model, data, cmd, last_action, actuator_names)
            raw_action = session.run(None, {input_name: obs})[0][0]
            last_action = raw_action.copy()

            # Convert to joint position targets: target = default + action * scale
            targets = DEFAULT_JOINT_POS + raw_action * ACTION_SCALE
            for i, jn in enumerate(MJLAB_JOINT_ORDER):
                current_ctrl[mjlab_to_act[i]] = targets[i]

        data.ctrl[:] = current_ctrl
        mujoco.mj_step(model, data)

        times_log[step] = data.time
        ctrl_log[step] = current_ctrl.copy()
        mujoco.mj_getState(model, data, state_log[step], mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
        sensor_log[step] = data.sensordata.copy()

        if (step + 1) % (n_steps // 10) == 0:
            pct = (step + 1) / n_steps * 100
            z = data.qpos[2]
            print(f"    {pct:5.1f}% | t={data.time:.2f}s | pelvis_z={z:.3f}")

    return times_log, ctrl_log, sensor_log, state_log


# ============================================================================
# Real robot data loader
# ============================================================================

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _joint_to_csv(mj_name: str) -> str:
    """left_hip_roll_joint -> LeftHipRoll"""
    base = mj_name.replace("_joint", "")
    return "".join(w.capitalize() for w in base.split("_"))


def load_real_data(
    csv_path: str | Path,
    model: mujoco.MjModel,
    joints: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load real robot CSV and return (times, ctrl, sensor, state).

    Reads the CSV, maps joint columns into MuJoCo ctrl and sensor arrays,
    and runs a nominal sim rollout to produce the state array needed for
    clip segmentation.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    times = df["time"].values
    n = len(times)
    actuator_names = [model.actuator(i).name for i in range(model.nu)]

    ctrl = np.zeros((n, model.nu))
    sensor = np.zeros((n, model.nsensordata))

    for jn in joints:
        csv_jn = _joint_to_csv(jn)
        act_idx = actuator_names.index(jn) if jn in actuator_names else -1

        q_cmd_col = f"{csv_jn}_q_cmd"
        q_meas_col = f"{csv_jn}_q_meas"
        dq_meas_col = f"{csv_jn}_dq_meas"
        tau_meas_col = f"{csv_jn}_tau_meas"

        if q_cmd_col in df.columns and act_idx >= 0:
            ctrl[:, act_idx] = df[q_cmd_col].values

        base = jn.replace("_joint", "")
        if q_meas_col in df.columns:
            sid = model.sensor(f"{base}_pos").id
            sensor[:, model.sensor_adr[sid]] = df[q_meas_col].values
        if dq_meas_col in df.columns:
            sid = model.sensor(f"{base}_vel").id
            sensor[:, model.sensor_adr[sid]] = df[dq_meas_col].values
        if tau_meas_col in df.columns:
            sid = model.sensor(f"{base}_torque").id
            sensor[:, model.sensor_adr[sid]] = df[tau_meas_col].values

    # IMU columns (optional -- fill if present)
    imu_map = {
        "imu_quat": ["imu_quat_w", "imu_quat_x", "imu_quat_y", "imu_quat_z"],
        "imu_gyro": ["imu_gyro_x", "imu_gyro_y", "imu_gyro_z"],
        "imu_acc":  ["imu_acc_x", "imu_acc_y", "imu_acc_z"],
    }
    for sensor_name, cols in imu_map.items():
        if all(c in df.columns for c in cols):
            sid = model.sensor(sensor_name).id
            adr = model.sensor_adr[sid]
            for k, col in enumerate(cols):
                sensor[:, adr + k] = df[col].values

    # Build initial state from first measurement
    data_nom = mujoco.MjData(model)
    mujoco.mj_resetData(model, data_nom)
    data_nom.qpos[2] = 1.02
    for jn in joints:
        csv_jn = _joint_to_csv(jn)
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jnt_id >= 0:
            q_col = f"{csv_jn}_q_meas"
            dq_col = f"{csv_jn}_dq_meas"
            if q_col in df.columns:
                data_nom.qpos[model.jnt_qposadr[jnt_id]] = df[q_col].iloc[0]
            if dq_col in df.columns:
                data_nom.qvel[model.jnt_dofadr[jnt_id]] = df[dq_col].iloc[0]

    init_state = sysid.create_initial_state(
        model, data_nom.qpos, data_nom.qvel, data_nom.act,
    )
    state_sim, _ = rollout.rollout(model, data_nom, init_state, ctrl[:-1])
    state = np.squeeze(state_sim, axis=0)

    return times, ctrl, sensor, state


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 70)
    if REAL_DATA_CSV:
        print("Active-SPI Stage 1 — Real Robot Data")
    else:
        print("Active-SPI Stage 1 — MjLab Policy Sim-to-Sim")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load ONNX policy (only needed for sim data + comparison videos)
    # ------------------------------------------------------------------
    session = None
    if REAL_DATA_CSV is None:
        print(f"\n[1/5] Loading ONNX policy from:\n  {POLICY_ONNX}")
        session = ort.InferenceSession(POLICY_ONNX)
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]
        print(f"  Input:  {inp.name} {inp.shape}")
        print(f"  Output: {out.name} {out.shape}")
    else:
        print(f"\n[1/5] Using real robot data — skipping policy load")

    # ------------------------------------------------------------------
    # 2. Build "true" model with training env actuator parameters
    # ------------------------------------------------------------------
    print("\n[2/5] Building model with training env actuator parameters...")
    spec = mujoco.MjSpec.from_string(H1_2_FREESTANDING_XML)
    apply_actuator_config(spec, TRAINING_ACTUATORS)
    # Zero out XML default joint damping -- training env has no passive damping
    for joint in spec.joints:
        if joint.name != "pelvis_freejoint":
            joint.damping = 0.0
    model = spec.compile()
    data = mujoco.MjData(model)

    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    print(f"  Model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"  Actuators: {actuator_names[:6]}... ({model.nu} total)")

    # Print true PD gains for reference
    print("\n  True actuator parameters (to be recovered by sysID):")
    for jn in LEG_JOINT_NAMES:
        kp, kv = sysid.model_modifier.get_actuator_pd_gains(model, jn)
        arm = float(np.asarray(model.joint(jn).armature).item())
        print(f"    {jn:35s}  kp={float(kp):7.1f}  kv={float(kv):5.1f}  arm={arm:.4f}")

    # ------------------------------------------------------------------
    # 3. Get trajectory data (sim or real)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)

    if REAL_DATA_CSV is not None:
        print(f"\n[3/5] Loading real robot data from: {REAL_DATA_CSV}")
        times, ctrl, sensor, state = load_real_data(
            REAL_DATA_CSV, model, LEG_JOINT_NAMES,
        )
        sensor_noisy = sensor
        print(f"  Loaded {len(times)} steps, {times[-1] - times[0]:.1f}s")
    else:
        print(f"\n[3/5] Collecting data with {len(VELOCITY_CMDS)} velocity commands "
              f"({DURATION_PER_CMD}s each)...")
        all_times, all_ctrl, all_sensor, all_state = [], [], [], []
        t_offset = 0.0
        for i, (cmd, cmd_name) in enumerate(zip(VELOCITY_CMDS, VELOCITY_CMD_NAMES)):
            print(f"    [{i+1}/{len(VELOCITY_CMDS)}] cmd={cmd} ({cmd_name})")
            mujoco.mj_resetData(model, data)
            t, c, s, st = collect_walking_data(
                model, data, session, DURATION_PER_CMD, cmd,
            )
            t = t - t[0] + t_offset
            all_times.append(t)
            all_ctrl.append(c)
            all_sensor.append(s)
            all_state.append(st)
            t_offset = t[-1] + SIM_DT

        times = np.concatenate(all_times)
        ctrl = np.concatenate(all_ctrl)
        sensor = np.concatenate(all_sensor)
        state = np.concatenate(all_state)
        sensor_noisy = sensor + rng.normal(scale=NOISE_STD, size=sensor.shape)
        total_dur = times[-1] - times[0]
        print(f"  Total: {len(times)} steps, {total_dur:.1f}s across "
              f"{len(VELOCITY_CMDS)} motions")

    # ------------------------------------------------------------------
    # 4. Segment and build sysID problem
    # ------------------------------------------------------------------
    print(f"\n[4/5] Building sysID problem ({N_CLIPS} clips)...")

    clips = segment_trajectory(
        times, ctrl, sensor_noisy, state,
        CLIP_H_MIN, CLIP_H_MAX, N_CLIPS, rng,
    )
    print(f"  Clip lengths: {[len(c[0]) for c in clips[:5]]}...")

    # Store reference parameter values (true values in sim, nominal in real)
    TRUE_ARMATURE = {n: float(np.asarray(model.joint(n).armature).item()) for n in LEG_JOINT_NAMES}
    TRUE_FRICTION = {n: float(np.asarray(model.joint(n).frictionloss).item()) for n in LEG_JOINT_NAMES}
    TRUE_DAMPING = {n: float(np.asarray(model.joint(n).damping).item()) for n in LEG_JOINT_NAMES}
    TRUE_KP, TRUE_KV = {}, {}
    for n in LEG_JOINT_NAMES:
        kp, kv = sysid.model_modifier.get_actuator_pd_gains(model, n)
        TRUE_KP[n] = float(kp)
        TRUE_KV[n] = float(kv)
    using_real_data = REAL_DATA_CSV is not None

    # Build ParameterDict with randomized initial guesses per joint
    # Each joint gets an independent random perturbation of 5-50% from truth
    params = sysid.ParameterDict()
    init_rng = np.random.default_rng()  # unseeded = different each run

    def make_armature_mod(jn):
        def mod(s, p): s.joint(jn).armature = p.value[0]
        return mod
    def make_friction_mod(jn):
        def mod(s, p): s.joint(jn).frictionloss = p.value[0]
        return mod
    def make_damping_mod(jn):
        def mod(s, p): s.joint(jn).damping = p.value[0]
        return mod
    def make_kp_mod(jn):
        def mod(s, p): sysid.apply_pgain(s, jn, p.value[0])
        return mod
    def make_kv_mod(jn):
        def mod(s, p): sysid.apply_dgain(s, jn, p.value[0])
        return mod

    JOINT_BOUNDS = {
        "hip":   {"arm": (0.005, 0.1),  "fric": (0.0, 1.0), "damp": (0.0, 2.0),
                  "kp": (50.0, 180.0),   "kv": (1.0, 15.0)},
        "knee":  {"arm": (0.01, 0.15),   "fric": (0.0, 1.0), "damp": (0.0, 2.0),
                  "kp": (80.0, 250.0),   "kv": (3.0, 18.0)},
        "ankle": {"arm": (0.001, 0.05),  "fric": (0.0, 1.0), "damp": (0.0, 2.0),
                  "kp": (5.0, 50.0),     "kv": (0.1, 5.0)},
    }

    def _joint_type(name):
        if "knee" in name:
            return "knee"
        elif "ankle" in name:
            return "ankle"
        return "hip"

    def _perturb(true_val, lo, hi, rng_):
        """Perturb true_val by 5-50%, random sign, clamped to [lo, hi]."""
        pct = rng_.uniform(0.05, 0.50)
        sign = rng_.choice([-1.0, 1.0])
        scale = max(abs(true_val), 1e-4)
        perturbed = true_val + sign * pct * scale
        return float(np.clip(perturbed, lo, hi))

    # Store per-joint initial values for the results table and video
    JOINT_INIT_VALUES = {}

    print("  Randomized initial guesses (5-50% perturbation per joint):")
    for name in LEG_JOINT_NAMES:
        jt = _joint_type(name)
        bnds = JOINT_BOUNDS[jt]

        init_arm  = _perturb(TRUE_ARMATURE[name], bnds["arm"][0],  bnds["arm"][1],  init_rng)
        init_fric = _perturb(TRUE_FRICTION[name], bnds["fric"][0], bnds["fric"][1], init_rng)
        init_damp = _perturb(TRUE_DAMPING[name],  bnds["damp"][0], bnds["damp"][1], init_rng)
        init_kp   = _perturb(TRUE_KP[name],       bnds["kp"][0],   bnds["kp"][1],   init_rng)
        init_kv   = _perturb(TRUE_KV[name],       bnds["kv"][0],   bnds["kv"][1],   init_rng)

        JOINT_INIT_VALUES[name] = {
            "arm": init_arm, "fric": init_fric, "damp": init_damp,
            "kp": init_kp, "kv": init_kv,
        }

        short = name.replace("_joint", "").replace("left_", "L_").replace("right_", "R_")
        print(f"    {short:20s}  kp={init_kp:7.1f} (true={TRUE_KP[name]:7.1f})  "
              f"kv={init_kv:5.2f} (true={TRUE_KV[name]:5.2f})  "
              f"arm={init_arm:.4f} (true={TRUE_ARMATURE[name]:.4f})")

        params.add(sysid.Parameter(
            f"{name}_armature", nominal=TRUE_ARMATURE[name],
            min_value=bnds["arm"][0], max_value=bnds["arm"][1],
            modifier=make_armature_mod(name),
        ))
        params[f"{name}_armature"].value[:] = init_arm

        params.add(sysid.Parameter(
            f"{name}_friction", nominal=TRUE_FRICTION[name],
            min_value=bnds["fric"][0], max_value=bnds["fric"][1],
            modifier=make_friction_mod(name),
        ))
        params[f"{name}_friction"].value[:] = init_fric

        params.add(sysid.Parameter(
            f"{name}_damping", nominal=TRUE_DAMPING[name],
            min_value=bnds["damp"][0], max_value=bnds["damp"][1],
            modifier=make_damping_mod(name),
        ))
        params[f"{name}_damping"].value[:] = init_damp

        params.add(sysid.Parameter(
            f"{name}_kp", nominal=TRUE_KP[name],
            min_value=bnds["kp"][0], max_value=bnds["kp"][1],
            modifier=make_kp_mod(name),
        ))
        params[f"{name}_kp"].value[:] = init_kp

        params.add(sysid.Parameter(
            f"{name}_kv", nominal=TRUE_KV[name],
            min_value=bnds["kv"][0], max_value=bnds["kv"][1],
            modifier=make_kv_mod(name),
        ))
        params[f"{name}_kv"].value[:] = init_kv

    n_free = sum(p.value.size for p in params.parameters.values() if not p.frozen)
    print(f"  Parameters: {n_free} free DOFs")

    # Build observations + residual
    enabled_obs = []
    for jn in LEG_JOINT_NAMES:
        base_name = jn.replace("_joint", "")
        enabled_obs.extend([
            (f"{base_name}_pos", timeseries.SignalType.MjSensor),
            (f"{base_name}_vel", timeseries.SignalType.MjSensor),
            (f"{base_name}_torque", timeseries.SignalType.MjSensor),
        ])
    enabled_obs.extend([
        ("imu_quat", timeseries.SignalType.MjSensor),
        ("imu_gyro", timeseries.SignalType.MjSensor),
        ("imu_acc", timeseries.SignalType.MjSensor),
    ])

    model_sequences = []
    for i, (t_clip, c_clip, s_clip, s0) in enumerate(clips):
        qpos0 = s0[1:1 + model.nq]
        qvel0 = s0[1 + model.nq:1 + model.nq + model.nv]
        nact = model.na
        act0 = s0[1+model.nq+model.nv:1+model.nq+model.nv+nact] if nact > 0 else np.zeros(0)
        init_state = sysid.create_initial_state(model, qpos0, qvel0, act0)
        ctrl_ts = sysid.TimeSeries(t_clip, c_clip)
        sensor_ts = sysid.TimeSeries.from_names(t_clip, s_clip, model)
        ms = sysid.ModelSequences(
            "h1_2_walk", spec, f"clip_{i}", init_state, ctrl_ts, sensor_ts,
        )
        model_sequences.append(ms)

    residual_fn = sysid.build_residual_fn(
        models_sequences=model_sequences,
        enabled_observations=enabled_obs,
    )
    print(f"  Residual built over {len(model_sequences)} clips")

    # ------------------------------------------------------------------
    # 5. Optimization
    # ------------------------------------------------------------------
    t0 = time.time()

    if OPTIMIZER == "mujoco":
        print("\n[5/5] Running MuJoCo NLS optimizer (gradient-based)...")
        opt_params, opt_result = sysid.optimize(
            initial_params=params,
            residual_fn=residual_fn,
            optimizer="mujoco",
        )
        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed:.1f}s")
        if hasattr(opt_result, 'cost'):
            print(f"  Final cost: {opt_result.cost:.6f}")
        elif hasattr(opt_result, 'fun'):
            print(f"  Final cost: {opt_result.fun:.6f}")
        es = None
    else:
        print(f"\n[5/5] Running CMA-ES (pop={CMAES_POPSIZE}, maxiter={CMAES_MAXITER})...")
        x0 = params.as_vector()
        lb, ub = params.get_bounds()

        def spi_cost(x):
            p = params.copy()
            p.update_from_vector(x)
            res, _, _ = residual_fn(x, p)
            return sum(float(np.sum(r**2)) for r in res)

        opts = cma.CMAOptions()
        opts["bounds"] = [lb.tolist(), ub.tolist()]
        opts["popsize"] = CMAES_POPSIZE
        opts["maxiter"] = CMAES_MAXITER
        opts["verbose"] = 1
        opts["tolfun"] = 1e-8

        es = cma.CMAEvolutionStrategy(x0.tolist(), CMAES_SIGMA0, opts)
        iteration = 0
        while not es.stop():
            X = es.ask()
            costs = [spi_cost(np.array(c)) for c in X]
            es.tell(X, costs)
            if iteration % 10 == 0:
                print(f"  iter {iteration:4d} | best={es.best.f:.4f} | "
                      f"mean={np.mean(costs):.4f}")
            iteration += 1

        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed:.1f}s ({iteration} iterations)")
        print(f"  Final best cost: {es.best.f:.6f}")
        opt_params = params.copy()
        opt_params.update_from_vector(np.array(es.best.x))

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    if using_real_data:
        print("RESULTS: Identified Parameters (real robot data)")
    else:
        print("RESULTS: Recovered vs True Parameters (sim-to-sim)")
    print("=" * 70)
    ref_label = "Nominal" if using_real_data else "True"
    print(f"\n{'Parameter':<40s} {ref_label:>10s} {'Initial':>10s} {'Recovered':>10s} {'AbsErr':>10s}")
    print("-" * 80)

    suffix_to_init_key = {
        "armature": "arm", "friction": "fric", "damping": "damp",
        "kp": "kp", "kv": "kv",
    }
    abs_errors = []
    for name in LEG_JOINT_NAMES:
        jiv = JOINT_INIT_VALUES[name]
        for suffix, true_dict in [
            ("armature", TRUE_ARMATURE), ("friction", TRUE_FRICTION),
            ("damping", TRUE_DAMPING), ("kp", TRUE_KP), ("kv", TRUE_KV),
        ]:
            key = f"{name}_{suffix}"
            tv = float(true_dict[name])
            iv = jiv[suffix_to_init_key[suffix]]
            ov = float(opt_params[key].value[0])
            ae = abs(ov - tv)
            abs_errors.append(ae)
            print(f"  {key:<38s} {tv:10.4f} {iv:10.4f} {ov:10.4f} {ae:10.4f}")

    print(f"\n  Mean absolute error:   {np.mean(abs_errors):.4f}")
    print(f"  Median absolute error: {np.median(abs_errors):.4f}")

    # Relative errors only for non-zero true values
    rel_errors = []
    for name in LEG_JOINT_NAMES:
        for suffix, true_dict in [
            ("armature", TRUE_ARMATURE), ("kp", TRUE_KP), ("kv", TRUE_KV),
        ]:
            key = f"{name}_{suffix}"
            tv = float(true_dict[name])
            if abs(tv) > 1e-6:
                ov = float(opt_params[key].value[0])
                rel_errors.append(abs(ov - tv) / abs(tv) * 100)
    if rel_errors:
        print(f"  Mean relative error (arm/kp/kv): {np.mean(rel_errors):.1f}%")
        print(f"  Median relative error:           {np.median(rel_errors):.1f}%")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Comparison videos
    # ------------------------------------------------------------------
    import mediapy as media

    if using_real_data:
        print("\n  Skipping comparison videos (real data mode — no sim policy)")
        print("  (Videos require closed-loop policy rollout on colored models)")
    elif session is None:
        print("\n  Skipping comparison videos (no policy loaded)")

    if session is not None and not using_real_data:
        print("\n  Generating closed-loop comparison videos...")

        def set_body_rgba(body, rgba):
            for geom in body.geoms:
                geom.rgba = rgba
            for child in body.bodies:
                set_body_rgba(child, rgba)

        def make_colored_model(base_spec, rgba, param_dict):
            s = base_spec.copy()
            for joint in s.joints:
                if joint.name != "pelvis_freejoint":
                    joint.damping = 0.0
            for key, param in param_dict.parameters.items():
                if not param.frozen:
                    param.apply_modifier(s)
            set_body_rgba(s.worldbody, rgba)
            return s.compile()

        def run_policy_closed_loop(vid_model, onnx_session, duration_s, cmd):
            vid_data = mujoco.MjData(vid_model)
            act_names = [vid_model.actuator(i).name for i in range(vid_model.nu)]
            mjlab_to_act = [act_names.index(jn) for jn in MJLAB_JOINT_ORDER]
            mujoco.mj_resetData(vid_model, vid_data)
            vid_data.qpos[2] = 1.02
            cur_ctrl = np.zeros(vid_model.nu)
            for i, jn in enumerate(MJLAB_JOINT_ORDER):
                jid = mujoco.mj_name2id(vid_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    vid_data.qpos[vid_model.jnt_qposadr[jid]] = DEFAULT_JOINT_POS[i]
                    cur_ctrl[mjlab_to_act[i]] = DEFAULT_JOINT_POS[i]
            vid_data.ctrl[:] = cur_ctrl
            for _ in range(200):
                mujoco.mj_step(vid_model, vid_data)
            n_steps = int(duration_s / SIM_DT)
            nstate = mujoco.mj_stateSize(vid_model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
            state_log = np.zeros((n_steps, nstate))
            last_act = np.zeros(27, dtype=np.float32)
            input_name = onnx_session.get_inputs()[0].name
            for step in range(n_steps):
                if step % DECIMATION == 0:
                    obs = build_obs(vid_model, vid_data, cmd, last_act, act_names)
                    raw = onnx_session.run(None, {input_name: obs})[0][0]
                    last_act = raw.copy()
                    targets = DEFAULT_JOINT_POS + raw * ACTION_SCALE
                    for i, jn in enumerate(MJLAB_JOINT_ORDER):
                        cur_ctrl[mjlab_to_act[i]] = targets[i]
                vid_data.ctrl[:] = cur_ctrl
                mujoco.mj_step(vid_model, vid_data)
                mujoco.mj_getState(vid_model, vid_data, state_log[step],
                                   mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
            return state_log

        init_params = params.copy()
        for name in LEG_JOINT_NAMES:
            jiv = JOINT_INIT_VALUES[name]
            init_params[f"{name}_armature"].value[:] = jiv["arm"]
            init_params[f"{name}_friction"].value[:] = jiv["fric"]
            init_params[f"{name}_damping"].value[:] = jiv["damp"]
            init_params[f"{name}_kp"].value[:] = jiv["kp"]
            init_params[f"{name}_kv"].value[:] = jiv["kv"]

        true_params = params.copy()
        for name in LEG_JOINT_NAMES:
            true_params[f"{name}_armature"].value[:] = TRUE_ARMATURE[name]
            true_params[f"{name}_friction"].value[:] = TRUE_FRICTION[name]
            true_params[f"{name}_damping"].value[:] = TRUE_DAMPING[name]
            true_params[f"{name}_kp"].value[:] = TRUE_KP[name]
            true_params[f"{name}_kv"].value[:] = TRUE_KV[name]

        green = [0.2, 0.8, 0.2, 0.7]
        red   = [1.0, 0.2, 0.2, 0.7]
        blue  = [0.2, 0.4, 1.0, 0.7]

        truth_model = make_colored_model(spec, green, true_params)
        init_model  = make_colored_model(spec, red, init_params)
        opt_model   = make_colored_model(spec, blue, opt_params)

        vid_duration = 10.0
        fps = 30
        camera = "head_on"

        vid_cmd = np.array([0.4, 0.0, 0.4], dtype=np.float32)  # walk forward + turn
        print(f"    Using velocity command: {vid_cmd}")
        print("    Running policy on TRUTH model (green)...")
        state_truth = run_policy_closed_loop(truth_model, session, vid_duration, vid_cmd)
        print("    Running policy on INITIAL model (red)...")
        state_init = run_policy_closed_loop(init_model, session, vid_duration, vid_cmd)
        print("    Running policy on OPTIMIZED model (blue)...")
        state_opt = run_policy_closed_loop(opt_model, session, vid_duration, vid_cmd)

        print("    Rendering BEFORE (initial=red vs truth=green)...")
        state_before = np.stack([state_init, state_truth], axis=0)
        models_before = [init_model, truth_model]
        datas_before = [mujoco.MjData(m) for m in models_before]
        frames_before = sysid.render_rollout(
            models_before, datas_before[0], state_before,
            framerate=fps, camera=camera, height=720, width=1280,
        )

        print("    Rendering AFTER (optimized=blue vs truth=green)...")
        state_after = np.stack([state_opt, state_truth], axis=0)
        models_after = [opt_model, truth_model]
        datas_after = [mujoco.MjData(m) for m in models_after]
        frames_after = sysid.render_rollout(
            models_after, datas_after[0], state_after,
            framerate=fps, camera=camera, height=720, width=1280,
        )

        print("    Compositing side-by-side...")
        n_frames = min(len(frames_before), len(frames_after))
        composite = []
        for i in range(n_frames):
            fb = frames_before[i]
            fa = frames_after[i]
            h = min(fb.shape[0], fa.shape[0])
            w = min(fb.shape[1], fa.shape[1])
            combined = np.concatenate([fb[:h, :w], fa[:h, :w]], axis=1)
            composite.append(combined)

        media.write_video("mjlab_spi_comparison.mp4", composite, fps=fps)
        media.write_video("mjlab_spi_before.mp4", frames_before, fps=fps)
        media.write_video("mjlab_spi_after.mp4", frames_after, fps=fps)
        print("  Saved: mjlab_spi_comparison.mp4")
        print("  Saved: mjlab_spi_before.mp4  (initial=red vs truth=green)")
        print("  Saved: mjlab_spi_after.mp4   (optimized=blue vs truth=green)")

    # ------------------------------------------------------------------
    # Cost landscapes: kp vs kv for ALL 12 leg joints
    # ------------------------------------------------------------------
    if not PLOT_COST_LANDSCAPES:
        print("\n  Skipping cost landscape plots (PLOT_COST_LANDSCAPES=False)")
        out_path = Path("mjlab_spi_stage1_results.png")
        return opt_params, es

    n_joints = len(LEG_JOINT_NAMES)
    n_cols = 4
    n_rows = (n_joints + n_cols - 1) // n_cols

    def eval_grid(key_x, grid_x, key_y, grid_y):
        cost = np.zeros((len(grid_x), len(grid_y)))
        p = opt_params.copy()
        for i, vx in enumerate(grid_x):
            for j, vy in enumerate(grid_y):
                p[key_x].value[:] = vx
                p[key_y].value[:] = vy
                x = p.as_vector()
                res, _, _ = residual_fn(x, p)
                cost[i, j] = sum(np.sum(r**2) for r in res)
        return cost

    print(f"\n  Computing kp-vs-kv cost landscapes for all {n_joints} joints...")
    all_costs = {}
    all_grids = {}
    for idx, jn in enumerate(LEG_JOINT_NAMES):
        jt = _joint_type(jn)
        bnds = JOINT_BOUNDS[jt]
        kp_grid = np.linspace(bnds["kp"][0], bnds["kp"][1], 8)
        kv_grid = np.linspace(bnds["kv"][0], bnds["kv"][1], 8)
        short = jn.replace("_joint", "").replace("left_", "L ").replace("right_", "R ")
        print(f"    [{idx+1}/{n_joints}] {short}...")
        all_costs[jn] = eval_grid(f"{jn}_kp", kp_grid, f"{jn}_kv", kv_grid)
        all_grids[jn] = (kp_grid, kv_grid)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                             layout="constrained")
    axes_flat = axes.flatten()

    for idx, jn in enumerate(LEG_JOINT_NAMES):
        ax = axes_flat[idx]
        cost = all_costs[jn]
        kp_grid, kv_grid = all_grids[jn]
        jiv = JOINT_INIT_VALUES[jn]
        short = jn.replace("_joint", "").replace("left_", "L ").replace("right_", "R ")

        X, Y = np.meshgrid(kp_grid, kv_grid)
        log_cost = np.log10(cost + 1e-12)
        levels = np.linspace(log_cost.min(), log_cost.max(), 25)
        cf = ax.contourf(X, Y, log_cost.T, levels=levels, cmap="viridis")

        tx, ty = TRUE_KP[jn], TRUE_KV[jn]
        ox = float(opt_params[f"{jn}_kp"].value[0])
        oy = float(opt_params[f"{jn}_kv"].value[0])
        ax.plot(tx, ty, "r*", markersize=14, label="True", zorder=5)
        ax.plot(jiv["kp"], jiv["kv"], "ws", markersize=8, label="Initial", zorder=5)
        ax.plot(ox, oy, marker="X", color="gold", markeredgecolor="k",
                markeredgewidth=1, markersize=11, linestyle="none",
                label="Optimized", zorder=5)

        jt = _joint_type(jn)
        bkp = JOINT_BOUNDS[jt]["kp"]
        bkv = JOINT_BOUNDS[jt]["kv"]
        rect = patches.Rectangle(
            (bkp[0], bkv[0]), bkp[1] - bkp[0], bkv[1] - bkv[0],
            linewidth=1.5, edgecolor="white", facecolor="none",
            linestyle="--", label="Bounds", zorder=6,
        )
        ax.add_patch(rect)

        ax.set_title(short, fontsize=10, fontweight="bold")
        ax.set_xlabel("kp", fontsize=8)
        ax.set_ylabel("kv", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15)
        fig.colorbar(cf, ax=ax, label=r"$\log_{10}$", shrink=0.85, pad=0.02)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=6)

    for idx in range(n_joints, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Active-SPI Stage 1 — kp vs kv cost landscapes (all leg joints)",
                 fontsize=14, fontweight="bold")
    out_path = Path("mjlab_spi_stage1_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()

    return opt_params, es


if __name__ == "__main__":
    opt_params, es = main()
