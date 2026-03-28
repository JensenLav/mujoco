"""Data collection for Active-SPI system identification.

Runs a user-provided policy on the free-standing H1_2 humanoid in MuJoCo
and logs joint + IMU trajectories to CSV files compatible with both
h1_2_sysid.py (real-data mode) and h1_2_sysid_active_spi.py.

Usage:
    # With the built-in example policies:
    cd python/mujoco/sysid
    python data_collect.py --policy standing --duration 10
    python data_collect.py --policy walking --duration 20
    python data_collect.py --policy squatting --duration 15

    # With your own policy (import and register it):
    python data_collect.py --policy my_module.MyPolicy --duration 20

    # All 12 leg joints are logged by default. Override with --joints:
    python data_collect.py --policy walking --joints LeftKnee RightKnee
"""

from __future__ import annotations

import argparse
import importlib
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

try:
    from mujoco.sysid._src.actuator_config import (
        H1_2_POSITION_ACTUATORS,
        apply_actuator_config,
    )
except ModuleNotFoundError:
    from _src.actuator_config import (
        H1_2_POSITION_ACTUATORS,
        apply_actuator_config,
    )

# ============================================================================
# MuJoCo joint name <-> CamelCase CSV name mapping
# ============================================================================

_CAMEL_TO_SNAKE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def joint_to_csv_name(mj_name: str) -> str:
    """left_hip_roll_joint -> LeftHipRoll"""
    base = mj_name.replace("_joint", "")
    return "".join(w.capitalize() for w in base.split("_"))


def csv_name_to_joint(csv_name: str) -> str:
    """LeftHipRoll -> left_hip_roll_joint"""
    return _CAMEL_TO_SNAKE.sub("_", csv_name).lower() + "_joint"


# All 12 leg joints in MuJoCo name order
ALL_LEG_JOINTS = [
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
]


# ============================================================================
# Policy interface
# ============================================================================


class Policy(ABC):
    """Base class for data-collection policies.

    Subclass this and implement `get_action` to plug in your own RL policy.
    """

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Called once before the collection episode starts."""

    @abstractmethod
    def get_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        obs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return ctrl vector (joint position targets), shape (nu,).

        Args:
            model: compiled MjModel
            data: current MjData (after mj_step)
            obs: dictionary with pre-extracted observations:
                "qpos"       : all joint positions (nq,)
                "qvel"       : all joint velocities (nv,)
                "imu_quat"   : torso IMU orientation quaternion (4,)
                "imu_gyro"   : torso angular velocity (3,)
                "imu_acc"    : torso linear acceleration (3,)
                "joint_pos"  : dict {joint_name: float} for all actuated joints
                "joint_vel"  : dict {joint_name: float} for all actuated joints
                "joint_torque": dict {joint_name: float} for all actuated joints
                "time"       : float, simulation time
        """
        ...


# ============================================================================
# Built-in example policies
# ============================================================================


class StandingPolicy(Policy):
    """Hold a stable standing pose."""

    def get_action(self, model, data, obs):
        ctrl = np.zeros(model.nu)
        names = [model.actuator(i).name for i in range(model.nu)]
        standing = {
            "left_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.25,
            "left_ankle_pitch_joint": -0.15,
            "right_hip_pitch_joint": -0.1,
            "right_knee_joint": 0.25,
            "right_ankle_pitch_joint": -0.15,
        }
        for jn, val in standing.items():
            if jn in names:
                ctrl[names.index(jn)] = val
        return ctrl


class WalkingPolicy(Policy):
    """Simple sinusoidal walking gait."""

    def get_action(self, model, data, obs):
        ctrl = np.zeros(model.nu)
        names = [model.actuator(i).name for i in range(model.nu)]
        t = obs["time"]
        omega = 2.0 * np.pi / 0.8

        gait = {
            "left_hip_pitch_joint":   -0.25 * np.sin(omega * t),
            "left_knee_joint":         0.4 + 0.35 * max(0, np.sin(omega * t)),
            "left_ankle_pitch_joint":  0.12 * np.sin(omega * t),
            "right_hip_pitch_joint":  -0.25 * np.sin(omega * t + np.pi),
            "right_knee_joint":        0.4 + 0.35 * max(0, np.sin(omega * t + np.pi)),
            "right_ankle_pitch_joint": 0.12 * np.sin(omega * t + np.pi),
        }
        for jn, val in gait.items():
            if jn in names:
                ctrl[names.index(jn)] = val
        return ctrl


class SquattingPolicy(Policy):
    """Symmetric periodic squatting motion."""

    def get_action(self, model, data, obs):
        ctrl = np.zeros(model.nu)
        names = [model.actuator(i).name for i in range(model.nu)]
        t = obs["time"]
        omega = 2.0 * np.pi / 2.0
        depth = 0.5 * (1.0 - np.cos(omega * t))

        gait = {
            "left_hip_pitch_joint":   -0.3 * depth,
            "left_knee_joint":         0.6 * depth,
            "left_ankle_pitch_joint":  0.15 * depth,
            "right_hip_pitch_joint":  -0.3 * depth,
            "right_knee_joint":        0.6 * depth,
            "right_ankle_pitch_joint": 0.15 * depth,
        }
        for jn, val in gait.items():
            if jn in names:
                ctrl[names.index(jn)] = val
        return ctrl


BUILTIN_POLICIES: dict[str, type[Policy]] = {
    "standing": StandingPolicy,
    "walking": WalkingPolicy,
    "squatting": SquattingPolicy,
}


# ============================================================================
# Observation extraction
# ============================================================================


def extract_obs(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    """Extract a standard observation dict from MjData."""
    actuator_names = [model.actuator(i).name for i in range(model.nu)]

    joint_pos = {}
    joint_vel = {}
    joint_torque = {}
    for name in actuator_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id >= 0:
            joint_pos[name] = float(data.qpos[model.jnt_qposadr[jnt_id]])
            joint_vel[name] = float(data.qvel[model.jnt_dofadr[jnt_id]])
        base = name.replace("_joint", "")
        try:
            tau_id = model.sensor(f"{base}_torque").id
            joint_torque[name] = float(
                data.sensordata[model.sensor_adr[tau_id]]
            )
        except KeyError:
            joint_torque[name] = 0.0

    imu_quat = np.zeros(4)
    imu_gyro = np.zeros(3)
    imu_acc = np.zeros(3)
    try:
        s = model.sensor("imu_quat")
        imu_quat = data.sensordata[model.sensor_adr[s.id]:model.sensor_adr[s.id] + 4].copy()
    except KeyError:
        pass
    try:
        s = model.sensor("imu_gyro")
        imu_gyro = data.sensordata[model.sensor_adr[s.id]:model.sensor_adr[s.id] + 3].copy()
    except KeyError:
        pass
    try:
        s = model.sensor("imu_acc")
        imu_acc = data.sensordata[model.sensor_adr[s.id]:model.sensor_adr[s.id] + 3].copy()
    except KeyError:
        pass

    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "imu_quat": imu_quat,
        "imu_gyro": imu_gyro,
        "imu_acc": imu_acc,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "joint_torque": joint_torque,
        "time": data.time,
    }


# ============================================================================
# Data collection loop
# ============================================================================


def collect_data(
    policy: Policy,
    duration: float = 20.0,
    dt: float = 0.002,
    joints: list[str] | None = None,
    settle_steps: int = 500,
    xml_string: str | None = None,
) -> tuple[dict, mujoco.MjModel]:
    """Run a policy and collect trajectory data.

    Args:
        policy: Policy instance whose get_action() will be called each step.
        duration: episode length in seconds.
        dt: simulation timestep (should match model).
        joints: list of MuJoCo joint names to log. Defaults to ALL_LEG_JOINTS.
        settle_steps: number of steps to let the robot settle before recording.
        xml_string: optional MJCF XML. Defaults to the freestanding H1_2 model.

    Returns:
        (data_dict, model) where data_dict has keys ready for CSV export.
    """
    if joints is None:
        joints = list(ALL_LEG_JOINTS)

    if xml_string is None:
        from h1_2_sysid_active_spi import H1_2_FREESTANDING_XML
        xml_string = H1_2_FREESTANDING_XML

    spec = mujoco.MjSpec.from_string(xml_string)
    apply_actuator_config(spec, H1_2_POSITION_ACTUATORS)
    model = spec.compile()
    data = mujoco.MjData(model)
    actuator_names = [model.actuator(i).name for i in range(model.nu)]

    # Settle into standing pose
    mujoco.mj_resetData(model, data)
    standing = {
        "left_hip_pitch_joint": -0.1, "left_knee_joint": 0.25,
        "left_ankle_pitch_joint": -0.15,
        "right_hip_pitch_joint": -0.1, "right_knee_joint": 0.25,
        "right_ankle_pitch_joint": -0.15,
    }
    for name, val in standing.items():
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id >= 0:
            data.qpos[model.jnt_qposadr[jnt_id]] = val
    for name, val in standing.items():
        if name in actuator_names:
            data.ctrl[actuator_names.index(name)] = val
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    policy.reset(model, data)

    n_steps = int(duration / dt)
    csv_joint_names = [joint_to_csv_name(j) for j in joints]

    # Pre-allocate storage
    log = {"time": np.zeros(n_steps)}
    for csv_jn in csv_joint_names:
        for suffix in ["q_cmd", "dq_cmd", "tau_cmd",
                       "q_meas", "dq_meas", "ddq_meas", "tau_meas"]:
            log[f"{csv_jn}_{suffix}"] = np.zeros(n_steps)
    log["imu_quat_w"] = np.zeros(n_steps)
    log["imu_quat_x"] = np.zeros(n_steps)
    log["imu_quat_y"] = np.zeros(n_steps)
    log["imu_quat_z"] = np.zeros(n_steps)
    log["imu_gyro_x"] = np.zeros(n_steps)
    log["imu_gyro_y"] = np.zeros(n_steps)
    log["imu_gyro_z"] = np.zeros(n_steps)
    log["imu_acc_x"] = np.zeros(n_steps)
    log["imu_acc_y"] = np.zeros(n_steps)
    log["imu_acc_z"] = np.zeros(n_steps)

    prev_vel = {j: 0.0 for j in joints}

    print(f"Collecting {duration}s at {1/dt:.0f} Hz ({n_steps} steps)...")
    for step in range(n_steps):
        obs = extract_obs(model, data)
        ctrl = policy.get_action(model, data, obs)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        # Read post-step sensors
        obs_post = extract_obs(model, data)
        log["time"][step] = obs_post["time"]

        for mj_jn, csv_jn in zip(joints, csv_joint_names):
            act_idx = actuator_names.index(mj_jn) if mj_jn in actuator_names else -1

            log[f"{csv_jn}_q_cmd"][step] = float(ctrl[act_idx]) if act_idx >= 0 else 0.0
            log[f"{csv_jn}_dq_cmd"][step] = 0.0
            log[f"{csv_jn}_tau_cmd"][step] = 0.0

            q_meas = obs_post["joint_pos"].get(mj_jn, 0.0)
            dq_meas = obs_post["joint_vel"].get(mj_jn, 0.0)
            tau_meas = obs_post["joint_torque"].get(mj_jn, 0.0)
            ddq_meas = (dq_meas - prev_vel[mj_jn]) / dt
            prev_vel[mj_jn] = dq_meas

            log[f"{csv_jn}_q_meas"][step] = q_meas
            log[f"{csv_jn}_dq_meas"][step] = dq_meas
            log[f"{csv_jn}_ddq_meas"][step] = ddq_meas
            log[f"{csv_jn}_tau_meas"][step] = tau_meas

        log["imu_quat_w"][step] = obs_post["imu_quat"][0]
        log["imu_quat_x"][step] = obs_post["imu_quat"][1]
        log["imu_quat_y"][step] = obs_post["imu_quat"][2]
        log["imu_quat_z"][step] = obs_post["imu_quat"][3]
        log["imu_gyro_x"][step] = obs_post["imu_gyro"][0]
        log["imu_gyro_y"][step] = obs_post["imu_gyro"][1]
        log["imu_gyro_z"][step] = obs_post["imu_gyro"][2]
        log["imu_acc_x"][step] = obs_post["imu_acc"][0]
        log["imu_acc_y"][step] = obs_post["imu_acc"][1]
        log["imu_acc_z"][step] = obs_post["imu_acc"][2]

        if (step + 1) % (n_steps // 10) == 0:
            pct = (step + 1) / n_steps * 100
            print(f"  {pct:5.1f}% ({step+1}/{n_steps})")

    return log, model


# ============================================================================
# Save to CSV + meta
# ============================================================================


def save_trajectory(
    log: dict,
    joints: list[str],
    policy_name: str,
    duration: float,
    dt: float,
    output_dir: str | Path = "data",
) -> tuple[Path, Path]:
    """Save collected data to CSV + meta file.

    Output format is compatible with h1_2_sysid.py (real-data mode) and
    h1_2_sysid_active_spi.py.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_joint_names = [joint_to_csv_name(j) for j in joints]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "_".join(csv_joint_names) + f"_{policy_name}_{timestamp}"

    csv_path = output_dir / f"{stem}.csv"
    meta_path = output_dir / f"{stem}_meta.txt"

    # Build column order: time, then per-joint columns, then IMU
    columns = ["time"]
    for csv_jn in csv_joint_names:
        for suffix in ["q_cmd", "dq_cmd", "tau_cmd",
                       "q_meas", "dq_meas", "ddq_meas", "tau_meas"]:
            columns.append(f"{csv_jn}_{suffix}")
    for imu_col in ["imu_quat_w", "imu_quat_x", "imu_quat_y", "imu_quat_z",
                     "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
                     "imu_acc_x", "imu_acc_y", "imu_acc_z"]:
        columns.append(imu_col)

    n_steps = len(log["time"])
    arr = np.column_stack([log[c] for c in columns])

    header = ",".join(columns)
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")

    n_steps = len(log["time"])
    meta_lines = [
        f"joint_names: {csv_joint_names}",
        f"policy: {policy_name}",
        f"control_dt: {dt}",
        f"duration: {duration}",
        f"num_samples: {n_steps}",
        f"timestamp: {timestamp}",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {meta_path}")
    return csv_path, meta_path


# ============================================================================
# CLI
# ============================================================================


def load_policy(policy_spec: str) -> Policy:
    """Load a policy by name (builtin) or dotted module path.

    Examples:
        "walking"                -> WalkingPolicy()
        "my_module.MyPolicy"     -> imports my_module, instantiates MyPolicy()
    """
    if policy_spec in BUILTIN_POLICIES:
        return BUILTIN_POLICIES[policy_spec]()

    if "." in policy_spec:
        module_path, class_name = policy_spec.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        instance = cls()
        if not isinstance(instance, Policy):
            raise TypeError(
                f"{policy_spec} must be a subclass of data_collect.Policy"
            )
        return instance

    raise ValueError(
        f"Unknown policy '{policy_spec}'. "
        f"Builtins: {list(BUILTIN_POLICIES.keys())}. "
        f"Or use 'module.ClassName' for custom policies."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect H1_2 trajectory data for Active-SPI system identification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--policy", type=str, default="walking",
        help="Policy name: 'standing', 'walking', 'squatting', or "
             "'module.ClassName' for a custom policy.",
    )
    parser.add_argument(
        "--duration", type=float, default=20.0,
        help="Episode duration in seconds (default: 20).",
    )
    parser.add_argument(
        "--dt", type=float, default=0.002,
        help="Simulation timestep (default: 0.002 = 500 Hz).",
    )
    parser.add_argument(
        "--joints", nargs="+", default=None,
        help="CamelCase joint names to log (e.g. LeftKnee RightKnee). "
             "Default: all 12 leg joints.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory for CSV + meta files (default: data/).",
    )
    args = parser.parse_args()

    joints = None
    if args.joints:
        joints = [csv_name_to_joint(n) for n in args.joints]

    policy = load_policy(args.policy)
    print(f"Policy: {args.policy} ({type(policy).__name__})")
    print(f"Duration: {args.duration}s, dt: {args.dt}s")
    if joints:
        print(f"Joints: {joints}")
    else:
        print(f"Joints: all 12 leg joints")

    log, model = collect_data(
        policy=policy,
        duration=args.duration,
        dt=args.dt,
        joints=joints,
    )

    mj_joints = joints if joints else ALL_LEG_JOINTS
    save_trajectory(
        log=log,
        joints=mj_joints,
        policy_name=args.policy,
        duration=args.duration,
        dt=args.dt,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
