# Real robot data for H1_2 MuJoCo sysID (`run_mjlab_policy_sysid.py`)

This document lists what to log on the physical robot so CSV files load correctly and the Stage-1 residual (joints + IMU) is meaningful. The loader is `load_real_data()` in `run_mjlab_policy_sysid.py`.

## Purpose of the data

- **Open-loop replay**: The optimizer rolls out the MuJoCo model with your recorded **commands** as `ctrl` (position targets sent to the low-level stack), then compares predicted sensors to **measured** joint and IMU streams.
- **Fitted parameters (current script)**: Per leg joint — `kp`, `kv`, `armature`, `frictionloss`, `damping` — so the simulation matches the real plant under the same command history.

## Required signals (minimum)

Log **time** and, for **each leg joint** you want to identify (the script uses all **12** leg joints), these columns:

| Quantity   | CSV column pattern        | Role in sysID |
|-----------|---------------------------|----------------|
| Time      | `time`                    | Synchronizes all channels (seconds, monotonic). |
| Command   | `{Joint}_q_cmd`           | **Position target** in radians, same convention as MjLab / deploy (what you send to the PD or position controller). Mapped to MuJoCo `ctrl`. |
| Measured position | `{Joint}_q_meas`  | Joint angle (rad), encoder or estimated. Maps to `{base}_pos` sensors. |
| Measured velocity | `{Joint}_dq_meas` | Joint velocity (rad/s). Maps to `{base}_vel` sensors. |
| Measured torque   | `{Joint}_tau_meas`| Joint torque (N·m), estimated from current × gear ratio or measured if available. Maps to `{base}_torque` sensors. |

`{Joint}` is **CamelCase** with `_joint` dropped from the MuJoCo name, e.g. `left_hip_yaw_joint` → `LeftHipYaw`.

### Leg joints and CSV prefixes (all 12)

| MuJoCo name                 | CSV prefix `{Joint}` |
|----------------------------|----------------------|
| `left_hip_yaw_joint`       | `LeftHipYaw`         |
| `left_hip_pitch_joint`     | `LeftHipPitch`       |
| `left_hip_roll_joint`      | `LeftHipRoll`        |
| `left_knee_joint`          | `LeftKnee`           |
| `left_ankle_pitch_joint`   | `LeftAnklePitch`     |
| `left_ankle_roll_joint`    | `LeftAnkleRoll`      |
| `right_hip_yaw_joint`      | `RightHipYaw`        |
| `right_hip_pitch_joint`    | `RightHipPitch`      |
| `right_hip_roll_joint`     | `RightHipRoll`       |
| `right_knee_joint`         | `RightKnee`          |
| `right_ankle_pitch_joint`  | `RightAnklePitch`    |
| `right_ankle_roll_joint`   | `RightAnkleRoll`     |

Example column names: `LeftKnee_q_cmd`, `LeftKnee_q_meas`, `LeftKnee_dq_meas`, `LeftKnee_tau_meas`.

## Strongly recommended: IMU on the torso (base link)

The default residual includes IMU sensors (`imu_quat`, `imu_gyro`, `imu_acc`). If these CSV columns are **missing**, the loader leaves those sensor slots at zero and the fit degrades.

Include, if available from the same clock as `time`:

| Quantity        | CSV columns |
|----------------|-------------|
| Orientation    | `imu_quat_w`, `imu_quat_x`, `imu_quat_y`, `imu_quat_z` (unit quaternion; match the convention your estimator outputs — align with the MuJoCo model’s IMU frame if needed). |
| Angular rate   | `imu_gyro_x`, `imu_gyro_y`, `imu_gyro_z` (rad/s, **body frame** on the robot, consistent with how the policy / MjLab expects base gyro). |
| Linear accel   | `imu_acc_x`, `imu_acc_y`, `imu_acc_z` (m/s², same frame as modeled in MJCF). |

## Sampling rate and duration

- **Rate**: Prefer **uniform** sampling. The sysID model timestep in the script is **0.002 s (500 Hz)**; if your log is slower (e.g. 200–500 Hz), ensure timestamps are accurate — the pipeline resamples/rolls out from your `time` column.
- **Duration**: Several **multi-second** segments (e.g. 5–30 s each) beat one short clip. Diversity matters more than total seconds in a single gait.
- **Synchronization**: `q_cmd`, `q_meas`, `dq_meas`, `tau_meas`, and IMU should share a **common time base** (or document fixed known offsets if you correct offline).

## Motions to record (excitation)

Match how you will **deploy** the policy (same controller structure and command semantics). For identifiability, include **varied** behaviors, for example:

- Walking forward at several speeds  
- In-place / slow stepping  
- Sidesteps or lateral motion  
- Forward motion combined with turning  
- Brief standing or very slow weight shifts (helps separate friction / damping from dynamic terms)

Avoid only a single repetitive stride if you want stable estimates across `kp`, `kv`, armature, and friction.

## Units and conventions (must match simulation)

- **Joints**: `q` in **radians**, `dq` in **rad/s**, `tau` in **N·m**.  
- **Signs and zero**: Same definition as in your URDF / MJCF (e.g. hip pitch zero pose). A sign flip on one joint will break the fit.  
- **Command**: `q_cmd` must be the **same** quantity the policy outputs after any scaling/offset your deploy stack applies **before** the motor PD (i.e. what MjLab would call the position target for that joint).

## Optional columns (not required by current loader)

`data_collect.py` can also write `dq_cmd`, `tau_cmd`, `ddq_meas` for traceability. **`run_mjlab_policy_sysid.load_real_data` does not read those** today; they are still useful for debugging and future residuals.

## Full-body commands

The MuJoCo model has actuators for **arms and torso** as well. If your CSV only fills the 12 leg `*_q_cmd` columns, other actuators default to **0** in `ctrl` during replay. For best fidelity when the real robot holds non-zero arm poses:

- Log **`q_cmd` for every actuated joint** you care about, or  
- At minimum, log constant nominal targets for arms/torso if they are held fixed during the log.

## Meta file (documentation)

The sim collector writes `{stem}_meta.txt` with `joint_names`, `control_dt`, etc. The real-robot loader **does not parse** that file; it only reads the CSV. Keeping a small sidecar note (rate, date, firmware, estimator version) is still recommended for your records.

## Pre-flight checklist

- [ ] `time` column monotonic, same length as all series  
- [ ] All 12 leg joints: `_q_cmd`, `_q_meas`, `_dq_meas`, `_tau_meas` present  
- [ ] CamelCase names match the table above  
- [ ] IMU quaternion + gyro + accel present and time-aligned (if using default observations)  
- [ ] Units: rad, rad/s, N·m, m/s² for accel  
- [ ] Multiple clips with different velocities / turns  
- [ ] Torque estimate validated (bad `tau_meas` will distort gains and friction)

## Using the CSV in the pipeline

Set in `run_mjlab_policy_sysid.py`:

```python
REAL_DATA_CSV = "/path/to/your_log.csv"
```

Run from `python/mujoco/sysid` with the same Python environment that has MuJoCo and dependencies installed.

## Reference implementation

Simulated logs in the format expected here can be generated with `data_collect.py` (`save_trajectory`), which produces CSV headers compatible with this workflow.
