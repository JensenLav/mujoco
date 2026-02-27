# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MjLab-style actuator config: define actuators in code and add them to MjSpec.

This module provides position actuator configuration and helpers so the model
and actuators can be built programmatically (like MjLab's BuiltinPositionActuator
and create_position_actuator), instead of encoding actuators only in XML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco


@dataclass(frozen=True)
class PositionActuatorCfg:
  """Configuration for a single MuJoCo built-in position actuator.

  Under the hood this creates a <position> actuator: force = kp*(ctrl - q) - kv*qvel,
  and sets the target joint's armature and frictionloss (MjLab-style).
  Mirrors MjLab's create_position_actuator helper for joint transmissions.
  """

  stiffness: float
  """PD proportional gain (kp)."""
  damping: float
  """PD derivative gain (kv)."""
  effort_limit: float | None = None
  """Maximum actuator force/torque (symmetric ±limit). If None, no limit."""
  armature: float = 0.098
  """Target joint armature (reflected inertia). Set on the joint like MjLab."""
  frictionloss: float = 2.0
  """Target joint friction loss. Set on the joint like MjLab."""


def create_position_actuator(
    spec: mujoco.MjSpec,
    joint_name: str,
    cfg: PositionActuatorCfg,
    name: str | None = None,
) -> mujoco.MjsActuator:
  """Add one position actuator and set target joint properties (MjLab-style).

  Creates a position actuator (force = kp*(ctrl-q) - kv*qvel) and sets the
  target joint's armature and frictionloss so behavior matches MjLab's
  create_position_actuator for TransmissionType.JOINT.
  """
  # Set target joint properties first (like MjLab's create_position_actuator).
  joint = spec.joint(joint_name)
  if joint is not None:
    joint.armature = cfg.armature
    joint.frictionloss = cfg.frictionloss

  act_name = name if name is not None else joint_name
  actuator = spec.add_actuator(name=act_name, target=joint_name)

  # Match MjLab's create_position_actuator actuator settings.
  actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE

  # Set stiffness and damping (force = kp*(ctrl - q) - kv*qvel).
  actuator.gainprm[0] = cfg.stiffness
  actuator.biasprm[1] = -cfg.stiffness
  actuator.biasprm[2] = -cfg.damping

  # Do NOT limit ctrl (MjLab keeps ctrllimited False for position actuators).
  actuator.ctrllimited = False

  # Effort limits.
  if cfg.effort_limit is not None:
    actuator.forcelimited = True
    actuator.forcerange = [-float(cfg.effort_limit), float(cfg.effort_limit)]
  else:
    actuator.forcelimited = False

  return actuator


def apply_actuator_config(
    spec: mujoco.MjSpec,
    config: Sequence[tuple[str, PositionActuatorCfg]],
) -> None:
  """Clear existing actuators and add position actuators from config.

  Config is a sequence of (joint_name, PositionActuatorCfg). Actuator names
  are set to joint_name so downstream code that indexes by actuator name
  continues to work.
  """
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for joint_name, cfg in config:
    create_position_actuator(spec, joint_name, cfg, name=joint_name)


# -----------------------------------------------------------------------------
# H1 humanoid position actuator config (legs kp=100 kv=3, torso 80/3, arms 60/3)
# -----------------------------------------------------------------------------

H1_2_POSITION_ACTUATORS: list[tuple[str, PositionActuatorCfg]] = [
    ("left_hip_yaw_joint", PositionActuatorCfg(100, 3, 200)),
    ("left_hip_pitch_joint", PositionActuatorCfg(100, 3, 200)),
    ("left_hip_roll_joint", PositionActuatorCfg(100, 3, 200)),
    ("left_knee_joint", PositionActuatorCfg(100, 3, 300)),
    ("left_ankle_pitch_joint", PositionActuatorCfg(100, 3, 60)),
    ("left_ankle_roll_joint", PositionActuatorCfg(100, 3, 40)),
    ("right_hip_yaw_joint", PositionActuatorCfg(100, 3, 200)),
    ("right_hip_pitch_joint", PositionActuatorCfg(100, 3, 200)),
    ("right_hip_roll_joint", PositionActuatorCfg(100, 3, 200)),
    ("right_knee_joint", PositionActuatorCfg(100, 3, 300)),
    ("right_ankle_pitch_joint", PositionActuatorCfg(100, 3, 60)),
    ("right_ankle_roll_joint", PositionActuatorCfg(100, 3, 40)),
    ("torso_joint", PositionActuatorCfg(80, 3, 200)),
    ("left_shoulder_pitch_joint", PositionActuatorCfg(60, 3, 40)),
    ("left_shoulder_roll_joint", PositionActuatorCfg(60, 3, 40)),
    ("left_shoulder_yaw_joint", PositionActuatorCfg(60, 3, 18)),
    ("left_elbow_joint", PositionActuatorCfg(60, 3, 18)),
    ("left_wrist_roll_joint", PositionActuatorCfg(60, 3, 19)),
    ("left_wrist_pitch_joint", PositionActuatorCfg(60, 3, 19)),
    ("left_wrist_yaw_joint", PositionActuatorCfg(60, 3, 19)),
    ("right_shoulder_pitch_joint", PositionActuatorCfg(60, 3, 40)),
    ("right_shoulder_roll_joint", PositionActuatorCfg(60, 3, 40)),
    ("right_shoulder_yaw_joint", PositionActuatorCfg(60, 3, 18)),
    ("right_elbow_joint", PositionActuatorCfg(60, 3, 18)),
    ("right_wrist_roll_joint", PositionActuatorCfg(60, 3, 19)),
    ("right_wrist_pitch_joint", PositionActuatorCfg(60, 3, 19)),
    ("right_wrist_yaw_joint", PositionActuatorCfg(60, 3, 19)),
]
