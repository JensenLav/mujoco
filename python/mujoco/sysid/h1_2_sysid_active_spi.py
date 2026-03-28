"""Active-SPI Stage 1: Sampling-based Parameter Identification for H1_2.

Implements Stage 1 of the SPI-Active framework (CoRL 2025):
  1. Generate walking/squatting trajectories on a free-standing H1_2 humanoid
  2. Segment long trajectories into variable-length clips
  3. Identify joint mechanics, PD gains, and body inertia via CMA-ES
  4. Evaluate recovered parameters against ground truth

Usage:
    cd python/mujoco/sysid
    python h1_2_sysid_active_spi.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import cma
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mujoco
import mujoco.rollout as rollout
import numpy as np
from absl import logging
from mujoco import sysid
from mujoco.sysid._src import timeseries

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

logging.set_verbosity("INFO")

# ============================================================================
# Configuration
# ============================================================================

DURATION_WALK = 10.0
DURATION_SQUAT = 8.0
DT = 0.002
NOISE_STD = 0.02

CLIP_H_MIN = 200
CLIP_H_MAX = 500
N_CLIPS = 30

CMAES_POPSIZE = 32
CMAES_MAXITER = 150
CMAES_SIGMA0 = 0.3

LEG_JOINT_NAMES = [
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

INERTIA_BODIES = ["pelvis"]

# Perturbation applied to true parameters to create the "initial guess"
PARAM_PERTURBATION = {
    "armature": 0.25,
    "friction": 0.5,
    "damping": 2.0,
    "kp": 60.0,
    "kv": 8.0,
}

# ============================================================================
# Free-Standing H1_2 MJCF Model
# ============================================================================

H1_2_FREESTANDING_XML = """\
<mujoco model="h1_2_freestanding">
  <compiler angle="radian" meshdir="meshes/" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>

  <visual>
    <global offwidth="1920" offheight="1440"/>
    <rgba haze="0.7 0.78 0.88 1"/>
  </visual>

  <default>
    <joint damping="0.7" armature="0.098" frictionloss="2.0"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.88 0.94" rgb2="0.6 0.65 0.78" width="512" height="512"/>
    <mesh name="pelvis" file="pelvis.STL"/>
    <mesh name="left_hip_yaw_link" file="left_hip_yaw_link.STL"/>
    <mesh name="left_hip_pitch_link" file="left_hip_pitch_link.STL"/>
    <mesh name="left_hip_roll_link" file="left_hip_roll_link.STL"/>
    <mesh name="left_knee_link" file="left_knee_link.STL"/>
    <mesh name="left_ankle_pitch_link" file="left_ankle_pitch_link.STL"/>
    <mesh name="left_ankle_roll_link" file="left_ankle_roll_link.STL"/>
    <mesh name="right_hip_yaw_link" file="right_hip_yaw_link.STL"/>
    <mesh name="right_hip_pitch_link" file="right_hip_pitch_link.STL"/>
    <mesh name="right_hip_roll_link" file="right_hip_roll_link.STL"/>
    <mesh name="right_knee_link" file="right_knee_link.STL"/>
    <mesh name="right_ankle_pitch_link" file="right_ankle_pitch_link.STL"/>
    <mesh name="right_ankle_roll_link" file="right_ankle_roll_link.STL"/>
    <mesh name="torso_link" file="torso_link.STL"/>
    <mesh name="left_shoulder_pitch_link" file="left_shoulder_pitch_link.STL"/>
    <mesh name="left_shoulder_roll_link" file="left_shoulder_roll_link.STL"/>
    <mesh name="left_shoulder_yaw_link" file="left_shoulder_yaw_link.STL"/>
    <mesh name="left_elbow_link" file="left_elbow_link.STL"/>
    <mesh name="left_wrist_roll_link" file="left_wrist_roll_link.STL"/>
    <mesh name="left_wrist_pitch_link" file="left_wrist_pitch_link.STL"/>
    <mesh name="wrist_yaw_link" file="wrist_yaw_link.STL"/>
    <mesh name="right_shoulder_pitch_link" file="right_shoulder_pitch_link.STL"/>
    <mesh name="right_shoulder_roll_link" file="right_shoulder_roll_link.STL"/>
    <mesh name="right_shoulder_yaw_link" file="right_shoulder_yaw_link.STL"/>
    <mesh name="right_elbow_link" file="right_elbow_link.STL"/>
    <mesh name="right_wrist_roll_link" file="right_wrist_roll_link.STL"/>
    <mesh name="right_wrist_pitch_link" file="right_wrist_pitch_link.STL"/>
    <mesh name="logo_link" file="logo_link.STL"/>
  </asset>

  <worldbody>
    <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.25 0.45 0.85 1" friction="1.0 0.005 0.0001"/>
    <light name="sun" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.5 0.5 0.5" directional="true"/>
    <camera name="head_on" pos="3 0 1.3" xyaxes="0 1 0 0 0 1" fovy="45" mode="trackcom"/>
    <camera name="behind" pos="-3 0 1.3" xyaxes="0 -1 0 0 0 1"/>
    <camera name="right" pos="0 -3 1.3" xyaxes="1 0 0 0 0 1"/>
    <camera name="left" pos="0 3 1.3" xyaxes="-1 0 0 0 0 1"/>

    <body name="pelvis" pos="0 0 0.98">
      <freejoint name="pelvis_freejoint"/>
      <inertial pos="-0.0004 3.7e-05 -0.046864" quat="0.497097 0.496809 -0.503132 0.502925" mass="5.983" diaginertia="0.0531565 0.0491678 0.00902583"/>
      <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="pelvis"/>
      <geom size="0.05" rgba="0.1 0.1 0.1 1"/>

      <!-- Left leg -->
      <body name="left_hip_yaw_link" pos="0 0.0875 -0.1632">
        <inertial pos="0 -0.026197 0.006647" quat="0.704899 -0.0553755 0.0548434 0.705013" mass="2.829" diaginertia="0.00574303 0.00455361 0.00349461"/>
        <joint name="left_hip_yaw_joint" axis="0 0 1" range="-0.43 0.43" actuatorfrcrange="-200 200"/>
        <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="left_hip_yaw_link"/>
        <body name="left_hip_pitch_link" pos="0 0.0755 0">
          <inertial pos="-0.00781 -0.004724 -6.3e-05" quat="0.701575 0.711394 0.0330266 0.0249149" mass="2.92" diaginertia="0.00560661 0.00445055 0.00385068"/>
          <joint name="left_hip_pitch_joint" axis="0 1 0" range="-3.14 2.5" actuatorfrcrange="-200 200"/>
          <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="left_hip_pitch_link"/>
          <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="left_hip_pitch_link"/>
          <body name="left_hip_roll_link">
            <inertial pos="0.004171 -0.008576 -0.194509" quat="0.634842 0.0146079 0.0074063 0.772469" mass="4.962" diaginertia="0.0480229 0.0462788 0.00887409"/>
            <joint name="left_hip_roll_joint" axis="1 0 0" range="-0.43 3.14" actuatorfrcrange="-200 200"/>
            <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="left_hip_roll_link"/>
            <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="left_hip_roll_link"/>
            <body name="left_knee_link" pos="0 0 -0.4">
              <inertial pos="0.000179 0.000121 -0.168936" quat="0.416585 0.0104983 0.00514003 0.909021" mass="3.839" diaginertia="0.0391044 0.038959 0.00501125"/>
              <joint name="left_knee_joint" axis="0 1 0" range="-0.12 2.19" actuatorfrcrange="-300 300"/>
              <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="left_knee_link"/>
              <geom size="0.04 0.1" pos="0 0 -0.2" type="cylinder" rgba="0.1 0.1 0.1 1"/>
              <body name="left_ankle_pitch_link" pos="0 0 -0.4">
                <inertial pos="-0.000294 0 -0.010794" quat="0.999984 0 -0.00574445 0" mass="0.102" diaginertia="2.39454e-05 2.1837e-05 1.34126e-05"/>
                <joint name="left_ankle_pitch_joint" axis="0 1 0" range="-0.897334 0.523598" actuatorfrcrange="-60 60"/>
                <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="left_ankle_pitch_link"/>
                <body name="left_ankle_roll_link" pos="0 0 -0.02">
                  <inertial pos="0.029589 0 -0.015973" quat="0 0.725858 0 0.687845" mass="0.747" diaginertia="0.00359178 0.00343534 0.000640307"/>
                  <joint name="left_ankle_roll_joint" axis="1 0 0" range="-0.261799 0.261799" actuatorfrcrange="-40 40"/>
                  <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="left_ankle_roll_link"/>
                  <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="left_ankle_roll_link"/>
                  <geom name="left_foot_sole" type="box" size="0.11 0.06 0.01" pos="0.03 0 -0.035" rgba="0.3 0.3 0.3 1" friction="1.0 0.005 0.0001"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>

      <!-- Right leg -->
      <body name="right_hip_yaw_link" pos="0 -0.0875 -0.1632">
        <inertial pos="0 0.026197 0.006647" quat="0.705013 0.0548434 -0.0553755 0.704899" mass="2.829" diaginertia="0.00574303 0.00455361 0.00349461"/>
        <joint name="right_hip_yaw_joint" axis="0 0 1" range="-0.43 0.43" actuatorfrcrange="-200 200"/>
        <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="right_hip_yaw_link"/>
        <body name="right_hip_pitch_link" pos="0 -0.0755 0">
          <inertial pos="-0.00781 0.004724 -6.3e-05" quat="0.711394 0.701575 -0.0249149 -0.0330266" mass="2.92" diaginertia="0.00560661 0.00445055 0.00385068"/>
          <joint name="right_hip_pitch_joint" axis="0 1 0" range="-3.14 2.5" actuatorfrcrange="-200 200"/>
          <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="right_hip_pitch_link"/>
          <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="right_hip_pitch_link"/>
          <body name="right_hip_roll_link">
            <inertial pos="0.004171 0.008576 -0.194509" quat="0.772469 0.0074063 0.0146079 0.634842" mass="4.962" diaginertia="0.0480229 0.0462788 0.00887409"/>
            <joint name="right_hip_roll_joint" axis="1 0 0" range="-3.14 0.43" actuatorfrcrange="-200 200"/>
            <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="right_hip_roll_link"/>
            <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="right_hip_roll_link"/>
            <body name="right_knee_link" pos="0 0 -0.4">
              <inertial pos="0.000179 -0.000121 -0.168936" quat="0.909021 0.00514003 0.0104983 0.416585" mass="3.839" diaginertia="0.0391044 0.038959 0.00501125"/>
              <joint name="right_knee_joint" axis="0 1 0" range="-0.12 2.19" actuatorfrcrange="-300 300"/>
              <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="right_knee_link"/>
              <geom size="0.04 0.1" pos="0 0 -0.2" type="cylinder" rgba="0.1 0.1 0.1 1"/>
              <body name="right_ankle_pitch_link" pos="0 0 -0.4">
                <inertial pos="-0.000294 0 -0.010794" quat="0.999984 0 -0.00574445 0" mass="0.102" diaginertia="2.39454e-05 2.1837e-05 1.34126e-05"/>
                <joint name="right_ankle_pitch_joint" axis="0 1 0" range="-0.897334 0.523598" actuatorfrcrange="-60 60"/>
                <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="right_ankle_pitch_link"/>
                <body name="right_ankle_roll_link" pos="0 0 -0.02">
                  <inertial pos="0.029589 0 -0.015973" quat="0 0.725858 0 0.687845" mass="0.747" diaginertia="0.00359178 0.00343534 0.000640307"/>
                  <joint name="right_ankle_roll_joint" axis="1 0 0" range="-0.261799 0.261799" actuatorfrcrange="-40 40"/>
                  <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="right_ankle_roll_link"/>
                  <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="right_ankle_roll_link"/>
                  <geom name="right_foot_sole" type="box" size="0.11 0.06 0.01" pos="0.03 0 -0.035" rgba="0.3 0.3 0.3 1" friction="1.0 0.005 0.0001"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>

      <!-- Torso + arms (unchanged from original) -->
      <body name="torso_link">
        <inertial pos="0.000489 0.002797 0.20484" quat="0.999989 -0.00130808 -0.00282289 -0.00349105" mass="17.789" diaginertia="0.487315 0.409628 0.127837"/>
        <joint name="torso_joint" axis="0 0 1" range="-2.35 2.35" ref="0" actuatorfrcrange="-200 200"/>
        <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="torso_link"/>
        <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="torso_link"/>
        <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="1 1 1 1" mesh="logo_link"/>
        <site name="imu" size="0.01" pos="-0.04452 -0.01891 0.27756"/>
        <body name="left_shoulder_pitch_link" pos="0 0.14806 0.42333" quat="0.991445 0.130526 0 0">
          <inertial pos="0.003053 0.06042 -0.0059" quat="0.761799 0.645681 -0.0378496 -0.0363943" mass="1.327" diaginertia="0.000588757 0.00053309 0.000393023"/>
          <joint name="left_shoulder_pitch_joint" axis="0 1 0" range="-3.14 1.57" actuatorfrcrange="-40 40"/>
          <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="left_shoulder_pitch_link"/>
          <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="left_shoulder_pitch_link"/>
          <body name="left_shoulder_roll_link" pos="0.0342 0.061999 -0.0060011" quat="0.991445 -0.130526 0 0">
            <inertial pos="-0.030932 -1e-06 -0.10609" quat="0.986055 0.000456937 0.166408 0.00213553" mass="1.393" diaginertia="0.00200869 0.00193464 0.000449847"/>
            <joint name="left_shoulder_roll_joint" axis="1 0 0" range="-0.38 3.4" actuatorfrcrange="-40 40"/>
            <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="left_shoulder_roll_link"/>
            <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="left_shoulder_roll_link"/>
            <body name="left_shoulder_yaw_link" pos="-0.0342 0 -0.1456">
              <inertial pos="0.004583 0.001128 -0.001128" quat="0.663644 -0.0108866 -0.0267235 0.747492" mass="1.505" diaginertia="0.00431782 0.00420697 0.000645658"/>
              <joint name="left_shoulder_yaw_joint" axis="0 0 1" range="-2.66 3.01" actuatorfrcrange="-18 18"/>
              <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="left_shoulder_yaw_link"/>
              <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="left_shoulder_yaw_link"/>
              <body name="left_elbow_link" pos="0.006 0.0329 -0.182">
                <inertial pos="0.077092 -0.028751 -0.009714" quat="0.544921 0.610781 0.423352 0.388305" mass="0.691" diaginertia="0.000942091 0.000905273 0.00023025"/>
                <joint name="left_elbow_joint" axis="0 1 0" range="-0.95 3.18" actuatorfrcrange="-18 18"/>
                <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="left_elbow_link"/>
                <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="left_elbow_link"/>
                <body name="left_wrist_roll_link" pos="0.121 -0.0329 -0.011">
                  <inertial pos="0.035281 -0.00232 0.000337" quat="0.334998 0.622198 -0.240131 0.66557" mass="0.683" diaginertia="0.00034681 0.000328248 0.000294628"/>
                  <joint name="left_wrist_roll_joint" axis="1 0 0" range="-3.01 2.75" actuatorfrcrange="-19 19"/>
                  <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="left_wrist_roll_link"/>
                  <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="left_wrist_roll_link"/>
                  <body name="left_wrist_pitch_link" pos="0.087 0 0">
                    <inertial pos="0.020395 3.6e-05 -0.002973" quat="0.915893 -0.228405 -0.327262 -0.0432527" mass="0.484" diaginertia="7.25675e-05 7.00325e-05 6.9381e-05"/>
                    <joint name="left_wrist_pitch_joint" axis="0 1 0" range="-0.4625 0.4625" actuatorfrcrange="-19 19"/>
                    <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="left_wrist_pitch_link"/>
                    <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="left_wrist_pitch_link"/>
                    <body name="left_wrist_yaw_link" pos="0.02 0 0">
                      <inertial pos="0.027967 9.6e-05 0.000739" quat="0.704961 -0.0198461 -0.019614 0.708697" mass="0.124" diaginertia="0.000169999 0.000137463 8.46436e-05"/>
                      <joint name="left_wrist_yaw_joint" axis="0 0 1" range="-1.27 1.27" actuatorfrcrange="-19 19"/>
                      <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="wrist_yaw_link"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
        <body name="right_shoulder_pitch_link" pos="0 -0.14806 0.42333" quat="0.991445 -0.130526 0 0">
          <inertial pos="0.003053 -0.06042 -0.0059" quat="0.645681 0.761799 0.0363943 0.0378496" mass="1.327" diaginertia="0.000588757 0.00053309 0.000393023"/>
          <joint name="right_shoulder_pitch_joint" axis="0 1 0" range="-3.14 1.57" actuatorfrcrange="-40 40"/>
          <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="right_shoulder_pitch_link"/>
          <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="right_shoulder_pitch_link"/>
          <body name="right_shoulder_roll_link" pos="0.0342 -0.061999 -0.0060011" quat="0.991445 0.130526 0 0">
            <inertial pos="-0.030932 1e-06 -0.10609" quat="0.986055 -0.000456937 0.166408 -0.00213553" mass="1.393" diaginertia="0.00200869 0.00193464 0.000449847"/>
            <joint name="right_shoulder_roll_joint" axis="1 0 0" range="-3.4 0.38" actuatorfrcrange="-40 40"/>
            <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="right_shoulder_roll_link"/>
            <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="right_shoulder_roll_link"/>
            <body name="right_shoulder_yaw_link" pos="-0.0342 0 -0.1456">
              <inertial pos="0.004583 -0.001128 -0.001128" quat="0.747492 -0.0267235 -0.0108866 0.663644" mass="1.505" diaginertia="0.00431782 0.00420697 0.000645658"/>
              <joint name="right_shoulder_yaw_joint" axis="0 0 1" range="-3.01 2.66" actuatorfrcrange="-18 18"/>
              <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="right_shoulder_yaw_link"/>
              <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="right_shoulder_yaw_link"/>
              <body name="right_elbow_link" pos="0.006 -0.0329 -0.182">
                <inertial pos="0.077092 0.028751 -0.009714" quat="0.388305 0.423352 0.610781 0.544921" mass="0.691" diaginertia="0.000942091 0.000905273 0.00023025"/>
                <joint name="right_elbow_joint" axis="0 1 0" range="-0.95 3.18" actuatorfrcrange="-18 18"/>
                <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="right_elbow_link"/>
                <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="right_elbow_link"/>
                <body name="right_wrist_roll_link" pos="0.121 0.0329 -0.011">
                  <inertial pos="0.035281 -0.00232 0.000337" quat="0.334998 0.622198 -0.240131 0.66557" mass="0.683" diaginertia="0.00034681 0.000328248 0.000294628"/>
                  <joint name="right_wrist_roll_joint" axis="1 0 0" range="-2.75 3.01" actuatorfrcrange="-19 19"/>
                  <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="right_wrist_roll_link"/>
                  <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="right_wrist_roll_link"/>
                  <body name="right_wrist_pitch_link" pos="0.087 0 0">
                    <inertial pos="0.020395 3.6e-05 -0.002973" quat="0.915893 -0.228405 -0.327262 -0.0432527" mass="0.484" diaginertia="7.25675e-05 7.00325e-05 6.9381e-05"/>
                    <joint name="right_wrist_pitch_joint" axis="0 1 0" range="-0.4625 0.4625" actuatorfrcrange="-19 19"/>
                    <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="right_wrist_pitch_link"/>
                    <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="right_wrist_pitch_link"/>
                    <body name="right_wrist_yaw_link" pos="0.02 0 0">
                      <inertial pos="0.027967 -9.6e-05 0.000739" quat="0.708697 -0.019614 -0.0198461 0.704961" mass="0.124" diaginertia="0.000169999 0.000137463 8.46436e-05"/>
                      <joint name="right_wrist_yaw_joint" axis="0 0 1" range="-1.27 1.27" actuatorfrcrange="-19 19"/>
                      <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.1 0.1 0.1 1" mesh="wrist_yaw_link"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <sensor>
    <jointpos name="left_hip_yaw_pos" joint="left_hip_yaw_joint"/>
    <jointpos name="left_hip_pitch_pos" joint="left_hip_pitch_joint"/>
    <jointpos name="left_hip_roll_pos" joint="left_hip_roll_joint"/>
    <jointpos name="left_knee_pos" joint="left_knee_joint"/>
    <jointpos name="left_ankle_pitch_pos" joint="left_ankle_pitch_joint"/>
    <jointpos name="left_ankle_roll_pos" joint="left_ankle_roll_joint"/>
    <jointpos name="right_hip_yaw_pos" joint="right_hip_yaw_joint"/>
    <jointpos name="right_hip_pitch_pos" joint="right_hip_pitch_joint"/>
    <jointpos name="right_hip_roll_pos" joint="right_hip_roll_joint"/>
    <jointpos name="right_knee_pos" joint="right_knee_joint"/>
    <jointpos name="right_ankle_pitch_pos" joint="right_ankle_pitch_joint"/>
    <jointpos name="right_ankle_roll_pos" joint="right_ankle_roll_joint"/>
    <jointpos name="torso_joint_pos" joint="torso_joint"/>
    <jointpos name="left_shoulder_pitch_pos" joint="left_shoulder_pitch_joint"/>
    <jointpos name="left_shoulder_roll_pos" joint="left_shoulder_roll_joint"/>
    <jointpos name="left_shoulder_yaw_pos" joint="left_shoulder_yaw_joint"/>
    <jointpos name="left_elbow_pos" joint="left_elbow_joint"/>
    <jointpos name="left_wrist_roll_pos" joint="left_wrist_roll_joint"/>
    <jointpos name="left_wrist_pitch_pos" joint="left_wrist_pitch_joint"/>
    <jointpos name="left_wrist_yaw_pos" joint="left_wrist_yaw_joint"/>
    <jointpos name="right_shoulder_pitch_pos" joint="right_shoulder_pitch_joint"/>
    <jointpos name="right_shoulder_roll_pos" joint="right_shoulder_roll_joint"/>
    <jointpos name="right_shoulder_yaw_pos" joint="right_shoulder_yaw_joint"/>
    <jointpos name="right_elbow_pos" joint="right_elbow_joint"/>
    <jointpos name="right_wrist_roll_pos" joint="right_wrist_roll_joint"/>
    <jointpos name="right_wrist_pitch_pos" joint="right_wrist_pitch_joint"/>
    <jointpos name="right_wrist_yaw_pos" joint="right_wrist_yaw_joint"/>

    <jointvel name="left_hip_yaw_vel" joint="left_hip_yaw_joint"/>
    <jointvel name="left_hip_pitch_vel" joint="left_hip_pitch_joint"/>
    <jointvel name="left_hip_roll_vel" joint="left_hip_roll_joint"/>
    <jointvel name="left_knee_vel" joint="left_knee_joint"/>
    <jointvel name="left_ankle_pitch_vel" joint="left_ankle_pitch_joint"/>
    <jointvel name="left_ankle_roll_vel" joint="left_ankle_roll_joint"/>
    <jointvel name="right_hip_yaw_vel" joint="right_hip_yaw_joint"/>
    <jointvel name="right_hip_pitch_vel" joint="right_hip_pitch_joint"/>
    <jointvel name="right_hip_roll_vel" joint="right_hip_roll_joint"/>
    <jointvel name="right_knee_vel" joint="right_knee_joint"/>
    <jointvel name="right_ankle_pitch_vel" joint="right_ankle_pitch_joint"/>
    <jointvel name="right_ankle_roll_vel" joint="right_ankle_roll_joint"/>
    <jointvel name="torso_joint_vel" joint="torso_joint"/>
    <jointvel name="left_shoulder_pitch_vel" joint="left_shoulder_pitch_joint"/>
    <jointvel name="left_shoulder_roll_vel" joint="left_shoulder_roll_joint"/>
    <jointvel name="left_shoulder_yaw_vel" joint="left_shoulder_yaw_joint"/>
    <jointvel name="left_elbow_vel" joint="left_elbow_joint"/>
    <jointvel name="left_wrist_roll_vel" joint="left_wrist_roll_joint"/>
    <jointvel name="left_wrist_pitch_vel" joint="left_wrist_pitch_joint"/>
    <jointvel name="left_wrist_yaw_vel" joint="left_wrist_yaw_joint"/>
    <jointvel name="right_shoulder_pitch_vel" joint="right_shoulder_pitch_joint"/>
    <jointvel name="right_shoulder_roll_vel" joint="right_shoulder_roll_joint"/>
    <jointvel name="right_shoulder_yaw_vel" joint="right_shoulder_yaw_joint"/>
    <jointvel name="right_elbow_vel" joint="right_elbow_joint"/>
    <jointvel name="right_wrist_roll_vel" joint="right_wrist_roll_joint"/>
    <jointvel name="right_wrist_pitch_vel" joint="right_wrist_pitch_joint"/>
    <jointvel name="right_wrist_yaw_vel" joint="right_wrist_yaw_joint"/>

    <jointactuatorfrc name="left_hip_yaw_torque" joint="left_hip_yaw_joint"/>
    <jointactuatorfrc name="left_hip_pitch_torque" joint="left_hip_pitch_joint"/>
    <jointactuatorfrc name="left_hip_roll_torque" joint="left_hip_roll_joint"/>
    <jointactuatorfrc name="left_knee_torque" joint="left_knee_joint"/>
    <jointactuatorfrc name="left_ankle_pitch_torque" joint="left_ankle_pitch_joint"/>
    <jointactuatorfrc name="left_ankle_roll_torque" joint="left_ankle_roll_joint"/>
    <jointactuatorfrc name="right_hip_pitch_torque" joint="right_hip_pitch_joint"/>
    <jointactuatorfrc name="right_hip_roll_torque" joint="right_hip_roll_joint"/>
    <jointactuatorfrc name="right_hip_yaw_torque" joint="right_hip_yaw_joint"/>
    <jointactuatorfrc name="right_knee_torque" joint="right_knee_joint"/>
    <jointactuatorfrc name="right_ankle_pitch_torque" joint="right_ankle_pitch_joint"/>
    <jointactuatorfrc name="right_ankle_roll_torque" joint="right_ankle_roll_joint"/>
    <jointactuatorfrc name="torso_joint_torque" joint="torso_joint"/>
    <jointactuatorfrc name="left_shoulder_yaw_torque" joint="left_shoulder_yaw_joint"/>
    <jointactuatorfrc name="left_shoulder_pitch_torque" joint="left_shoulder_pitch_joint"/>
    <jointactuatorfrc name="left_shoulder_roll_torque" joint="left_shoulder_roll_joint"/>
    <jointactuatorfrc name="left_elbow_torque" joint="left_elbow_joint"/>
    <jointactuatorfrc name="left_wrist_roll_torque" joint="left_wrist_roll_joint"/>
    <jointactuatorfrc name="left_wrist_pitch_torque" joint="left_wrist_pitch_joint"/>
    <jointactuatorfrc name="left_wrist_yaw_torque" joint="left_wrist_yaw_joint"/>
    <jointactuatorfrc name="right_shoulder_pitch_torque" joint="right_shoulder_pitch_joint"/>
    <jointactuatorfrc name="right_shoulder_roll_torque" joint="right_shoulder_roll_joint"/>
    <jointactuatorfrc name="right_shoulder_yaw_torque" joint="right_shoulder_yaw_joint"/>
    <jointactuatorfrc name="right_elbow_torque" joint="right_elbow_joint"/>
    <jointactuatorfrc name="right_wrist_roll_torque" joint="right_wrist_roll_joint"/>
    <jointactuatorfrc name="right_wrist_pitch_torque" joint="right_wrist_pitch_joint"/>
    <jointactuatorfrc name="right_wrist_yaw_torque" joint="right_wrist_yaw_joint"/>

    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_acc" site="imu"/>
    <framepos name="frame_pos" objtype="site" objname="imu"/>
    <framelinvel name="frame_vel" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""


# ============================================================================
# Utility Functions
# ============================================================================


def generate_walking_ctrl(model, duration: float, dt: float) -> np.ndarray:
    """Generate a simple sinusoidal walking gait as position targets.

    Returns ctrl array of shape (n_steps, nu) with joint angle targets.
    """
    n_steps = int(duration / dt)
    t = np.arange(n_steps) * dt
    ctrl = np.zeros((n_steps, model.nu))
    actuator_names = [model.actuator(i).name for i in range(model.nu)]

    walk_period = 0.8
    omega = 2.0 * np.pi / walk_period

    gait = {
        "left_hip_pitch_joint":    lambda tt: -0.25 * np.sin(omega * tt),
        "left_hip_roll_joint":      lambda tt:  0.0 * np.ones_like(tt),
        "left_hip_yaw_joint":       lambda tt:  0.0 * np.ones_like(tt),
        "left_knee_joint":          lambda tt:  0.4 + 0.35 * np.maximum(0, np.sin(omega * tt)),
        "left_ankle_pitch_joint":   lambda tt:  0.12 * np.sin(omega * tt),
        "left_ankle_roll_joint":    lambda tt:  0.0 * np.ones_like(tt),
        "right_hip_pitch_joint":   lambda tt: -0.25 * np.sin(omega * tt + np.pi),
        "right_hip_roll_joint":     lambda tt:  0.0 * np.ones_like(tt),
        "right_hip_yaw_joint":      lambda tt:  0.0 * np.ones_like(tt),
        "right_knee_joint":         lambda tt:  0.4 + 0.35 * np.maximum(0, np.sin(omega * tt + np.pi)),
        "right_ankle_pitch_joint":  lambda tt:  0.12 * np.sin(omega * tt + np.pi),
        "right_ankle_roll_joint":   lambda tt:  0.0 * np.ones_like(tt),
    }

    for name, fn in gait.items():
        if name in actuator_names:
            ctrl[:, actuator_names.index(name)] = fn(t)

    return ctrl


def generate_squatting_ctrl(model, duration: float, dt: float) -> np.ndarray:
    """Generate a symmetric squatting motion as position targets."""
    n_steps = int(duration / dt)
    t = np.arange(n_steps) * dt
    ctrl = np.zeros((n_steps, model.nu))
    actuator_names = [model.actuator(i).name for i in range(model.nu)]

    squat_period = 2.0
    omega = 2.0 * np.pi / squat_period
    squat_depth = 0.5 * (1.0 - np.cos(omega * t))

    gait = {
        "left_hip_pitch_joint":    -0.3 * squat_depth,
        "left_knee_joint":          0.6 * squat_depth,
        "left_ankle_pitch_joint":   0.15 * squat_depth,
        "right_hip_pitch_joint":   -0.3 * squat_depth,
        "right_knee_joint":         0.6 * squat_depth,
        "right_ankle_pitch_joint":  0.15 * squat_depth,
    }

    for name, values in gait.items():
        if name in actuator_names:
            ctrl[:, actuator_names.index(name)] = values

    return ctrl


def segment_trajectory(
    times: np.ndarray,
    ctrl: np.ndarray,
    sensor: np.ndarray,
    state: np.ndarray,
    h_min: int,
    h_max: int,
    n_clips: int,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Segment a long trajectory into variable-length clips (SPI-style).

    Returns list of (times_clip, ctrl_clip, sensor_clip, state0) tuples.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    max_start = len(times) - h_max
    if max_start <= 0:
        max_start = max(1, len(times) - h_min)
    clips = []
    for _ in range(n_clips):
        h = rng.integers(h_min, h_max + 1)
        h = min(h, len(times) - 1)
        start = rng.integers(0, max(1, len(times) - h))
        end = min(start + h, len(times))
        t_clip = times[start:end] - times[start]
        clips.append((
            t_clip,
            ctrl[start:end],
            sensor[start:end],
            state[start],
        ))
    return clips


def simulate_trajectory(model, data, ctrl, initial_state):
    """Run a forward rollout and return (state, sensor) arrays."""
    state_out, sensor_out = rollout.rollout(
        model, data, initial_state, ctrl[:-1]
    )
    state_out = np.squeeze(state_out, axis=0)
    sensor_out = np.squeeze(sensor_out, axis=0)
    return state_out, sensor_out


def set_standing_pose(model, data):
    """Set a stable standing pose for the free-standing H1_2."""
    mujoco.mj_resetData(model, data)
    actuator_names = [model.actuator(i).name for i in range(model.nu)]

    standing = {
        "left_hip_pitch_joint": -0.1,
        "left_knee_joint": 0.25,
        "left_ankle_pitch_joint": -0.15,
        "right_hip_pitch_joint": -0.1,
        "right_knee_joint": 0.25,
        "right_ankle_pitch_joint": -0.15,
    }

    for name, val in standing.items():
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id >= 0:
            data.qpos[model.jnt_qposadr[jnt_id]] = val

    # Pelvis floating base: pos (0,0,0.98) + identity quat already set by XML
    # Settle for a few hundred steps to find stable ground contact
    for _ in range(500):
        mujoco.mj_step(model, data)


# ============================================================================
# Main Pipeline
# ============================================================================


def main():
    print("=" * 70)
    print("Active-SPI Stage 1: H1_2 System Identification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Compile free-standing model
    # ------------------------------------------------------------------
    print("\n[1/7] Compiling free-standing H1_2 model...")
    spec = mujoco.MjSpec.from_string(H1_2_FREESTANDING_XML)
    apply_actuator_config(spec, H1_2_POSITION_ACTUATORS)
    model = spec.compile()
    data = mujoco.MjData(model)
    actuator_names = [model.actuator(i).name for i in range(model.nu)]

    print(f"  Model: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    print(f"  Sensors: {model.nsensor} ({model.nsensordata} dims)")
    print(f"  Timestep: {model.opt.timestep}s")

    # ------------------------------------------------------------------
    # 2. Generate ground-truth walking + squatting trajectories
    # ------------------------------------------------------------------
    print("\n[2/7] Generating walking + squatting trajectories...")

    set_standing_pose(model, data)
    stand_state = sysid.create_initial_state(model, data.qpos, data.qvel, data.act)

    ctrl_walk = generate_walking_ctrl(model, DURATION_WALK, DT)
    ctrl_squat = generate_squatting_ctrl(model, DURATION_SQUAT, DT)

    state_walk, sensor_walk = simulate_trajectory(model, data, ctrl_walk, stand_state)
    times_walk = state_walk[:, 0]

    state_squat, sensor_squat = simulate_trajectory(model, data, ctrl_squat, stand_state)
    times_squat = state_squat[:, 0]

    rng = np.random.default_rng(seed=0)
    sensor_walk_noisy = sensor_walk + rng.normal(scale=NOISE_STD, size=sensor_walk.shape)
    sensor_squat_noisy = sensor_squat + rng.normal(scale=NOISE_STD, size=sensor_squat.shape)

    print(f"  Walking:  {len(times_walk)} steps, {DURATION_WALK}s")
    print(f"  Squatting: {len(times_squat)} steps, {DURATION_SQUAT}s")
    print(f"  Sensor noise std: {NOISE_STD}")

    # ------------------------------------------------------------------
    # 3. Segment into variable-length clips
    # ------------------------------------------------------------------
    print(f"\n[3/7] Segmenting into {N_CLIPS} clips per motion (H ~ U[{CLIP_H_MIN}, {CLIP_H_MAX}])...")

    clips_walk = segment_trajectory(
        times_walk, ctrl_walk[:len(times_walk)], sensor_walk_noisy,
        state_walk, CLIP_H_MIN, CLIP_H_MAX, N_CLIPS, rng,
    )
    clips_squat = segment_trajectory(
        times_squat, ctrl_squat[:len(times_squat)], sensor_squat_noisy,
        state_squat, CLIP_H_MIN, CLIP_H_MAX, N_CLIPS, rng,
    )
    all_clips = clips_walk + clips_squat
    clip_lengths = [len(c[0]) for c in all_clips]
    print(f"  Total clips: {len(all_clips)}")
    print(f"  Clip lengths: min={min(clip_lengths)}, max={max(clip_lengths)}, "
          f"mean={np.mean(clip_lengths):.0f}")

    # ------------------------------------------------------------------
    # 4. Build ParameterDict (all leg joints + pelvis inertia)
    # ------------------------------------------------------------------
    print("\n[4/7] Building parameter space...")

    TRUE_ARMATURE = {n: float(model.joint(n).armature) for n in LEG_JOINT_NAMES}
    TRUE_FRICTION = {n: float(model.joint(n).frictionloss) for n in LEG_JOINT_NAMES}
    TRUE_DAMPING = {n: float(model.joint(n).damping) for n in LEG_JOINT_NAMES}
    TRUE_KP, TRUE_KV = {}, {}
    for n in LEG_JOINT_NAMES:
        kp, kv = sysid.model_modifier.get_actuator_pd_gains(model, n)
        TRUE_KP[n] = kp
        TRUE_KV[n] = kv

    params = sysid.ParameterDict()

    def make_armature_modifier(jn):
        def modifier(s, p):
            s.joint(jn).armature = p.value[0]
        return modifier

    def make_friction_modifier(jn):
        def modifier(s, p):
            s.joint(jn).frictionloss = p.value[0]
        return modifier

    def make_damping_modifier(jn):
        def modifier(s, p):
            s.joint(jn).damping = p.value[0]
        return modifier

    def make_kp_modifier(act_name):
        def modifier(s, p):
            sysid.apply_pgain(s, act_name, p.value[0])
        return modifier

    def make_kv_modifier(act_name):
        def modifier(s, p):
            sysid.apply_dgain(s, act_name, p.value[0])
        return modifier

    for name in LEG_JOINT_NAMES:
        params.add(sysid.Parameter(
            f"{name}_armature",
            nominal=TRUE_ARMATURE[name], min_value=0.01, max_value=0.6,
            modifier=make_armature_modifier(name),
        ))
        params[f"{name}_armature"].value[:] = PARAM_PERTURBATION["armature"]

        params.add(sysid.Parameter(
            f"{name}_friction",
            nominal=TRUE_FRICTION[name], min_value=0.01, max_value=10.0,
            modifier=make_friction_modifier(name),
        ))
        params[f"{name}_friction"].value[:] = PARAM_PERTURBATION["friction"]

        params.add(sysid.Parameter(
            f"{name}_damping",
            nominal=TRUE_DAMPING[name], min_value=0.1, max_value=12.0,
            modifier=make_damping_modifier(name),
        ))
        params[f"{name}_damping"].value[:] = PARAM_PERTURBATION["damping"]

        params.add(sysid.Parameter(
            f"{name}_kp",
            nominal=TRUE_KP[name], min_value=30.0, max_value=500.0,
            modifier=make_kp_modifier(name),
        ))
        params[f"{name}_kp"].value[:] = PARAM_PERTURBATION["kp"]

        params.add(sysid.Parameter(
            f"{name}_kv",
            nominal=TRUE_KV[name], min_value=1.0, max_value=30.0,
            modifier=make_kv_modifier(name),
        ))
        params[f"{name}_kv"].value[:] = PARAM_PERTURBATION["kv"]

    # Body inertia (log-Cholesky / Pseudo) for pelvis
    for body_name in INERTIA_BODIES:
        inertia_param = sysid.body_inertia_param(
            spec, model, body_name,
            inertia_type=sysid.InertiaType.Pseudo,
            scale_rot_inertia=True,
            mass_bound_mult=np.array([0.5, 2.0]),
            ipos_bound_off=np.array([-0.05, 0.05]),
        )
        params.add(inertia_param)

    n_free = sum(p.value.size for p in params.parameters.values() if not p.frozen)
    print(f"  Parameters: {len(params)} entries, {n_free} free DOFs")
    print(f"  Leg joints: {len(LEG_JOINT_NAMES)} x 5 (arm, fric, damp, kp, kv) = "
          f"{len(LEG_JOINT_NAMES) * 5}")
    print(f"  Body inertia: {INERTIA_BODIES} (Pseudo/log-Cholesky)")

    # ------------------------------------------------------------------
    # 5. Build ModelSequences with clips + IMU observations
    # ------------------------------------------------------------------
    print("\n[5/7] Building residual function with IMU observations...")

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
    print(f"  Enabled observations: {len(enabled_obs)} sensor groups")
    print(f"  Includes: joint pos/vel/torque + IMU quat/gyro/acc")

    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)

    model_sequences = []
    for i, (t_clip, c_clip, s_clip, s0_full) in enumerate(all_clips):
        qpos0 = s0_full[1:1 + model.nq]
        qvel0 = s0_full[1 + model.nq:1 + model.nq + model.nv]
        nact = model.na
        act0 = s0_full[1 + model.nq + model.nv:1 + model.nq + model.nv + nact] if nact > 0 else np.zeros(0)
        init_state = sysid.create_initial_state(model, qpos0, qvel0, act0)

        ctrl_ts = sysid.TimeSeries(t_clip, c_clip)
        sensor_ts = sysid.TimeSeries.from_names(t_clip, s_clip, model)

        motion_type = "walk" if i < N_CLIPS else "squat"
        ms = sysid.ModelSequences(
            f"clip_{motion_type}_{i}",
            spec, f"clip_{i}", init_state, ctrl_ts, sensor_ts,
        )
        model_sequences.append(ms)

    residual_fn = sysid.build_residual_fn(
        models_sequences=model_sequences,
        enabled_observations=enabled_obs,
    )
    print(f"  Built residual over {len(model_sequences)} clips")

    # ------------------------------------------------------------------
    # 6. CMA-ES Optimization (SPI Stage 1)
    # ------------------------------------------------------------------
    print(f"\n[6/7] Running CMA-ES optimization (pop={CMAES_POPSIZE}, "
          f"maxiter={CMAES_MAXITER})...")

    x0 = params.as_vector()
    lb, ub = params.get_bounds()
    lb = lb.tolist()
    ub = ub.tolist()

    def spi_cost(x):
        p_copy = params.copy()
        p_copy.update_from_vector(x)
        res, _, _ = residual_fn(x, p_copy)
        return sum(float(np.sum(r**2)) for r in res)

    opts = cma.CMAOptions()
    opts["bounds"] = [lb, ub]
    opts["popsize"] = CMAES_POPSIZE
    opts["maxiter"] = CMAES_MAXITER
    opts["verbose"] = 1
    opts["tolfun"] = 1e-8
    opts["tolx"] = 1e-10

    t_start = time.time()
    es = cma.CMAEvolutionStrategy(x0.tolist(), CMAES_SIGMA0, opts)

    iteration = 0
    while not es.stop():
        candidates = es.ask()
        costs = [spi_cost(np.array(c)) for c in candidates]
        es.tell(candidates, costs)
        if iteration % 10 == 0:
            print(f"  iter {iteration:4d} | best cost: {es.best.f:.6f} | "
                  f"mean cost: {np.mean(costs):.6f}")
        iteration += 1

    elapsed = time.time() - t_start
    print(f"\n  CMA-ES finished in {elapsed:.1f}s ({iteration} iterations)")
    print(f"  Final best cost: {es.best.f:.6f}")

    opt_params = params.copy()
    opt_params.update_from_vector(np.array(es.best.x))

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    print("\n[7/7] Evaluating results...")
    print(f"\n{'Parameter':<40s} {'True':>10s} {'Initial':>10s} {'Optimized':>10s} {'Error%':>10s}")
    print("-" * 80)

    for name in LEG_JOINT_NAMES:
        for suffix, true_dict, init_val in [
            ("armature", TRUE_ARMATURE, PARAM_PERTURBATION["armature"]),
            ("friction", TRUE_FRICTION, PARAM_PERTURBATION["friction"]),
            ("damping", TRUE_DAMPING, PARAM_PERTURBATION["damping"]),
            ("kp", TRUE_KP, PARAM_PERTURBATION["kp"]),
            ("kv", TRUE_KV, PARAM_PERTURBATION["kv"]),
        ]:
            key = f"{name}_{suffix}"
            true_v = true_dict[name]
            opt_v = float(opt_params[key].value[0])
            err_pct = abs(opt_v - true_v) / max(abs(true_v), 1e-8) * 100
            print(f"  {key:<38s} {true_v:10.4f} {init_val:10.4f} {opt_v:10.4f} {err_pct:9.1f}%")

    # Summary statistics
    errors = []
    for name in LEG_JOINT_NAMES:
        for suffix, true_dict in [
            ("armature", TRUE_ARMATURE), ("friction", TRUE_FRICTION),
            ("damping", TRUE_DAMPING), ("kp", TRUE_KP), ("kv", TRUE_KV),
        ]:
            key = f"{name}_{suffix}"
            true_v = true_dict[name]
            opt_v = float(opt_params[key].value[0])
            errors.append(abs(opt_v - true_v) / max(abs(true_v), 1e-8) * 100)

    print(f"\n  Mean parameter error: {np.mean(errors):.1f}%")
    print(f"  Median parameter error: {np.median(errors):.1f}%")
    print(f"  Max parameter error: {np.max(errors):.1f}%")

    # ------------------------------------------------------------------
    # Cost landscape visualization (first joint)
    # ------------------------------------------------------------------
    joint_0 = LEG_JOINT_NAMES[0]
    print(f"\n  Plotting cost landscapes for {joint_0}...")

    name_arm = f"{joint_0}_armature"
    name_fric = f"{joint_0}_friction"
    name_kp = f"{joint_0}_kp"
    name_kv = f"{joint_0}_kv"

    armature_grid = np.linspace(0.0, 0.8, 8)
    friction_grid = np.linspace(0.0, 12.0, 8)
    kp_grid = np.linspace(0.0, 600.0, 8)
    kv_grid = np.linspace(0.0, 35.0, 8)

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

    cost_af = eval_grid(name_arm, armature_grid, name_fric, friction_grid)
    cost_kpkv = eval_grid(name_kp, kp_grid, name_kv, kv_grid)

    bnds = dict(arm=(0.01, 0.6), fric=(0.01, 10.0), kp=(30.0, 500.0), kv=(1.0, 30.0))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    panels = [
        (armature_grid, friction_grid, cost_af, "Armature", "Friction",
         TRUE_ARMATURE[joint_0], TRUE_FRICTION[joint_0],
         PARAM_PERTURBATION["armature"], PARAM_PERTURBATION["friction"],
         float(opt_params[name_arm].value[0]), float(opt_params[name_fric].value[0]),
         *bnds["arm"], *bnds["fric"]),
        (kp_grid, kv_grid, cost_kpkv, "kp (stiffness)", "kv (damping)",
         TRUE_KP[joint_0], TRUE_KV[joint_0],
         PARAM_PERTURBATION["kp"], PARAM_PERTURBATION["kv"],
         float(opt_params[name_kp].value[0]), float(opt_params[name_kv].value[0]),
         *bnds["kp"], *bnds["kv"]),
    ]

    for ax, (gx, gy, cost, xlabel, ylabel, tx, ty, ix, iy, ox, oy,
             bx_lo, bx_hi, by_lo, by_hi) in zip(axes, panels):
        X, Y = np.meshgrid(gx, gy)
        log_cost = np.log10(cost + 1e-12)
        levels = np.linspace(log_cost.min(), log_cost.max(), 30)
        cf = ax.contourf(X, Y, log_cost.T, levels=levels, cmap="viridis")
        ax.plot(tx, ty, "r*", markersize=15, label="True", zorder=5)
        ax.plot(ix, iy, "g*", markersize=15, label="Initial", zorder=5)
        ax.plot(ox, oy, marker="X", color="gold", markeredgecolor="k",
                markeredgewidth=1, markersize=12, linestyle="none",
                label="Optimized", zorder=5)
        rect = patches.Rectangle(
            (bx_lo, by_lo), bx_hi - bx_lo, by_hi - by_lo,
            linewidth=2, edgecolor="black", facecolor="none",
            linestyle="--", label="Param bounds", zorder=6,
        )
        ax.add_patch(rect)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)
        fig.colorbar(cf, ax=ax, label=r"$\log_{10}$(cost)", shrink=0.9)
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(f"SPI Stage 1 — Pairwise cost landscapes for {joint_0}", fontsize=14)
    plt.savefig("spi_stage1_cost_landscape.png", dpi=150, bbox_inches="tight")
    print("  Saved: spi_stage1_cost_landscape.png")
    plt.show()

    print("\n" + "=" * 70)
    print("Active-SPI Stage 1 complete.")
    print("=" * 70)

    return opt_params, es


if __name__ == "__main__":
    opt_params, es = main()
