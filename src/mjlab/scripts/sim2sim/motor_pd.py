"""Shared deploy-style motor PD helpers for native MuJoCo sim2sim."""

from __future__ import annotations

from typing import Literal

import mujoco
import numpy as np
import numpy.typing as npt

PdMode = Literal["implicit", "explicit"]

# Deploy sim motor PD gains (SDK actuator order), from football_mjlab_depth/deploy.yaml.
DEPLOY_MOTOR_STIFFNESS = np.asarray(
  [
    40.1792386345,
    99.0984277767,
    40.1792386345,
    99.0984277767,
    28.5012461957,
    28.5012461957,
    40.1792386345,
    99.0984277767,
    40.1792386345,
    99.0984277767,
    28.5012461957,
    28.5012461957,
    40.1792386345,
    28.5012461957,
    28.5012461957,
    14.2506230979,
    14.2506230979,
    14.2506230979,
    14.2506230979,
    14.2506230979,
    16.7783274809,
    16.7783274809,
    14.2506230979,
    14.2506230979,
    14.2506230979,
    14.2506230979,
    14.2506230979,
    16.7783274809,
    16.7783274809,
  ],
  dtype=np.float64,
)
DEPLOY_MOTOR_DAMPING = np.asarray(
  [
    2.5578897754,
    6.3088018535,
    2.5578897754,
    6.3088018535,
    1.8144456866,
    1.8144456866,
    2.5578897754,
    6.3088018535,
    2.5578897754,
    6.3088018535,
    1.8144456866,
    1.8144456866,
    2.5578897754,
    1.8144456866,
    1.8144456866,
    0.9072228433,
    0.9072228433,
    0.9072228433,
    0.9072228433,
    0.9072228433,
    1.0681415022,
    1.0681415022,
    0.9072228433,
    0.9072228433,
    0.9072228433,
    0.9072228433,
    0.9072228433,
    1.0681415022,
    1.0681415022,
  ],
  dtype=np.float64,
)


def uses_explicit_motor_pd(
  pd_mode: PdMode, *, legacy_ablation_flag: bool = False
) -> bool:
  """Resolve the clear PD mode and the legacy ablation flag.

  ``legacy_ablation_flag`` remains supported so existing experiment commands
  continue to select deploy-style explicit torque PD.
  """
  return pd_mode == "explicit" or legacy_ablation_flag


def scale_training_position_pd_gains(
  model: mujoco.MjModel, *, kp_scale: float, kd_scale: float
) -> None:
  """Scale BuiltinPositionActuator kp/kd baked into the compiled MjModel."""
  if kp_scale == 1.0 and kd_scale == 1.0:
    return
  for act_id in range(model.nu):
    if model.actuator_biastype[act_id] != mujoco.mjtBias.mjBIAS_AFFINE:
      continue
    model.actuator_gainprm[act_id, 0] *= kp_scale
    model.actuator_biasprm[act_id, 1] *= kp_scale
    model.actuator_biasprm[act_id, 2] *= kd_scale


def apply_joint_targets(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  target: npt.NDArray[np.float32],
  actuator_ids: npt.NDArray[np.int32],
  *,
  motor_pd_control: bool,
  kp_scale: float = 1.0,
  kd_scale: float = 1.0,
) -> None:
  """Write position setpoints or deploy-style motor PD torques into ``data.ctrl``."""
  if not motor_pd_control:
    data.ctrl[actuator_ids] = target
    return
  num_motor = model.nu
  if num_motor != len(DEPLOY_MOTOR_STIFFNESS):
    raise RuntimeError(
      f"Motor PD ablation expects {len(DEPLOY_MOTOR_STIFFNESS)} motors, got {num_motor}."
    )
  if model.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE:
    raise RuntimeError(
      "Motor PD ablation requires XML <motor> actuators, but the compiled model "
      "still uses position actuators."
    )
  for policy_idx, act_id in enumerate(actuator_ids):
    sdk_motor = int(act_id)
    if sdk_motor < 0 or sdk_motor >= num_motor:
      raise RuntimeError(f"Invalid actuator id {sdk_motor} for motor PD ablation.")
    q = data.sensordata[sdk_motor]
    dq = data.sensordata[sdk_motor + num_motor]
    kp = DEPLOY_MOTOR_STIFFNESS[sdk_motor] * kp_scale
    kd = DEPLOY_MOTOR_DAMPING[sdk_motor] * kd_scale
    data.ctrl[sdk_motor] = kp * (target[policy_idx] - q) + kd * (0.0 - dq)
