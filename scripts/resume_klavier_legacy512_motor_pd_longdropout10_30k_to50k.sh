#!/usr/bin/env bash
set -euo pipefail

repo_dir="${MJLAB_SOCCER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
train_entrypoint="$repo_dir/.venv/bin/train"
experiment_name="g1_velocity_football_klavier_legacy512_motor_pd"
source_run="${SOURCE_RUN:-2026-09-04_15-40-20_KlavierLegacy512_MotorPD_IdealPd_Envelope30_ActionAcc01_BallNoise0cm_FromWalk20k_seed42_30k_wandb}"
source_checkpoint="${SOURCE_CHECKPOINT:-model_29999.pt}"
checkpoint="$repo_dir/logs/rsl_rl/$experiment_name/$source_run/$source_checkpoint"
num_envs="${NUM_ENVS:-4096}"

task_id="Mjlab-Velocity-Football-KlavierReplica-Legacy512-MotorPD-IdealPd-NoPushCurr-LegacyRewards-LongDropout10-BallNoise0-BallTemporal-Flat-Unitree-G1"
run_name="KlavierLegacy512_MotorPD_IdealPd_LongDropout10_Envelope30_ActionAcc01_BallNoise0cm_resume30k_to50k_seed42_wandb"

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: missing MotorPD checkpoint: $checkpoint" >&2
  exit 1
fi
if [[ ! -x "$train_entrypoint" ]]; then
  echo "ERROR: missing project train entrypoint: $train_entrypoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mjlab-uv-cache}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/mjlab-warp-motorpd-longdropout10}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mjlab-mpl-motorpd-longdropout10}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/mjlab-xdg-motorpd-longdropout10}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

exec "$train_entrypoint" \
  "$task_id" \
  --env.scene.num-envs "$num_envs" \
  --agent.seed 42 \
  --agent.resume True \
  --agent.load-run "$source_run" \
  --agent.load-checkpoint "$source_checkpoint" \
  --agent.max-iterations 20000 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
