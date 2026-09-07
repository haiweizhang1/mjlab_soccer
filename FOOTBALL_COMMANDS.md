# Football 常用指令

正式 checkpoint 与哈希见 [BASELINES.md](BASELINES.md)。除非特别说明，先进入项目目录：

```bash
cd /home/ut/football_project/mjlab_soccer
```

## 常用 Task ID

| 角色 | Task ID |
|---|---|
| 环境 smoke | `Mjlab-Velocity-Football-Flat-Unitree-G1` |
| 坐标 Teacher 基线 | `Mjlab-Velocity-Football-A1R0-LongDropout10-Envelope30-LegacyCurriculum-Flat-Unitree-G1` |
| DepthStudent 基线 | `Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeVisualDR-FrozenMLP-Distillation-Flat-Unitree-G1` |
| DepthStudent 候选 | `Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeStrongVisualDR-ConstrainedMLP-Distillation-Flat-Unitree-G1` |
| Klavier LegacyRewards Stage 1 | `Mjlab-Velocity-Football-Depth-KlavierLegacyRewardsNoise5Teacher-LegacyStage1DR-FrozenMLP-NoSym-ActionOnlyDistillation-Flat-Unitree-G1` |
| Klavier LegacyRewards Stage 2 | `Mjlab-Velocity-Football-Depth-KlavierLegacyRewardsNoise5Teacher-LegacyStage1DR-ConstrainedLastMLP-NoSym-LatentDistillation-Flat-Unitree-G1` |

## 训练

重建坐标 Teacher：

```bash
bash scripts/run_longdropout10_isaac_actor_dr_from_walk_to50k_seed42.sh
```

从正式坐标 Teacher 训练冻结 MLP 的 DepthStudent：

```bash
bash scripts/run_depth_student_mount_range_frozen_mlp_teacher_rollout_10k_seed42.sh
```

从 `model_4000.pt` 继续训练强随机化、受约束的 DepthStudent：

```bash
bash scripts/resume_depth_student_mount_range_strong_constrained_model4000_to10k_seed42.sh
```

三个脚本均使用仓库外的 `/home/ut/football_project/log_old/logs` checkpoint，并在启动前检查文件是否存在。

## Play / 导出

`play` 直接加载 `.pt` checkpoint。训练保存 checkpoint 时会在同目录导出同名 `.onnx`，sim2sim 使用 `.onnx`：

```bash
uv run play TASK_ID --checkpoint-file /ABS/PATH/TO/model_XXXX.pt
```

## Sim2sim 四种保留模式

部署保留两个独立观测入口，并统一使用 `--pd-mode implicit|explicit` 选择
PD 实现。旧参数 `--ablate-motor-pd-control True` 仍兼容，等价于
`--pd-mode explicit`。

| 观测 | `--pd-mode implicit` | `--pd-mode explicit` |
|---|---|---|
| 球位置 | MuJoCo 隐式位置 PD | XML motor + 显式力矩 PD |
| 深度图 | MuJoCo 隐式位置 PD | XML motor + 显式力矩 PD |

球位置模式必须显式使用 `--ball-observer mujoco`，否则坐标入口默认走
RoboCup RGB + YOLO 观测：

```bash
# 球位置 + 隐式位置 PD
uv run sim2sim-g1-football \
  --policy /ABS/PATH/TO/coordinate_policy.onnx \
  --ball-observer mujoco \
  --pd-mode implicit

# 球位置 + 显式力矩 PD
uv run sim2sim-g1-football \
  --policy /ABS/PATH/TO/coordinate_policy.onnx \
  --ball-observer mujoco \
  --pd-mode explicit
```

## Sim2sim 深度策略

深度 sim2sim 默认 task 是旧 `DEPTH_CANDIDATE_TASK_ID`。只适合旧 `MountRangeStrongVisualDR-ConstrainedMLP` 谱系：

```bash
# 深度图 + 隐式位置 PD
uv run sim2sim-g1-football-depth \
  --policy /ABS/PATH/TO/depth_policy.onnx \
  --pd-mode implicit

# 深度图 + 显式力矩 PD
uv run sim2sim-g1-football-depth \
  --policy /ABS/PATH/TO/depth_policy.onnx \
  --pd-mode explicit
```

Klavier LegacyRewards Stage 2 必须显式传 task id，否则会用旧 robot spec 评估新策略：

```bash
uv run sim2sim-g1-football-depth \
  --task-id Mjlab-Velocity-Football-Depth-KlavierLegacyRewardsNoise5Teacher-LegacyStage1DR-ConstrainedLastMLP-NoSym-LatentDistillation-Flat-Unitree-G1 \
  --policy /ABS/PATH/TO/policy.onnx
```

常用可选参数：

```bash
uv run sim2sim-g1-football-depth \
  --task-id TASK_ID \
  --policy /ABS/PATH/TO/policy.onnx \
  --duration 120 \
  --command-x 0.5 \
  --command-y 0.0 \
  --command-yaw 0.0
```

无窗口快速 smoke：

```bash
uv run sim2sim-g1-football-depth \
  --task-id TASK_ID \
  --policy /ABS/PATH/TO/policy.onnx \
  --headless \
  --duration 10
```

## Deploy sim2sim（MuJoCo-in-the-loop）

验证 **C++ 部署栈**（`deploy.yaml` + OrtRunner + 共享内存深度相机），与上方 Python sim2sim 是两条独立路径。

| | mjlab sim2sim | deploy sim2sim |
|---|---|---|
| 入口 | `g1_football_depth.py` | `unitree_mujoco` + `g1_ctrl` |
| 观测/动作 | 训练 env 直接构造 | `football_mjlab_depth/params/deploy.yaml` |
| 深度来源 | env 内渲染 | MuJoCo 共享内存 → `camera_depth_policy` |
| 用途 | 训练 parity | 部署代码验证 |

深度策略 **不需要** `football_vision`（无 YOLO / 球坐标）。

### 前置：编译与 ONNX

```bash
# 仿真器（只需一次）
cd /home/ut/football_project/klavier_rl_deploy-isaacsim5.1_de/simulate
cmake -S . -B build && cmake --build build -j8

# 控制器（改 deploy.yaml 或 C++ 后需重编）
cd /home/ut/football_project/klavier_rl_deploy-isaacsim5.1_de/deploy/robots/g1
cmake -S . -B build && cmake --build build -j8
```

将 stage2 Klavier ONNX 放到 deploy 策略目录（FSM `Football_MJLab` 默认加载 `policy.onnx`）：

```bash
cp /ABS/PATH/TO/stage2.onnx \
  /home/ut/football_project/klavier_rl_deploy-isaacsim5.1_de/deploy/robots/g1/config/policy/velocity/football_mjlab_depth/exported/policy.onnx
```

Klavier 策略还需确认 `football_mjlab_depth/params/deploy.yaml` 中：
- `joint_ids_map` 为 Klavier 顺序（非恒等 `[0..28]`）
- `default_joint_pos` / `scale` / `offset` 与 ONNX metadata 一致（Klavier 排列）
- `raw_clip` / `clip` / `max_delta` 显式设为 `null`（不可省略字段，否则 yaml-cpp 崩溃）
- `stiffness` / `damping` 保持 SDK motor 顺序，不要 permute

### 三终端启动（顺序固定）

`simulate/config.yaml` 已指向虚拟手柄 FIFO：`/tmp/unitree_virtual_js0`。

**终端 1 — 虚拟手柄（必须先起）：**

```bash
python3 /home/ut/football_project/klavier_rl_deploy-isaacsim5.1_de/simulate/tools/virtual_joystick.py \
  --device /tmp/unitree_virtual_js0
```

**终端 2 — MuJoCo（等终端 1 出现 “Waiting for reader” 后启动）：**

```bash
cd /home/ut/football_project/klavier_rl_deploy-isaacsim5.1_de
./simulate/build/unitree_mujoco \
  --scene unitree_model/g1_description/g1_29dof_football.xml \
  --use_camera
```

**终端 3 — 部署控制器：**

```bash
cd /home/ut/football_project/klavier_rl_deploy-isaacsim5.1_de/deploy/robots/g1
./build/g1_ctrl --network=lo --deployment-profile=sim
```

### 进入深度策略与操控

在 **终端 1**（virtual_joystick）按键：

| 键 | 动作 |
|---|---|
| `f` | Passive → FixStand |
| `g` | → **Football_MJLab**（`football_mjlab_depth` 深度策略） |
| `w` / `s` | 前 / 后 |
| `a` / `d` | 左 / 右 |
| `q` / `e` | 左转 / 右转 |
| `x` | 停（摇杆归零） |
| `p` | → Passive |

也可从 FixStand 直接 `g`（config 里 `RT+Y` 快捷路径）。真机手柄对应：`L2+Up` → FixStand，`RT+Y` → Football_MJLab。

### 常见问题

- **`Joystick open failed`**：先起 virtual_joystick，再起 unitree_mujoco。
- **`other process is using the lowcmd channel`**：已有 `g1_ctrl` 在跑，先 `pkill g1_ctrl` 再重启终端 3。
- **策略行为异常**：检查 `policy.onnx` 关节顺序是否与 `deploy.yaml` 的 `joint_ids_map` 匹配（Klavier vs leg-by-leg）。

## 配置检查

查看某个 run 的训练配置：

```bash
ls -la /ABS/PATH/TO/RUN_DIR/params
```

比较两个 run 的保存配置：

```bash
diff -u /ABS/OLD_RUN/params/agent.yaml /ABS/NEW_RUN/params/agent.yaml
diff -u /ABS/OLD_RUN/params/env.yaml /ABS/NEW_RUN/params/env.yaml
```

检查 ONNX metadata 的关节顺序、默认姿态和 action scale：

```bash
uv run python - <<'PY'
from pathlib import Path
import onnxruntime as ort

policy = Path("/ABS/PATH/TO/policy.onnx")
session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
meta = session.get_modelmeta().custom_metadata_map
for key in ("joint_names", "default_joint_pos", "action_scale", "observation_names"):
  print(f"{key}: {meta.get(key)}")
PY
```

## 检查进程

```bash
ps -ef | rg "Mjlab-Velocity-Football|g1_football_depth|sim2sim-g1-football|unitree_mujoco|g1_ctrl|virtual_joystick|wandb-core"
```

停止训练时向训练主进程发送 `SIGINT`，使 logger 有机会正常收尾：

```bash
kill -INT PID
```
