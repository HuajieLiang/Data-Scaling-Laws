# pick_cube 训练脚本

pick_cube 只保留四组物理/相机组合，每组都提供“输入实际 TCP 坐标”和“不输入实际 TCP
坐标”两个版本：

| 数据 | 1 相机（有坐标 / 无坐标） | 2 相机（有坐标 / 无坐标） |
|---|---|---|
| UMI | `pick_cube_1cam_umi.sh` / `pick_cube_1cam_umi_no_state.sh` | `pick_cube_2cam_umi.sh` / `pick_cube_2cam_umi_no_state.sh` |
| 真机 | `pick_cube_1cam_real.sh` / `pick_cube_1cam_real_no_state.sh` | `pick_cube_2cam_real.sh` / `pick_cube_2cam_real_no_state.sh` |

脚本共享 `_pick_cube_train_common.sh`，只有数据路径、来源、相机数量、采样频率和是否
启用 proprioception 不同。所有脚本都使用同一个官方 UMI 训练入口：
`train_diffusion_unet_timm_umi_workspace` + `UmiDataset`。

## 坐标和训练含义

两个 Zarr 都保存固定桌面坐标系下的绝对 TCP：

```text
+X 向镜头前方，+Y 向左，+Z 向上
robot0_eef_pos             = absolute p_D_TCP(t), metre
robot0_eef_rot_axis_angle  = absolute R_D_TCP(t), radian
robot0_gripper_width       = physical opening width, metre
```

训练配置显式保持：

```text
task.pose_repr.obs_pose_repr=relative
task.pose_repr.action_pose_repr=relative
```

这表示只在 `UmiDataset` 取样时将绝对 TCP 转换为 current-TCP-relative observation 和
future action，数据文件本身不会逐 episode 归零。`*_no_state.sh` 仅设置
`task.ignore_proprioception=true`，不改变 action、坐标转换或图像处理；有坐标版本则保留
`robot0_eef_pos`、旋转和夹爪作为策略输入。

相机字段与 Zarr 一致：USB/wrist 为 `camera0_rgb`，Top 为 `camera1_rgb`。1cam 使用
`umi` task，2cam 使用 `umi_2cam` task；后者显式启用第二相机并共享 RGB encoder。

默认仅有这些数据相关差异：UMI 使用数据统计得到的 `23.12959389342191 Hz`，真机使用
`20 Hz`；1cam 的 observation/action downsample 为 `3/3`，2cam 为 `1/1`。其余优化器、
策略、相对位姿语义和 BF16 启动方式由公共脚本统一。

## 运行和覆盖参数

从 `Data-Scaling-Laws` 根目录执行，例如：

```bash
./train_scripts/pick_cube_1cam_umi.sh
./train_scripts/pick_cube_2cam_real_no_state.sh
```

常用覆盖方式：

```bash
DATASET_PATH=data/other/dataset.zarr.zip \
PYTHON_BIN=/path/to/python \
MIXED_PRECISION=bf16 \
NUM_EPOCHS=300 \
CHECKPOINT_EVERY=10 \
CHECKPOINT_TOPK_K=100 \
./train_scripts/pick_cube_1cam_real.sh
```

pick_cube 默认每 10 个 epoch 触发一次 checkpoint，并最多保留 100 个按验证指标筛选的
top-k `.ckpt`。`latest.ckpt` 是用于续训的独立最新断点，因此目录中最多可能有 101 个
`.ckpt` 文件；若要严格限制总数为 100，可将 `CHECKPOINT_TOPK_K` 设为 `99`。

默认 `MIXED_PRECISION=bf16`，通过 `accelerate --mixed_precision bf16` 启动，适用于支持
BF16 的 GPU。若需要诊断，可显式设置 `MIXED_PRECISION=fp16` 或 `no`；脚本会拒绝其它
值。每次运行会检查数据文件存在，并生成独立的 `data/outputs/<日期>/...` 目录。
