# sorting_0915：合并 LeRobot 与转换 Zarr

这两个脚本只读取
`data/umi_data_lerobot_preprocess/sorting_0915`，不会修改或删除已经完成的
crop、二维码移除和时间同步结果。已有输出默认也不会被覆盖；只有显式加入
`--overwrite` 才会替换目标产物。

## 1. 合并为可训练的 LeRobot v3

```bash
/home/smore/miniconda3/envs/smolvla/bin/python \
  Data-Scaling-Laws/scripts/merge_sorting_0915_lerobot.py
```

默认输出：

```text
Data-Scaling-Laws/data/umi_data_lerobot_preprocess/sorting_0915_merged
```

输出是一个包含全部 episode 的 LeRobot v3 数据集，而不是 39 个彼此独立的
dataset。每个 episode 只有一张 parquet 表，列为：

- `observation.state`：16D，顺序为 left 的
  `[tx,ty,tz,qw,qx,qy,qz,gripper]`，随后是 right 的同样 8D；
- `action`：同顺序的 16D 下一帧目标；
- `timestamp`：LeRobot 要求的单时间轴，严格为 `frame_index / 60`；
- `source_timestamp`：预处理后 top/left/right 共用的原始同步时钟；
- `frame_index`、`episode_index`、全局 `index` 和 `task_index`。

原视频的帧数和画面不变。脚本只在 MP4 封装层把 H.264 packet 的 PTS/DTS
改成严格 60 Hz，使 LeRobot 默认的 `1e-4 s` 解码容差能够成立；没有视频解码、
重编码、补帧、丢帧或换序。

先做全量只读检查：

```bash
/home/smore/miniconda3/envs/smolvla/bin/python \
  Data-Scaling-Laws/scripts/merge_sorting_0915_lerobot.py --dry-run
```

只转换最前一条到临时目录：

```bash
/home/smore/miniconda3/envs/smolvla/bin/python \
  Data-Scaling-Laws/scripts/merge_sorting_0915_lerobot.py \
  --max-episodes 1 --output-root /tmp/sorting_0915_lerobot_smoke
```

## 2. 转换为三相机双臂 Zarr

```bash
LD_PRELOAD=/home/smore/miniconda3/envs/zarr-convert/lib/libstdc++.so.6 \
  /home/smore/miniconda3/envs/zarr-convert/bin/python \
  Data-Scaling-Laws/scripts/convert_sorting_0915_to_zarr.py
```

默认输出：

```text
Data-Scaling-Laws/data/dataset_umi_zarr/sorting_0915_3cam/dataset.zarr.zip
```

相机映射为：

- `camera0_rgb` = left；
- `camera1_rgb` = right；
- `camera2_rgb` = top。

图像使用已 crop/mask 且同步后的 RGB，只直接 resize 到 `224x224`，不再执行
crop、二维码检测或时间对齐。

低维字段遵循仓库 `UmiDataset` 的双臂命名：每只手都有
`eef_pos`、`eef_rot_axis_angle`、`gripper_width`、`demo_start_pose` 和
`demo_end_pose`，另有 14D `action`、原同步 `timestamp`、
`episode_frame_index` 和 `meta/episode_ends`。robot0 是 left，robot1 是 right。

原始两条手轨迹都是各自的 `first_frame_delta`。为了让双臂处于同一坐标系，
脚本使用每个 episode 元数据里的 `T_left_right`：

```text
T_common_robot0(t) = T_left_delta(t)
T_common_robot1(t) = T_left_right(frame0) @ T_right_delta(t)
```

因此 Zarr 的公共坐标系是“该 episode 的左手源首帧”，不是固定桌面绝对 TCP
坐标系。这个约定及每条 episode 实际使用的矩阵都写入 Zarr attrs 和相邻的
`dataset.zarr.zip.conversion.json`，避免后续误读。

只做检查或冒烟转换：

```bash
LD_PRELOAD=/home/smore/miniconda3/envs/zarr-convert/lib/libstdc++.so.6 \
  /home/smore/miniconda3/envs/zarr-convert/bin/python \
  Data-Scaling-Laws/scripts/convert_sorting_0915_to_zarr.py --dry-run

LD_PRELOAD=/home/smore/miniconda3/envs/zarr-convert/lib/libstdc++.so.6 \
  /home/smore/miniconda3/envs/zarr-convert/bin/python \
  Data-Scaling-Laws/scripts/convert_sorting_0915_to_zarr.py \
  --max-episodes 1 --output /tmp/sorting_0915_smoke.zarr.zip
```

## 3. 两相机版本：不使用 top

如果只需要 wrist 的 left/right 两路图像，使用独立的 2cam 脚本。它们仍然读取同一份
已完成预处理的数据，不删除原始数据，也不依赖 3cam 输出。

### 3.1 两相机 LeRobot

```bash
/home/smore/miniconda3/envs/smolvla/bin/python \
  Data-Scaling-Laws/scripts/merge_sorting_0915_lerobot_2cam.py
```

默认输出：

```text
Data-Scaling-Laws/data/umi_data_lerobot_preprocess/sorting_0915_merged_2cam
```

输出 parquet 的 `observation.state` 和 `action` 仍然是双臂 16D，定义与 3cam
一致；视频特征只包含：

- `observation.images.left`；
- `observation.images.right`。

先做只读检查：

```bash
/home/smore/miniconda3/envs/smolvla/bin/python \
  Data-Scaling-Laws/scripts/merge_sorting_0915_lerobot_2cam.py --dry-run
```

只转换最前一条到临时目录：

```bash
/home/smore/miniconda3/envs/smolvla/bin/python \
  Data-Scaling-Laws/scripts/merge_sorting_0915_lerobot_2cam.py \
  --max-episodes 1 --output-root /tmp/sorting_0915_lerobot_2cam_smoke
```

### 3.2 两相机 Zarr

```bash
LD_PRELOAD=/home/smore/miniconda3/envs/zarr-convert/lib/libstdc++.so.6 \
  /home/smore/miniconda3/envs/zarr-convert/bin/python \
  Data-Scaling-Laws/scripts/convert_sorting_0915_to_zarr_2cam.py
```

默认输出：

```text
Data-Scaling-Laws/data/dataset_umi_zarr/sorting_0915_2cam/dataset.zarr.zip
```

相机映射为：

- `camera0_rgb` = left；
- `camera1_rgb` = right。

低维轨迹、双臂公共坐标系、14D action 和 3cam Zarr 保持一致，只是不写
`camera2_rgb`。

只做检查或冒烟转换：

```bash
LD_PRELOAD=/home/smore/miniconda3/envs/zarr-convert/lib/libstdc++.so.6 \
  /home/smore/miniconda3/envs/zarr-convert/bin/python \
  Data-Scaling-Laws/scripts/convert_sorting_0915_to_zarr_2cam.py --dry-run

LD_PRELOAD=/home/smore/miniconda3/envs/zarr-convert/lib/libstdc++.so.6 \
  /home/smore/miniconda3/envs/zarr-convert/bin/python \
  Data-Scaling-Laws/scripts/convert_sorting_0915_to_zarr_2cam.py \
  --max-episodes 1 --output /tmp/sorting_0915_2cam_smoke.zarr.zip
```
