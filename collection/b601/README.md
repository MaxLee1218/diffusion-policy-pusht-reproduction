# B601-DM 棋子抓放数据采集

这套工具用一台 Mac、一台 reBot B601-DM、USB-CAN 转接器和一台相机录制示教。操作员用键盘移动末端和开合夹爪，每次完整的抓、搬、放保存为一条 LeRobot episode。另一台电脑只分析采集流程或读取文件时，无需连接机械臂。

| 文件 | 作用 |
| --- | --- |
| [b601_pick_place.py](b601_pick_place.py) | 相机读取、只读关节探测、运动学与控制循环、终端示教和数据写入 |
| [b601_teach_app.py](b601_teach_app.py) | macOS 界面、相机选择、键盘和鼠标输入、开始录制和保存 |
| [workspace-bounds.example.json](workspace-bounds.example.json) | 原采集台的 XYZ 工作区示例，单位为米 |
| [models/ReBot_Arm_DM.urdf](models/ReBot_Arm_DM.urdf) | 采集器读取关节原点、转轴、限位和末端变换的模型 |

## 在采集电脑上运行

用 Python 3.11 创建独立的采集环境。在仓库根目录执行：

```bash
python3.11 -m venv .venv-lerobot
source .venv-lerobot/bin/activate
python -m pip install -r collection/b601/requirements.txt
```

先用当前机器的串口和相机编号替换下面的示例。相机编号以本次预览为准，macOS 可能会请求摄像头权限。姿态命令读取六个关节和夹爪的反馈，并输出模型计算的末端位置。

```bash
ls /dev/cu.usbmodem*
port=/dev/cu.usbmodem00000000050C1
camera=1
python collection/b601/b601_pick_place.py camera --camera "$camera" --output /tmp/b601-preview.jpg
open /tmp/b601-preview.jpg
python collection/b601/b601_pick_place.py pose --port "$port"
```

将工作区示例复制为 `workspace-bounds.local.json` 并根据现场测量调整，顺序为 `[XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX]`，相对于机械臂底座。边界约束模型的末端参考点 `end_link`。测量时要给夹指、桌面和其他物体留出空间，并核对关节零位和夹爪方向。参见 [Seeed 的 B601-DM 上手与零位流程](https://wiki.seeedstudio.com/rebot_b601_dm_getting_started/)。

```bash
cp collection/b601/workspace-bounds.example.json collection/b601/workspace-bounds.local.json
# 编辑 workspace-bounds.local.json，填入当前工位测得的工作区。
open -n "collection/b601/B601 Teach.app" --args \
  --port "$port" --camera "$camera" \
  --bounds "$PWD/collection/b601/workspace-bounds.local.json"
```

也可以用终端采集单条示教。把六个大写占位符替换成测量值，并为每条示教指定一个新目录：

```bash
python collection/b601/b601_pick_place.py record \
  --port "$port" --camera "$camera" \
  --bounds XMIN XMAX YMIN YMAX ZMIN ZMAX \
  --output data/b601/pick-place-pilot-01
```

## 操作和保存

1. 预览画面，确认能看到棋子、蓝色目标格和夹爪。图形界面点「检查连接」读取姿态并验证工作区，通过后点「使能电机」。终端版通过检查后会要求输入 `ENABLE`。
2. 摆好棋子和机械臂起始位置，按 `R` 开始录制。用 `W/S` 沿 X 正向或负向移动，`A/D` 沿 Y 正向或负向移动，`E/C` 升降，`O/P` 打开或闭合夹爪。腕部保持会话启动时的朝向。
3. `F` 切换细调和快档。水平细调每次 2 mm，快档优先尝试 6 mm；升降分别从 4 mm 或 6 mm 起步，受限时尝试更小步长。图形界面的水平快档按住移动，松开、拖出按钮或窗口失焦时取消前方目标并保持当前位置。
4. 棋子放下后松夹，确认棋子直立、底座中心留在蓝色目标格内。按 Enter 保存，等到界面显示已保存的帧数。图形界面可直接录下一条，并通过「打开数据目录」定位文件。
5. 把机械臂停在有稳固支撑的位置后结束示教。终端版用 `Q` 或 Esc。退出会解除电机扭矩，程序不会自动回零。

## 数据怎样写入

录制循环以 10 Hz 为目标。每次循环读取相机和实测状态，处理操作员输入，更新并记录实际下发的位置目标。如果这一帧没有发送新命令，`action` 继续记录上一次已应用的目标；录制会话启动时用实测位置初始化目标。相机时间是 Mac 收到图像的时间；分析实际采样间隔时用 `timing.csv`，LeRobot 的 `timestamp` 按帧号和标称帧率生成。

| 字段或文件 | 内容 |
| --- | --- |
| `observation.images.front` | 单路 RGB 图像，采集请求 640×480，实际形状见 `meta/info.json`；图像字节保存在 Parquet 中 |
| `observation.state` | 七维 float32，六个关节和夹爪的实测电机角度 |
| `action` | 七维 float32，实际应用的位置目标，与实测状态使用同一套电机角度单位和顺序 |
| `timing.csv` | 帧号、图像收到时间、反馈读取完成时间、命令发送完成时间（单调时钟，纳秒）、输入键，以及六个关节的速度和扭矩反馈 |
| `meta/` | 数据格式、字段顺序、统计量、任务和 episode 元数据 |

每条示教单独保存在 `data/b601/pick-place-…/`。转移原始数据时保留完整目录。读取时核对 `meta/info.json`、Parquet 和 `timing.csv` 的帧数，再检查最终松夹画面。整理训练集时按整条示教划分，并排除空目录和仅用于保存起始姿态的静止采样。原始录制和本机工作区配置由仓库根目录的 `.gitignore` 排除。

## 不接机械臂的检查

下面的命令用合成图像验证六个方向的运动学求解，以及 LeRobot 数据写入和读回。每次用新的输出目录。

```bash
scratch=$(mktemp -d)
python collection/b601/b601_pick_place.py smoke --output "$scratch/episode"
```

采集器和界面迁自 `vla-chess-robot` 的 `tools/`。URDF 取自该项目的 `6f14c236` 版本，保留 [CERN-OHL-W-2.0 许可证](models/LICENSE-CERN-OHL-W-2.0.txt)。采集器只解析关节和末端运动学，URDF 中的渲染网格引用需要原项目的模型资产。
