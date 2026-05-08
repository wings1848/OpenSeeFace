# Linux端的vtube-studio面捕方案

[![OSF.png](Images/OSF.png)](https://github.com/emilianavt/OpenSeeFace)

> 📖 **English documentation: [README.md](README_en.md)** | 英文文档请见 [README.md](README_en.md)


> 本项目基于OpenSeeFace魔改 
> **基于 MobileNetV3 的人脸特征点检测项目**  
> 通过摄像头或视频文件进行实时面部追踪，通过 UDP 协议将数据发送给 VTube studio 等应用。

---

## 目录

- [快速开始](#快速开始)
- [VTube Studio 配置（Linux）](#vtube-studio-配置linux)
- [启动脚本](#启动脚本)
- [手动运行](#手动运行)
- [参数说明](#参数说明)
- [模型选择](#模型选择)
- [GPU 加速](#gpu-加速-cuda)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

---

## 快速开始

## VTube Studio 配置（Linux）

在 Linux 上使用 **VTube Studio** 接收 OpenSeeFace 的面部捕捉数据，需要配置 UDP 连接。
VTube Studio 通过读取 `ip.txt` 文件来知道从哪里接收追踪数据。

### 1. 确定 VTube Studio 数据目录

根据 Steam 安装方式不同，路径有所差异：

| Steam 安装方式 | StreamingAssets 路径 |
|---|---|
| **默认 (原生)** | `~/.local/share/Steam/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets/` |
| **Flatpak** | `~/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets/` |

### 2. 创建 ip.txt

`ip.txt` 使用 `key=value` 格式，每行一个参数：

```bash
# 创建目录（如果不存在）
mkdir -p "$HOME/.local/share/Steam/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets"

# 写入 UDP 配置（默认值）
cat > "$HOME/.local/share/Steam/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets/ip.txt" << 'EOF'
ip=127.0.0.1
port=11573
EOF
```

### 3. 启动顺序

```bash
# 1. 先启动 OpenSeeFace 面部追踪
./start_opensseface.sh

# 2. 再启动 VTube Studio
# VTube Studio 启动时自动读取 ip.txt，开始接收追踪数据
```

### 4. 注意事项

- `ip.txt` 使用 `key=value` 格式，`ip=` 和 `port=` 的值必须与 OpenSeeFace 的 `-i` / `-p` 参数一致
- 默认值 `127.0.0.1` / `11573`（本地回环），无需修改
- 如果 VTube Studio 在 OpenSeeFace 之后启动，可能需要**重启 VTube Studio**才能识别 `ip.txt` 的变更
- 两台机器之间传输：将追踪机器的 `ip.txt` 中 `ip=` 设为局域网 IP（如 `192.168.1.100`），并在 OpenSeeFace 启动时指定 `-i 0.0.0.0` 以允许外部连接

```bash
# ip.txt 示例（跨机器）
ip=192.168.1.100
port=11573
```

```bash
# OpenSeeFace 允许外部连接
python facetracker.py -i 0.0.0.0 -p 11573 --model 2 --try-hard 1
```

> **提示**：如果 VTube Studio 显示 "未连接"，请检查：① `ip.txt` 格式和内容是否正确（`key=value` 每行一对）；② OpenSeeFace 是否正在运行；③ 防火墙是否阻止了 UDP 端口。

---

## 启动脚本

本项目提供了一个交互式启动脚本 `start_opensseface.sh`，可以引导您配置所有参数并以后台进程方式运行。

```bash
./start_opensseface.sh            # 首次运行：交互式配置向导
./start_opensseface.sh            # 后续运行：直接使用已保存配置静默启动
./start_opensseface.sh reconfig   # 强制重新配置
./start_opensseface.sh --help     # 显示帮助
```

### 自动环境初始化

脚本**自动使用 `uv` 创建虚拟环境**（如果不存在），并安装所有必需的依赖。

依赖通过 `pyproject.toml` 中的 extras 声明：

```bash
uv pip install -e ".[cpu]"   # CPU 版（onnxruntime）
uv pip install -e ".[gpu]"   # GPU 加速版（onnxruntime-gpu + CUDA 12.x）
```

首次运行时，脚本**自动检测 NVIDIA GPU**：
- **检测到 GPU** → 安装 `.[gpu]` extra（CUDA 加速）
- **未检测到**   → 安装 `.[cpu]` extra（CPU 推理）

无需手动执行 `pip install`，直接运行脚本即可。

### 配置持久化

首次运行后，所有参数自动保存在 `.facetracker_config` 文件中。后续运行时**跳过交互式问答直接启动**，无需每次手动配置。如需更改配置，运行 `reconfig` 子命令重新进入配置向导。

脚本启动前还会自动终止所有残留的 `facetracker.py` 进程，确保每次重启干净。

### 交互式配置项

| 配置步骤 | 说明 |
|---|---|
| **运行模式** | 摄像头、视频文件、Benchmark 测试 |
| **摄像头设置** | 设备编号、分辨率、帧率、是否镜像 |
| **质量预设** | 5 种预设 + 自定义（见下方表格） |
| **多人脸设置** | 最大追踪人脸数（1~4） |
| **UDP 输出** | 发送目标 IP 和端口 |
| **日志选项** | 是否记录追踪数据和控制台输出 |
| **线程数** | 根据 CPU 核心数调整 |

### 质量预设速查

| 预设 | 模型 | 检测阈值 | 追踪阈值 | 3D自适应 | Gaze追踪 | 适用场景 |
|---|---|---|---|---|---|---|
| 🚀 **极致性能** | -1 | 0.4 | 0.6 | 关闭 | 关闭 | 低配设备/对延迟极度敏感 |
| ⚡ **快速** | 0 | 0.5 | 0.7 | 关闭 | 关闭 | 平衡速度与质量 |
| 🎯 **均衡（默认）** | 2 | 0.6 | 0.8 | 关闭 | 开启 | 大多数场景推荐 |
| ✨ **高质量** | 3 | 0.6 | 0.85 | 开启 | 开启 | 追求最佳追踪精度 |
| 😉 **眨眼优化** | 4 | 0.6 | 0.8 | 关闭 | 开启 | 对眨眼检测有特殊需求 |

### 后台管理

启动脚本使用 `nohup` 在后台运行追踪程序，并自动管理：

```bash
# 查看实时日志
tail -f .facetracker_console.log

# 停止追踪
./start_opensseface.sh stop
# 或手动: kill $(cat .facetracker.pid) 2>/dev/null || rm -f .facetracker.pid

# 检查是否运行中
ps aux | grep facetracker
```

---

## 手动运行

```bash
# 基本用法（默认摄像头 + 均衡预设）
python facetracker.py

# 完整参数示例
python facetracker.py \
  -c 0 \                         # 摄像头编号
  -F 30 \                        # 帧率
  -W 640 -H 480 \                # 分辨率
  --model 3 \                    # 模型质量
  --try-hard 1 \                 # 尽力找脸
  --gaze-tracking 1 \            # 眼球追踪
  --no-3d-adapt 0 \              # 3D自适应
  --detection-threshold 0.6 \    # 检测阈值
  --threshold 0.8 \              # 追踪阈值
  --max-threads 4 \              # 线程数
  -i 127.0.0.1 -p 11573 \        # UDP 目标
  --visualize 1 \                # 可视化显示
  --faces 1 \                    # 最大人脸数
  -M \                           # 镜像模式
  -c video.mp4 \                 # 视频文件
  --repeat-video 1               # 视频循环
```

### 可视化模式

```
--visualize 0   不显示画面（默认，节省资源）
--visualize 1   显示追踪画面
--visualize 2   显示面部 ID
--visualize 3   显示置信度数值
--visualize 4   显示地标点编号
```

---

## 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-c, --capture` | `0` | 摄像头编号或视频文件路径 |
| `-W, --width` | `640` | 画面宽度 |
| `-H, --height` | `360` | 画面高度 |
| `-F, --fps` | `24` | 帧率 |
| `-M, --mirror-input` | 关 | 镜像画面 |

### 追踪参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `3` | 模型质量 (-3~4) |
| `--detection-threshold` | `0.6` | 人脸检测阈值，越低越灵敏 |
| `--threshold` | 自动 | 追踪置信度阈值，越低越易追踪 |
| `--no-3d-adapt` | `1` | 关闭 3D 自适应（关闭更快） |
| `--try-hard` | `0` | 尽力找脸模式 |
| `--gaze-tracking` | `1` | 眼球追踪 |
| `--faces` | `1` | 最大追踪人脸数，越多人脸越慢 |
| `--scan-every` | `3` | 多人脸时每隔多少帧扫描一次 |

### 性能参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--max-threads` | `1` | 最大线程数，建议设为 CPU 核心数 |
| `--silent` | `0` | 静默模式，关闭控制台输出 |

### 输出参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-i, --ip` | `127.0.0.1` | UDP 发送地址 |
| `-p, --port` | `11573` | UDP 发送端口 |
| `--log-data` | 空 | 追踪数据日志文件 |
| `--log-output` | 空 | 控制台输出日志文件 |
| `--video-out` | 空 | 保存追踪可视化视频 |

---

## 模型选择

| 模型 | 质量 | 速度 | 说明 |
|---|---|---|---|
| **-3** | 极低 | 极快 | 介于 0 和 -1 之间 |
| **-2** | 低 | 很快 | 等效于模型 1 |
| **-1** | 很低 | 最快 | 56×56 输入，极低精度高灵敏度 |
| **0** | 低 | 快 | 适合低配设备 |
| **1** | 中 | 中 | 偏刚性，眨眼检测较弱 |
| **2** | 良好 | 中 | 推荐的平衡选择 |
| **3** (默认) | 最高 | 较慢 | 最佳追踪精度 |
| **4** | 高 | 较慢 | 眨眼检测优化 |

> **速度参考**（单核，单脸）：模型 -1 ~213 FPS，模型 3 ~44 FPS

---

## GPU 加速 (CUDA)

OpenSeeFace 支持 **NVIDIA GPU 加速**，通过 onnxruntime-gpu + CUDA 12.x 实现显著性能提升。

| 模型 | CPU FPS | **GPU FPS (GTX 1660 Ti)** | 加速比 |
|---|---|---|---|
| 3 (高质量) | 125 | **210** | **×1.68** |
| 2 | 133 | **231** | **×1.74** |
| 1 | 169 | **278** | **×1.65** |
| 0 | 142 | **288** | **×2.03** |
| -1 (极速) | 299 | **512** | **×1.71** |
| -2 | 176 | **294** | **×1.67** |
| -3 | 236 | **330** | **×1.40** |

### 自动 GPU 配置

启动脚本**自动检测 NVIDIA GPU** 并安装对应依赖：
- **检测到 GPU** → `uv pip install -e ".[gpu]"`（onnxruntime-gpu + CUDA 12.x 运行时）
- **未检测到**   → `uv pip install -e ".[cpu]"`（onnxruntime，纯 CPU）

所有 GPU 依赖在 `pyproject.toml` 中以 `[gpu]` extra 声明：
```toml
[tool.poetry.extras]
cpu = ["onnxruntime"]
gpu = ["onnxruntime-gpu", "nvidia-cublas-cu12",
       "nvidia-cuda-runtime-cu12", "nvidia-cufft-cu12"]
```

手动安装（如需要）：
```bash
uv pip install -e ".[gpu]"   # GPU 加速版
uv pip install -e ".[cpu]"   # CPU 版
```

### 工作原理

1. **自动检测**：onnxruntime-gpu 报告 `CUDAExecutionProvider` 可用性
2. **模型切换**：Tracker 自动选择 `_gpu.onnx` 模型文件（`FusedConv` 已分解为 `Conv + 激活函数`）
3. **优先级**：`CUDAExecutionProvider` → `CPUExecutionProvider`（自动回退）
4. **无 GPU 环境**：自动使用 CPU `_opt.onnx` 模型，行为不变

启动脚本 `./start_opensseface.sh` 会自动设置 CUDA 运行时路径。

## 性能优化

本项目已应用多项性能优化，详情见 [优化报告](Docs/OPTIMIZATION_REPORT.md)。

| 优化项 | 作用 |
|---|---|
| **人脸检测自适应退避** | 无人脸时自动降频，从每帧检测降到每3帧、每10帧一次 |
| **Gaze 模型按需跳过** | `--gaze-tracking 0` 时完全跳过眼球追踪计算 |
| **adjust_3d 间隔控制** | 可配置 `adjust_3d()` 执行间隔，降低 CPU 占用 |
| **低置信度跳过 3D 拟合** | 置信度 ≤ 0.3 时跳过耗时 3D 求解 |
| **UDP 批量打包** | 18 次 struct.pack 合并为 5 次，减少 Python 调用开销 |
| **GC 降频** | 从每帧 GC 改为每 30 帧 GC，减少卡顿 |
| **🚀 GPU 加速** | 集成 CUDA 12.x，**1.4x–2.0x** FPS 提升 |

所有优化保持 **API 完全向后兼容**，UDP 二进制格式逐字节一致。

---

## 常见问题

**Q: 找不到摄像头？**  
A: 检查 `-c` 参数是否正确。Linux 下可使用 `v4l2-ctl --list-devices` 查看设备列表。

**Q: 帧率太低 / CPU 占用太高？**  
A: 尝试更低质量的模型（`--model -1` 或 `--model 0`），降低帧率（`-F 20`），或增加线程数（`--max-threads 4`）。

**Q: 检测不到人脸？**  
A: 尝试开启 `--try-hard 1`，降低 `--detection-threshold`（如 0.4），或者关闭 `--no-3d-adapt 1`。

**Q: 如何测试模型性能？**  
A: 运行 `python facetracker.py --benchmark 1` 或使用启动脚本的 Benchmark 模式。

**Q: 如何在 Unity 中使用？**  
A: 参考 `Unity/` 目录下的 `OpenSee` 组件和 `Examples/` 目录中的示例。

---

## 相关链接

- [原项目 GitHub](https://github.com/emilianavt/OpenSeeFace)
- [VSeeFace (3D模型驱动)](https://www.vseeface.icu/)
- [VTube Studio (Live2D驱动)](https://denchisoft.com/)

## 许可

本项目基于 BSD 2-Clause 许可协议发布。第三方库的许可见 `Licenses/` 目录。
