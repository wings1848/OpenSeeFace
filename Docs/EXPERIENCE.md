# 可复用经验总结

> 本文档记录本次优化过程中的关键经验和可复用模式，供后续类似项目参考。

---

## 一、Python 性能优化模式

### 1.1 避免 `str()` 作为字典键

```python
# ❌ 差：每帧生成字符串
groups[str(rect)] = group

# ✅ 好：tuple 天然可哈希
groups[tuple(rect)] = group
```

**适用场景**: 任何需要将 tuple/list 作为 dict key 的场景。`tuple()` 比 `str()` 快数倍且内存更省。

### 1.2 批量 `struct.pack` 减少调用开销

```python
# ❌ 差：18 次独立调用
buf = bytearray()
buf += struct.pack('<I', val1)
buf += struct.pack('<f', val2)
# ... ×16

# ✅ 好：合并为 5 次批量调用
buf += struct.pack('<If3f2I...', val1, val2, ...)
```

**关键**: 使用 `=` 前缀确保标准对齐，保证跨平台二进制兼容。

### 1.3 自适应频率退避

```python
# 当条件不满足时逐步降低操作频率
no_face_consecutive += 1
if no_face_consecutive > 90:
    detect_every = 10       # 极低频率
elif no_face_consecutive > 30:
    detect_every = 3        # 低频率
else:
    detect_every = 1        # 正常

if frame_count % detect_every == 0:
    run_detection()
```

**适用场景**: 任何周期性检测/计算，在"无用"状态下自动降频。

### 1.4 提前跳过昂贵计算

```python
# 按计算成本从低到高排列检查
if not enabled:              # 最便宜的检查
    return
if confidence < threshold:   # 次便宜
    use_fallback()
    return
expensive_computation()      # 昂贵的计算
```

### 1.5 GC 阈值调优（实时应用）

```python
import gc
gc.set_threshold(700, 10, 10)  # 提高 gen0 阈值
# 降低 collect 频率（从每帧 → 每 N 帧）
if frame_count % 30 == 0:
    gc.collect()
```

---

## 二、ONNX + GPU 加速模式

### 2.1 FusedConv 分解

CPU EP 优化的 `FusedConv`（Conv+激活融合）在 GPU EP 上不支持。需分解：

```python
import onnx
from onnx import helper

# 找到 FusedConv 节点 → 替换为 Conv + Activation
for node in model.graph.node:
    if node.op_type == 'FusedConv':
        # 提取 activation 参数
        activation = [a for a in node.attribute if a.name == 'activation']
        # 创建普通 Conv + 独立 Activation 节点
        ...
```

**命名约定**: `model_opt.onnx`（CPU） → `model_gpu.onnx`（GPU）

### 2.2 Provider 优先级 + 自动回退

```python
import onnxruntime as ort

# 检测可用 provider
available = ort.get_available_providers()
if 'CUDAExecutionProvider' in available:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    model_suffix = '_gpu'
else:
    providers = ['CPUExecutionProvider']
    model_suffix = '_opt'

# 所有子模型统一使用同一 provider 列表
session = ort.InferenceSession(f'model{model_suffix}.onnx', providers=providers)
```

**陷阱**: 不要用 `get_available_providers()` 作为 session 的 providers 参数 —— 在装有 TensorRT 但未配置的环境中会导致初始化错误。应显式指定 provider 列表。

### 2.3 pip CUDA 运行时安装

```bash
# onnxruntime-gpu 需要 CUDA 运行时库，可通过 pip 安装
uv pip install 'onnxruntime-gpu==1.23.2' \
    nvidia-cublas-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cufft-cu12

# 设置 LD_LIBRARY_PATH 指向 pip 安装的 .so 文件
export LD_LIBRARY_PATH="$VENV/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
# 对每个 nvidia 包的 lib 目录重复...
```

**优势**: 无需系统级 CUDA 安装，不污染全局环境。

---

## 三、Bash 脚本最佳实践

### 3.1 全局 IFS 的坑

```bash
# ❌ 危险：IFS=$'\n\t' 导致 ${array[*]} 以换行拼接
IFS=$'\n\t'
echo "${CMD_ARGS[*]}"   # 每个元素一行

# ✅ 安全：临时覆盖 IFS
FULL_CMD=$(IFS=' '; echo "${CMD_ARGS[*]}")
```

**教训**: `IFS=$'\n\t'` 是常见 strict mode 设置，但它会影响所有 `${array[*]}` 展开。需要空格拼接时务必用 subshell 临时覆盖。

### 3.2 `local` 作用域限制

```bash
# ❌ 错误：local 只能在函数内使用
if [ "$cond" = "true" ]; then
    local x=1   # 报错！
fi

# ✅ 正确：主脚本中不加 local
if [ "$cond" = "true" ]; then
    x=1         # 普通全局变量
fi
```

### 3.3 函数定义必须在调用之前

```bash
# ❌ 错误：stop handler 在函数定义之前
if [ "$1" = "stop" ]; then
    _stop_all_trackers   # 未找到
    exit 0
fi
_stop_all_trackers() { ... }

# ✅ 正确：函数定义在前，handler 在后（或在函数之后）
_stop_all_trackers() { ... }
if [ "$1" = "stop" ]; then
    _stop_all_trackers
    exit 0
fi
```

### 3.4 配置持久化模式

```bash
# 用关联数组管理变量名 + 默认值
declare -A defaults=(
    [MODEL]=3  [FACES]=1  [PORT]=11573
)
_save_config() {
    for key in "${!defaults[@]}"; do
        printf '%s="%s"\n' "$key" "${!key:-${defaults[$key]}}"
    done | sort > "$CONFIG_FILE"
}
_load_config() {
    [ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE"
}
```

### 3.5 进程清理模式

```bash
# 优雅终止 → 等待 → 强制终止
_kill_one() {
    local pid="$1"
    kill -0 "$pid" 2>/dev/null || return 0
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
}

# PID 文件 + pgrep 双层保障
_stop_all() {
    # 1. 从 PID 文件读取
    if [ -f "$PID_FILE" ]; then
        _kill_one "$(cat "$PID_FILE")"
        rm -f "$PID_FILE"
    fi
    # 2. pgrep 扫描残留
    for pid in $(pgrep -f "pattern" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        _kill_one "$pid"
    done
}
```

### 3.6 uv 虚拟环境自动管理

```bash
_ensure_venv() {
    if ! command -v uv &>/dev/null; then
        echo "请先安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    if [ ! -f ".venv/bin/activate" ]; then
        uv venv -p 3.10 .venv
    fi
}

# GPU 自动检测 → 选择对应 extra
_check_gpu() { command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; }

if _check_gpu; then
    uv pip install -e ".[gpu]"
else
    uv pip install -e ".[cpu]"
fi
```

---

## 四、pyproject.toml 模式

### 4.1 扁平项目结构配置

```toml
[tool.poetry]
name = "myproject"
packages = [
    { include = "main.py" },
    { include = "module_a.py" },
    { include = "module_b.py" },
]
```

**适用**: 非 `src/` 布局的简单项目，无需创建 `__init__.py` 包目录。

### 4.2 Optional 依赖 + Extras

```toml
[tool.poetry.dependencies]
cpu-lib = { version = ">=1.0", optional = true }
gpu-lib = { version = ">=2.0", optional = true }

[tool.poetry.extras]
cpu = ["cpu-lib"]
gpu = ["gpu-lib", "cuda-runtime"]
```

安装: `uv pip install -e ".[cpu]"` 或 `uv pip install -e ".[gpu]"`

### 4.3 版本约束选择

```toml
# ❌ 太紧：^1.21.3 锁定次版本，导致 numpy 2.x 被拒绝
numpy = "^1.21.3"

# ✅ 适度：>=1.21.3 允许大版本升级（已验证兼容）
numpy = ">=1.21.3"

# ✅ 精确锁定（仅适用于已知不兼容的情况）
cuda-lib = "*"   # 版本由 onnxruntime-gpu 决定
```

---

## 五、文档组织模式

### 5.1 README 双语策略

```
README.md      ← GitHub 默认显示（英文），顶部链接到中文版
README_zh.md   ← 完整中文版，顶部链接到英文版
```

两个文件内容对等，便于不同语言的读者。

### 5.2 变更日志结构

```
Docs/CHANGELOG.md    ← 按模块分类，记录所有改动及其效果
Docs/EXPERIENCE.md   ← 提取通用模式和踩坑经验，供后续复用
```

---

## 六、VTube Studio 集成要点

### 6.1 ip.txt 配置

```
# VTube Studio StreamingAssets/ip.txt
# 格式：key=value，每行一对
ip=127.0.0.1
port=11573
```

### 6.2 Linux Steam 路径

| 安装方式 | StreamingAssets 路径 |
|---|---|
| 原生 Steam | `~/.local/share/Steam/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets/` |
| Flatpak | `~/.var/app/com.valvesoftware.Steam/.local/share/Steam/.../` |

### 6.3 启动顺序

1. 先启动 OpenSeeFace（开始发送 UDP 数据）
2. 再启动 VTube Studio（读取 ip.txt 建立连接）
3. 如未连接，重启 VTube Studio

---

## 七、踩坑清单

| # | 问题 | 原因 | 解法 |
|---|---|---|---|
| 1 | `FusedConv` 在 GPU 上报错 | CPU EP 特有算子 | 分解为 Conv + Activation |
| 2 | TensorRT 初始化错误 | `get_available_providers()` 包含未配置的 TRT | 显式指定 provider 列表 |
| 3 | `uv pip install -e .` 报 `ModuleOrPackageNotFoundError` | 扁平项目无包目录 | 添加 `packages = [{include = "*.py"}]` |
| 4 | `IFS=$'\n\t'` 导致命令显示换行 | IFS 首字符是 `\n` | 临时 `(IFS=' '; echo "${arr[*]}")` |
| 5 | `local` 在主脚本报错 | `local` 仅函数内有效 | 主脚本用普通变量 |
| 6 | stop handler 在函数定义前 | 脚本顺序执行 | 函数定义移到 handler 之前 |
| 7 | `nvidia-cufft-cu12` 没有 12.x 版本 | pip 上最新为 11.4 | 用 `*` 通配版本约束 |
| 8 | `^` 约束导致 uv 强制降级 numpy | caret 锁定次版本 | 改为 `>=` 约束 |
