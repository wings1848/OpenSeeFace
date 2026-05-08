#!/usr/bin/env bash
# =============================================================================
# OpenSeeFace 后台启动脚本
# 交互式配置面捕质量参数，以守护进程方式运行 facetracker.py
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# ---------- 帮助信息 ----------
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "用法: $(basename "$0") [reconfig|stop]"
    echo ""
    echo "  (无参数)  使用已保存的配置直接启动（有配置时跳过交互式问答）"
    echo "  reconfig  强制进入交互式配置向导，覆盖已保存的配置"
    echo "  stop       安全停止后台运行的 facetracker 并清理 PID 文件"
    echo "  -h, --help  显示本帮助"
    echo ""
    echo "配置保存在: $(cd "$(dirname "$0")" && pwd)/.facetracker_config"
    echo "首次运行时会自动进入交互式向导并保存配置。"
    exit 0
fi

# ---------- 路径 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"
FACETRACKER="$SCRIPT_DIR/facetracker.py"

# ---------- CUDA 加速环境 ----------
# 自动设置 LD_LIBRARY_PATH 以加载 CUDA 12.x 运行时库
# (在虚拟环境激活后调用)
_set_cuda_env() {
    local venv_site
    venv_site="$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "$SCRIPT_DIR/.venv/lib/python3.10/site-packages")"
    for _cd in "$venv_site/nvidia/cublas/lib" "$venv_site/nvidia/cuda_runtime/lib" \
               "$venv_site/nvidia/cufft/lib" "$venv_site/nvidia/cuda_nvrtc/lib"; do
        if [ -d "$_cd" ]; then
            export LD_LIBRARY_PATH="$_cd${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
    done
    # 抑制 TensorRT 未安装的错误提示
    export ORT_TENSORRT_UNAVAILABLE=1
}

# ---------- 颜色输出 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }
header(){ echo -e "\n${BOLD}━━━ $* ━━━${NC}"; }

# ---------- 公共：终止单个进程（优雅 → 强制）----------
_kill_one() {
    local pid="$1"
    kill -0 "$pid" 2>/dev/null || return 0
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -0 "$pid" 2>/dev/null || return 0
    kill -9 "$pid" 2>/dev/null || true
    sleep 0.5
}

# ---------- 公共：通过 PID 文件 + pgrep 扫描终止所有 tracker ----------
_stop_all_trackers() {
    local pid_file="${1:-$SCRIPT_DIR/.facetracker.pid}"
    local killed=0

    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null)
        if [ -n "$pid" ]; then
            if kill -0 "$pid" 2>/dev/null; then
                echo "正在停止 facetracker (PID: $pid)..."
                _kill_one "$pid"
                if kill -0 "$pid" 2>/dev/null; then
                    echo "✗ 无法终止进程 $pid"
                else
                    echo "✓ 已停止"
                    killed=1
                fi
            else
                echo "PID 文件存在但进程已退出，清理文件"
            fi
        fi
        rm -f "$pid_file"
    fi

    local pids
    pids=$(pgrep -f "facetracker.py" 2>/dev/null || true)
    for pid in $pids; do
        [ "$pid" = "$$" ] && continue
        kill -0 "$pid" 2>/dev/null || continue
        echo "正在停止 facetracker (PID: $pid)..."
        _kill_one "$pid"
        echo "✓ 已停止"
        killed=1
    done

    return "$killed"
}

# ---------- 停止已运行的 facetracker ----------
if [ "${1:-}" = "stop" ] || [ "${1:-}" = "--stop" ]; then
    _stop_all_trackers "$SCRIPT_DIR/.facetracker.pid" || true
    rm -f "$SCRIPT_DIR/.facetracker_console.log"
    exit 0
fi

# ---------- 自动创建/激活虚拟环境 ----------
_ensure_venv() {
    # 1. 检查 uv 是否可用
    if ! command -v uv &>/dev/null; then
        err "未找到 uv 命令"
        err "请先安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        err "或手动创建虚拟环境: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi

    # 2. 如果 venv 不存在则创建
    if [ ! -f "$VENV_ACTIVATE" ]; then
        warn "未找到虚拟环境，正在使用 uv 创建..."
        uv venv -p 3.10 "$SCRIPT_DIR/.venv" 2>&1 | sed 's/^/  /'
        if [ ! -f "$VENV_ACTIVATE" ]; then
            err "虚拟环境创建失败"
            exit 1
        fi
        ok "虚拟环境已创建: $SCRIPT_DIR/.venv"
    fi

    # 3. 始终使用 POSIX 兼容的 activate（脚本由 bash shebang 执行）
    VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"
    export VENV_ACTIVATE
}

_ensure_venv
source "$VENV_ACTIVATE"
ok "虚拟环境已激活 (Python $(python3 --version | cut -d' ' -f2))"

# ---------- 自动安装依赖 ----------
# 检测 NVIDIA GPU 是否可用
_check_gpu_available() {
    command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null 2>&1
}

# 检测核心依赖是否可导入（不区分 CPU/GPU 版 onnxruntime）
_check_imports() {
    python3 -c "import numpy, cv2, PIL.Image, onnxruntime" 2>/dev/null
}

_ensure_deps() {
    if _check_imports; then
        return 0
    fi

    warn "依赖尚未安装，正在使用 uv 自动安装..."

    local extra
    if _check_gpu_available; then
        ok "检测到 NVIDIA GPU，将安装 CUDA 加速依赖..."
        extra="gpu"
    else
        extra="cpu"
    fi

    uv pip install -e "$SCRIPT_DIR[$extra]" 2>&1 | sed 's/^/  /'

    # GPU 模式下设置 CUDA 库路径
    [ "$extra" = "gpu" ] && _set_cuda_env

    if _check_imports; then
        ok "依赖安装完成"
    else
        warn "部分依赖安装可能不完整，尝试强制重装..."
        uv pip install -e "$SCRIPT_DIR[$extra]" --reinstall 2>&1 | sed 's/^/  /' || true
        if _check_imports; then
            ok "依赖安装完成"
        else
            warn "仍有依赖缺失，请手动执行: uv pip install -e '.[$extra]'"
        fi
    fi
}
_ensure_deps

# ---------- 检查 facetracker.py ----------
if [ ! -f "$FACETRACKER" ]; then
    err "找不到 $FACETRACKER"
    err "请确保脚本放置在 OpenSeeFace 项目根目录下运行"
    exit 1
fi

# ---------- 配置 CUDA 加速（在 venv 激活后）----------
_set_cuda_env
# 缓存 GPU 检测结果
GPU_LABEL="$(python3 -c "import onnxruntime; print('GPU' if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else 'CPU')" 2>/dev/null || echo "CPU")"
if [ "$GPU_LABEL" = "GPU" ]; then
    ok "CUDA 加速已就绪 (GPU 可用)"
else
    info "使用 CPU 推理 (如需 GPU 加速请安装 onnxruntime-gpu)"
fi

CONFIG_FILE="$SCRIPT_DIR/.facetracker_config"

# ---------- 配置持久化 ----------
_save_config() {
    # 变量名 → 默认值的映射
    local -A defaults=(
        [RUN_MODE]=1           [CAM_ID]=0        [CAM_FPS]=24
        [CAM_W]=640            [CAM_H]=480        [MIRROR]=0
        [VIDEO_PATH]=""        [QUALITY_PRESET]=3  [MODEL]=3
        [DETECTION_THRESHOLD]=0.6  [THRESHOLD]=0.8  [NO_3D_ADAPT]=1
        [TRY_HARD]=1           [GAZE]=1           [HEADLESS]="均衡"
        [FACES]=1              [UDP_IP]="127.0.0.1"  [UDP_PORT]=11573
        [LOG_DATA]=""          [LOG_OUTPUT]=""    [MAX_THREADS]=4
        [VISUALIZE]=0
    )
    {
        echo "# OpenSeeFace 启动配置（自动生成，请勿手动编辑）"
        for key in "${!defaults[@]}"; do
            printf '%s="%s"\n' "$key" "${!key:-${defaults[$key]}}"
        done | sort
    } > "$CONFIG_FILE"
    ok "配置已保存到 $CONFIG_FILE"
}

_load_config() {
    if [ -f "$CONFIG_FILE" ]; then
        source "$CONFIG_FILE"
        return 0
    fi
    return 1
}

# ---------- 检测可用摄像头（一趟扫描，同时输出列表和计数）----------
# 用法: detect_cameras  → 设置全局 CAM_LIST 和 CAM_COUNT
detect_cameras() {
    CAM_LIST=()
    for dev in /dev/video*; do
        [ -e "$dev" ] || continue
        CAM_LIST+=("$dev")
    done
    CAM_COUNT="${#CAM_LIST[@]}"
}

# =============================================================================
# 配置加载 / 交互式配置
# =============================================================================

# 尝试加载已保存的配置
if [ "${1:-}" = "reconfig" ]; then
    warn "强制重新配置模式（将覆盖已保存配置）"
    rm -f "$CONFIG_FILE"
    goto_build=false
elif _load_config; then
    # 已有配置 → 静默启动
    ok "已加载配置，跳过交互式询问（如需重新配置请运行: $0 reconfig）"
    goto_build=true
else
    goto_build=false
fi

if [ "$goto_build" = "false" ]; then
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║       OpenSeeFace 面捕质量配置向导                  ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ---------- 1. 运行模式 ----------
header "运行模式"

echo "请选择输入源:"
echo "  1) 摄像头 (默认)"
echo "  2) 视频文件"
echo "  3) Benchmark 模式 (测试各模型性能)"
read -p "请输入 [1-3, 默认 1]: " RUN_MODE
RUN_MODE="${RUN_MODE:-1}"

if [ "$RUN_MODE" = "3" ]; then
    header "Benchmark 模式"
    echo "将依次测试所有 7 个模型并报告 FPS。"
    echo "可选择单线程或多线程运行:"
    read -p "最大线程数 [默认 4]: " BM_THREADS
    BM_THREADS="${BM_THREADS:-4}"
    read -p "是否静默输出 (0=显示详细, 1=只显示FPS) [默认 1]: " BM_SILENT
    BM_SILENT="${BM_SILENT:-1}"

    header "启动 Benchmark"
    CMD=("python3" "$FACETRACKER" "--benchmark" "1" "--max-threads" "$BM_THREADS" "--silent" "$BM_SILENT")
    echo -e "${YELLOW}执行: ${CMD[*]}${NC}"
    echo ""

    "${CMD[@]}"
    exit 0
fi

# ---------- 2. 视频/摄像头源 ----------
if [ "$RUN_MODE" = "2" ]; then
    header "视频文件"
    read -p "请输入视频文件路径: " VIDEO_PATH
    while [ ! -f "$VIDEO_PATH" ]; do
        err "文件不存在: $VIDEO_PATH"
        read -p "请重新输入 (或输入 q 退出): " VIDEO_PATH
        [ "$VIDEO_PATH" = "q" ] && exit 1
    done
    CAPTURE_ARG="-c \"$VIDEO_PATH\""
    CAM_COUNT=0
else
    header "摄像头"
    detect_cameras
    if [ "$CAM_COUNT" -gt 0 ]; then
        ok "检测到 $CAM_COUNT 个摄像头设备"
        for dev in "${CAM_LIST[@]}"; do
            idx="${dev#/dev/video}"
            if command -v v4l2-ctl &>/dev/null; then
                name=$(v4l2-ctl --device="$dev" --all 2>/dev/null | grep "Card type" | sed 's/.*Card type\s*:\s*//' || echo "未知设备")
                echo "  /dev/video$idx → $name"
            else
                echo "  /dev/video$idx"
            fi
        done
    fi
    read -p "摄像头编号 [默认 0]: " CAM_ID
    CAM_ID="${CAM_ID:-0}"
    CAPTURE_ARG="-c $CAM_ID"

    read -p "摄像头帧率 FPS [默认 24]: " CAM_FPS
    CAM_FPS="${CAM_FPS:-24}"

    read -p "图像宽度 [默认 640]: " CAM_W
    CAM_W="${CAM_W:-640}"
    read -p "图像高度 [默认 480]: " CAM_H
    CAM_H="${CAM_H:-480}"

    read -p "是否镜像输入? (0=否, 1=是) [默认 0]: " MIRROR
    MIRROR="${MIRROR:-0}"
fi

# ---------- 3. 质量/速度预设 ----------
header "质量/速度预设"

echo "请选择质量预设 (会影响模型、检测阈值等):"
echo "  1) 🚀 极致性能  — 最低延迟, 适合低配设备 (模型 -1)"
echo "  2) ⚡ 快速      — 平衡速度与质量 (模型 0)"
echo "  3) 🎯 均衡      — 推荐默认 (模型 2)"
echo "  4) ✨ 高质量    — 最佳追踪精度 (模型 3)"
echo "  5) 😉 眨眼优化  — 针对眨眼检测优化 (模型 4)"
echo "  6) 🔧 自定义    — 自行配置各项参数"
read -p "请输入 [1-6, 默认 3]: " QUALITY_PRESET
QUALITY_PRESET="${QUALITY_PRESET:-3}"

case "$QUALITY_PRESET" in
    1)
        MODEL=-1
        DETECTION_THRESHOLD=0.4
        THRESHOLD=0.6
        NO_3D_ADAPT=1
        TRY_HARD=0
        GAZE=0
        HEADLESS="极致性能"
        ;;
    2)
        MODEL=0
        DETECTION_THRESHOLD=0.5
        THRESHOLD=0.7
        NO_3D_ADAPT=1
        TRY_HARD=1
        GAZE=0
        HEADLESS="快速"
        ;;
    3)
        MODEL=2
        DETECTION_THRESHOLD=0.6
        THRESHOLD=0.8
        NO_3D_ADAPT=1
        TRY_HARD=1
        GAZE=1
        HEADLESS="均衡"
        ;;
    4)
        MODEL=3
        DETECTION_THRESHOLD=0.6
        THRESHOLD=0.85
        NO_3D_ADAPT=0
        TRY_HARD=1
        GAZE=1
        HEADLESS="高质量"
        ;;
    5)
        MODEL=4
        DETECTION_THRESHOLD=0.6
        THRESHOLD=0.8
        NO_3D_ADAPT=1
        TRY_HARD=1
        GAZE=1
        HEADLESS="眨眼优化"
        ;;
    6)
        HEADLESS="自定义"
        header "模型选择"
        echo "可用模型 (-3 ~ 4, 越高越精细越慢):"
        echo "  -3 : 超极速 (56×56, 极低精度)"
        echo "  -2 : 快速型 (等效模型 1)"
        echo "  -1 : 极速型 (低精度高灵敏度)"
        echo "   0 : 快速型"
        echo "   1 : 标准型 (偏向刚性)"
        echo "   2 : 精细型 (推荐)"
        echo "   3 : 高质量 (默认)"
        echo "   4 : 眨眼优化"
        read -p "模型编号 [默认 3]: " MODEL
        MODEL="${MODEL:-3}"
        # 校验模型范围
        if ! [[ "$MODEL" =~ ^-?[0-9]+$ ]] || [ "$MODEL" -lt -3 ] || [ "$MODEL" -gt 4 ]; then
            warn "无效模型编号 $MODEL，已重置为 3"
            MODEL=3
        fi

        header "阈值配置"
        read -p "人脸检测阈值 (0.0~1.0, 越低越灵敏) [默认 0.6]: " DETECTION_THRESHOLD
        DETECTION_THRESHOLD="${DETECTION_THRESHOLD:-0.6}"
        read -p "追踪置信度阈值 (0.0~1.0, 越低越易追踪) [默认 0.8]: " THRESHOLD
        THRESHOLD="${THRESHOLD:-0.8}"

        header "功能开关"
        read -p "启用 3D 自适应? (0=关闭更快, 1=开启更准) [默认 0]: " NO_3D_ADAPT
        NO_3D_ADAPT="${NO_3D_ADAPT:-0}"
        read -p "启用 Try-Hard (尽力找脸)? (0=关闭, 1=开启) [默认 1]: " TRY_HARD
        TRY_HARD="${TRY_HARD:-1}"
        read -p "启用 Gaze 眼球追踪? (0=关闭更快, 1=开启) [默认 1]: " GAZE
        GAZE="${GAZE:-1}"
        ;;
esac

# ---------- 4. 多人脸设置 ----------
header "多人脸设置"
read -p "最大追踪人脸数 (1=单人, 2/3/4=多人, 越多人脸越慢) [默认 1]: " FACES
FACES="${FACES:-1}"
if ! [[ "$FACES" =~ ^[1-4]$ ]]; then
    warn "无效人脸数 $FACES，已重置为 1"
    FACES=1
fi

# ---------- 5. UDP 输出 ----------
header "UDP 数据输出"
read -p "发送 IP 地址 [默认 127.0.0.1]: " UDP_IP
UDP_IP="${UDP_IP:-127.0.0.1}"
read -p "发送端口 [默认 11573]: " UDP_PORT
UDP_PORT="${UDP_PORT:-11573}"

# ---------- 6. 日志 ----------
header "日志选项"
read -p "是否记录追踪数据到文件? (留空=不记录, 输入文件名如 track.log): " LOG_DATA
read -p "是否记录控制台输出到文件? (留空=不记录, 输入文件名如 output.log): " LOG_OUTPUT

# ---------- 7. 线程数 ----------
header "线程配置"
read -p "最大线程数 (越高CPU占用越高但可能更快) [默认 4]: " MAX_THREADS
MAX_THREADS="${MAX_THREADS:-4}"
if ! [[ "$MAX_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    warn "无效线程数 $MAX_THREADS，已重置为 4"
    MAX_THREADS=4
fi

# ---------- 8. 显示/静默 ----------
header "显示设置"
read -p "可视化级别 (0=不显示, 1=显示, 2=显示ID, 3=显示置信度) [默认 0]: " VISUALIZE
VISUALIZE="${VISUALIZE:-0}"

_save_config
fi  # end goto_build=false

# =============================================================================
# 构建命令
# =============================================================================

header "启动命令预览"

CMD_ARGS=()

# 输入源
if [ "$RUN_MODE" = "2" ]; then
    CMD_ARGS+=(-c "$VIDEO_PATH")
else
    CMD_ARGS+=(-c "$CAM_ID")
    CMD_ARGS+=(-F "$CAM_FPS")
    CMD_ARGS+=(-W "$CAM_W")
    CMD_ARGS+=(-H "$CAM_H")
    [ "$MIRROR" = "1" ] && CMD_ARGS+=(-M)
fi

# 模型/质量
CMD_ARGS+=(--model "$MODEL")
CMD_ARGS+=(--detection-threshold "$DETECTION_THRESHOLD")
CMD_ARGS+=(--threshold "$THRESHOLD")
CMD_ARGS+=(--no-3d-adapt "$NO_3D_ADAPT")
CMD_ARGS+=(--try-hard "$TRY_HARD")
CMD_ARGS+=(--gaze-tracking "$GAZE")

# 多人脸
CMD_ARGS+=(--faces "$FACES")

# UDP
CMD_ARGS+=(-i "$UDP_IP")
CMD_ARGS+=(-p "$UDP_PORT")

# 线程
CMD_ARGS+=(--max-threads "$MAX_THREADS")

# 可视化
CMD_ARGS+=(-v "$VISUALIZE")
CMD_ARGS+=(--silent 1)

# 日志
[ -n "$LOG_DATA" ]   && CMD_ARGS+=(--log-data "$LOG_DATA")
[ -n "$LOG_OUTPUT" ] && CMD_ARGS+=(--log-output "$LOG_OUTPUT")

# ---------- 显示预览 ----------
echo ""
echo -e "${BOLD}配置摘要:${NC}"
echo -e "  质量预设 : ${CYAN}${HEADLESS}${NC}"
echo -e "  模型     : ${CYAN}${MODEL}${NC}"

if [ "$RUN_MODE" = "1" ]; then
    echo -e "  摄像头   : ${CYAN}${CAM_ID}${NC} (${CAM_W}x${CAM_H} @ ${CAM_FPS}FPS)"
    [ "$MIRROR" = "1" ] && echo -e "  镜像     : ${CYAN}是${NC}"
else
    echo -e "  视频文件 : ${CYAN}${VIDEO_PATH}${NC}"
fi

echo -e "  阈值     : 检测=${CYAN}${DETECTION_THRESHOLD}${NC}, 追踪=${CYAN}${THRESHOLD}${NC}"
echo -e "  3D自适应 : ${CYAN}$([ "$NO_3D_ADAPT" = "0" ] && echo "开启" || echo "关闭")${NC}"
echo -e "  Try-Hard : ${CYAN}$([ "$TRY_HARD" = "1" ] && echo "开启" || echo "关闭")${NC}"
echo -e "  Gaze追踪 : ${CYAN}$([ "$GAZE" = "1" ] && echo "开启" || echo "关闭")${NC}"
echo -e "  最大人脸 : ${CYAN}${FACES}${NC}"
echo -e "  UDP 输出 : ${CYAN}${UDP_IP}:${UDP_PORT}${NC}"
echo -e "  线程数   : ${CYAN}${MAX_THREADS}${NC}"
echo -e "  加速器   : ${CYAN}${GPU_LABEL}${NC}"
[ -n "$LOG_DATA" ]   && echo -e "  数据日志 : ${CYAN}${LOG_DATA}${NC}"
[ -n "$LOG_OUTPUT" ] && echo -e "  输出日志 : ${CYAN}${LOG_OUTPUT}${NC}"
echo ""

# 用空格 IFS 拼接命令行（全局 IFS 为换行，需临时覆盖）
FULL_CMD="python3 \"$FACETRACKER\" $(IFS=' '; echo "${CMD_ARGS[*]}")"
echo -e "${YELLOW}完整命令:${NC}"
echo "  $ cd $(pwd)"
echo "  $ source .venv/bin/activate"
echo "  $ $FULL_CMD"
echo ""

# ---------- 确认启动 ----------
if [ "${goto_build:-false}" = "true" ]; then
    CONFIRM="y"
    info "静默模式，自动确认启动"
else
    read -p "是否以上述配置启动? (y=启动, n=取消, e=编辑参数) [默认 y]: " CONFIRM
    CONFIRM="${CONFIRM:-y}"
fi

if [ "$CONFIRM" = "n" ]; then
    echo "已取消启动。"
    exit 0
elif [ "$CONFIRM" = "e" ]; then
    echo "请输入额外参数 (直接追加到命令末尾):"
    read -r EXTRA_ARGS
    CMD_ARGS+=($EXTRA_ARGS)
fi

# =============================================================================
# 后台启动
# =============================================================================

header "启动 OpenSeeFace 面捕"

PID_FILE="$SCRIPT_DIR/.facetracker.pid"
LOG_FILE="$SCRIPT_DIR/.facetracker_console.log"

# ---------- 清理所有旧进程 ----------
info "检查旧进程..."
_stop_all_trackers "$PID_FILE" && { ok "旧进程已全部终止"; sleep 1; } || true

# 创建日志文件
touch "$LOG_FILE"

# 后台启动 (nohup)
nohup python3 "$FACETRACKER" "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1 &
BGPID=$!
echo "$BGPID" > "$PID_FILE"

# 等待几秒检查是否启动成功
sleep 2
if kill -0 "$BGPID" 2>/dev/null; then
    ok "OpenSeeFace 已成功在后台启动 (PID: $BGPID)"
    echo ""
    echo -e "  ${BOLD}日志文件:${NC} $LOG_FILE"
    [ -n "$LOG_DATA" ]   && echo -e "  ${BOLD}数据日志:${NC} $LOG_DATA"
    [ -n "$LOG_OUTPUT" ] && echo -e "  ${BOLD}输出日志:${NC} $LOG_OUTPUT"
    echo -e "  ${BOLD}PID 文件:${NC} $PID_FILE"
    echo ""
    echo -e "  ${YELLOW}常用管理命令:${NC}"
    echo -e "  查看日志  : tail -f $LOG_FILE"
    echo -e "  停止进程  : $(basename "$0") stop"
    echo -e "  检查状态  : ps aux | grep facetracker"
    echo ""
else
    err "进程启动失败! 查看日志获取详情:"
    err "  tail -50 $LOG_FILE"
    tail -5 "$LOG_FILE" | while IFS= read -r line; do
        err "  | $line"
    done
    rm -f "$PID_FILE"
    exit 1
fi

# ---------- 显示最近日志 ----------
echo -e "${BOLD}最近日志 (tail -5):${NC}"
tail -5 "$LOG_FILE" | while IFS= read -r line; do
    echo "  $line"
done

echo ""
ok "启动完成！"
