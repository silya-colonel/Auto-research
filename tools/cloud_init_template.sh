#!/usr/bin/env bash
# cloud_init_template.sh — 云GPU实例初始化
#
# 用法 (在云GPU实例上执行):
#   bash cloud_init.sh [选项]
#
# 选项:
#   --data-source URL     数据集来源 (oss://bucket/path 或本地路径)
#   --max-hours N         最大运行时间，超时自动关机 (默认: 8)
#   --clearml-queue NAME  ClearML队列名 (默认: yolo-cloud)
#   --gpu-ids LIST        GPU列表 (默认: 0)

set -euo pipefail

DATA_SOURCE=""
MAX_HOURS=8
CLEARML_QUEUE="yolo-cloud"
GPU_IDS="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-source)   DATA_SOURCE="${2:?}"; shift 2 ;;
        --max-hours)     MAX_HOURS="${2:?}"; shift 2 ;;
        --clearml-queue) CLEARML_QUEUE="${2:?}"; shift 2 ;;
        --gpu-ids)       GPU_IDS="${2:?}"; shift 2 ;;
        -h|--help)
            sed -n '2,13p' "$0" | sed 's/^# //'; exit 0 ;;
        *) echo "未知参数: $1"; exit 2 ;;
    esac
done

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
ok() { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }

# ── 检测云平台 ──────────────────────────────────────
detect_platform() {
    if [[ -d "/root/autodl-tmp" ]]; then
        echo "autodl"
    elif [[ -n "${VAST_CONTAINER_ID:-}" ]]; then
        echo "vast"
    else
        echo "generic"
    fi
}

PLATFORM=$(detect_platform)
ok "检测到云平台: $PLATFORM"

# ── Step 1: 硬件检查 ────────────────────────────────
echo ""
echo "═══ Step 1/5: 硬件检查 ═══"
nvidia-smi --query-gpu=name --format=csv,noheader
ok "GPU可用: $(nvidia-smi -L | wc -l) 块"
df -h / | awk 'NR==2{print "磁盘可用: " $4}'

# ── Step 2: 安装依赖 ────────────────────────────────
echo ""
echo "═══ Step 2/5: 安装依赖 ═══"

# 根据平台确定工作目录
case "$PLATFORM" in
    autodl)
        WORK_DIR="/root/autodl-tmp/yolo"
        DATA_DIR="/root/autodl-tmp/datasets"
        ;;
    vast)
        WORK_DIR="/workspace/yolo"
        DATA_DIR="/workspace/datasets"
        ;;
    *)
        WORK_DIR="$HOME/work/yolo"
        DATA_DIR="$HOME/datasets"
        ;;
esac

mkdir -p "$WORK_DIR" "$DATA_DIR"

pip install --upgrade pip -q
pip install ultralytics clearml clearml-agent pyyaml -q
ok "Python 依赖安装完成"

# 验证 PyTorch CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用!'; print(f'PyTorch CUDA OK, 设备数: {torch.cuda.device_count()}')"
ok "PyTorch CUDA 验证通过"

# ── Step 3: 拉取代码 ────────────────────────────────
echo ""
echo "═══ Step 3/5: 拉取代码 ═══"

if [[ -f "$WORK_DIR/train_yolo_clearml.py" ]]; then
    cd "$WORK_DIR"
    git pull --ff-only 2>/dev/null || ok "代码目录已存在，跳过git pull"
else
    warn "代码目录为空，请手动 clone:"
    echo "  cd $(dirname $WORK_DIR) && git clone <你的仓库> $(basename $WORK_DIR)"
fi

# ── Step 4: 拉取数据 ────────────────────────────────
echo ""
echo "═══ Step 4/5: 拉取数据 ═══"

if [[ -n "$DATA_SOURCE" ]]; then
    case "$DATA_SOURCE" in
        oss://*|s3://*|cos://*)
            if command -v rclone &>/dev/null; then
                rclone copy "$DATA_SOURCE" "$DATA_DIR" --progress
                ok "数据已从对象存储拉取"
            else
                warn "rclone 未安装，可以: curl https://rclone.org/install.sh | sudo bash"
            fi
            ;;
        http*)
            wget -O "$DATA_DIR/dataset.zip" "$DATA_SOURCE"
            unzip "$DATA_DIR/dataset.zip" -d "$DATA_DIR"
            ok "数据已下载解压"
            ;;
        *)
            if [[ -d "$DATA_SOURCE" ]]; then
                cp -r "$DATA_SOURCE"/* "$DATA_DIR/"
                ok "数据已从本地路径复制"
            else
                warn "无法识别的数据源: $DATA_SOURCE"
            fi
            ;;
    esac
else
    warn "未指定数据源，请手动上传数据到: $DATA_DIR"
    echo "  或用 --data-source 指定"
fi

# ── Step 5: ClearML + 后台服务 ──────────────────────
echo ""
echo "═══ Step 5/5: ClearML配壳 + 后台服务 ═══"

if [[ -n "${CLEARML_API_ACCESS_KEY:-}" ]]; then
    cat > ~/clearml.conf <<EOF
api {
    web_server: ${CLEARML_API_WEB_SERVER:-https://app.clear.ml}
    api_server: ${CLEARML_API_HOST:-https://api.clear.ml}
    files_server: ${CLEARML_FILES_HOST:-https://files.clear.ml}
    credentials {
        access_key: ${CLEARML_API_ACCESS_KEY}
        secret_key: ${CLEARML_API_SECRET_KEY:-}
    }
}
EOF
    ok "ClearML 配置完成"
else
    warn "未设置 ClearML 环境变量，请手动执行 clearml-init"
fi

# 启动 watchdog (后台)
if [[ -f "$WORK_DIR/tools/watchdog.py" ]]; then
    nohup python3 "$WORK_DIR/tools/watchdog.py" \
        --base-dir /tmp/aris-watchdog --interval 60 \
        > /tmp/watchdog.log 2>&1 &
    ok "watchdog 已启动 (pid=$!)"
fi

# 启动 ClearML Agent (后台)
nohup clearml-agent daemon --queue "$CLEARML_QUEUE" --foreground \
    > /tmp/clearml-agent.log 2>&1 &
ok "ClearML Agent 已启动 (队列: $CLEARML_QUEUE, pid=$!)"

# ── 费用保护：定时关机 ───────────────────────────────
echo ""
echo "═══ 费用保护 ═══"
SHUTDOWN_MINUTES=$((MAX_HOURS * 60))
echo "→ 设置 $MAX_HOURS 小时后自动关机"
sudo shutdown -h +$SHUTDOWN_MINUTES 2>/dev/null || {
    warn "无sudo权限，无法设置自动关机"
    echo "  请使用云平台控制台设置自动释放"
}
ok "定时关机已设置: $(date -d "+$SHUTDOWN_MINUTES minutes" '+%Y-%m-%d %H:%M' 2>/dev/null || date -v+${SHUTDOWN_MINUTES}M '+%Y-%m-%d %H:%M' 2>/dev/null || echo "${MAX_HOURS}h后")"

# ── 完成 ────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  云GPU实例初始化完成!"
echo "══════════════════════════════════════════"
echo ""
echo "  运行训练:"
echo "    cd $WORK_DIR"
echo "    python train_yolo_clearml.py --task-name baseline_yolo11n_640 \\"
echo "      --data-yaml $DATA_DIR/data.yaml --model yolo11n.pt --imgsz 640 --epochs 100"
echo ""
echo "  查看状态:"
echo "    python3 $WORK_DIR/tools/watchdog.py --status"
echo "    cat /tmp/watchdog.log"
echo ""
echo "  取消自动关机:"
echo "    sudo shutdown -c"
echo ""
echo "  ClearML Agent 已监听队列: $CLEARML_QUEUE"
echo "  从 Mac/Codex 提交任务即可自动接取"
echo ""
