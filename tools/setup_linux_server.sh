#!/usr/bin/env bash
# setup_linux_server.sh — 自有Linux服务器一键部署
#
# 用法 (在服务器上执行):
#   bash setup_linux_server.sh [选项]
#
# 或从Mac远程执行:
#   ssh lab-server "bash -s" < setup_linux_server.sh
#
# 选项:
#   --project-dir PATH    项目目录 (默认: ~/work/yolo)
#   --conda-env NAME      conda环境名 (默认: yolo)
#   --python-ver VERSION  Python版本 (默认: 3.10)
#   --clearml-queue NAME  ClearML队列名 (默认: yolo-linux)
#   --gpu-ids LIST        使用的GPU列表 (默认: 0. 示例: 0,1)
#   --no-systemd          不安装systemd服务 (手动管理)
#   --dry-run             只检查不安装

set -euo pipefail

PROJECT_DIR="${HOME}/work/yolo"
CONDA_ENV="yolo"
PYTHON_VER="3.10"
CLEARML_QUEUE="yolo-linux"
GPU_IDS="0"
NO_SYSTEMD=false
DRY_RUN=false

usage() {
    sed -n '2,17p' "$0" | sed 's/^# //'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-dir)   PROJECT_DIR="${2:?}"; shift 2 ;;
        --conda-env)     CONDA_ENV="${2:?}"; shift 2 ;;
        --python-ver)    PYTHON_VER="${2:?}"; shift 2 ;;
        --clearml-queue) CLEARML_QUEUE="${2:?}"; shift 2 ;;
        --gpu-ids)       GPU_IDS="${2:?}"; shift 2 ;;
        --no-systemd)    NO_SYSTEMD=true; shift ;;
        --dry-run)       DRY_RUN=true; shift ;;
        -h|--help)       usage ;;
        *) echo "未知参数: $1"; exit 2 ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
err()  { echo -e "${RED}✗${NC} $*"; }

$DRY_RUN && echo "【DRY-RUN 模式，只检查不安装】" && echo ""

# ── Step 1: 硬件检查 ────────────────────────────────
echo "══════════ Step 1/7: 硬件检查 ══════════"

if ! command -v nvidia-smi &>/dev/null; then
    err "未找到 nvidia-smi，请安装NVIDIA驱动"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ',' -)
ok "GPU: $GPU_COUNT 块 ($GPU_NAMES)"

CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' 2>/dev/null || echo "unknown")
ok "CUDA 驱动版本: $CUDA_VER"

RAM=$(free -h | awk '/Mem:/{print $2}')
ok "内存: $RAM"

DISK=$(df -h "$HOME" | awk 'NR==2{print $4}')
ok "可用磁盘 ($HOME): $DISK"

# ── Step 2: Conda 检查 ───────────────────────────────
echo ""
echo "══════════ Step 2/7: Conda环境 ══════════"

if command -v conda &>/dev/null; then
    ok "conda 已安装: $(conda --version 2>/dev/null)"
else
    warn "conda 未安装，尝试用 python venv..."
fi

# ── Step 3: 项目目录检查 ─────────────────────────────
echo ""
echo "══════════ Step 3/7: 项目代码 ══════════"
if [[ -d "$PROJECT_DIR" ]]; then
    ok "项目目录存在: $PROJECT_DIR"
else
    warn "项目目录不存在: $PROJECT_DIR"
    echo "  请先 clone 项目:"
    echo "  git clone <你的GitHub仓库> $PROJECT_DIR"
    echo "  cd $PROJECT_DIR && git pull"
fi

if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
    ok "requirements.txt 存在"
else
    warn "requirements.txt 不存在，将使用内置依赖列表"
fi

# ── Step 4: 安装Python依赖 ───────────────────────────
echo ""
echo "══════════ Step 4/7: Python依赖 ══════════"

if $DRY_RUN; then
    echo "  (dry-run) 创建环境并安装依赖"
else

if command -v conda &>/dev/null; then
    if conda env list | grep -q "^${CONDA_ENV} "; then
        ok "conda环境 $CONDA_ENV 已存在"
    else
        echo "→ 创建 conda 环境: $CONDA_ENV (Python $PYTHON_VER)"
        conda create -n "$CONDA_ENV" python="$PYTHON_VER" -y
        ok "conda环境创建完成"
    fi

    # 安装/更新依赖
    CONDA_PYTHON="$(conda run -n "$CONDA_ENV" which python)"
    CONDA_PIP="$(conda run -n "$CONDA_ENV" which pip)"

    echo "→ 安装 PyTorch + CUDA..."
    conda run -n "$CONDA_ENV" pip install --upgrade pip

    if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
        conda run -n "$CONDA_ENV" pip install -r "$PROJECT_DIR/requirements.txt"
    else
        conda run -n "$CONDA_ENV" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        conda run -n "$CONDA_ENV" pip install ultralytics clearml clearml-agent pyyaml
    fi
    ok "依赖安装完成"
else
    python3 -m venv "${PROJECT_DIR}/.venv"
    source "${PROJECT_DIR}/.venv/bin/activate"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install ultralytics clearml clearml-agent pyyaml
    ok "venv环境依赖安装完成"
fi

fi  # dry-run

# ── Step 5: PyTorch CUDA 验证 ────────────────────────
echo ""
echo "══════════ Step 5/7: CUDA验证 ══════════"

if ! $DRY_RUN; then
    if command -v conda &>/dev/null; then
        CUDA_OK=$(conda run -n "$CONDA_ENV" python -c "import torch; print(torch.cuda.is_available())")
    else
        source "${PROJECT_DIR}/.venv/bin/activate"
        CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())")
    fi

    if [[ "$CUDA_OK" == "True" ]]; then
        ok "PyTorch CUDA 可用"
    else
        err "PyTorch 看不到 CUDA! 请检查 PyTorch 和 CUDA 版本匹配"
        echo "  pip list | grep torch"
        echo "  nvidia-smi"
        exit 1
    fi
fi

# ── Step 6: ClearML 配置 ─────────────────────────────
echo ""
echo "══════════ Step 6/7: ClearML配置 ══════════"

if $DRY_RUN; then
    echo "  (dry-run) 配置 ClearML"
else
    if [[ -n "${CLEARML_API_ACCESS_KEY:-}" ]] && [[ -n "${CLEARML_API_SECRET_KEY:-}" ]]; then
        echo "→ 从环境变量配置 ClearML..."
        conda run -n "$CONDA_ENV" clearml-init 2>/dev/null || true

        # 写入配置文件
        CLEARML_CONF="$HOME/clearml.conf"
        cat > "$CLEARML_CONF" <<EOF
api {
    web_server: ${CLEARML_API_WEB_SERVER:-http://localhost:8080}
    api_server: ${CLEARML_API_HOST:-http://localhost:8008}
    files_server: ${CLEARML_FILES_HOST:-http://localhost:8081}
    credentials {
        access_key: ${CLEARML_API_ACCESS_KEY}
        secret_key: ${CLEARML_API_SECRET_KEY}
    }
}
EOF
        ok "ClearML 配置已写入 ~/clearml.conf"
    else
        warn "未检测到 CLEARML_API_ACCESS_KEY 和 CLEARML_API_SECRET_KEY"
        echo "  请手动执行: clearml-init"
        echo "  然后粘贴 ClearML Web UI 提供的凭证"
    fi

    # 验证连通性
    echo "→ 验证 ClearML 连通性..."
    if conda run -n "$CONDA_ENV" python -c "
from clearml import Task
t = Task.init(project_name='yolo', task_name='server_setup_check')
print(f'ClearML OK, task_id={t.id}')
t.close()
" 2>/dev/null; then
        ok "ClearML 连通性验证成功"
    else
        warn "ClearML 连通性验证失败，请检查 clearml-init 配置"
    fi
fi

# ── Step 7: 安装看门狗和 Agent (systemd) ─────────────
echo ""
echo "══════════ Step 7/7: 后台服务 ══════════"

if $NO_SYSTEMD; then
    warn "跳过 systemd 安装 (--no-systemd)"
    echo "  手动启动:"
    echo "    clearml-agent daemon --queue $CLEARML_QUEUE --foreground"
    echo "    python3 tools/watchdog.py --base-dir /tmp/aris-watchdog"
elif $DRY_RUN; then
    echo "  (dry-run) 安装 systemd 服务"
else

# 生成 systemd service 文件
SERVICE_DIR="$HOME/.config/systemd/user"
mkdir -p "$SERVICE_DIR"

# ClearML Agent service
cat > "$SERVICE_DIR/clearml-agent-yolo.service" <<SVC
[Unit]
Description=ClearML Agent for YOLO ($CLEARML_QUEUE)
After=network.target

[Service]
Type=simple
ExecStart=$(which clearml-agent 2>/dev/null || echo "$HOME/miniconda3/envs/$CONDA_ENV/bin/clearml-agent") daemon --queue $CLEARML_QUEUE --foreground
Restart=on-failure
RestartSec=30
Environment="PATH=$HOME/miniconda3/envs/$CONDA_ENV/bin:/usr/local/cuda/bin:/usr/bin:/bin"

[Install]
WantedBy=default.target
SVC
ok "systemd: clearml-agent-yolo.service"

# ARIS Watchdog service
cat > "$SERVICE_DIR/aris-watchdog.service" <<SVC
[Unit]
Description=ARIS Watchdog — GPU训练监控
After=network.target

[Service]
Type=simple
ExecStart=$(conda run -n "$CONDA_ENV" which python3 || echo "python3") $PROJECT_DIR/tools/watchdog.py --base-dir /tmp/aris-watchdog --interval 60
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
SVC
ok "systemd: aris-watchdog.service"

# 启用 linger (允许用户服务在登出后继续运行)
loginctl enable-linger "$USER" 2>/dev/null || true

# 重新加载并启用
systemctl --user daemon-reload
systemctl --user enable clearml-agent-yolo.service
systemctl --user enable aris-watchdog.service

echo ""
ok "systemd 服务已安装并启用"
echo ""
echo "  管理命令:"
echo "    systemctl --user start clearml-agent-yolo   # 启动ClearML Agent"
echo "    systemctl --user start aris-watchdog        # 启动看门狗"
echo "    systemctl --user status clearml-agent-yolo  # 查看状态"
echo "    journalctl --user -u clearml-agent-yolo -f  # 查看日志"
fi

# ── 完成 ────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  服务器部署完成!"
echo "══════════════════════════════════════════"
echo ""
echo "  快速启动:"
echo "    systemctl --user start clearml-agent-yolo"
echo "    systemctl --user start aris-watchdog"
echo ""
echo "  验证:"
echo "    conda activate $CONDA_ENV"
echo "    python -c 'import torch; print(torch.cuda.is_available())'"
echo "    python -c 'from clearml import Task; Task.init(project_name=\"yolo\", task_name=\"check\")'"
echo ""
