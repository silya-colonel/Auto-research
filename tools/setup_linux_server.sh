#!/usr/bin/env bash
# setup_linux_server.sh — 自有Linux服务器一键部署
#
# 用法 (在服务器上执行):
#   bash setup_linux_server.sh [选项]
#
# 或从Mac远程执行:
#   ssh lab-server "cd ~/Auto-research && bash tools/setup_linux_server.sh"
#
# 选项:
#   --project-dir PATH    项目目录 (默认: 当前仓库目录；否则 ~/Auto-research)
#   --conda-env NAME      conda环境名 (默认: silya)
#   --python-ver VERSION  Python版本 (默认: 3.11)
#   --gpu-ids LIST        使用的GPU列表 (默认: 0. 示例: 0,1)
#   --no-systemd          不安装systemd服务 (手动管理)
#   --dry-run             只检查不安装

set -euo pipefail

if [[ -f "$PWD/train_yolo.py" && -d "$PWD/tools" ]]; then
    DEFAULT_PROJECT_DIR="$PWD"
else
    DEFAULT_PROJECT_DIR="${HOME}/Auto-research"
fi

PROJECT_DIR="$DEFAULT_PROJECT_DIR"
CONDA_ENV="silya"
PYTHON_VER="3.11"
GPU_IDS="0"
NO_SYSTEMD=false
DRY_RUN=false

usage() {
    sed -n '2,16p' "$0" | sed 's/^# *//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-dir)   PROJECT_DIR="${2:?}"; shift 2 ;;
        --conda-env)     CONDA_ENV="${2:?}"; shift 2 ;;
        --python-ver)    PYTHON_VER="${2:?}"; shift 2 ;;
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
echo "══════════ Step 1/5: 硬件检查 ══════════"

if ! command -v nvidia-smi &>/dev/null; then
    err "未找到 nvidia-smi，请安装NVIDIA驱动"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ',' -)
ok "GPU: $GPU_COUNT 块 ($GPU_NAMES)"

CUDA_VER=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([^ |]*\).*/\1/p' | head -n 1)
CUDA_VER="${CUDA_VER:-unknown}"
ok "CUDA 驱动版本: $CUDA_VER"

RAM=$(free -h | awk '/Mem:/{print $2}')
ok "内存: $RAM"

DISK=$(df -h "$HOME" | awk 'NR==2{print $4}')
ok "可用磁盘 ($HOME): $DISK"

# ── Step 2: Conda 检查 ───────────────────────────────
echo ""
echo "══════════ Step 2/5: Conda环境 ══════════"

if command -v conda &>/dev/null; then
    ok "conda 已安装: $(conda --version 2>/dev/null)"
else
    warn "conda 未安装，尝试用 python venv..."
fi

# ── Step 3: 项目目录检查 ─────────────────────────────
echo ""
echo "══════════ Step 3/5: 项目代码 ══════════"
if [[ -d "$PROJECT_DIR" ]]; then
    ok "项目目录存在: $PROJECT_DIR"
else
    warn "项目目录不存在: $PROJECT_DIR"
    echo "  请先 clone 项目:"
    echo "  git clone git@github.com:<你的GitHub用户名或组织名>/Auto-research.git $PROJECT_DIR"
    echo "  cd $PROJECT_DIR"
    if ! $DRY_RUN; then
        err "项目目录不存在，停止安装。clone 完成后重新运行本脚本。"
        exit 1
    fi
fi

if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
    ok "requirements.txt 存在"
else
    warn "requirements.txt 不存在，将使用内置依赖列表"
fi

# ── Step 4: 安装Python依赖 ───────────────────────────
echo ""
echo "══════════ Step 4/5: Python依赖 ══════════"

if $DRY_RUN; then
    echo "  (dry-run) 创建环境并安装依赖"
else

if command -v conda &>/dev/null; then
    if conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
        ok "conda环境 $CONDA_ENV 已存在"
    else
        echo "→ 创建 conda 环境: $CONDA_ENV (Python $PYTHON_VER)"
        conda create -n "$CONDA_ENV" python="$PYTHON_VER" -y
        ok "conda环境创建完成"
    fi

    # 安装/更新依赖
    echo "→ 安装 PyTorch + CUDA..."
    conda run -n "$CONDA_ENV" pip install --upgrade pip
    conda run -n "$CONDA_ENV" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
        conda run -n "$CONDA_ENV" pip install -r "$PROJECT_DIR/requirements.txt"
    else
        conda run -n "$CONDA_ENV" pip install ultralytics pyyaml kagglehub
    fi
    ok "依赖安装完成"
else
    python3 -m venv "${PROJECT_DIR}/.venv"
    source "${PROJECT_DIR}/.venv/bin/activate"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install ultralytics pyyaml kagglehub
    ok "venv环境依赖安装完成"
fi

fi  # dry-run

# ── Step 5: PyTorch CUDA 验证 ────────────────────────
echo ""
echo "══════════ Step 5/5: CUDA验证 ══════════"

if ! $DRY_RUN; then
    if command -v conda &>/dev/null; then
        CUDA_OK=$(conda run -n "$CONDA_ENV" python -c "import torch; print(torch.cuda.is_available())")
    else
        source "${PROJECT_DIR}/.venv/bin/activate"
        CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())")
    fi

    if [[ "$CUDA_OK" == *"True"* ]]; then
        ok "PyTorch CUDA 可用"
    else
        err "PyTorch 看不到 CUDA! 请检查 PyTorch 和 CUDA 版本匹配"
        echo "  pip list | grep torch"
        echo "  nvidia-smi"
        exit 1
    fi
fi

# ── Step 5: 后台服务 (systemd) ────────────────────────
echo ""
echo "══════════ Step 5/5: 后台服务 ══════════"

if $NO_SYSTEMD; then
    warn "跳过 systemd 安装 (--no-systemd)"
    echo "  手动启动:"
    if command -v conda &>/dev/null; then
        echo "    cd $PROJECT_DIR && conda run -n $CONDA_ENV python tools/watchdog.py --base-dir /tmp/aris-watchdog"
    else
        echo "    cd $PROJECT_DIR && .venv/bin/python tools/watchdog.py --base-dir /tmp/aris-watchdog"
    fi
elif $DRY_RUN; then
    echo "  (dry-run) 安装 systemd 服务"
else

# 生成 systemd service 文件
SERVICE_DIR="$HOME/.config/systemd/user"
mkdir -p "$SERVICE_DIR"

if command -v conda &>/dev/null; then
    WATCHDOG_PYTHON="$(conda run -n "$CONDA_ENV" python -c 'import sys; print(sys.executable)')"
else
    WATCHDOG_PYTHON="$PROJECT_DIR/.venv/bin/python"
fi

# ARIS Watchdog service
cat > "$SERVICE_DIR/aris-watchdog.service" <<SVC
[Unit]
Description=ARIS Watchdog — GPU训练监控
After=network.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
ExecStart=$WATCHDOG_PYTHON $PROJECT_DIR/tools/watchdog.py --base-dir /tmp/aris-watchdog --interval 60
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
systemctl --user enable aris-watchdog.service

echo ""
ok "systemd 服务已安装并启用"
echo ""
echo "  管理命令:"
echo "    systemctl --user start aris-watchdog        # 启动看门狗"
echo "    systemctl --user status aris-watchdog       # 查看状态"
echo "    journalctl --user -u aris-watchdog -f       # 查看日志"
fi

# ── 完成 ────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  服务器部署完成!"
echo "══════════════════════════════════════════"
echo ""
echo "  快速启动:"
echo "    systemctl --user start aris-watchdog"
echo ""
echo "  验证:"
if command -v conda &>/dev/null; then
    echo "    conda run -n $CONDA_ENV python -c 'import torch; print(torch.cuda.is_available())'"
    echo "    cd $PROJECT_DIR && conda run -n $CONDA_ENV python train_yolo.py train --task-name smoke_test --data-yaml <data.yaml> --epochs 1"
else
    echo "    $PROJECT_DIR/.venv/bin/python -c 'import torch; print(torch.cuda.is_available())'"
    echo "    cd $PROJECT_DIR && .venv/bin/python train_yolo.py train --task-name smoke_test --data-yaml <data.yaml> --epochs 1"
fi
echo ""
