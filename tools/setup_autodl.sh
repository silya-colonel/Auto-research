#!/usr/bin/env bash
# setup_autodl.sh — AutoDL 实例一键部署 (YOLO + ClearML)
#
# 用法 (在 AutoDL 实例终端中执行):
#   source /root/.bashrc
#   git clone <你的仓库> /root/autodl-tmp/Auto-research
#   bash /root/autodl-tmp/Auto-research/tools/setup_autodl.sh [选项]
#
# 选项:
#   --project-dir PATH    项目目录 (默认: /root/autodl-tmp/Auto-research)
#   --conda-env NAME      conda环境名 (默认: yolo)
#   --python-ver VERSION  Python版本 (默认: 3.10)
#   --clearml-queue NAME  ClearML队列名 (默认: yolo-autodl)

set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/Auto-research"
CONDA_ENV="yolo"
PYTHON_VER="3.10"
CLEARML_QUEUE="yolo-autodl"

usage() {
    sed -n '2,9p' "$0" | sed 's/^# //'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-dir)   PROJECT_DIR="${2:?}"; shift 2 ;;
        --conda-env)     CONDA_ENV="${2:?}"; shift 2 ;;
        --python-ver)    PYTHON_VER="${2:?}"; shift 2 ;;
        --clearml-queue) CLEARML_QUEUE="${2:?}"; shift 2 ;;
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

# ── AutoDL 环境检查 ──────────────────────────────────
echo "══════════ AutoDL 环境概览 ══════════"

source /root/.bashrc 2>/dev/null || true

ok "系统盘: / (30GB)"
ok "数据盘: /root/autodl-tmp"
df -h / /root/autodl-tmp 2>/dev/null | grep -v "^Filesystem" || true
echo ""

if command -v nvidia-smi &>/dev/null; then
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) 块"
    ok "CUDA: $(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}')"
else
    err "未检测到 GPU，请确认实例已分配 GPU"
    exit 1
fi

ok "conda: $(conda --version 2>/dev/null || echo '未安装')"
ok "Python: $(python --version 2>&1)"

# ── Step 1: 创建 conda 环境 ────────────────────────────
echo ""
echo "══════════ Step 1/5: Conda环境 ══════════"

if conda env list | grep -q "^${CONDA_ENV} "; then
    ok "conda环境 $CONDA_ENV 已存在，跳过创建"
else
    echo "→ 创建 conda 环境: $CONDA_ENV (Python $PYTHON_VER)"
    conda create -n "$CONDA_ENV" python="$PYTHON_VER" -y
    ok "conda环境创建完成"
fi

# ── Step 2: 安装 Python 依赖 ──────────────────────────
echo ""
echo "══════════ Step 2/5: Python依赖 ══════════"

echo "→ 安装 PyTorch + CUDA..."
conda run -n "$CONDA_ENV" pip install --upgrade pip

if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
    conda run -n "$CONDA_ENV" pip install -r "$PROJECT_DIR/requirements.txt"
else
    conda run -n "$CONDA_ENV" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    conda run -n "$CONDA_ENV" pip install ultralytics clearml clearml-agent pyyaml
fi
ok "依赖安装完成"

# ── Step 3: CUDA 验证 ─────────────────────────────────
echo ""
echo "══════════ Step 3/5: CUDA验证 ══════════"

CUDA_OK=$(conda run -n "$CONDA_ENV" python -c "import torch; print(torch.cuda.is_available())")

if [[ "$CUDA_OK" == "True" ]]; then
    ok "PyTorch CUDA 可用"
    conda run -n "$CONDA_ENV" python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.version.cuda}')
print(f'  GPU:     {torch.cuda.get_device_name(0)}')
print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"
else
    err "PyTorch 看不到 CUDA，请检查 PyTorch 和 CUDA 版本匹配"
    exit 1
fi

# ── Step 4: ClearML 配置 ──────────────────────────────
echo ""
echo "══════════ Step 4/5: ClearML配置 ══════════"

if [[ -n "${CLEARML_API_ACCESS_KEY:-}" ]] && [[ -n "${CLEARML_API_SECRET_KEY:-}" ]]; then
    echo "→ 从环境变量配置 ClearML..."

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
    warn "未检测到 CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY"
    echo "  手动配置方法:"
    echo "    conda activate $CONDA_ENV"
    echo "    clearml-init"
    echo "    # 粘贴 ClearML Web UI 的 credentials"
fi

# ── Step 5: 启动 ClearML Agent (后台) ─────────────────
echo ""
echo "══════════ Step 5/5: ClearML Agent ══════════"

# 用 nohup 后台运行 (容器内不适合 systemd)
AGENT_PID=$(pgrep -f "clearml-agent daemon.*$CLEARML_QUEUE" 2>/dev/null || true)
if [[ -n "$AGENT_PID" ]]; then
    ok "ClearML Agent 已在运行 (PID: $AGENT_PID)"
else
    echo "→ 启动 ClearML Agent (队列: $CLEARML_QUEUE)..."
    nohup conda run -n "$CONDA_ENV" clearml-agent daemon \
        --queue "$CLEARML_QUEUE" \
        --foreground \
        > /root/autodl-tmp/clearml-agent.log 2>&1 &
    sleep 2

    if pgrep -f "clearml-agent daemon.*$CLEARML_QUEUE" > /dev/null; then
        ok "ClearML Agent 已启动 (日志: /root/autodl-tmp/clearml-agent.log)"
    else
        warn "ClearML Agent 启动失败，查看日志: /root/autodl-tmp/clearml-agent.log"
    fi
fi

# ── 完成 ──────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  AutoDL 实例部署完成!"
echo "══════════════════════════════════════════"
echo ""
echo "  AutoDL 目录速查:"
echo "    数据盘 (IO快,放项目/数据集): /root/autodl-tmp"
echo "    文件存储 (跨实例共享):       /root/autodl-fs"
echo "    公共数据 (只读):             /root/autodl-pub"
echo "    JupyterLab 入口:             http://<实例IP>:<端口>"
echo ""
echo "  环境激活:"
echo "    conda activate $CONDA_ENV"
echo ""
echo "  验证命令:"
echo "    python -c 'import torch; print(torch.cuda.is_available())'"
echo "    clearml-agent --version"
echo ""
echo "  启动训练 (二选一):"
echo "    # 方式1: 直接训练"
echo "    cd $PROJECT_DIR && conda activate $CONDA_ENV"
echo "    python train_yolo_clearml.py"
echo ""
echo "    # 方式2: 通过 ClearML 远程提交"
echo "    clearml-task --project yolo --name test_run \\"
echo "      --script $PROJECT_DIR/train_yolo_clearml.py"
echo ""
