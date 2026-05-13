#!/usr/bin/env bash
# init_yolo_project.sh — Mac端一键创建YOLO项目
#
# 用法:
#   bash tools/init_yolo_project.sh <项目名> [选项]
#
# 选项:
#   --server-type TYPE     linux-lab(默认) 或 windows
#   --github-remote URL    GitHub私有仓库URL (可选，跳过则手动添加)
#   --data-dir PATH        数据集本地路径 (Mac端，可选)
#   --dry-run              只打印不执行

set -euo pipefail

# ── 参数解析 ────────────────────────────────────────
PROJECT_NAME=""
SERVER_TYPE="linux-lab"
GITHUB_REMOTE=""
DATA_DIR=""
DRY_RUN=false

usage() {
    cat <<'HELP'
init_yolo_project.sh — Mac端一键创建YOLO缺陷检测项目

用法:
  bash tools/init_yolo_project.sh <项目名> [选项]

选项:
  --server-type TYPE     部署模式: linux-lab(默认) 或 windows
  --github-remote URL    GitHub私有仓库URL (git@github.com:user/repo.git)
  --data-dir PATH        本地数据集路径 (将生成data.yaml)
  --dry-run              只打印不执行

示例:
  # Linux 服务器
  bash tools/init_yolo_project.sh yolo-defect --github-remote git@github.com:silya/yolo-defect.git

  # Windows 服务器
  bash tools/init_yolo_project.sh yolo-defect --server-type windows --github-remote git@github.com:silya/yolo-defect.git
HELP
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-type)   SERVER_TYPE="${2:?}"; shift 2 ;;
        --github-remote) GITHUB_REMOTE="${2:?}"; shift 2 ;;
        --data-dir)      DATA_DIR="${2:?}"; shift 2 ;;
        --dry-run)       DRY_RUN=true; shift ;;
        -h|--help)       usage ;;
        *)
            if [[ -z "$PROJECT_NAME" ]]; then PROJECT_NAME="$1"
            else echo "未知参数: $1"; exit 2; fi
            shift ;;
    esac
done

[[ -z "$PROJECT_NAME" ]] && { echo "错误: 请指定项目名"; usage; }

if [[ "$SERVER_TYPE" != "linux-lab" && "$SERVER_TYPE" != "windows" ]]; then
    echo "错误: --server-type 必须是 linux-lab 或 windows"
    exit 2
fi

# ── 定位 Auto-research 仓库 ─────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARIS_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATES_DIR="$ARIS_REPO/templates"

[[ -d "$TEMPLATES_DIR" ]] || { echo "错误: 找不到模板目录 $TEMPLATES_DIR"; exit 1; }

PROJECT_DIR="$(pwd)/$PROJECT_NAME"

# ── 主流程 ──────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   ARIS YOLO 项目初始化                    ║"
echo "╠══════════════════════════════════════════╣"
echo "║  项目名:   $PROJECT_NAME"
echo "║  部署模式: $SERVER_TYPE"
echo "║  目标目录: $PROJECT_DIR"
echo "╚══════════════════════════════════════════╝"
echo ""

$DRY_RUN && echo "【DRY-RUN 模式，不执行实际操作】" && echo ""

# 1. 创建目录结构
echo "→ 创建目录结构..."
dirs=(
    "$PROJECT_DIR"
    "$PROJECT_DIR/experiments"
    "$PROJECT_DIR/results"
    "$PROJECT_DIR/docs"
    "$PROJECT_DIR/notes"
    "$PROJECT_DIR/configs"
    "$PROJECT_DIR/tools"
)
for d in "${dirs[@]}"; do
    if $DRY_RUN; then
        echo "  mkdir -p $d"
    else
        mkdir -p "$d"
        echo "  ✓ $d"
    fi
done

# 2. 复制模板
echo "→ 复制模板文件..."

copy_template() {
    local src="$TEMPLATES_DIR/$1"
    local dst="$PROJECT_DIR/$2"
    if [[ ! -f "$src" ]]; then
        echo "  ⚠ 模板不存在，跳过: $src"
        return
    fi
    if $DRY_RUN; then
        echo "  cp $src → $dst"
    else
        cp "$src" "$dst"
        echo "  ✓ $2"
    fi
}

copy_template "AGENTS_TEMPLATE.md" "AGENTS.md"
copy_template "train_yolo.py" "train_yolo.py"
copy_template "YOLO_GITIGNORE_TEMPLATE.txt" ".gitignore"
copy_template "YOLO_REQUIREMENTS_TEMPLATE.txt" "requirements.txt"
copy_template "YOLO_EXPERIMENT_MANIFEST_TEMPLATE.md" "experiments/YOLO_EXPERIMENT_MANIFEST.md"

# 3. 生成服务器初始化脚本
echo "→ 生成服务器端初始化脚本..."

if $DRY_RUN; then
    echo "  cp setup_linux_server.sh → $PROJECT_DIR/tools/"
    echo "  cp setup_windows_server.ps1 → $PROJECT_DIR/tools/"
else
    cp "$ARIS_REPO/tools/setup_linux_server.sh" "$PROJECT_DIR/tools/"
    chmod +x "$PROJECT_DIR/tools/setup_linux_server.sh"
    echo "  ✓ tools/setup_linux_server.sh"
    cp "$ARIS_REPO/tools/setup_windows_server.ps1" "$PROJECT_DIR/tools/"
    echo "  ✓ tools/setup_windows_server.ps1"
fi

# 4. 提示配置 AGENTS.md
if ! $DRY_RUN && [[ -f "$PROJECT_DIR/AGENTS.md" ]]; then
    echo "→ AGENTS.md 已生成，请编辑填写服务器信息"
fi

# 6. 初始化 Git
echo "→ 初始化 Git..."
if $DRY_RUN; then
    echo "  cd $PROJECT_DIR && git init"
    [[ -n "$GITHUB_REMOTE" ]] && echo "  git remote add origin $GITHUB_REMOTE"
else
    cd "$PROJECT_DIR"
    git init
    if [[ -n "$GITHUB_REMOTE" ]]; then
        git remote add origin "$GITHUB_REMOTE"
        echo "  ✓ Git 仓库已初始化，remote: $GITHUB_REMOTE"
    else
        echo "  ✓ Git 仓库已初始化 (未设置remote，请手动添加)"
    fi
    cd - > /dev/null
fi

# 7. 复制 watchdog
echo "→ 复制 watchdog..."
copy_template "../tools/watchdog.py" "tools/watchdog.py"
if ! $DRY_RUN && [[ -f "$PROJECT_DIR/tools/watchdog.py" ]]; then
    chmod +x "$PROJECT_DIR/tools/watchdog.py"
fi

# ── 完成 ────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  项目初始化完成!                          ║"
echo "╠══════════════════════════════════════════╣"
echo "║  项目目录: $PROJECT_DIR"
echo "║  下一步:                                  ║"
echo "╚══════════════════════════════════════════╝"
echo ""

echo "  1. 编辑 AGENTS.md — 填写服务器IP、用户名、路径"
if [[ "$SERVER_TYPE" == "windows" ]]; then
    echo "  2. 在 Windows 服务器上准备数据:"
    echo "     C:\\datasets\\defect\\data.yaml  (data.yaml 和图片/标签一起放数据集目录)"
else
    echo "  2. 在 Linux 服务器上准备数据:"
    echo "     ~/datasets/defect/data.yaml  (data.yaml 和图片/标签一起放数据集目录)"
fi
echo "  3. 把项目推送到 GitHub:"
echo "     cd $PROJECT_NAME && git add -A && git commit -m 'init project' && git push -u origin main"
echo "  4. 在服务器上运行初始化:"
if [[ "$SERVER_TYPE" == "windows" ]]; then
    echo "     ssh <Windows服务器> 'powershell -NoProfile -ExecutionPolicy Bypass -Command \"cd \$env:USERPROFILE\\Auto-research; .\\tools\\setup_windows_server.ps1\"'"
else
    echo "     ssh <Linux服务器> 'cd ~/Auto-research && bash tools/setup_linux_server.sh'"
fi
echo ""

echo "  然后回到 Mac 用 Codex 执行:"
echo "    /yolo-pipeline \"基于YOLO的缺陷检测\""
echo ""
