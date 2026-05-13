param(
    [string]$ProjectDir = "$env:USERPROFILE\Auto-research",
    [string]$CondaEnv = "silya",
    [string]$PythonVer = "3.11",
    [string]$GpuIds = "0",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Ok($Message) { Write-Host "[OK] $Message" -ForegroundColor Green }
function Warn($Message) { Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Fail($Message) { Write-Host "[ERROR] $Message" -ForegroundColor Red; exit 1 }

Write-Host "==== Step 1/4: Hardware check ===="
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Fail "nvidia-smi not found. Install NVIDIA driver first."
}

nvidia-smi --query-gpu=name --format=csv,noheader
Ok "NVIDIA GPU is visible"

Write-Host ""
Write-Host "==== Step 2/4: Project check ===="
if (Test-Path $ProjectDir) {
    Ok "Project directory exists: $ProjectDir"
} else {
    Warn "Project directory does not exist: $ProjectDir"
    Write-Host "Clone it first:"
    Write-Host "  git clone git@github.com:<your-github-owner>/Auto-research.git `"$ProjectDir`""
    if (-not $DryRun) {
        Fail "Project directory missing. Clone the project and run this script again."
    }
}

Write-Host ""
Write-Host "==== Step 3/4: Conda environment ===="
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Fail "conda not found. Install Miniconda or Anaconda, then open a new PowerShell."
}

$envExists = $false
try {
    $envJson = conda env list --json | ConvertFrom-Json
    $envExists = @($envJson.envs | ForEach-Object { Split-Path $_ -Leaf }) -contains $CondaEnv
} catch {
    Warn "Could not parse conda env list; will try to create or use the env directly."
}

if ($envExists) {
    Ok "Conda env exists: $CondaEnv"
} else {
    Write-Host "Creating conda env: $CondaEnv (Python $PythonVer)"
    if (-not $DryRun) {
        conda create -n $CondaEnv python=$PythonVer -y
    }
}

Write-Host ""
Write-Host "==== Step 4/4: Python dependencies ===="
if ($DryRun) {
    Write-Host "(dry-run) skip dependency installation"
} else {
    conda run -n $CondaEnv python -m pip install --upgrade pip
    conda run -n $CondaEnv python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    $requirements = Join-Path $ProjectDir "requirements.txt"
    if (Test-Path $requirements) {
        conda run -n $CondaEnv python -m pip install -r $requirements
    } else {
        conda run -n $CondaEnv python -m pip install ultralytics pyyaml kagglehub
    }

    $cudaOk = conda run -n $CondaEnv python -c "import torch; print(torch.cuda.is_available())"
    if ($cudaOk -match "True") {
        Ok "PyTorch CUDA is available"
    } else {
        Fail "PyTorch cannot see CUDA. Check NVIDIA driver and PyTorch CUDA package."
    }
}

Write-Host ""
Write-Host "==== Windows server setup complete ===="
Write-Host "Verify:"
Write-Host "  conda run -n $CondaEnv python -c `"import torch; print(torch.cuda.is_available())`""
Write-Host "Smoke test:"
Write-Host "  cd `"$ProjectDir`""
Write-Host "  conda run -n $CondaEnv python train_yolo.py train --task-name smoke_test --data-yaml C:\datasets\defect\data.yaml --epochs 1"
