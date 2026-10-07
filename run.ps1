# One command per stage. Usage (from the repo folder):
#   powershell -ExecutionPolicy Bypass -File run.ps1 <setup|check|smoke|fold0|rest|merge>
# Every stage except setup writes its console output to logs\<stage>.log as well.
param([Parameter(Mandatory)][ValidateSet("setup", "check", "smoke", "fold0", "rest", "merge")][string]$Stage)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$py = ".venv\Scripts\python.exe"

if ($Stage -eq "setup") {
    python -m venv .venv
    & $py -m pip install --upgrade pip
    # RTX 50xx (Blackwell) needs a PyTorch build for CUDA 12.8 or newer
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    & $py -m pip install -r requirements.txt
    exit
}

New-Item -ItemType Directory -Force logs | Out-Null
$cmd = @{
    check = "-c `"import torch, pathlib; print(torch.__version__, torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GB'); assert pathlib.Path('data/Dataset_Final/train.csv').exists(), 'data/Dataset_Final missing'; import peft, bitsandbytes, mistral_common; print('setup OK')`""
    smoke = "-u src/15_voxtral_qlora.py --folds 0 --limit 16 --epochs 1"
    fold0 = "-u src/15_voxtral_qlora.py --folds 0"
    rest  = "-u src/15_voxtral_qlora.py --folds 1 2 3 4"
    merge = "-u src/15_voxtral_qlora.py --merge"
}[$Stage]
cmd /c "$py $cmd 2>&1" | Tee-Object -FilePath "logs\$Stage.log"
