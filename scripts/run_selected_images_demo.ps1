param(
    [string]$DemoRoot = ".demo-data\selected-images-demo",
    [string]$ReleaseTag = "demo-assets-v0.1",
    [double]$Threshold = 0.4,
    [string]$Device = "cpu",
    [switch]$SkipInstall,
    [switch]$ForceDownload
)

$ErrorActionPreference = "Stop"
$Repo = "UARK-NED3/BubbleID-Workflow"
$PythonExe = ".venv-bubbleid-demo\Scripts\python.exe"
$ImagesZip = Join-Path $DemoRoot "selected-images.zip"
$ImagesDir = Join-Path $DemoRoot "selected-images"
$WeightsPath = Join-Path $DemoRoot "model_1class.pth"
$OutputDir = Join-Path $DemoRoot "outputs"

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-Python310 {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        & py -3.10 --version *> $null
        if ($LASTEXITCODE -eq 0) {
            return "py -3.10"
        }
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return "python"
    }
    throw "Python was not found. Install Python 3.10, then rerun this script."
}

function Test-PythonImport($ModuleName) {
    & $PythonExe -c "import $ModuleName" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Install-Detectron2 {
    if (Test-PythonImport "detectron2") {
        Write-Host "Detectron2 is already installed."
        return
    }

    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "Visual Studio Build Tools were not found. Install Microsoft C++ Build Tools, then rerun this script."
    }

    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) {
        throw "Visual Studio Build Tools are installed, but the C++ compiler component was not found."
    }

    $vsDevCmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $vsDevCmd)) {
        throw "Could not find VsDevCmd.bat at $vsDevCmd"
    }

    Write-Step "Building Detectron2 from source"
    & cmd.exe /c "`"$vsDevCmd`" -arch=x64 && set DISTUTILS_USE_SDK=1 && `"$PythonExe`" -m pip install --no-build-isolation `"git+https://github.com/facebookresearch/detectron2.git`""
    if ($LASTEXITCODE -ne 0) {
        throw "Detectron2 installation failed."
    }
}

Write-Step "Preparing demo folders"
New-Item -ItemType Directory -Force -Path $DemoRoot, $ImagesDir, $OutputDir | Out-Null

if (-not (Test-Path $PythonExe)) {
    Write-Step "Creating Python virtual environment"
    $pythonCommand = Get-Python310
    Invoke-Expression "$pythonCommand -m venv .venv-bubbleid-demo"
}

if (-not $SkipInstall) {
    Write-Step "Installing Python dependencies"
    & $PythonExe -m pip install --upgrade pip setuptools wheel
    & $PythonExe -m pip install -e .
    & $PythonExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    & $PythonExe -m pip install opencv-python matplotlib pycocotools fvcore iopath yacs
    Install-Detectron2
}

Write-Step "Downloading demo assets"
$baseUrl = "https://github.com/$Repo/releases/download/$ReleaseTag"
if ($ForceDownload -or -not (Test-Path $ImagesZip)) {
    Invoke-WebRequest "$baseUrl/selected-images.zip" -OutFile $ImagesZip
}
if ($ForceDownload -or -not (Test-Path $WeightsPath)) {
    Invoke-WebRequest "$baseUrl/model_1class.pth" -OutFile $WeightsPath
}

Write-Step "Extracting selected images"
if ($ForceDownload -or -not (Get-ChildItem $ImagesDir -Filter *.jpg -ErrorAction SilentlyContinue)) {
    Expand-Archive -Path $ImagesZip -DestinationPath $ImagesDir -Force
}

Write-Step "Running BubbleID Workflow segmentation"
& $PythonExe -m bubbleid_agent.cli segment-images $ImagesDir $WeightsPath $OutputDir --threshold $Threshold --device $Device

Write-Step "Building overlay contact sheet"
$contactSheetScript = Join-Path $DemoRoot "make_contact_sheet.py"
@"
from pathlib import Path
from PIL import Image, ImageDraw
import math

overlay_dir = Path(r"$OutputDir") / "overlays"
paths = sorted(overlay_dir.glob("*_overlay.jpg"))
thumb_w, thumb_h = 320, 240
cols = 4
rows = math.ceil(len(paths) / cols)
label_h = 44
sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
draw = ImageDraw.Draw(sheet)
for idx, path in enumerate(paths):
    img = Image.open(path).convert("RGB")
    img.thumbnail((thumb_w, thumb_h))
    x = (idx % cols) * thumb_w
    y = (idx // cols) * (thumb_h + label_h)
    sheet.paste(img, (x + (thumb_w - img.width) // 2, y))
    draw.text((x + 6, y + thumb_h + 4), path.name.replace("_overlay.jpg", "")[:42], fill=(0, 0, 0))
out = Path(r"$OutputDir") / "overlay_contact_sheet.jpg"
sheet.save(out, quality=90)
print(out.resolve())
"@ | Set-Content -Path $contactSheetScript -Encoding UTF8
& $PythonExe $contactSheetScript

Write-Step "Demo complete"
Write-Host "CSV:       $(Resolve-Path (Join-Path $OutputDir 'vapor_fraction_results.csv'))"
Write-Host "Summary:   $(Resolve-Path (Join-Path $OutputDir 'summary.json'))"
Write-Host "Overlays:  $(Resolve-Path (Join-Path $OutputDir 'overlays'))"
Write-Host "Contact:   $(Resolve-Path (Join-Path $OutputDir 'overlay_contact_sheet.jpg'))"
