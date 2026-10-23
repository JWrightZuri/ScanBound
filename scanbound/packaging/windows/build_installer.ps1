# Build the ScanBound Windows installer: dist\ScanBound-Setup-<version>.exe
#
#   powershell -ExecutionPolicy Bypass -File packaging\windows\build_installer.ps1
#
# Steps: download the official Python runtime from nuget.org (cached), install ScanBound and its
# standard dependencies into it (not YOLO / PyTorch, which the app installs on request), trim it,
# check that every ScanBound module imports, compile ScanBound.exe, make the icon and wizard
# images from the logo, and run Inno Setup (winget install JRSoftware.InnoSetup).
param(
    [string]$PythonVersion = "3.12.10",
    [switch]$SkipRuntime            # reuse the staged runtime from the last build (only the launcher / installer changed)
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is far slower with its progress bar

$Here = $PSScriptRoot
$Root = (Resolve-Path "$Here\..\..").Path
$Work = Join-Path $Here "build"
$Stage = Join-Path $Work "stage"
$Assets = Join-Path $Work "assets"
$Runtime = Join-Path $Stage "runtime"
$Out = Join-Path $Root "dist"
$Logo = Join-Path $Root "src\scanbound\app\resources\sb_logo.png"

$Version = (Select-String -Path "$Root\pyproject.toml" -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
Write-Host "Building ScanBound $Version installer"

function Invoke-Checked([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe $($arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

$Iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
          "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) { throw "Inno Setup 6 not found. Install it with:  winget install JRSoftware.InnoSetup" }
$Csc = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path $Csc)) { throw "The .NET Framework C# compiler was not found at $Csc" }

New-Item -ItemType Directory -Force $Work, $Out | Out-Null

if (-not $SkipRuntime) {
    # 1. the Python runtime: python.org's own build, published on nuget.org as a plain zip
    $Package = Join-Path $Work "python-$PythonVersion.zip"
    if (-not (Test-Path $Package)) {
        Write-Host "Downloading Python $PythonVersion ..."
        Invoke-WebRequest "https://www.nuget.org/api/v2/package/python/$PythonVersion" -OutFile "$Package.part"
        Move-Item "$Package.part" $Package
    }
    if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
    $Unpacked = Join-Path $Work "python-$PythonVersion"
    if (Test-Path $Unpacked) { Remove-Item -Recurse -Force $Unpacked }
    Expand-Archive $Package $Unpacked
    New-Item -ItemType Directory -Force $Stage | Out-Null
    Move-Item (Join-Path $Unpacked "tools") $Runtime
    Remove-Item -Recurse -Force $Unpacked
    $Python = Join-Path $Runtime "python.exe"

    # 2. ScanBound and its standard dependencies (the same extras run_scanbound.bat installs)
    $env:PYTHONNOUSERSITE = "1"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    Invoke-Checked $Python @("-m", "ensurepip", "--upgrade")
    Invoke-Checked $Python @("-m", "pip", "install", "--upgrade", "--no-warn-script-location", "pip")
    Invoke-Checked $Python @("-m", "pip", "install", "--no-warn-script-location", "--no-cache-dir",
                             "$Root[gui,clusters,formats]")

    # 3. trim: test suites, the pip launchers in Scripts (they hold absolute paths), build leftovers
    Get-ChildItem $Runtime\Lib\site-packages -Recurse -Directory -Filter tests | Remove-Item -Recurse -Force
    foreach ($d in @("Scripts", "Lib\site-packages\pyqtgraph\examples", "Lib\test", "Lib\idlelib", "Lib\tkinter")) {
        $p = Join-Path $Runtime $d
        if (Test-Path $p) { Remove-Item -Recurse -Force $p }
    }
    if (Test-Path "$Root\build") { Remove-Item -Recurse -Force "$Root\build" }   # setuptools' wheel build folder

    # the Qt parts ScanBound never loads: QML / Quick, 3D, multimedia (with its FFmpeg), PDF, Designer,
    # devices and networking modules.  It uses widgets, OpenGL and SVG.  This also keeps the deepest
    # paths (qml\QtQuick\Controls\...) out, which broke installs into long folder paths.
    $PyQt = Join-Path $Runtime "Lib\site-packages\PyQt6"
    $Unused = @("QAxContainer", "QtBluetooth", "QtDesigner", "QtHelp", "QtMultimedia", "QtMultimediaWidgets", "QtNfc",
                "QtPdf", "QtPdfWidgets", "QtPositioning", "QtQml", "QtQuick", "QtQuick3D", "QtQuickWidgets",
                "QtRemoteObjects", "QtSensors", "QtSerialPort", "QtSpatialAudio", "QtSql", "QtStateMachine",
                "QtTextToSpeech", "QtWebChannel", "QtWebSockets")
    foreach ($m in $Unused) { Remove-Item -Force "$PyQt\$m.pyd", "$PyQt\$m.pyi" -ErrorAction SilentlyContinue }
    $UnusedDlls = @("Qt6Quick*", "Qt6Qml*", "Qt6Labs*", "Qt63D*", "Qt6Multimedia*", "Qt6SpatialAudio*", "Qt6Pdf*",
                    "Qt6Designer*", "Qt6Help*", "Qt6Bluetooth*", "Qt6Nfc*", "Qt6Positioning*", "Qt6Sensors*",
                    "Qt6SerialPort*", "Qt6TextToSpeech*", "Qt6WebChannel*", "Qt6WebSockets*", "Qt6WebView*",
                    "Qt6RemoteObjects*", "Qt6Sql*", "Qt6StateMachine*", "Qt6Scxml*", "Qt6ShaderTools*",
                    "av*.dll", "sw*.dll")
    foreach ($p in $UnusedDlls) { Remove-Item -Force "$PyQt\Qt6\bin\$p" -ErrorAction SilentlyContinue }
    foreach ($d in @("qml", "qsci", "plugins\assetimporters", "plugins\geometryloaders", "plugins\help",
                     "plugins\multimedia", "plugins\position", "plugins\qmllint", "plugins\qmlls", "plugins\renderers",
                     "plugins\sceneparsers", "plugins\scxmldatamodel", "plugins\sensors", "plugins\sqldrivers",
                     "plugins\texttospeech", "plugins\webview")) {
        $p = Join-Path "$PyQt\Qt6" $d
        if (Test-Path $p) { Remove-Item -Recurse -Force $p }
    }

    # 4. every ScanBound module must import from the trimmed runtime (YOLO-only modules may say YOLO is missing)
    $Check = @'
import importlib, pkgutil, sys, scanbound
bad = []
for m in pkgutil.walk_packages(scanbound.__path__, "scanbound."):
    try:
        importlib.import_module(m.name)
    except ImportError as e:
        if getattr(e, "name", "") not in ("ultralytics", "torch", "torchvision"):
            bad.append(f"{m.name}: {e!r}")
    except Exception as e:
        bad.append(f"{m.name}: {e!r}")
print("\n".join(bad) or "all ScanBound modules import")
if not bad:                          # and the main window opens (loads the Qt platform and image plugins)
    from PyQt6.QtWidgets import QApplication
    from scanbound.app.mainwindow import MainWindow
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.processEvents()
    w.close()
    print("main window opens")
sys.exit(1 if bad else 0)
'@
    $CheckFile = Join-Path $Work "check_imports.py"     # a file: PowerShell 5 mangles quotes in -c arguments
    Set-Content -Path $CheckFile -Value $Check -Encoding ascii
    $env:QT_QPA_PLATFORM = "offscreen"
    Invoke-Checked $Python @("-I", $CheckFile)
    Remove-Item Env:QT_QPA_PLATFORM
}

# 5. icon, wizard images and the launcher
Invoke-Checked (Join-Path $Runtime "python.exe") @("$Here\make_assets.py", $Logo, $Assets)
Invoke-Checked $Csc @("/nologo", "/target:winexe", "/optimize+", "/platform:anycpu",
                      "/win32icon:$Assets\scanbound.ico", "/out:$Stage\ScanBound.exe",
                      "/reference:System.Windows.Forms.dll", "$Here\ScanBound.cs")

# 6. the installer
Invoke-Checked $Iscc @("/Q", "/DAppVersion=$Version", "/DStage=$Stage", "/DAssets=$Assets", "/DOutDir=$Out",
                       "$Here\scanbound.iss")
$Setup = Join-Path $Out "ScanBound-Setup-$Version.exe"
Write-Host ("Built {0} ({1:N0} MB)" -f $Setup, ((Get-Item $Setup).Length / 1MB))
