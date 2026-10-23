; Inno Setup script for the ScanBound installer.  build_installer.ps1 stages the files and runs:
;   ISCC.exe /DAppVersion=<version> /DStage=<staged files> /DAssets=<icon and images> /DOutDir=<dist> scanbound.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6B0E4C8A-3F1D-4C55-9C2E-5B7A1D2E9F40}
AppName=ScanBound
AppVersion={#AppVersion}
AppVerName=ScanBound {#AppVersion}
AppPublisher=ScanBound
VersionInfoVersion={#AppVersion}
VersionInfoProductName=ScanBound
; a per-user install needs no administrator rights, and leaves the install folder writable so that
; Machine learning -> Install YOLO... can add packages to the bundled Python
PrivilegesRequired=lowest
DefaultDirName={autopf}\ScanBound
DisableDirPage=no
DefaultGroupName=ScanBound
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
WizardImageFile={#Assets}\wizard.bmp,{#Assets}\wizard@2x.bmp
WizardSmallImageFile={#Assets}\wizard_small.bmp,{#Assets}\wizard_small@2x.bmp
SetupIconFile={#Assets}\scanbound.ico
UninstallDisplayIcon={app}\ScanBound.exe
UninstallDisplayName=ScanBound
OutputDir={#OutDir}
OutputBaseFilename=ScanBound-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; start from a clean runtime on upgrades, so packages from the old version cannot linger
Type: filesandordirs; Name: "{app}\runtime"

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ScanBound"; Filename: "{app}\ScanBound.exe"; AppUserModelID: "ScanBound.ScanBound"
Name: "{group}\Uninstall ScanBound"; Filename: "{uninstallexe}"
Name: "{autodesktop}\ScanBound"; Filename: "{app}\ScanBound.exe"; AppUserModelID: "ScanBound.ScanBound"; Tasks: desktopicon

[Run]
Filename: "{app}\ScanBound.exe"; Description: "{cm:LaunchProgram,ScanBound}"; Flags: nowait postinstall skipifsilent
; Help -> Check for updates runs this installer with /SILENT /RELAUNCH=1: start the new version when it is done
Filename: "{app}\ScanBound.exe"; Flags: nowait; Check: RelaunchAfterUpdate

[Code]
function RelaunchAfterUpdate: Boolean;
begin
  Result := WizardSilent and (ExpandConstant('{param:RELAUNCH|0}') = '1');
end;

[UninstallDelete]
; packages added after installation (YOLO, PyTorch) and compiled Python files are not in the install log
Type: filesandordirs; Name: "{app}\runtime"
