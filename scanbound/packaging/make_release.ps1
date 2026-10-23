# Publish a ScanBound release on GitHub, which the in-app updater then offers to everyone.
#
#   powershell -ExecutionPolicy Bypass -File packaging\make_release.ps1
#
# Before running it: raise the version in pyproject.toml and src\scanbound\__init__.py, describe the
# changes under "## <version>" in CHANGELOG.md, then commit and push.  The script checks all of that,
# builds the Windows installer (packaging\windows\build_installer.ps1) and creates the release
# v<version> on the pushed commit, with the CHANGELOG section as the notes (the list the update
# prompt shows) and the installer attached.  Needs the GitHub CLI, signed in:  winget install GitHub.cli
# then  gh auth login.
param(
    [switch]$SkipBuild,             # use the installer already in dist\
    [string]$Repo = "JWrightZuri/ScanBound"
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..").Path

$Version = (Select-String -Path "$Root\pyproject.toml" -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
$InCode = (Select-String -Path "$Root\src\scanbound\__init__.py" -Pattern '^__version__\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
if ($Version -ne $InCode) { throw "pyproject.toml says $Version but src\scanbound\__init__.py says $InCode" }
$Tag = "v$Version"

# the release notes: the CHANGELOG section of this version
$Lines = Get-Content "$Root\CHANGELOG.md" -Encoding utf8
$Start = [array]::FindIndex($Lines, [Predicate[string]]{ param($l) $l -match "^##\s+\[?$([regex]::Escape($Version))\]?(\s|$)" })
if ($Start -lt 0) { throw "CHANGELOG.md has no '## $Version' section: describe the changes there first" }
$End = [array]::FindIndex($Lines, $Start + 1, [Predicate[string]]{ param($l) $l -match '^##\s' })
if ($End -lt 0) { $End = $Lines.Length }
$Notes = ($Lines[($Start + 1)..($End - 1)] -join "`n").Trim()
if (-not $Notes) { throw "the '## $Version' section of CHANGELOG.md is empty" }

# the release is made from committed, pushed code
& gh --version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "the GitHub CLI is missing: winget install GitHub.cli, then gh auth login" }
Push-Location $Root
try {
    if (git status --porcelain -- .) { throw "the scanbound folder has uncommitted changes: commit and push them first" }
    git fetch --quiet origin
    $Head = (git rev-parse HEAD).Trim()
    if (-not (git branch -r --contains $Head)) { throw "commit $Head is not pushed yet: git push first" }
    & gh release view $Tag --repo $Repo *> $null
    if ($LASTEXITCODE -eq 0) { throw "release $Tag exists already: raise the version for a new release" }

    if (-not $SkipBuild) { & "$Root\packaging\windows\build_installer.ps1" }
    $Setup = Join-Path $Root "dist\ScanBound-Setup-$Version.exe"
    if (-not (Test-Path $Setup)) { throw "$Setup is missing: build it first (or leave out -SkipBuild)" }

    $NotesFile = New-TemporaryFile
    Set-Content -Path $NotesFile -Value $Notes -Encoding utf8
    & gh release create $Tag $Setup --repo $Repo --target $Head --title "ScanBound $Version" --notes-file $NotesFile
    if ($LASTEXITCODE -ne 0) { throw "gh release create failed" }
    Remove-Item $NotesFile
    Write-Host "Released ScanBound $Version. Running copies offer it the next time they start."
} finally {
    Pop-Location
}
