$ErrorActionPreference = 'Stop'
$current = git config --local --get core.hooksPath
if ($LASTEXITCODE -notin @(0, 1)) { throw 'Unable to inspect Git hook configuration.' }
if ($current -and $current -ne '.githooks') {
    $answer = Read-Host 'A custom hooks path exists. Replace it? Type YES to continue'
    if ($answer -cne 'YES') { throw 'Hook installation cancelled.' }
}
git config --local core.hooksPath .githooks
if ($LASTEXITCODE -ne 0) { throw 'Unable to configure Git hooks.' }
Write-Host 'Repository hooks configured. Ensure hook files are executable on Unix.'
