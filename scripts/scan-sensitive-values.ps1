param([Parameter(Mandatory=$true)][ValidateSet('staged','outgoing','tracked')][string]$Mode)
$ErrorActionPreference = 'Stop'
& python (Join-Path $PSScriptRoot 'scan_sensitive_values.py') $Mode
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
