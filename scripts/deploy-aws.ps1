param([switch]$Deploy)
$ErrorActionPreference = 'Stop'
$root = (git rev-parse --show-toplevel)
if ($LASTEXITCODE -ne 0) { throw 'Not in a Git repository.' }
if ((git branch --show-current) -ne 'main') { throw 'Production requires main.' }
if (git status --porcelain) { throw 'Working tree must be clean.' }
. (Join-Path $PSScriptRoot 'load-account-parameters.ps1')
& python (Join-Path $PSScriptRoot 'scan_sensitive_values.py') tracked
if ($LASTEXITCODE -ne 0) { throw 'Security scan failed.' }
$account = aws sts get-caller-identity --query Account --output text --region $env:PRA_AWS_REGION
if ($LASTEXITCODE -ne 0 -or $account -ne $env:PRA_AWS_ACCOUNT_ID) { throw 'AWS identity does not match the configured account.' }
Push-Location (Join-Path $root 'infra')
try {
    cdk synth --quiet
    if ($LASTEXITCODE -ne 0) { throw 'CDK synthesis failed.' }
    cdk diff
    if ($LASTEXITCODE -ne 0) { throw 'CDK diff failed.' }
    if ($Deploy) {
        if ((Read-Host 'Review the diff. Type DEPLOY to confirm production deployment') -cne 'DEPLOY') { throw 'Deployment cancelled.' }
        cdk deploy
        if ($LASTEXITCODE -ne 0) { throw 'CDK deployment failed.' }
    }
} finally { Pop-Location }
