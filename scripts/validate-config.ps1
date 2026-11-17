param([string]$Path = (Join-Path $PSScriptRoot '../account-parameters.json'))
$ErrorActionPreference = 'Stop'
try {
    $config = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    function Check-Keys($object) {
        if ($object -is [System.Array]) { foreach ($item in $object) { Check-Keys $item }; return }
        if ($object -isnot [pscustomobject]) { return }
        foreach ($property in $object.PSObject.Properties) {
            if ($property.Name -match '(?i)secret|password|token|credential|access.?key|private.?key') { throw 'Secret-like configuration field is not permitted.' }
            Check-Keys $property.Value
        }
    }
    Check-Keys $config
    if ($config.aws.account_id -notmatch '^\d{12}$' -or $config.aws.account_id -eq '000000000000') { throw 'Invalid AWS account ID.' }
    if ($config.aws.region -notmatch '^[a-z]{2}(-[a-z0-9]+)+-\d+$') { throw 'Invalid AWS region.' }
    if ($config.github.deploy_role_name -notmatch '^[A-Za-z0-9+=,.@_-]{1,64}$') { throw 'Invalid deployment role name.' }
    if ($config.cdk.qualifier -notmatch '^[A-Za-z0-9_-]{1,10}$') { throw 'Invalid CDK qualifier.' }
    if ($config.cognito.user_pool_id -notmatch '^[a-z0-9-]+_[A-Za-z0-9]+$') { throw 'Invalid Cognito pool ID.' }
    $callback = $null
    if (-not [Uri]::TryCreate([string]$config.cognito.callback_logout_url, [UriKind]::Absolute, [ref]$callback) -or $callback.Scheme -ne 'https' -or -not $callback.Host -or $callback.UserInfo -or $callback.Fragment -or $config.cognito.callback_logout_url -match '\s') { throw 'Invalid Cognito callback/logout URL.' }
    if ($config.website.domain -notmatch '^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}$') { throw 'Invalid website hostname.' }
    $email = [string]$config.notifications.owner_email
    if ($email -match '[\r\n]') { throw 'Set notifications.owner_email to a valid, non-placeholder email address.' }
    $email = $email.Trim()
    if ($email -notmatch '^[A-Za-z0-9.!#$%&''*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$' -or $email -match '(?i)COST_ALERT_EMAIL|REPLACE_WITH|@example\.(com|org|net)$|placeholder') { throw 'Set notifications.owner_email to a valid, non-placeholder email address.' }
} catch { throw "Configuration validation failed: $($_.Exception.Message)" }
