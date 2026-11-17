param([string]$Path = (Join-Path $PSScriptRoot '../account-parameters.json'))
. (Join-Path $PSScriptRoot 'validate-config.ps1') -Path $Path
$config = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
$env:PRA_AWS_ACCOUNT_ID = $config.aws.account_id
$env:PRA_AWS_REGION = $config.aws.region
$env:PRA_GITHUB_DEPLOY_ROLE_NAME = $config.github.deploy_role_name
$env:PRA_CDK_QUALIFIER = $config.cdk.qualifier
$env:PRA_COGNITO_USER_POOL_ID = $config.cognito.user_pool_id
$env:PRA_COGNITO_CALLBACK_LOGOUT_URL = $config.cognito.callback_logout_url
$env:PRA_WEBSITE_DOMAIN = $config.website.domain
$env:PRA_COST_ALERT_EMAIL = ([string]$config.notifications.owner_email).Trim()
