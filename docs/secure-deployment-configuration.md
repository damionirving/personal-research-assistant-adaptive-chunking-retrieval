# Secure Deployment Configuration

> **Release gate:** The scanner and deployment wrapper require further verification, and compatibility with the live stack has not been established. Do not deploy until the checks and a CDK diff have been reviewed.

The repository uses a root-local `account-parameters.json` for deployment metadata. This file is ignored by Git and must not contain credentials. The placeholder-only example is at `config/account-parameters.example.json`. Keep credentials in an appropriate secret store, such as AWS Secrets Manager, rather than either JSON file.

## Account Configuration Example

Create `account-parameters.json` at the repository root using the structure below. Replace each placeholder locally with the settings for your existing deployment. Do not commit this file or include access keys, passwords, tokens, or other secrets.

```json
{
  "aws": {
    "account_id": "<AWS_ACCOUNT_ID>",
    "region": "<AWS_REGION>"
  },
  "github": {
    "deploy_role_name": "<GITHUB_DEPLOY_ROLE_NAME>"
  },
  "cdk": {
    "qualifier": "<CDK_QUALIFIER>"
  },
  "cognito": {
    "user_pool_id": "<COGNITO_USER_POOL_ID>",
    "callback_logout_url": "https://<your-domain.com>/research-assistant/"
  },
  "website": {
    "domain": "<your-domain.com>"
  },
  "notifications": {
    "owner_email": "you@example.com"
  }
}
```

`account_id` and `region` identify the target AWS environment. `deploy_role_name` identifies the GitHub deployment role, while `qualifier` identifies the CDK bootstrap configuration. `user_pool_id` must identify the existing Cognito user pool: supplying a different pool can change the authentication resources. These fields are deployment metadata, not credentials. The example values are placeholders and will not pass validation until replaced.

`website.domain` is a hostname without a scheme, path, or port. The loader sets `PRA_WEBSITE_DOMAIN`; CDK uses it for the site's HTTPS CORS origin. `cognito.callback_logout_url` is the full HTTPS URL registered for both Cognito callbacks and logout, independently of the CORS hostname. For GitHub deployments, set the production environment variables `WEBSITE_DOMAIN` and `COGNITO_CALLBACK_LOGOUT_URL` to the corresponding values. Old Firebase origins are not included. Update DNS and hosting separately, and review `cdk diff` before deployment.

Run the validator before dot-sourcing the loader. The loader exports the values only into the current PowerShell process; GitHub Actions uses separately configured GitHub variables and does not read this local file. The current implementation reads the qualifier into an environment variable, but its use by CDK has not yet been verified. Review the CDK bootstrap configuration and deployment diff before deploying.

The PowerShell validator and loader are `scripts/validate-config.ps1` and `scripts/load-account-parameters.ps1`. The loader must be dot-sourced to set variables in the current PowerShell process:

```powershell
. ./scripts/load-account-parameters.ps1
```

CDK uses `PRA_AWS_ACCOUNT_ID` and `PRA_AWS_REGION` before the corresponding CDK context values. Supply the existing Cognito pool through `PRA_COGNITO_USER_POOL_ID` or the `cognito_user_pool_id` context value. Missing pool configuration fails rather than silently creating a replacement. This change has not been checked against the live stack; run and review `cdk diff` before deployment.

The manual GitHub deployment workflow reads `AWS_ACCOUNT_ID`, `AWS_REGION`, `AWS_DEPLOY_ROLE_NAME`, `COGNITO_USER_POOL_ID`, `COGNITO_CALLBACK_LOGOUT_URL`, and `WEBSITE_DOMAIN` from GitHub variables and assumes the configured role using OIDC. Its production environment and main-branch checks do not themselves configure GitHub protection or AWS IAM trust. An administrator must verify those settings separately.

The SNS alarm subscription requires an explicit notification address. Set `notifications.owner_email` to a real address in the ignored local JSON; the loader validates it and exports `PRA_COST_ALERT_EMAIL`. Set `COST_ALERT_EMAIL` to the same address in GitHub Actions variables: GitHub runners cannot access the local file. The workflow passes it to both synth and deploy with `-c owner_email=...`; using the same value prevents the reviewed template and deployment from diverging. After deployment, confirm the subscription using the email sent by SNS. A subscription is not active until confirmed.

If an earlier stack is in `ROLLBACK_FAILED` or `DELETE_FAILED`, inspect CloudFormation stack events and resolve the failed resources or deletion before retrying. Do not delete retained resources without reviewing their contents and recovery requirements.

## Operator Setup

For Windows prerequisites, version checks, and troubleshooting, see the [Windows CLI toolchain guide](personal-research-assistant-windows-cli-toolchain.md). Its reported baseline is historical and does not establish that this checkout is ready to deploy.

1. Install Python, PowerShell, AWS CLI, CDK CLI, Docker, and the repository dependencies described in the infrastructure README. Authenticate with a short-lived AWS profile or SSO session.
2. Prepare the ignored root configuration from the example and enter the existing deployment settings. The pool ID must identify the already deployed pool. Do not put passwords, access keys, or tokens in this file.
3. Run `powershell -NoProfile -File scripts/validate-config.ps1`. Run `powershell -NoProfile -File scripts/install-git-hooks.ps1` to opt in to repository-local hooks. The installer asks before replacing a different hooks path.
4. Run `powershell -NoProfile -File scripts/deploy-aws.ps1` on a clean `main` branch. It verifies AWS identity and runs synth and diff without deploying. Review the diff for replacement or deletion, especially Cognito, storage, API, Lambda, and IAM resources. Stop if any change is unexpected.
5. Only after review and authorization, `powershell -NoProfile -File scripts/deploy-aws.ps1 -Deploy` repeats the checks and requires a confirmation before running CDK deploy. This repository's changes have not been verified against the live stack.

For GitHub deployment, configure the listed variables in the protected `production` environment. Require reviewers and restrict the environment to `main`; protect the branch and require reviews. Configure the AWS OIDC trust with audience `sts.amazonaws.com` and subject `repo:damionirving/personal-research-assistant-adaptive-chunking-retrieval:environment:production`. Verify the actual trust policy and CDK bootstrap permissions in AWS. The runner does not use the local JSON file. GitHub variables are configuration, not credential storage.

## Security Checks and Limits

`python scripts/scan_sensitive_values.py staged` scans index blobs; `outgoing` consumes pre-push ref updates and scans reachable outgoing commits; `tracked` scans the index in CI. Checks identify selected key patterns, private-key headers, account-specific IAM/OIDC ARNs, prohibited paths, and identifiers from the local configuration when present. Data-directory content is checked for these credential patterns but excluded from the local-identifier check. These checks are not a substitute for a dedicated secret scanner: they cannot recognize every secret or guarantee that excluded content is safe. Hooks are bypassable; branch protection, CI, production approvals and IAM are the enforcement boundaries.

If validation or identity checks fail, correct the local configuration or AWS profile; do not disable TLS verification. A failed scanner reports file names and rule labels rather than values. Historical commits and copies can still contain sensitive data; rotate exposed credentials and plan history cleanup separately. Do not treat an account ID as an authentication secret.

## Deployment Architecture

These supplied diagrams are illustrative. Their details have not been verified against the deployed infrastructure.

![Secure AWS deployment workflow](Secure%20AWS%20Deployment%20Workflow.png)

*Prepare and verify the target account, review the synthesized changes, then deploy through GitHub OIDC and CDK roles using temporary credentials. The diagram describes the intended workflow; verify the review gates, IAM trust, and runtime secret access in the target environment.*

![Personal Assistant Hotel Deployment Strategy](Personal%20Assistant%20Hotel%20Deployment%20Strategy.png)

*Deployment strategy overview.*

![Secure Modular Deployment Architecture](Secure%20Modular%20Deployment%20Architecture.png)

*Illustrative secure modular deployment architecture.*
