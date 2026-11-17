# GitHub OIDC Deployment

This project can deploy to AWS from GitHub Actions without storing long-lived AWS access keys. GitHub issues a short-lived OpenID Connect token, AWS validates that token, and the workflow assumes an IAM role dedicated to CDK deployment.

The deploy workflow is defined in:

```text
.github/workflows/deploy.yml
```

Production deployment is intentionally manual. GitHub Actions runs the production workflow only through `workflow_dispatch` from `main`, requires an explicit production confirmation input, and pauses at the protected `production` environment for administrator approval.

## Current AWS Target

| Setting | Value |
| --- | --- |
| AWS account | `000000000000` |
| Region | `us-east-1` |
| GitHub owner | `damionirving` |
| GitHub repo | `personal-research-assistant-adaptive-chunking-retrieval` |
| CDK stack | `PersonalResearchAssistantStack` |
| GitHub environment | `production` |

## Workflow Deployment Identifiers

Configure these non-secret variables in the protected production environment:

| Variable | Value |
| --- | --- |
| `AWS_ACCOUNT_ID` | `<AWS_ACCOUNT_ID>` |
| `AWS_REGION` | `<AWS_REGION>` |
| `AWS_DEPLOY_ROLE_NAME` | `<GITHUB_DEPLOY_ROLE_NAME>` |
| `COGNITO_USER_POOL_ID` | `<COGNITO_USER_POOL_ID>` |
| `COGNITO_CALLBACK_LOGOUT_URL` | `https://<your-domain.com>/research-assistant/` |
| `WEBSITE_DOMAIN` | `<your-domain.com>` |

These values identify the target account, region, and role name. They are not AWS credentials. No `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, or `AWS_SESSION_TOKEN` is stored in GitHub.

LangSmith tracing now follows the same principle. The current MVP expects the LangSmith API key to live in an existing AWS Secrets Manager secret named `LANGSMITH_API_KEY`, where the deployed Lambda reads it at runtime along with the optional `LANGSMITH_ENDPOINT` value. GitHub Actions does not need the LangSmith API key for this deployment path.

## AWS OIDC Provider

Create the GitHub Actions OIDC provider once per AWS account:

```cmd
aws iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com --client-id-list sts.amazonaws.com --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 --profile admin-programatic
```

If the provider already exists, AWS will return an entity-exists error. That is fine; use the existing provider.

Expected provider ARN:

```text
arn:aws:iam::000000000000:oidc-provider/token.actions.githubusercontent.com
```

## Deployment Role Trust Policy

Create a local file named `github-oidc-trust-policy.json` outside the repo or in a temporary folder:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::000000000000:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringEquals": {
          "token.actions.githubusercontent.com:sub": "repo:damionirving/personal-research-assistant-adaptive-chunking-retrieval:environment:production"
        }
      }
    }
  ]
}
```

Create the deploy role. The trust policy must use the exact public repository subject shown above; do not replace it with a wildcard:

```cmd
aws iam create-role --role-name GitHubPersonalResearchAssistantDeployRole --assume-role-policy-document file://github-oidc-trust-policy.json --profile admin-programatic
```

## Deployment Role Permissions

The role should deploy through the already bootstrapped CDK roles instead of carrying broad permanent deployment permissions itself.

Create `github-cdk-deploy-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AssumeCdkBootstrapRoles",
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Resource": [
        "arn:aws:iam::000000000000:role/cdk-hnb659fds-deploy-role-000000000000-us-east-1",
        "arn:aws:iam::000000000000:role/cdk-hnb659fds-file-publishing-role-000000000000-us-east-1",
        "arn:aws:iam::000000000000:role/cdk-hnb659fds-image-publishing-role-000000000000-us-east-1",
        "arn:aws:iam::000000000000:role/cdk-hnb659fds-lookup-role-000000000000-us-east-1"
      ]
    },
    {
      "Sid": "ReadCdkBootstrapAssets",
      "Effect": "Allow",
      "Action": [
        "cloudformation:DescribeStacks",
        "cloudformation:DescribeStackEvents",
        "cloudformation:DescribeStackResources",
        "cloudformation:GetTemplate",
        "s3:GetBucketLocation",
        "s3:ListBucket"
      ],
      "Resource": "*"
    }
  ]
}
```

Attach the policy:

```cmd
aws iam put-role-policy --role-name GitHubPersonalResearchAssistantDeployRole --policy-name PersonalResearchAssistantCdkDeploy --policy-document file://github-cdk-deploy-policy.json --profile admin-programatic
```

## CDK Bootstrap Requirement

GitHub OIDC deployment expects the account and region to already be bootstrapped:

```cmd
cd C:\path\to\your\project\infra
cdk.cmd bootstrap aws://000000000000/us-east-1 --profile admin-programatic
```

An administrator must verify the bootstrap stack independently before deployment.

## Run A Deployment

Production deployment is not triggered by pull requests or ordinary pushes. Run it manually in GitHub:

```text
Repository -> Actions -> Deploy -> Run workflow -> main -> confirm_production=true -> Run workflow
```

The workflow will:

1. Check out the repository at `main`.
2. Require explicit production confirmation.
3. Pause for the protected `production` environment approval.
4. Install Python and Node.
5. Install CDK and project dependencies.
6. Run `python -m pytest`.
7. Assume the AWS deploy role through OIDC only after approval.
8. Run `cdk synth`.
9. Run `cdk deploy --require-approval never --force`.

The `production` environment must have required reviewers and a deployment branch policy restricting deployments to `main`. Do not allow pull requests or forks to use this environment.

Runtime resiliency changes, such as Bedrock retry/backoff behavior, are deployed through this same workflow. A successful deployment updates the Lambda package and environment wiring; it does not change Bedrock model access in the AWS console.

Cognito Hosted UI auth-flow changes, including the authorization code + PKCE app-client configuration and token lifetimes, are also deployed through this workflow because they are defined in the CDK stack.

## Troubleshooting

If role assumption fails, check:

- The GitHub repository variables are set.
- The workflow was manually started from `main` with `confirm_production=true`.
- The GitHub environment name is `production`.
- The IAM trust policy allows only `repo:damionirving/personal-research-assistant:environment:production`.
- The OIDC provider exists in account `000000000000`.

If CDK cannot publish assets or update the stack, check:

- The CDK bootstrap stack exists in `us-east-1`.
- The GitHub deploy role can assume the CDK bootstrap roles.
- The bootstrap roles trust principals from the same account.

If the deployed chat endpoint intermittently reports that the LLM provider is unavailable, check the Lambda CloudWatch logs for `bedrock_generation_retry` and `bedrock_generation_failed`. Retries indicate transient Bedrock throttling, timeout, or temporary service errors. Immediate failures usually indicate model access, IAM, validation, or credential configuration issues.

If upload or chat reports `Please Sign in Again. MM/DD/YYYY HH:MM AM/PM`, the browser could not refresh the Cognito session or the API rejected the token. Sign in again through the Hosted UI. If the issue appears immediately after a deployment, verify that the deployed app client allows authorization code grant and that callback/logout URLs match the API URL.
