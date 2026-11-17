# Personal Research Assistant — Windows CLI / PowerShell / CMD Toolchain

> The version and test results in this checklist are reported observations from a previous environment, not verification of the current checkout. The secure deployment implementation and live-stack compatibility remain under review. Check the repository's dependency files and CI configuration for the current supported setup.

This file defines the local command-line software needed to develop, test, synthesize, diff, and deploy the **Personal Research Assistant** AWS/CDK application on Windows.

> **Recommended baseline:** Windows 10/11 64-bit, PowerShell, Python 3.12, Node.js 24.x (22.x minimum for AWS CDK), AWS CDK CLI 2.1148.1, AWS CLI v2, Docker Desktop using Linux containers, and Git.

---

## 1. Required software

| Software | Project target / version | Required? | Purpose | Verify |
|---|---:|---|---|---|
| **Windows** | Windows 10/11 64-bit | Yes | Local development host | `winver` |
| **PowerShell** | Windows PowerShell 5.1+ or PowerShell 7.x | Yes | Runs project `.ps1` configuration, validation, security, hook-install, and deployment scripts | `$PSVersionTable.PSVersion` |
| **Python** | **3.12.x** | Yes | Project runtime, local virtual environment, tests, CDK Python app, Lambda runtime alignment | `python --version` |
| **pip** | Current pip compatible with Python 3.12 | Yes | Installs Python dependencies | `python -m pip --version` |
| **venv** | Python 3.12 built-in `venv` | Yes | Isolated local Python environment | `python -m venv --help` |
| **Node.js** | **24.14.1 currently validated locally**; AWS CDK requires **22.x or later** | Yes | Required by AWS CDK/jsii backend | `node --version` |
| **npm** | Version bundled with installed Node.js | Yes | Installs/updates AWS CDK CLI | `npm --version` |
| **AWS CDK CLI** | **2.1148.1** | Yes | `cdk synth`, `cdk diff`, `cdk deploy`, bootstrap/toolkit operations | `cdk --version` |
| **AWS CLI** | **AWS CLI v2** | Yes | AWS authentication, STS identity checks, profiles, SSO login | `aws --version` |
| **Docker Desktop** | Current supported Docker Desktop release, **Linux container engine enabled** | Yes for this project | CDK Lambda bundling and container-image asset builds | `docker version` |
| **Docker Engine / CLI** | Bundled with Docker Desktop | Yes | Executes bundling containers used during CDK synth/diff/deploy | `docker info` |
| **Git** | Current Git for Windows 2.x | Yes | Source control and project Git hooks | `git --version` |

---

## 2. Project-specific Python infrastructure packages

These are Python packages inside `.venv`, not separate Windows applications.

| Package | Validated version | Purpose | Verify |
|---|---:|---|---|
| `aws-cdk-lib` | **2.276.0** | AWS CDK construct library used by the Python infrastructure app | `python -m pip show aws-cdk-lib` |
| `jsii` | **1.142.0** | Runtime bridge used by Python CDK constructs | `python -m pip show jsii` |
| `constructs` | **10.8.1** | CDK constructs programming model | `python -m pip show constructs` |
| `aws-cdk.cloud-assembly-schema` | **55.6.0** observed in the validated environment | Cloud assembly schema package | `python -m pip show aws-cdk.cloud-assembly-schema` |
| `pytest` | Project dev dependency, **8.3+** | Test runner; current project test suite passed with 321 tests | `python -m pytest --version` |

### Important CDK compatibility note

The project was successfully synthesized after using:

```text
aws-cdk-lib 2.276.0
AWS CDK CLI 2.1148.1
jsii 1.142.0
constructs 10.8.1
Node.js 24.14.1
```

Do not casually upgrade `aws-cdk-lib` beyond the CLI-supported Cloud Assembly schema. A prior local install of `aws-cdk-lib 2.277.0` produced a Cloud Assembly schema mismatch with CDK CLI 2.1148.1.

---

## 3. Recommended version policy

### Pin or tightly control

These should be pinned or deliberately upgraded together:

```text
aws-cdk-lib==2.276.0
constructs>=10.5.0,<11.0.0
```

The local validated environment currently has:

```text
constructs==10.8.1
jsii==1.142.0
```

For reproducible infrastructure builds, avoid an unnecessarily broad CDK dependency such as:

```text
aws-cdk-lib>=2.180.0,<3.0.0
```

because a fresh install can silently pull a newer CDK library that produces a Cloud Assembly schema newer than the installed CLI supports.

### Node.js

AWS CDK currently requires:

```text
Node.js 22.x or later
```

The project has successfully run CDK with:

```text
Node.js v24.14.1
```

There is no current project requirement to downgrade Node 24.

### Python

Use:

```text
Python 3.12.x
```

The application and Lambda bundling use Python 3.12. The repository itself requires Python `>=3.12`.

---

## 4. Docker requirements

Docker Desktop must be running before CDK synthesis because the stack bundles Python Lambda assets inside AWS SAM build containers and also builds the RAGAS worker container image.

Required operating mode:

```text
Docker Desktop
└── Linux containers
```

Useful checks:

```powershell
docker version
docker info
docker desktop status
```

If Docker Desktop is installed but stopped:

```powershell
docker desktop start
```

If the CDK bundling process reports this type of error:

```text
failed to connect to the docker API at
npipe:////./pipe/dockerDesktopLinuxEngine
```

start Docker Desktop and verify the Linux engine is available.

A basic Docker DNS/network check:

```powershell
docker run --rm python:3.12-slim python -c "import socket; print(socket.gethostbyname('pypi.org'))"
```

HTTPS test:

```powershell
docker run --rm python:3.12-slim python -c "import urllib.request; print(urllib.request.urlopen('https://pypi.org', timeout=10).status)"
```

---

## 5. AWS authentication requirements

### AWS CLI

Use AWS CLI **v2**.

Verify:

```powershell
aws --version
```

Verify the active AWS identity:

```powershell
aws sts get-caller-identity
```

For an AWS SSO / IAM Identity Center profile:

```powershell
aws sso login --profile <profile-name>
aws sts get-caller-identity --profile <profile-name>
```

Run CDK against a named profile:

```powershell
cdk diff --profile <profile-name>
```

### Expired local temporary credentials

If temporary environment credentials have expired, inspect them:

```powershell
Get-ChildItem Env:AWS* | Select-Object -ExpandProperty Name
```

Clear stale temporary values if necessary:

```powershell
Remove-Item Env:AWS_ACCESS_KEY_ID -ErrorAction SilentlyContinue
Remove-Item Env:AWS_SECRET_ACCESS_KEY -ErrorAction SilentlyContinue
Remove-Item Env:AWS_SESSION_TOKEN -ErrorAction SilentlyContinue
```

Then authenticate again using the intended AWS CLI profile.

> GitHub Actions production deployment uses **GitHub OIDC → AWS STS**. Local `cdk diff` still requires valid local AWS credentials.

---

## 6. Git / Git hooks

Git is required for source control and for the repository's security hooks.

Verify:

```powershell
git --version
git status
```

Project hook installation:

```powershell
.\scripts\install-git-hooks.ps1
```

Expected hooks:

```text
.githooks/pre-commit
.githooks/pre-push
```

Security scanner commands:

```powershell
.venv\Scripts\python.exe scripts\scan_sensitive_values.py tracked
.venv\Scripts\python.exe scripts\scan_sensitive_values.py staged
```

---

## 7. PowerShell scripts used by this repository

Run these from the repository root unless otherwise noted.

### Activate virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Validate local account configuration

```powershell
.\scripts\validate-config.ps1
```

### Load local non-secret AWS/account parameters

```powershell
. .\scripts\load-account-parameters.ps1
```

### Install Git security hooks

```powershell
.\scripts\install-git-hooks.ps1
```

### Run security scanner wrapper

```powershell
.\scripts\scan-sensitive-values.ps1 -Mode tracked
```

### Controlled AWS deployment helper

```powershell
.\scripts\deploy-aws.ps1
```

Do not use the deploy helper until `cdk diff` has been reviewed.

---

## 8. Standard setup sequence

From the repository root:

```powershell
python --version
node --version
npm --version
aws --version
cdk --version
docker version
git --version
```

Create the virtual environment if it does not already exist:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the application dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Install infrastructure dependencies:

```powershell
python -m pip install -r infra\requirements.txt
```

For the currently validated CDK library:

```powershell
python -m pip install "aws-cdk-lib==2.276.0"
```

Install/update the CDK CLI:

```powershell
npm install -g aws-cdk@2.1148.1
```

Verify:

```powershell
cdk --version
python -m pip show aws-cdk-lib jsii constructs
```

---

## 9. Pre-CDK validation

From repository root:

```powershell
.\scripts\validate-config.ps1
. .\scripts\load-account-parameters.ps1
python -m pytest
.venv\Scripts\python.exe scripts\scan_sensitive_values.py tracked
.venv\Scripts\python.exe scripts\scan_sensitive_values.py staged
aws sts get-caller-identity
docker info
```

Expected test baseline from the validated local environment:

```text
321 passed
```

---

## 10. CDK workflow

### Synthesize

```powershell
cd infra
cdk synth
```

Current non-blocking warning:

```text
aws-cdk-lib.aws_lambda.FunctionOptions#logRetention is deprecated
use logGroup instead
```

This warning does not prevent synthesis.

The Docker bundling process may also print a pip cache warning such as:

```text
The directory '/.cache/pip' ... is not owned or is not writable
```

This is non-blocking when dependency installation and bundling complete successfully.

### Compare with deployed AWS stack

```powershell
cdk diff
```

or:

```powershell
cdk diff --profile <profile-name>
```

Review carefully for:

```text
[-] resource deletions
[+] unexpected resources
[~] modifications
replacement operations
Cognito changes
S3 replacement
API Gateway replacement
Lambda replacement
IAM changes
SQS changes
Step Functions changes
```

### Deploy

Only after a reviewed `cdk diff`:

```powershell
cdk deploy
```

For this repository, production deployment should normally occur through the approved GitHub Actions OIDC workflow rather than long-lived AWS access keys.

---

## 11. One-command environment inventory

Run this in PowerShell to record the main toolchain:

```powershell
Write-Host "=== PowerShell ==="
$PSVersionTable.PSVersion

Write-Host "`n=== Python ==="
python --version
python -m pip --version

Write-Host "`n=== Node / npm ==="
node --version
npm --version

Write-Host "`n=== AWS ==="
aws --version
cdk --version

Write-Host "`n=== Docker ==="
docker --version

Write-Host "`n=== Git ==="
git --version

Write-Host "`n=== Python CDK packages ==="
python -m pip show aws-cdk-lib jsii constructs aws-cdk.cloud-assembly-schema
```

---

## 12. Validated project baseline

At the time this checklist was created, the known-good local tooling combination was:

```text
Operating System:          Windows 64-bit
Shell:                     PowerShell
Python:                    3.12.x target
Node.js:                   24.14.1
AWS CDK CLI:               2.1148.1
aws-cdk-lib:               2.276.0
jsii:                      1.142.0
constructs:                10.8.1
cloud-assembly-schema:     55.6.0
AWS CLI:                   v2
Docker Desktop:            required; Linux engine
Git:                       Git for Windows 2.x
```

Known validation results:

```text
pytest:                    321 passed
tracked security scan:     PASS
staged security scan:      PASS
config validation:         PASS
account parameter loading: PASS
cdk synth:                 PASS
cdk diff:                  pending valid local AWS credentials
```

---

## 13. Optional tools

These are useful but not required for the CDK workflow itself.

| Tool | Suggested version | Use |
|---|---:|---|
| **GitHub CLI (`gh`)** | Current 2.x release | GitHub auth, repo operations, Actions inspection |
| **VS Code** | Current stable | Editing/debugging |
| **Windows Terminal** | Current stable | Better PowerShell/CLI terminal |
| **WSL2** | Current Windows-supported version | Docker Desktop backend / optional Linux shell |
| **nvm-windows** | Optional | Switching Node versions when multiple projects need different versions |

---

## 14. Software that is *not* required

The following are not required for this project's current Windows/Python CDK workflow:

```text
Java / JDK
Maven
.NET SDK
Go
TypeScript compiler
Terraform
kubectl
Helm
nvm-windows
```

They should only be added if a future project feature explicitly requires them.

---

## 15. Official references

- AWS CDK prerequisites: https://docs.aws.amazon.com/cdk/v2/guide/prerequisites.html
- AWS CDK versions: https://docs.aws.amazon.com/cdk/api/versions.html
- AWS CDK getting started / CLI: https://docs.aws.amazon.com/cdk/v2/guide/getting-started.html
- AWS CLI v2 installation: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
- Docker Desktop CLI: https://docs.docker.com/desktop/features/desktop-cli/
