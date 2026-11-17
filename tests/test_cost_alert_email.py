import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "email_validation", ROOT / "infra/pra_infra/email_validation.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_valid_owner_email_is_trimmed():
    assert module.validate_owner_email("  ops@company.test  ") == "ops@company.test"


@pytest.mark.parametrize("value", [None, "", "REPLACE_WITH_VALID_EMAIL", "you@example.com", "ops@company.test\n", "COST_ALERT_EMAIL=ops@company.test", "invalid@address"])
def test_invalid_owner_email_is_rejected(value):
    with pytest.raises(ValueError, match="owner_email"):
        module.validate_owner_email(value)


def test_github_deploy_uses_same_email_for_synth_and_deploy():
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
    assert workflow.count(' -c owner_email="$COST_ALERT_EMAIL"') == 2
    assert workflow.count("COST_ALERT_EMAIL: ${{ vars.COST_ALERT_EMAIL }}") == 3


@pytest.mark.skipif(not shutil.which("powershell"), reason="Windows PowerShell unavailable")
@pytest.mark.parametrize("email,valid", [(" ops@company.test ", True), ("", False), ("REPLACE_WITH_VALID_EMAIL", False), ("ops@company.test\n", False), ("COST_ALERT_EMAIL=ops@company.test", False)])
def test_local_configuration_email(tmp_path, email, valid):
    config = {
        "aws": {"account_id": "123456789012", "region": "us-east-1"},
        "github": {"deploy_role_name": "test-role"},
        "cdk": {"qualifier": "hnb659fds"},
        "cognito": {"user_pool_id": "us-east-1_Example", "callback_logout_url": "https://company.test/callback"},
        "website": {"domain": "company.test"},
        "notifications": {"owner_email": email},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    script = ROOT / "scripts/load-account-parameters.ps1"
    command = f". '{script}' -Path '{path}'; if ($env:PRA_COST_ALERT_EMAIL -cne 'ops@company.test') {{ exit 2 }}"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)
    assert (result.returncode == 0) is valid
