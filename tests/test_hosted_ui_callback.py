import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from pra_infra.callback_validation import validate_hosted_ui_callback_url


@pytest.mark.parametrize("value", [None, "", "  ", "http://company.test/", "https://", "https://company.test/#fragment", "https://user:pass@company.test/", "https://company.test/\nnext", "https://company.test:invalid/"])
def test_invalid_callback_rejected(value):
    with pytest.raises(ValueError, match="hosted_ui_callback_url"):
        validate_hosted_ui_callback_url(value)


def test_explicit_callback_accepted():
    assert validate_hosted_ui_callback_url(" https://company.test/callback ") == "https://company.test/callback"


def test_deployment_context_is_explicit():
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text()
    script = (ROOT / "scripts/deploy-aws.ps1").read_text()
    assert workflow.count('-c hosted_ui_callback_url="$PRA_COGNITO_CALLBACK_LOGOUT_URL"') == 2
    assert script.count('hosted_ui_callback_url=$env:PRA_COGNITO_CALLBACK_LOGOUT_URL') == 3


def test_legacy_identifier_not_in_tracked_files():
    identifier = "svr1" + "n7ghpk"
    result = subprocess.run(["git", "grep", "-I", "-l", "--", identifier], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout
