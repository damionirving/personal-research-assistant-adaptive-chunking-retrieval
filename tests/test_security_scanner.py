import os
import subprocess
import sys
from pathlib import Path


SCANNER = Path(__file__).resolve().parents[1] / "scripts" / "scan_sensitive_values.py"


def run(repo, *args, input=None):
    return subprocess.run(args, cwd=repo, input=input, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"})


def test_index_scanner_reads_staged_blobs(tmp_path):
    assert run(tmp_path, "git", "init").returncode == 0
    path = tmp_path / "sample.txt"
    path.write_text("<AWS_ACCOUNT_ID> and AWS_SECRET_ACCESS_KEY\n", encoding="utf-8")
    assert run(tmp_path, "git", "add", "sample.txt").returncode == 0
    assert run(tmp_path, sys.executable, str(SCANNER), "staged").returncode == 0
    path.write_text("AKIA" + "X" * 16, encoding="utf-8")
    assert run(tmp_path, sys.executable, str(SCANNER), "staged").returncode == 0
    assert run(tmp_path, "git", "add", "sample.txt").returncode == 0
    assert run(tmp_path, sys.executable, str(SCANNER), "staged").returncode != 0


def test_outgoing_scanner_requires_push_input(tmp_path):
    assert run(tmp_path, "git", "init").returncode == 0
    assert run(tmp_path, sys.executable, str(SCANNER), "outgoing", input=b"").returncode != 0


def test_outgoing_scans_earlier_commit_not_only_tip(tmp_path):
    assert run(tmp_path, "git", "init").returncode == 0
    assert run(tmp_path, "git", "config", "user.name", "Test User").returncode == 0
    assert run(tmp_path, "git", "config", "user.email", "test@example.invalid").returncode == 0
    path = tmp_path / "sample.txt"
    path.write_text("safe", encoding="utf-8")
    assert run(tmp_path, "git", "add", "sample.txt").returncode == 0
    assert run(tmp_path, "git", "commit", "-m", "base").returncode == 0
    base = run(tmp_path, "git", "rev-parse", "HEAD").stdout.decode().strip()
    path.write_text("ASIA" + "X" * 16, encoding="utf-8")
    assert run(tmp_path, "git", "add", "sample.txt").returncode == 0
    assert run(tmp_path, "git", "commit", "-m", "unsafe").returncode == 0
    path.write_text("safe again", encoding="utf-8")
    assert run(tmp_path, "git", "add", "sample.txt").returncode == 0
    assert run(tmp_path, "git", "commit", "-m", "safe tip").returncode == 0
    head = run(tmp_path, "git", "rev-parse", "HEAD").stdout.decode().strip()
    update = f"refs/heads/main {head} refs/heads/main {base}\n".encode()
    assert run(tmp_path, sys.executable, str(SCANNER), "outgoing", input=update).returncode != 0
