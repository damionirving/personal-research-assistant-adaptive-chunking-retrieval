# AWS CDK Recursive `cdk.out` Cleanup on Windows

## Problem

The automation encountered an apparent infinite loop and recursively created nested `cdk.out` directories until the drive ran out of space.

Example:

```text
cdk.out/
└── cdk.out/
    └── cdk.out/
        └── ...
```

Windows Explorer estimated that deleting the directory would take approximately five hours.

---

## Recommended Cleanup

Do not use Windows Explorer for deeply nested generated directories. Use **Command Prompt** instead.

### Stop Related Processes

In PowerShell:

```powershell
Get-Process node, python -ErrorAction SilentlyContinue |
    Stop-Process -Force
```

Be aware that this stops all matching Node.js and Python processes.

Close:

- VS Code
- Other terminals using the repository
- File Explorer windows opened inside the repository
- Any process still running `cdk synth`

---

## Delete `cdk.out` From Its Parent Directory

Open **Command Prompt as Administrator**.

Navigate to the directory that directly contains `cdk.out`:

```cmd
cd /d C:\path\to\your\repository
```

Verify the current directory:

```cmd
cd
```

Verify the deletion target:

```cmd
echo \\?\%CD%\cdk.out
```

Then delete it:

```cmd
rd /s /q "\\?\%CD%\cdk.out"
```

### Command Explanation

- `rd` removes a directory.
- `/s` removes all files and subdirectories.
- `/q` suppresses confirmation prompts.
- `%CD%` expands to the current working directory.
- `\\?\` enables Windows extended-length path handling.

> **Warning:** Confirm that the current directory is the repository containing the unwanted `cdk.out` directory before running the command. `rd /s /q` permanently deletes the target without asking for confirmation.

---

## Alternative: Empty-Directory Mirror

If `rd` still struggles, create an empty directory and mirror it over `cdk.out`:

```cmd
mkdir C:\empty_folder
robocopy C:\empty_folder "\\?\%CD%\cdk.out" /MIR /R:0 /W:0
rd /s /q "\\?\%CD%\cdk.out"
rd /s /q C:\empty_folder
```

Use `/MIR` carefully because it makes the destination match the source, including deleting destination contents.

---

## Check Recovered Disk Space

In PowerShell:

```powershell
Get-PSDrive C
```

---

## Prevent Recurrence

Add the following to `.gitignore`:

```gitignore
# AWS CDK generated output
cdk.out/

# Common generated directories
node_modules/
.venv/
__pycache__/
.pytest_cache/
```

Check `cdk.json` and the CDK application for dangerous configurations such as:

```json
{
  "app": "cd cdk.out && cdk synth"
}
```

or:

```json
{
  "output": "cdk.out/cdk.out"
}
```

Search the repository for explicit output configuration:

```powershell
Get-ChildItem -Recurse -File |
    Select-String -Pattern "outdir|CDK_OUTDIR|--output|cdk\.out"
```

Always run CDK commands from the repository root:

```powershell
cd C:\path\to\your\repository
cdk synth
```

---

## Suggested Automation Guardrail Prompt

```text
Work only from the repository root.

Never run CDK commands from inside cdk.out.
Treat cdk.out as generated, disposable output.
Do not inspect, copy, recursively traverse, modify, or create files inside cdk.out.
Do not change the CDK output directory.
Before running cdk synth, print the working directory and confirm that it is the repository root.
Run cdk synth no more than once unless the first invocation fails.
Do not retry commands automatically in a loop.
Stop immediately if a nested cdk.out directory appears.
```

---

## Original Chat Transcript

### User

An automation tool encountered an infinite loop and recursively created `cdk.out`, using up available hard-drive space.

### Assistant

This appears to be a recursive CDK synthesis loop in which `cdk.out` was created inside another `cdk.out`. The recommended response was to stop the active processes, remove the generated directory from the repository root, inspect `cdk.json` and output-path settings, update `.gitignore`, and rerun `cdk synth` only from the repository root.

### User

Erasing `cdk.out` using Windows Explorer takes five hours.

### Assistant

Windows Explorer is slow when processing millions of deeply nested files. The recommended approach was to use `rd /s /q` with the Windows extended-length path prefix, or use an empty-directory `robocopy /MIR` operation before removing the directory.

### User

Modify:

```cmd
rd /s /q "\\?\C:\path\to\your\repository\cdk.out"
```

so it works from the directory where `cdk.out` is located.

### Assistant

From inside the directory that contains `cdk.out`, run:

```cmd
rd /s /q "\\?\%CD%\cdk.out"
```

First verify the target:

```cmd
echo \\?\%CD%\cdk.out
```

Use Command Prompt rather than PowerShell.
