# Git Branch Strategy

Use a separate branch for RAGAS evaluation work so it can be reviewed, tested,
deployed, or reverted independently from the locked retrieval-performance path.

## Recommended Branch

```text
feature/ragas-evaluation
```

The branch should start from the current stable branch and contain only the
RAGAS evaluation changes and their tests. Existing retrieval-only evaluation
behavior should remain unchanged.

## Create And Switch To The Branch

Run these commands from the repository directory in PowerShell:

```powershell
cd "C:\path\to\your\project"
git status
git switch -c feature/ragas-evaluation
git branch --show-current
```

The last command should print:

```text
feature/ragas-evaluation
```

If the branch already exists locally, switch to it instead:

```powershell
git switch feature/ragas-evaluation
```

## How The Assistant Knows The Branch

The coding assistant reads the current Git checkout in the shared repository workspace. There
is no separate branch registration step. After `git switch`, the assistant sees the
active branch automatically. Confirm it before asking the assistant to make changes:

```powershell
git branch --show-current
git status --short
```

## Work And Verify

Ask the coding assistant to implement the RAGAS design while this branch is active. Before
committing, review the changed files and run the relevant tests:

```powershell
git diff --stat
git diff
python -m pytest -q
```

Do not commit unrelated changes from another task.

## Stage And Commit

Stage the intended files, inspect the staged diff, and create a focused commit:

```powershell
git add .
git diff --cached --stat
git diff --cached
git commit -m "add async ragas evaluation workflow"
```

## Push The Branch

Publish the branch to the configured remote:

```powershell
git push -u origin feature/ragas-evaluation
```

If Git reports that the branch has no upstream, it means the local branch is
not yet linked to a matching branch on `origin`. The `-u` flag creates that
tracking link, so future `git push` and `git pull` commands can use the branch
name without repeating the remote.

If you prefer Git to create upstream tracking automatically for new branches,
you can set:

```powershell
git config --global push.autoSetupRemote true
```

Afterward, verify the commit and clean working tree:

```powershell
git log -1 --oneline
git status
```

## Return To The Base Branch

After the RAGAS branch is pushed or merged, return to the base branch used by
the repository, for example:

```powershell
git switch main
git pull --ff-only
```

Use `git branch` to list local branches if the base branch is not named
`main`.
