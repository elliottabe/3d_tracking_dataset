# Git Scripts for Multi-Repo Management

These scripts help manage the 3d_tracking_dataset repository and its submodules (stac-mjx).

## Initial Setup

### Setup Submodules
```bash
./scripts/git/setup_submodules.sh
```
**Run this once** after cloning the repository or if submodules aren't working correctly. This script:
- Initializes all submodules defined in `.gitmodules`
- Handles cases where directories exist as standalone repos
- Properly configures submodules in `.git/config`
- Clones submodules on the correct branch

Use this if you see "Warning: Not found or not initialized" errors.

## Daily Workflow Scripts

### Check Status
```bash
./scripts/git/status.sh
```
Shows branch and status for all repos, including unpushed commits.

### Sync (Pull Latest)
```bash
./scripts/git/sync.sh
```
Pulls latest changes from remote for all repos on their current branches.

### Commit Changes
```bash
./scripts/git/commit.sh "Your commit message"
```
Commits changes in submodules first, then main repo. Prompts to push afterwards.

### Push Commits
```bash
./scripts/git/push.sh
```
Pushes all unpushed commits to remote (submodules first, then main repo).

## Branch Management

### Create Feature Branch
```bash
./scripts/git/branch.sh create feature/my-feature
```
Creates and pushes a new branch across all repos, updates .gitmodules.

### Checkout Existing Branch
```bash
./scripts/git/branch.sh checkout feature/my-feature
```
Checks out an existing branch in all repos.

### Merge to Main
```bash
./scripts/git/merge.sh feature/my-feature
```
Merges feature branch to main in all repos (submodules first, then main).

## Typical Workflows

### Start New Feature
```bash
./scripts/git/branch.sh create feature/descending-control
# Make changes...
./scripts/git/commit.sh "Implement descending control"
# Commits and prompts to push automatically
```

### Daily Development
```bash
./scripts/git/sync.sh                          # Pull latest
# Make changes...
./scripts/git/commit.sh "Fix sensor mapping"   # Commit and push
```

### Complete Feature
```bash
./scripts/git/status.sh                        # Check everything is committed
./scripts/git/merge.sh feature/descending-control  # Merge to main
./scripts/git/branch.sh checkout main          # Switch back to main
```

## Notes

- All scripts handle submodules automatically
- Submodules are always processed before the main repo
- Scripts work from any directory in the repo
- Use `set -e` to exit on any error
- All scripts are idempotent and safe to re-run

## Troubleshooting

### Submodules Not Initializing

If you see warnings like "Warning: Not found or not initialized":

```bash
./scripts/git/setup_submodules.sh
```

This will clean up and properly initialize all submodules.

### Submodule Directories Exist But Not Working

If submodule directories exist as standalone repos (not properly linked as submodules), run:

```bash
./scripts/git/setup_submodules.sh
```

The script will detect this and fix it automatically by removing and re-cloning them properly.

### Checking Submodule Status

```bash
git submodule status
```

Should show commit hashes for each submodule. If it shows nothing, run `setup_submodules.sh`.
