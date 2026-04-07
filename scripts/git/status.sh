#!/bin/bash
# Check status of main repo and all submodules
# Usage: ./scripts/git/status.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBMODULES=("stac-mjx")

cd "$REPO_ROOT"

echo "========================================"
echo "Git Status - All Repositories"
echo "========================================"
echo

# Main repo status
echo "[Main Repo] 3d_tracking_dataset"
echo "Branch: $(git symbolic-ref --short HEAD 2>/dev/null || echo 'DETACHED HEAD')"
echo "Status:"
git status --short
echo

# Submodule status
for submodule in "${SUBMODULES[@]}"; do
    if [ -e "$submodule/.git" ]; then
        echo "[$submodule]"
        cd "$REPO_ROOT/$submodule"
        echo "Branch: $(git symbolic-ref --short HEAD 2>/dev/null || echo 'DETACHED HEAD')"
        echo "Status:"
        git status --short
        
        # Check for unpushed commits
        BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null)
        if [ "$BRANCH" != "" ]; then
            UNPUSHED=$(git log origin/$BRANCH..HEAD --oneline 2>/dev/null | wc -l || echo "0")
            if [ "$UNPUSHED" -gt 0 ]; then
                echo "⚠ $UNPUSHED unpushed commit(s)"
            fi
        fi
        echo
        cd "$REPO_ROOT"
    fi
done

echo "========================================"
echo "Submodule References:"
git submodule status
echo "========================================"
