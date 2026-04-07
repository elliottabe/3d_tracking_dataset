#!/bin/bash
# Push commits in all repos
# Usage: ./scripts/git/push.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBMODULES=("stac-mjx")

cd "$REPO_ROOT"

echo "========================================"
echo "Pushing All Repositories"
echo "========================================"
echo

# Push each submodule first
for submodule in "${SUBMODULES[@]}"; do
    if [ -e "$submodule/.git" ]; then
        cd "$REPO_ROOT/$submodule"
        
        BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null)
        if [ -n "$BRANCH" ]; then
            if git ls-remote --exit-code --heads origin "$BRANCH" &>/dev/null; then
                UNPUSHED=$(git log "origin/$BRANCH..HEAD" --oneline | wc -l)
            else
                UNPUSHED=1
            fi
            if [ "$UNPUSHED" -gt 0 ]; then
                echo "[$submodule] Pushing to origin/$BRANCH..."
                git push --set-upstream origin "$BRANCH"
                echo "  ✓ Pushed"
            else
                echo "[$submodule] No commits to push"
            fi
        else
            echo "[$submodule] Detached HEAD - skipping"
        fi
        
        cd "$REPO_ROOT"
        echo
    fi
done

# Push main repo
cd "$REPO_ROOT"
BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null)
if [ -n "$BRANCH" ]; then
    if git ls-remote --exit-code --heads origin "$BRANCH" &>/dev/null; then
        UNPUSHED=$(git log "origin/$BRANCH..HEAD" --oneline | wc -l)
    else
        UNPUSHED=1
    fi
    if [ "$UNPUSHED" -gt 0 ]; then
        echo "[Main Repo] Pushing to origin/$BRANCH..."
        git push --set-upstream origin "$BRANCH"
        echo "  ✓ Pushed"
    else
        echo "[Main Repo] No commits to push"
    fi
else
    echo "[Main Repo] Detached HEAD - skipping"
fi

echo
echo "========================================"
echo "✓ Push Complete"
echo "========================================"
