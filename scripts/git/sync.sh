#!/bin/bash
# Sync all repos: pull latest changes from tracked branches
# Usage: ./scripts/git/sync.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBMODULES=("stac-mjx")

cd "$REPO_ROOT"

echo "========================================"
echo "Syncing All Repositories"
echo "========================================"
echo

# Get current branch
CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
echo "Current branch: $CURRENT_BRANCH"
echo

# Pull main repo
echo "[Main Repo] Pulling latest changes..."
git pull origin "$CURRENT_BRANCH"
echo

# Initialize submodules if needed
echo "Ensuring submodules are initialized..."
git submodule update --init --recursive
echo

# Pull each submodule (stash local changes, pull, then restore)
for submodule in "${SUBMODULES[@]}"; do
    if [ -d "$submodule/.git" ] || [ -f "$submodule/.git" ]; then
        echo "[$submodule] Pulling latest changes..."
        cd "$REPO_ROOT/$submodule"

        BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
        echo "  Current branch: $BRANCH"

        # Stash any local modifications so pull doesn't conflict
        STASH_MSG="sync-auto-stash"
        git stash push -m "$STASH_MSG" --quiet 2>/dev/null || true

        git pull origin "$BRANCH" || echo "  Warning: Could not pull from origin/$BRANCH"

        # Restore stashed changes if we stashed anything
        if git stash list | head -1 | grep -q "$STASH_MSG"; then
            echo "  Restoring local changes..."
            git stash pop --quiet 2>/dev/null || {
                echo "  Warning: Could not auto-restore stash (conflict?). Run 'cd $submodule && git stash pop' manually."
            }
        fi

        cd "$REPO_ROOT"
        echo
    else
        echo "[$submodule] Warning: Not found or not initialized"
        echo "  Path: $REPO_ROOT/$submodule"
        echo
    fi
done

# Commit updated submodule pointers if any changed
CHANGED_SUBMODULES=()
for submodule in "${SUBMODULES[@]}"; do
    if ! git diff --quiet "$submodule" 2>/dev/null; then
        CHANGED_SUBMODULES+=("$submodule")
    fi
done

if [ ${#CHANGED_SUBMODULES[@]} -gt 0 ]; then
    echo "Updating submodule pointers: ${CHANGED_SUBMODULES[*]}"
    git add "${CHANGED_SUBMODULES[@]}"
    git commit -m "sync: update submodule pointers"
    echo
fi

# Ensure submodule working copies match recorded pointers
echo "Resetting submodules to recorded commits..."
git submodule update --init --recursive
echo

echo "========================================"
echo "✓ Sync Complete"
echo "========================================"
echo
echo "Run './scripts/git/status.sh' to see current state"
