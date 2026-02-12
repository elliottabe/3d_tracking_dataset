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

# Pull each submodule
for submodule in "${SUBMODULES[@]}"; do
    if [ -d "$submodule/.git" ] || [ -f "$submodule/.git" ]; then
        echo "[$submodule] Pulling latest changes..."
        cd "$REPO_ROOT/$submodule"
        
        BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
        echo "  Current branch: $BRANCH"
        git pull origin "$BRANCH" || echo "  Warning: Could not pull from origin/$BRANCH"
        
        cd "$REPO_ROOT"
        echo
    else
        echo "[$submodule] Warning: Not found or not initialized"
        echo "  Path: $REPO_ROOT/$submodule"
        echo
    fi
done

echo "========================================"
echo "✓ Sync Complete"
echo "========================================"
echo
echo "Run './scripts/git/status.sh' to see current state"
