#!/bin/bash
# Commit changes in main repo and/or submodules
# Usage: ./scripts/git/commit.sh ["commit message"]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBMODULES=("stac-mjx")

cd "$REPO_ROOT"

# Get commit message
if [ -z "$1" ]; then
    echo "Enter commit message:"
    read -r COMMIT_MSG
    if [ -z "$COMMIT_MSG" ]; then
        echo "Error: Commit message cannot be empty"
        exit 1
    fi
else
    COMMIT_MSG="$1"
fi

echo "========================================"
echo "Committing Changes"
echo "Message: $COMMIT_MSG"
echo "========================================"
echo

# Track if anything was committed
COMMITTED=false
SUBMODULES_CHANGED=false

# Commit changes in each submodule
for submodule in "${SUBMODULES[@]}"; do
    if [ -e "$submodule/.git" ]; then
        cd "$REPO_ROOT/$submodule"
        
        if ! git diff-index --quiet HEAD -- 2>/dev/null; then
            echo "[$submodule] Committing changes..."
            git add -A
            git commit -m "$COMMIT_MSG"
            COMMITTED=true
            SUBMODULES_CHANGED=true
            echo "  ✓ Committed"
        else
            echo "[$submodule] No changes to commit"
        fi
        
        cd "$REPO_ROOT"
        echo
    fi
done

# Commit changes in main repo
cd "$REPO_ROOT"
if ! git diff-index --quiet HEAD -- 2>/dev/null || [ "$SUBMODULES_CHANGED" = true ]; then
    echo "[Main Repo] Committing changes..."
    
    # Add submodules if they changed
    if [ "$SUBMODULES_CHANGED" = true ]; then
        for submodule in "${SUBMODULES[@]}"; do
            if [ -e "$submodule/.git" ]; then
                git add "$submodule"
            fi
        done
    fi
    
    git add -A
    git commit -m "$COMMIT_MSG"
    COMMITTED=true
    echo "  ✓ Committed"
else
    echo "[Main Repo] No changes to commit"
fi
echo

if [ "$COMMITTED" = false ]; then
    echo "========================================"
    echo "Nothing to commit"
    echo "========================================"
    exit 0
fi

# Ask to push
echo "========================================"
read -p "Push changes to remote? (Y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Changes committed locally but not pushed"
    echo "Run './scripts/git/push.sh' to push later"
else
    "$SCRIPT_DIR/push.sh"
fi
