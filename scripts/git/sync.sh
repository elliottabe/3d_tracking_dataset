#!/bin/bash
# Sync all repos: pull latest changes on their tracking branches.
# Usage: ./scripts/git/sync.sh
#
# No `set -e`: one repo failing to pull must not abort the rest.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "$REPO_ROOT"

echo "========================================"
echo "Syncing All Repositories"
echo "========================================"
echo

# Get current branch of the main repo.
CURRENT_BRANCH=$(git symbolic-ref --short -q HEAD 2>/dev/null || echo "main")
echo "Current branch: $CURRENT_BRANCH"
echo

# Pull main repo.
echo "[Main Repo] Pulling latest changes..."
git pull origin "$CURRENT_BRANCH" || echo "  Warning: Could not pull origin/$CURRENT_BRANCH"
echo

# Make sure submodules are present (this may leave them detached; we reattach below).
echo "Ensuring submodules are initialized..."
git submodule update --init --recursive
echo

mapfile -t SUBMODULES < <(get_submodules)

# Pull each submodule on its tracking branch (reattach first, stash local edits).
for submodule in "${SUBMODULES[@]}"; do
    if [ ! -e "$submodule/.git" ]; then
        echo "[$submodule] Warning: Not found or not initialized"
        echo "  Path: $REPO_ROOT/$submodule"
        echo
        continue
    fi

    echo "[$submodule] Pulling latest changes..."
    # Reattach to the tracking branch so we don't pull into a detached HEAD.
    ensure_on_branch "$submodule"

    cd "$REPO_ROOT/$submodule"
    BRANCH=$(git symbolic-ref --short -q HEAD 2>/dev/null || echo "main")
    echo "  Current branch: $BRANCH"

    # Stash any local modifications so pull doesn't conflict.
    STASH_MSG="sync-auto-stash"
    git stash push -m "$STASH_MSG" --quiet 2>/dev/null || true

    git pull origin "$BRANCH" || echo "  Warning: Could not pull from origin/$BRANCH"

    # Restore stashed changes if we stashed anything.
    if git stash list | head -1 | grep -q "$STASH_MSG"; then
        echo "  Restoring local changes..."
        git stash pop --quiet 2>/dev/null || {
            echo "  Warning: Could not auto-restore stash (conflict?). Run 'cd $submodule && git stash pop' manually."
        }
    fi

    cd "$REPO_ROOT"
    echo
done

# Commit updated submodule pointers if any changed.
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

# Keep submodules on their tracking branches (do NOT run `git submodule update`
# here: it would re-detach them at the recorded commit).
echo "Ensuring submodules stay on their tracking branches..."
for submodule in "${SUBMODULES[@]}"; do
    [ -e "$submodule/.git" ] && ensure_on_branch "$submodule"
done
echo

echo "========================================"
echo "✓ Sync Complete"
echo "========================================"
echo
echo "Run './scripts/git/status.sh' to see current state"
