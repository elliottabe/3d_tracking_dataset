#!/bin/bash
# Merge feature branch to main across all repos
# Usage: ./scripts/git/merge.sh <feature-branch>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBMODULES=("stac-mjx")

FEATURE_BRANCH="$1"
MAIN_BRANCH="main"

if [ -z "$FEATURE_BRANCH" ]; then
    echo "Usage: ./scripts/git/merge.sh <feature-branch>"
    exit 1
fi

echo "========================================"
echo "Merging $FEATURE_BRANCH → $MAIN_BRANCH"
echo "========================================"
echo
echo "This will merge across all repos."
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

cd "$REPO_ROOT"

# Ensure we're on the feature branch in main repo
echo "[Main Repo] Checking out $FEATURE_BRANCH..."
git checkout "$FEATURE_BRANCH"
echo

# Merge submodules first
for submodule in "${SUBMODULES[@]}"; do
    if [ -e "$submodule/.git" ]; then
        echo "[$submodule] Merging..."
        cd "$REPO_ROOT/$submodule"
        
        git checkout "$MAIN_BRANCH"
        git pull origin "$MAIN_BRANCH"
        git merge "$FEATURE_BRANCH" --no-edit
        git push origin "$MAIN_BRANCH"
        
        echo "  ✓ Merged"
        cd "$REPO_ROOT"
        echo
    fi
done

# Update submodule references on feature branch
echo "[Main Repo] Updating submodule references on $FEATURE_BRANCH..."
cd "$REPO_ROOT"

# Update .gitmodules to track main branch
for submodule in "${SUBMODULES[@]}"; do
    if [ -e "$submodule/.git" ]; then
        git config -f .gitmodules submodule.$submodule.branch "$MAIN_BRANCH"
    fi
done

# Update submodules to point to their new main commits
git submodule update --remote --merge

# Add updated references
for submodule in "${SUBMODULES[@]}"; do
    if [ -e "$submodule/.git" ]; then
        git add "$submodule"
    fi
done
git add .gitmodules

# Commit updated references on feature branch
git commit -m "Update submodule references after merging to $MAIN_BRANCH" || echo "  No changes to commit"
git push origin "$FEATURE_BRANCH"
echo

# Now merge feature branch to main in main repo
echo "[Main Repo] Merging $FEATURE_BRANCH → $MAIN_BRANCH..."
git checkout "$MAIN_BRANCH"
git pull origin "$MAIN_BRANCH"
git merge "$FEATURE_BRANCH" --no-edit
git push origin "$MAIN_BRANCH"

echo
echo "========================================"
echo "✓ Merge Complete"
echo "========================================"
