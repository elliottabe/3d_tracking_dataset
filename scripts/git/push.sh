#!/bin/bash
# Push commits in all repos (submodules first, then main).
# Usage: ./scripts/git/push.sh
#
# Deliberately does NOT use `set -e`: a failure in one repo must not stop the
# others from being pushed. Failures are collected and reported at the end.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "$REPO_ROOT"

echo "========================================"
echo "Pushing All Repositories"
echo "========================================"
echo

FAILED=()

# Push the repo in the current working directory. $1 = display label.
push_current_repo() {
    local label="$1" branch unpushed
    branch="$(git symbolic-ref --short -q HEAD 2>/dev/null || true)"
    if [ -z "$branch" ]; then
        echo "[$label] Detached HEAD - skipping"
        FAILED+=("$label (detached HEAD)")
        return 0
    fi

    if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
        unpushed="$(git log "origin/$branch..HEAD" --oneline 2>/dev/null | wc -l | tr -d ' ')"
    else
        unpushed=1   # remote branch does not exist yet
    fi

    if [ "${unpushed:-0}" -gt 0 ]; then
        echo "[$label] Pushing $unpushed commit(s) to origin/$branch..."
        if git push --set-upstream origin "$branch"; then
            echo "  ✓ Pushed"
        else
            echo "  ✗ Push failed"
            FAILED+=("$label")
        fi
    else
        echo "[$label] No commits to push"
    fi
}

# Submodules first: reattach if detached, then push.
mapfile -t SUBMODULES < <(get_submodules)
for submodule in "${SUBMODULES[@]}"; do
    [ -e "$REPO_ROOT/$submodule/.git" ] || continue
    ensure_on_branch "$submodule" || FAILED+=("$submodule (reattach refused)")
    cd "$REPO_ROOT/$submodule"
    push_current_repo "$submodule"
    cd "$REPO_ROOT"
    echo
done

# Main repo last.
cd "$REPO_ROOT"
push_current_repo "Main Repo"
echo

echo "========================================"
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "✓ Push Complete"
    echo "========================================"
else
    echo "⚠ Push finished with failures:"
    for f in "${FAILED[@]}"; do
        echo "   - $f"
    done
    echo "========================================"
    exit 1
fi
