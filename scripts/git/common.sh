#!/bin/bash
# Shared helpers for the git multi-repo scripts.
# Source this from the other scripts:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
#
# Provides:
#   REPO_ROOT              - absolute path to the superproject root
#   GITMODULES            - absolute path to .gitmodules
#   get_submodules        - print submodule paths (one per line) as declared in .gitmodules
#   submodule_branch PATH - print the tracking branch declared for a submodule path (empty if none)
#   ensure_on_branch PATH - reattach a detached submodule to its tracking branch, preserving commits

GIT_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$GIT_COMMON_DIR/../.." && pwd)"
GITMODULES="$REPO_ROOT/.gitmodules"

# Print submodule paths declared in .gitmodules, one per line.
# Deriving from .gitmodules (instead of a hardcoded list) means every submodule
# is handled, including any added later.
get_submodules() {
    [ -f "$GITMODULES" ] || return 0
    git config --file "$GITMODULES" --get-regexp '^submodule\..*\.path$' \
        | awk '{print $2}'
}

# Print the tracking branch declared for a submodule PATH in .gitmodules (empty if none).
submodule_branch() {
    local path="$1" key name
    [ -f "$GITMODULES" ] || return 0
    key="$(git config --file "$GITMODULES" --get-regexp '^submodule\..*\.path$' \
            | awk -v p="$path" '$2==p {print $1; exit}')"
    [ -z "$key" ] && return 0
    name="${key#submodule.}"
    name="${name%.path}"
    git config --file "$GITMODULES" --get "submodule.$name.branch" 2>/dev/null || true
}

# Ensure a submodule is checked out on its tracking branch, reattaching if detached
# WITHOUT losing any commits that are only on the detached HEAD.
# Returns 0 on success, 1 if it refused to switch (divergence) so the caller can flag it.
ensure_on_branch() {
    local path="$1"
    local dir="$REPO_ROOT/$path"
    [ -e "$dir/.git" ] || return 0

    local branch
    branch="$(submodule_branch "$path")"
    [ -z "$branch" ] && branch="main"

    # Already on a branch -> nothing to do.
    if git -C "$dir" symbolic-ref --short -q HEAD >/dev/null 2>&1; then
        return 0
    fi

    echo "  [$path] detached HEAD -> reattaching to '$branch'"
    if git -C "$dir" show-ref --verify --quiet "refs/heads/$branch"; then
        if git -C "$dir" merge-base --is-ancestor "$branch" HEAD; then
            # Local branch is behind the current commit: fast-forward it up, keep the commit.
            git -C "$dir" branch -f "$branch" HEAD
            git -C "$dir" checkout -q "$branch"
        elif git -C "$dir" merge-base --is-ancestor HEAD "$branch"; then
            # Current commit already contained in the branch: safe to switch.
            git -C "$dir" checkout -q "$branch"
        else
            echo "  [$path] WARNING: local '$branch' has diverged from the checked-out commit"
            echo "           ($(git -C "$dir" rev-parse --short HEAD)); leaving detached to avoid data loss."
            return 1
        fi
    else
        # No local branch yet: create it at the current (detached) commit.
        git -C "$dir" checkout -q -b "$branch"
    fi

    # Track the remote branch if it exists.
    if git -C "$dir" ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
        git -C "$dir" branch -q --set-upstream-to="origin/$branch" "$branch" >/dev/null 2>&1 || true
    fi
    return 0
}
