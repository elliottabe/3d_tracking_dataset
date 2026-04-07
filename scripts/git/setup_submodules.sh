#!/bin/bash
# Setup script to properly initialize git submodules
# This script handles the case where submodule directories might already exist as standalone repos
# Usage: ./scripts/git/setup_submodules.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBMODULES=("stac-mjx")

cd "$REPO_ROOT"

echo "========================================"
echo "Setting Up Git Submodules"
echo "========================================"
echo

# Check if .gitmodules exists
if [ ! -f .gitmodules ]; then
    echo "Error: .gitmodules file not found!"
    exit 1
fi

# Function to check if a directory is a git repository
is_git_repo() {
    [ -d "$1/.git" ] || [ -f "$1/.git" ]
}

# Function to check if a submodule is properly initialized
is_submodule_initialized() {
    local submodule="$1"
    # Check if submodule is in git config and .git/modules exists or .git is a file
    git config --get "submodule.$submodule.url" > /dev/null 2>&1 && \
    ([ -d ".git/modules/$submodule" ] || [ -f "$submodule/.git" ])
}

# Parse .gitmodules to get submodule info
declare -A SUBMODULE_URLS
declare -A SUBMODULE_BRANCHES

while IFS= read -r line; do
    if [[ $line =~ ^\[submodule\ \"([^\"]+)\"\] ]]; then
        current_submodule="${BASH_REMATCH[1]}"
    elif [[ $line =~ ^[[:space:]]*url[[:space:]]*=[[:space:]]*(.+)$ ]]; then
        SUBMODULE_URLS[$current_submodule]="${BASH_REMATCH[1]}"
    elif [[ $line =~ ^[[:space:]]*branch[[:space:]]*=[[:space:]]*(.+)$ ]]; then
        SUBMODULE_BRANCHES[$current_submodule]="${BASH_REMATCH[1]}"
    fi
done < .gitmodules

# Process each submodule
for submodule in "${SUBMODULES[@]}"; do
    echo "--------------------------------------"
    echo "Processing: $submodule"
    echo "--------------------------------------"
    
    url="${SUBMODULE_URLS[$submodule]}"
    branch="${SUBMODULE_BRANCHES[$submodule]:-main}"
    
    if [ -z "$url" ]; then
        echo "Warning: No URL found in .gitmodules for $submodule"
        continue
    fi
    
    echo "  URL: $url"
    echo "  Branch: $branch"
    
    # Check if already properly initialized
    if is_submodule_initialized "$submodule"; then
        echo "  ✓ Already initialized as submodule"
        cd "$submodule"
        current_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo "detached")
        echo "  Current branch: $current_branch"
        cd "$REPO_ROOT"
        echo
        continue
    fi
    
    # If directory exists but is not a proper submodule
    if [ -d "$submodule" ]; then
        if is_git_repo "$submodule"; then
            echo "  ⚠ Directory exists as standalone git repo"
            echo "  Removing and re-cloning as submodule..."
            rm -rf "$submodule"
        else
            echo "  ⚠ Directory exists but is not a git repo"
            echo "  Removing..."
            rm -rf "$submodule"
        fi
    fi
    
    # Configure submodule in git config
    echo "  Configuring submodule in git..."
    git config -f .git/config "submodule.$submodule.url" "$url"
    git config -f .git/config "submodule.$submodule.active" true
    
    # Clone the repository
    echo "  Cloning $submodule..."
    if ! git clone -b "$branch" "$url" "$submodule"; then
        echo "  ✗ Failed to clone $submodule"
        echo "  Attempting with default branch..."
        git clone "$url" "$submodule"
        cd "$submodule"
        git checkout "$branch" 2>/dev/null || echo "  Warning: Could not checkout branch $branch"
        cd "$REPO_ROOT"
    fi
    
    # Convert to proper submodule format
    echo "  Converting to submodule format..."
    git submodule absorbgitdirs
    
    echo "  ✓ Successfully initialized $submodule"
    echo
done

echo "========================================"
echo "Verifying Submodule Status"
echo "========================================"
echo

# Run git submodule status to show final state
git submodule status

echo
echo "========================================"
echo "✓ Submodule Setup Complete"
echo "========================================"
echo
echo "You can now use:"
echo "  ./scripts/git/sync.sh    - to sync all repos"
echo "  ./scripts/git/status.sh  - to check status"
