#!/bin/bash
# Create or checkout a feature branch across all repos
# Usage: 
#   ./scripts/git/branch.sh create <branch-name>   # Create new branch
#   ./scripts/git/branch.sh checkout <branch-name> # Checkout existing branch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBMODULES=("stac-mjx")

ACTION="$1"
BRANCH_NAME="$2"

if [ -z "$ACTION" ] || [ -z "$BRANCH_NAME" ]; then
    echo "Usage:"
    echo "  ./scripts/git/branch.sh create <branch-name>   # Create new branch"
    echo "  ./scripts/git/branch.sh checkout <branch-name> # Checkout existing branch"
    exit 1
fi

cd "$REPO_ROOT"

if [ "$ACTION" = "create" ]; then
    echo "========================================"
    echo "Creating Feature Branch: $BRANCH_NAME"
    echo "========================================"
    echo
    
    # Create in main repo
    echo "[Main Repo] Creating branch..."
    git checkout -b "$BRANCH_NAME"
    git push -u origin "$BRANCH_NAME"
    echo
    
    # Create in submodules
    for submodule in "${SUBMODULES[@]}"; do
        if [ -e "$submodule/.git" ]; then
            echo "[$submodule] Creating branch..."
            cd "$REPO_ROOT/$submodule"
            
            if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
                echo "  Branch exists locally, checking out..."
                git checkout "$BRANCH_NAME"
            else
                git checkout -b "$BRANCH_NAME"
            fi
            
            git push -u origin "$BRANCH_NAME" 2>/dev/null || echo "  Remote branch may already exist"
            cd "$REPO_ROOT"
            echo
        fi
    done
    
    # Update .gitmodules
    echo "[Config] Updating .gitmodules to track $BRANCH_NAME..."
    for submodule in "${SUBMODULES[@]}"; do
        if [ -e "$submodule/.git" ]; then
            git config -f .gitmodules submodule.$submodule.branch "$BRANCH_NAME"
        fi
    done
    
    git add .gitmodules
    git commit -m "Configure submodules to track $BRANCH_NAME branch"
    git push
    
    echo "========================================"
    echo "✓ Feature Branch Created"
    echo "========================================"
    
elif [ "$ACTION" = "checkout" ]; then
    echo "========================================"
    echo "Checking Out Branch: $BRANCH_NAME"
    echo "========================================"
    echo
    
    # Checkout in main repo
    echo "[Main Repo] Checking out $BRANCH_NAME..."
    git checkout "$BRANCH_NAME"
    git pull origin "$BRANCH_NAME" 2>/dev/null || true
    echo
    
    # Checkout in submodules
    for submodule in "${SUBMODULES[@]}"; do
        if [ -e "$submodule/.git" ]; then
            echo "[$submodule] Checking out $BRANCH_NAME..."
            cd "$REPO_ROOT/$submodule"
            
            git checkout "$BRANCH_NAME"
            git pull origin "$BRANCH_NAME" 2>/dev/null || echo "  Warning: Could not pull from origin"
            
            cd "$REPO_ROOT"
            echo
        fi
    done
    
    echo "========================================"
    echo "✓ Checked Out $BRANCH_NAME"
    echo "========================================"
    
else
    echo "Error: Invalid action '$ACTION'"
    echo "Use 'create' or 'checkout'"
    exit 1
fi
