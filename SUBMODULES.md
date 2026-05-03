# Git Submodules Guide

This repository uses git submodules to manage dependencies.

## Current Submodules

- **stac-mjx** (`stac-mjx/`) - STAC inverse kinematics solver
  - Repository: `git@github.com:elliottabe/stac-mjx.git`
  - Branch: main

## Initial Setup

When cloning this repository for the first time, initialize the submodules:

```bash
# Clone the main repository
git clone git@github.com:YOUR_USERNAME/3d_tracking_dataset.git
cd 3d_tracking_dataset

# Initialize and clone all submodules
git submodule update --init --recursive
```

Or clone with submodules in one step:

```bash
git clone --recurse-submodules git@github.com:YOUR_USERNAME/3d_tracking_dataset.git
```

## Working with Submodules

### Updating Submodules

To update a submodule to the latest commit on its tracked branch:

```bash
# Update stac-mjx to latest main branch
cd stac-mjx
git checkout main
git pull
cd ..

# Commit the submodule update in the parent repo
git add stac-mjx
git commit -m "Update stac-mjx submodule to latest"
```

Or update all submodules at once:

```bash
git submodule update --remote --merge
```

### Check Submodule Status

```bash
# Show current commit for each submodule
git submodule status

# Show detailed info
git submodule foreach 'git status'
```

### Making Changes in Submodules

If you need to modify code in stac-mjx:

```bash
# 1. Navigate to submodule
cd stac-mjx

# 2. Create a branch and make changes
git checkout -b my-feature
# ... make changes ...
git add .
git commit -m "My changes"

# 3. Push to submodule repository
git push origin my-feature

# 4. Return to parent and update reference
cd ..
git add stac-mjx
git commit -m "Update stac-mjx to feature branch"
```

### Switching Branches

When switching branches in the parent repository, update submodules:

```bash
git checkout other-branch
git submodule update --recursive
```

### Common Issues

**Submodule directory exists but is empty:**
```bash
git submodule update --init --recursive
```

**Detached HEAD in submodule:**
This is normal - submodules track specific commits. To work on a branch:
```bash
cd stac-mjx
git checkout main
```

**Submodule has uncommitted changes:**
```bash
# Stash changes in submodule
cd stac-mjx
git stash
cd ..

# Or commit them
cd stac-mjx
git add .
git commit -m "WIP"
cd ..
```

## Integration with Batch Processing

The batch processing scripts can now reference the submodule:

```bash
# Run STAC IK using the submodule
cd stac-mjx
python run_stac.py paths=workstation dataset=free_running anatomy=v1 \
    paths.data_dir=/data2/users/eabe/datasets/Johnson_lab/free_running/Predictions_3D_20260202-171900
```

## Removing a Submodule

If you need to remove a submodule:

```bash
# 1. Remove from .gitmodules
git config -f .gitmodules --remove-section submodule.stac-mjx

# 2. Remove from .git/config
git config -f .git/config --remove-section submodule.stac-mjx

# 3. Remove from git index
git rm --cached stac-mjx

# 4. Remove directory
rm -rf stac-mjx

# 5. Remove from .git/modules
rm -rf .git/modules/stac-mjx

# 6. Commit the changes
git commit -m "Remove stac-mjx submodule"
```

## Resources

- [Git Submodules Documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [GitHub Submodules Guide](https://github.blog/2016-02-01-working-with-submodules/)
