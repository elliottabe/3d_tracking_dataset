#!/bin/bash

# Script to set up conda environment to unset LD_LIBRARY_PATH
# Usage: ./setup_conda_unset_ld_library_path.sh <environment_name>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <environment_name>"
    echo "Example: $0 my_env"
    exit 1
fi

ENV_NAME="$1"

# Find conda base directory
if [ -n "$CONDA_PREFIX" ]; then
    CONDA_BASE=$(dirname $(dirname $CONDA_PREFIX))
elif [ -n "$CONDA_EXE" ]; then
    CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
else
    # Try common locations
    if [ -d "$HOME/miniconda3" ]; then
        CONDA_BASE="$HOME/miniconda3"
    elif [ -d "$HOME/anaconda3" ]; then
        CONDA_BASE="$HOME/anaconda3"
    else
        echo "Error: Could not find conda installation"
        exit 1
    fi
fi

ENV_PATH="$CONDA_BASE/envs/$ENV_NAME"

# Check if environment exists
if [ ! -d "$ENV_PATH" ]; then
    echo "Error: Environment '$ENV_NAME' not found at $ENV_PATH"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo "Setting up LD_LIBRARY_PATH unset for environment: $ENV_NAME"
echo "Environment path: $ENV_PATH"

# Create activation script directory
ACTIVATE_DIR="$ENV_PATH/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"

# Create deactivation script directory
DEACTIVATE_DIR="$ENV_PATH/etc/conda/deactivate.d"
mkdir -p "$DEACTIVATE_DIR"

# Create activation script
cat > "$ACTIVATE_DIR/env_vars.sh" << 'EOF'
#!/bin/sh

# Save the current LD_LIBRARY_PATH
export OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

# Unset LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
EOF

# Create deactivation script
cat > "$DEACTIVATE_DIR/env_vars.sh" << 'EOF'
#!/bin/sh

# Restore the original LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
unset OLD_LD_LIBRARY_PATH
EOF

# Make scripts executable
chmod +x "$ACTIVATE_DIR/env_vars.sh"
chmod +x "$DEACTIVATE_DIR/env_vars.sh"

echo "✓ Created conda activation/deactivation scripts"

# Update Jupyter kernel if it exists
KERNEL_JSON="$ENV_PATH/share/jupyter/kernels/python3/kernel.json"
if [ -f "$KERNEL_JSON" ]; then
    echo "Found Jupyter kernel, updating configuration..."
    
    # Backup original kernel.json
    cp "$KERNEL_JSON" "$KERNEL_JSON.backup"
    
    # Check if kernel.json already has env section
    if grep -q '"env"' "$KERNEL_JSON"; then
        echo "Warning: kernel.json already has an 'env' section. Please update manually."
        echo "Backup saved at: $KERNEL_JSON.backup"
    else
        # Add env section to kernel.json
        python3 << 'PYTHON_EOF'
import json
import sys

kernel_json_path = sys.argv[1]

with open(kernel_json_path, 'r') as f:
    config = json.load(f)

# Add env section to unset LD_LIBRARY_PATH
config['env'] = {'LD_LIBRARY_PATH': ''}

with open(kernel_json_path, 'w') as f:
    json.dump(config, f, indent=1)

print("✓ Updated Jupyter kernel configuration")
PYTHON_EOF
        python3 -c "import json, sys; config = json.load(open('$KERNEL_JSON')); config['env'] = {'LD_LIBRARY_PATH': ''}; json.dump(config, open('$KERNEL_JSON', 'w'), indent=1)"
        echo "✓ Updated Jupyter kernel configuration"
    fi
else
    echo "No Jupyter kernel found (this is fine if not using Jupyter)"
fi

echo ""
echo "Setup complete! Changes will take effect when you:"
echo "1. Activate the environment: conda activate $ENV_NAME"
echo "2. Restart any running Jupyter kernels (if using notebooks)"
echo ""
echo "To test, run:"
echo "  conda activate $ENV_NAME"
echo "  echo \"LD_LIBRARY_PATH=\$LD_LIBRARY_PATH\""
