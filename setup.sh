#!/bin/bash

set -e  # Exit on any error

echo "🚀 Starting FitDiT setup..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install wget if not available
if ! command_exists wget; then
    echo "📦 Installing wget..."
    sudo yum install -y wget
fi

# Check if conda is already installed
if ! command_exists conda; then
    echo "🐍 Installing Miniconda..."
    
    # Download Miniconda installer
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # Install Miniconda silently
    bash miniconda.sh -b -p $HOME/miniconda3
    
    # Clean up installer
    rm miniconda.sh
    
    # Initialize conda for bash and zsh
    $HOME/miniconda3/bin/conda init bash
    $HOME/miniconda3/bin/conda init zsh 2>/dev/null || true
    
    # Add conda to current PATH
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    echo "✅ Miniconda installed successfully"
else
    echo "✅ Conda already installed"
fi

# Source conda setup to make it available in current session
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Ensure conda is in PATH
if ! command_exists conda; then
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

# Create conda environment if it doesn't exist
echo "🔧 Setting up conda environment..."
if conda env list | grep -q "^fitdit "; then
    echo "✅ Environment 'fitdit' already exists"
else
    echo "📦 Creating conda environment 'fitdit'..."
    conda create -n fitdit python=3.10 -y
fi

# Activate environment
echo "🔄 Activating conda environment..."
conda activate fitdit

# Install Python requirements
echo "📦 Installing Python requirements..."
pip install -r requirements.txt

# Install git-lfs
echo "📦 Installing git-lfs..."
if ! command_exists git-lfs; then
    sudo yum install -y git-lfs
    git lfs install
else
    echo "✅ git-lfs already installed"
fi

# Clone models if directory doesn't exist
if [ ! -d "models" ]; then
    echo "📥 Cloning FitDiT models..."
    git clone https://huggingface.co/BoyuanJiang/FitDiT models/
else
    echo "✅ Models directory already exists"
fi

# Add convenience alias to shell profiles
echo "🔧 Setting up shell convenience..."
ALIAS_LINE="alias fitdit='conda activate fitdit'"

# Add to .bashrc if it exists
if [ -f "$HOME/.bashrc" ]; then
    if ! grep -q "alias fitdit=" "$HOME/.bashrc"; then
        echo "" >> "$HOME/.bashrc"
        echo "# FitDiT convenience alias" >> "$HOME/.bashrc"
        echo "$ALIAS_LINE" >> "$HOME/.bashrc"
        echo "✅ Added 'fitdit' alias to .bashrc"
    fi
fi

# Add to .zshrc if it exists
if [ -f "$HOME/.zshrc" ]; then
    if ! grep -q "alias fitdit=" "$HOME/.zshrc"; then
        echo "" >> "$HOME/.zshrc"
        echo "# FitDiT convenience alias" >> "$HOME/.zshrc"
        echo "$ALIAS_LINE" >> "$HOME/.zshrc"
        echo "✅ Added 'fitdit' alias to .zshrc"
    fi
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Conda is now initialized in your shell. In new terminal sessions:"
echo "  • Conda will be available automatically"
echo "  • Use 'fitdit' command to quickly activate the environment"
echo "  • Or use 'conda activate fitdit'"
echo ""
echo "Current session: FitDiT environment is already active!"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"