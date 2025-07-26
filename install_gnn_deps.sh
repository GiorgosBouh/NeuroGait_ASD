#!/bin/bash
# Script to install PyTorch and PyTorch Geometric with proper CUDA support

echo "🔧 Installing PyTorch and PyTorch Geometric dependencies..."
echo "=================================================="

# Function to check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ CUDA detected"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
        return 0
    else
        echo "❌ No CUDA detected - will install CPU version"
        return 1
    fi
}

# Check Python version
echo "🐍 Python version:"
python --version

# Check for CUDA
echo ""
echo "🔍 Checking for CUDA..."
if check_cuda; then
    # CUDA is available
    echo ""
    echo "📦 Installing PyTorch with CUDA support..."
    
    # Get CUDA version
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo "CUDA Version detected: $cuda_version"
    
    # Install PyTorch based on CUDA version
    if [[ "$cuda_version" == "11.8" ]]; then
        echo "Installing PyTorch for CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$cuda_version" == "11.7" ]]; then
        echo "Installing PyTorch for CUDA 11.7..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    elif [[ "$cuda_version" == "12.1" ]]; then
        echo "Installing PyTorch for CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "⚠️  CUDA $cuda_version detected but using CUDA 11.8 wheels (usually compatible)"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
else
    # No CUDA - CPU only
    echo ""
    echo "📦 Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install PyTorch Geometric
echo ""
echo "📦 Installing PyTorch Geometric..."

# Get PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "PyTorch version: $TORCH_VERSION"

# Install PyTorch Geometric and dependencies
pip install torch-geometric

# Install additional dependencies
echo ""
echo "📦 Installing additional dependencies..."
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION%+*}.html

# Install other required packages
echo ""
echo "📦 Installing other required packages..."
pip install networkx neo4j python-dotenv

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "
import torch
import torch_geometric
import networkx

print('✅ PyTorch version:', torch.__version__)
print('✅ PyTorch Geometric version:', torch_geometric.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ CUDA device:', torch.cuda.get_device_name(0))
print('✅ NetworkX version:', networkx.__version__)
"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "💡 To use GNN analysis in neurogait.py:"
echo "   1. Make sure Neo4j is running"
echo "   2. Build the graph: python neurogait_kg_builder.py"
echo "   3. Run analysis: python neurogait.py (choose option 4)"
