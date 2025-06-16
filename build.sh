#!/bin/bash

echo "🚀 Starting Render build process..."

# Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Clear any cached packages
echo "🧹 Clearing pip cache..."
pip cache purge

# Uninstall anthropic if it exists (to avoid version conflicts)
echo "🔄 Removing any existing anthropic installation..."
pip uninstall -y anthropic || true

# Install exact version of anthropic first
echo "📥 Installing anthropic==0.25.0..."
pip install anthropic==0.25.0

# Install other requirements
echo "📥 Installing remaining requirements..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "import anthropic; print(f'Anthropic version: {anthropic.__version__}')"
python -c "import flask; print(f'Flask version: {flask.__version__}')"
python -c "import fitz; print('PyMuPDF imported successfully')"

echo "🎉 Build completed successfully!" 