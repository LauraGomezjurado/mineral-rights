#!/bin/bash

echo "Setting up OCR experiment environment..."

# Install system dependencies (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing Tesseract on macOS..."
    brew install tesseract
    brew install poppler  # for pdf2image
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p experiments/ocr_comparison/outputs
mkdir -p results/ocr_benchmarks

echo "Setup complete! To run the experiment:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run: python experiments/ocr_comparison/run_experiment.py --pdf_dir data/no-reservs" 