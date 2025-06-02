#!/bin/bash

echo "🔄 Restructuring mineral-rights project for optimized pipeline..."

# Create new directory structure
mkdir -p mineral-rights-agent/{architecture,notebooks,prompts,data/{samples,outputs},src}

# Copy sample files (using smaller ones for quick testing)
echo "📁 Moving sample files..."
cp "data/reservs/Brooke DB 74_249.pdf" "mineral-rights-agent/data/samples/reserv_deed_1.pdf" 2>/dev/null || echo "Sample reserv file not found"
cp "data/no-reservs/Butler DB 1895_80 - 4.23.2025.pdf" "mineral-rights-agent/data/samples/no_reserv_deed_1.pdf" 2>/dev/null || echo "Sample no-reserv file not found"

echo "✅ New optimized structure created in mineral-rights-agent/"
echo ""
echo "🚀 Quick Start:"
echo "   cd mineral-rights-agent"
echo "   python src/simulate_pipeline.py --input data/samples/reserv_deed_1.pdf"
echo ""
echo "📊 Explore notebooks:"
echo "   jupyter notebook notebooks/ocr_evaluation.ipynb"
echo "   jupyter notebook notebooks/prompt_experiments.ipynb"